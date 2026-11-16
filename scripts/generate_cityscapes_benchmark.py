"""Generate resumable, fully recorded candidates from the supplied Cityscapes LoRA."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--weights', type=Path, default=Path('pytorch_lora_weights.safetensors'))
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()

    import torch
    from diffusers import StableDiffusionPipeline
    from PIL.PngImagePlugin import PngInfo

    rows = json.loads(args.plan.read_text())
    assert len({r['image_id'] for r in rows}) == len(rows)
    if args.limit:
        rows = rows[:args.limit]
    root = args.output.resolve()
    images = root / 'candidates'
    images.mkdir(parents=True, exist_ok=True)
    (root / 'logs').mkdir(exist_ok=True)
    todo = []
    for row in rows:
        target = images / (row['image_id'] + '.png')
        metadata = images / (row['image_id'] + '.json')
        if target.exists() and metadata.exists():
            old = json.loads(metadata.read_text())
            assert all(old[key] == value for key, value in row.items()), row['image_id']
            continue
        assert not target.exists() and not metadata.exists(), f'Partial image needs review: {target}'
        todo.append(row)
    print(f'{len(rows)} planned; {len(todo)} to generate', flush=True)
    if not todo:
        return
    assert torch.cuda.is_available()
    model = 'stable-diffusion-v1-5/stable-diffusion-v1-5'
    pipe = StableDiffusionPipeline.from_pretrained(model, torch_dtype=torch.float16,
                                                  local_files_only=True).to('cuda')
    pipe.set_progress_bar_config(disable=True)
    weights = args.weights.resolve()
    pipe.load_lora_weights(str(weights.parent), weight_name=weights.name,
                           adapter_name='cityscapes', local_files_only=True)
    pipe.set_adapters('cityscapes', adapter_weights=1.0)
    assert pipe.get_active_adapters() == ['cityscapes']
    layers = [m for m in pipe.unet.modules() if hasattr(m, 'lora_A') and 'cityscapes' in m.lora_A]
    assert layers, 'LoRA was not loaded'
    b_norm = sum(float(m.lora_B['cityscapes'].weight.float().norm()) for m in layers)
    assert b_norm > 0, 'LoRA has no effective nonzero up-projection'
    hf = Path(os.environ['HF_HOME']) / 'hub/models--stable-diffusion-v1-5--stable-diffusion-v1-5/refs/main'
    runtime = {'base_model': model, 'base_revision': hf.read_text().strip(),
               'lora_path': str(weights), 'lora_sha256': digest(weights),
               'adapter': 'cityscapes', 'lora_layers': len(layers),
               'lora_ranks': sorted({m.lora_A['cityscapes'].weight.shape[0] for m in layers}),
               'lora_up_projection_norm_sum': b_norm, 'dtype': str(pipe.unet.dtype),
               'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__,
               'diffusers': importlib.metadata.version('diffusers'),
               'transformers': importlib.metadata.version('transformers'),
               'peft': importlib.metadata.version('peft'),
               'scheduler': type(pipe.scheduler).__name__, 'scheduler_config': dict(pipe.scheduler.config),
               'plan': str(args.plan.resolve()), 'plan_sha256': digest(args.plan),
               'safety_checker_enabled': pipe.safety_checker is not None}
    runtime_name = f'runtime_{args.plan.stem}_{os.getpid()}.json'
    (root / 'logs' / runtime_name).write_text(json.dumps(runtime, indent=2) + '\n')
    print(f'Loaded {len(layers)} rank-{runtime["lora_ranks"]} LoRA layers; hash {runtime["lora_sha256"]}', flush=True)
    groups = defaultdict(list)
    for row in todo:
        groups[(row['width'], row['height'], row['steps'], row['guidance_scale'], row['lora_scale'])].append(row)
    done = 0
    for (width, height, steps, cfg, scale), group in groups.items():
        pipe.set_adapters('cityscapes', adapter_weights=scale)
        for start in range(0, len(group), args.batch_size):
            batch = group[start:start + args.batch_size]
            generators = [torch.Generator(device='cuda').manual_seed(row['seed']) for row in batch]
            torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            with torch.inference_mode():
                result = pipe(prompt=[row['prompt'] for row in batch],
                              negative_prompt=[row['negative_prompt'] for row in batch],
                              generator=generators, width=width, height=height,
                              num_inference_steps=steps, guidance_scale=cfg)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
            for i, (row, image) in enumerate(zip(batch, result.images)):
                record = dict(row, **runtime, runtime_file=runtime_name, batch_size=len(batch),
                              batch_position=i, batch_image_ids=[r['image_id'] for r in batch],
                              nsfw_flagged=bool(result.nsfw_content_detected[i]) if result.nsfw_content_detected is not None else None,
                              elapsed_batch_seconds=elapsed, seconds_per_image=elapsed / len(batch),
                              peak_gpu_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                              image_file=f'candidates/{row["image_id"]}.png')
                record['prompt_token_ids'] = pipe.tokenizer(
                    row['prompt'], padding='max_length', max_length=77, truncation=True)['input_ids']
                record['negative_prompt_token_ids'] = pipe.tokenizer(
                    row['negative_prompt'], padding='max_length', max_length=77, truncation=True)['input_ids']
                record['prompt_untruncated_tokens'] = len(pipe.tokenizer(row['prompt'], truncation=False)['input_ids'])
                record['negative_prompt_untruncated_tokens'] = len(pipe.tokenizer(row['negative_prompt'], truncation=False)['input_ids'])
                record['pixel_sha256'] = hashlib.sha256(image.tobytes()).hexdigest()
                info = PngInfo()
                info.add_text('generation', json.dumps(record))
                target = images / (row['image_id'] + '.png')
                tmp = target.with_suffix('.tmp.png')
                image.save(tmp, pnginfo=info)
                tmp.replace(target)
                (images / (row['image_id'] + '.json')).write_text(json.dumps(record, indent=2) + '\n')
                with (root / 'generation.jsonl').open('a') as handle:
                    handle.write(json.dumps(record) + '\n')
                done += 1
                print(f'{done}/{len(todo)} {row["image_id"]} {row["family"]} {width}x{height} '
                      f'{elapsed / len(batch):.2f}s/image flagged={record["nsfw_flagged"]}', flush=True)


if __name__ == '__main__':
    main()
