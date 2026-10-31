"""Streaming GoCA-inspired CA constructions for the locked SG24 experiments.

The official GoCA implementation is https://github.com/Darkbblue/goca . We
reuse its projected-output dot-product head score and self/pseudo-global
affinity dot-product layer score, adapted to our 31-call SD1.5 trajectory.
"""
from __future__ import annotations

import math
import re
from collections import defaultdict

import torch
import torch.nn.functional as F

from src.attention_capture.memory import CompactAttentionRecorder
from src.segmentation import controlled_cv as cv

STOP = frozenset("a an the and or but if of to in on at by for from with as is are was were be been being this that these those it its into over under about around through during before after between against without within above below up down out off than then so not no nor do does did have has had i you he she we they me him her us them my your his our their there here who which what where when why how".split())
CONTENT = re.compile(r"[a-z0-9]+", re.I)
REF_SUFFIX = "up_blocks.2.attentions.1.transformer_blocks.0.attn2"


def content_positions(tokenizer, prompt: str, target: tuple[int, ...]) -> tuple[int, ...]:
    """Select content pieces by fast-tokenizer word offsets; keep all target pieces."""
    selected = set(target)
    if getattr(tokenizer, "is_fast", False):
        encoded = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length,
                            truncation=True, return_offsets_mapping=True)
        for index, (start, end) in enumerate(encoded["offset_mapping"]):
            word = prompt[start:end].strip().lower()
            if start != end and CONTENT.search(word) and word not in STOP:
                selected.add(index)
    else:
        # SD1.5 commonly ships the slow CLIPTokenizer. Its BPE pre-tokenizer
        # splits words and punctuation independently; reconstruct piece spans.
        full_raw = tokenizer(prompt, add_special_tokens=False, truncation=False)["input_ids"]
        cursor = 1  # CLIP beginning-of-text token.
        for match in re.finditer(r"\w+|[^\w\s]", prompt):
            word = match.group(0).lower()
            pieces = tokenizer(word, add_special_tokens=False)["input_ids"]
            if CONTENT.search(word) and word not in STOP:
                selected.update(range(cursor, cursor + len(pieces)))
            cursor += len(pieces)
        if cursor != len(full_raw) + 1:
            raise ValueError("slow CLIP tokenizer content-piece alignment failed")
        selected.intersection_update(range(1, min(len(full_raw), tokenizer.model_max_length - 2) + 1))
    return tuple(sorted(selected))


def head_weights(prob: torch.Tensor, value: torch.Tensor, out_weight: torch.Tensor) -> torch.Tensor:
    """GoCA SD1.5 dot-product *without clamp*; uniform zero-output fallback."""
    heads, pixels, _ = prob.shape
    width = value.shape[-1]
    if value.shape[0] != heads or out_weight.shape[1] != heads * width:
        raise ValueError("incompatible CA V/output projection shapes")
    post = prob @ value.float()
    projected = torch.stack([post[h] @ out_weight[:, h*width:(h+1)*width].T.float()
                             for h in range(heads)], dim=0)
    total = projected.sum(0)
    scores = (projected * total[None]).sum(-1)
    denominator = scores.sum(0, keepdim=True)
    return torch.where(denominator > 1e-20, scores / denominator.clamp_min(1e-20),
                       torch.full_like(scores, 1.0 / heads))


def affinity_similarity(query: torch.Tensor, key: torch.Tensor, reference: torch.Tensor,
                        *, both: bool = False) -> torch.Tensor:
    """Dot product of pseudo-global F affinity and paired mean-head SA affinity.

    Like GoCA, resize key axes bilinearly and query axes by nearest-neighbor.
    Heads are streamed so a 64x64 module never retains eight 4096x4096 maps.

    The resize path omits GoCA's second ``permute(2, 3, 0, 1)``, so a resized
    matrix is indexed [key, query] and the score is <M^T, P> (protocol Section
    107). That historical score is kept as the default. ``both=True`` returns
    ``(as implemented, GoCA orientation <M, P>)``; they coincide when no resize
    is needed.
    """
    ref_hw = math.isqrt(reference.shape[0])
    hw = math.isqrt(query.shape[-2])
    if ref_hw * ref_hw != reference.shape[0] or hw * hw != query.shape[-2]:
        raise ValueError("non-square spatial attention")
    similarity = torch.zeros((), device=reference.device)
    goca = torch.zeros((), device=reference.device)
    for h in range(query.shape[0]):
        q, k = query[h].to(reference.device).float(), key[h].to(reference.device).float()
        matrix = torch.softmax(q @ k.T / math.sqrt(q.shape[-1]), dim=-1)
        if hw != ref_hw:
            matrix = F.interpolate(matrix.reshape(1, hw*hw, hw, hw),
                                   size=(ref_hw, ref_hw), mode="bilinear", align_corners=False)
            matrix = matrix.reshape(hw, hw, ref_hw, ref_hw).permute(2, 3, 0, 1)
            matrix = F.interpolate(matrix.reshape(1, ref_hw*ref_hw, hw, hw),
                                   size=(ref_hw, ref_hw), mode="nearest")
            matrix = matrix.reshape(ref_hw*ref_hw, ref_hw*ref_hw)
            if both:
                goca += (matrix * reference.T).sum()  # <M^T,P^T> = <M,P>
        elif both:
            goca += (matrix * reference).sum()
        similarity += (matrix * reference).sum()
        del q, k, matrix
    if both:
        return similarity / query.shape[0], goca / query.shape[0]
    return similarity / query.shape[0]


class SG24Recorder(CompactAttentionRecorder):
    """Reduces each diffusion call to five CA seeds; keeps only selected SA Q/K."""

    pipe = None
    ca_modules: tuple[str, ...] = ()
    selected_modules: tuple[str, ...] = ()
    selected_sa = ""
    experiment = "a"
    enabled = ("A0", "A1", "A2", "A3", "A4")
    content: tuple[int, ...] = ()
    # Also accumulate "selected/A3G": A3 with GoCA-oriented layer weights.
    dual_orientation = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.target = self.target_positions
        if self.attribution_hidden_states is not None:
            prompt = self.attribution_prompt
            ids = self.pipe.tokenizer(prompt, padding="max_length",
                                      max_length=self.pipe.tokenizer.model_max_length,
                                      truncation=True, return_tensors="pt").input_ids.to(self.pipe.device)
            self.attribution_hidden_states = self.pipe.text_encoder(ids)[0]
        self.pending_ca = {}
        self.pending_self = {}
        self.reference_query = None
        self.sa_qk = []
        self.sums = defaultdict(lambda: torch.zeros((64, 64), device=self.pipe.device))
        self.weights = []
        self.content_mass = []
        self.content_mass_map = torch.zeros((64, 64), device=self.pipe.device)
        self.target_top_map = torch.zeros((64, 64), device=self.pipe.device)

    def record(self, module_name, kind, query, key, value, attention_mask, scale, attribution_key=None):
        step = self.call_counts[module_name]
        self.call_counts[module_name] += 1
        self.step_tensors.setdefault(step, {})
        spatial = math.isqrt(query.shape[-2])
        self.module_metadata[module_name].update(spatial_height=spatial, spatial_width=spatial,
                                                 spatial_tokens=query.shape[-2], head_dim=query.shape[-1])
        if kind == "self":
            if module_name == self.selected_sa and 10 <= step <= 20:
                self.sa_qk.append((query[1].detach().half().clone(), key[1].detach().half().clone()))
            paired = module_name[:-5] + "attn2"
            if paired in self.ca_modules:
                self.pending_self[paired] = (query[1].detach().half().clone(), key[1].detach().half().clone())
            return
        if module_name == REF_SUFFIX:
            self.reference_query = query[1].detach().permute(1, 0, 2).reshape(query.shape[-2], -1).float()
        if module_name not in self.ca_modules:
            return
        if attribution_key is not None:
            # COCO's attribution prompt is encoded separately. V must come from
            # the same text embedding as K or head-contribution weights are invalid.
            key = torch.stack((key[0], attribution_key[0]))
            module = self.pipe.unet.get_submodule(module_name)
            states = self.attribution_hidden_states
            if module.norm_cross:
                states = module.norm_encoder_hidden_states(states)
            v = module.to_v(states).reshape(1, states.shape[1], module.heads, -1)
            v = v.permute(0, 2, 1, 3)[0]
            attention_mask = None
        else:
            v = value[1]
        scores = query[1].float() @ key[1].float().transpose(-1, -2) * scale
        if attention_mask is not None:
            scores += attention_mask[1].float()
        prob = scores.softmax(-1)
        target = prob[..., list(self.target)].mean(-1)
        mass = prob[..., list(self.content)].sum(-1).clamp_min(1e-20)
        highest = (prob[..., list(self.target)].amax(-1) >=
                   prob[..., list(self.content)].amax(-1)).float().mean(0)
        for accumulator, observation in ((self.content_mass_map, mass.mean(0)),
                                         (self.target_top_map, highest)):
            resized = F.interpolate(observation.reshape(1,1,spatial,spatial),
                                    size=(64,64), mode="bilinear", align_corners=False)[0,0]
            accumulator += resized / (31 * len(self.ca_modules))
        if set(self.content) == set(self.target):
            # COCO class-only attribution has no competing content token.
            # Enforce the algebraic constant exactly, avoiding min-max
            # amplification of float32 roundoff into a fake spatial map.
            scaled = torch.full_like(target, 1.0 / len(self.target))
        else:
            scaled = (prob / mass[..., None])[..., list(self.target)].mean(-1)
        module = self.pipe.unet.get_submodule(module_name)
        hw = head_weights(prob, v, module.to_out[0].weight)
        maps = {
            "A0": target.half().float().mean(0),
            "A1": scaled.mean(0),
            "A2": (hw * target).sum(0),
            "A3": target.mean(0),
            "A4": (hw * scaled).sum(0),
        }
        if set(self.content) == set(self.target):
            maps["A4"] = torch.full_like(maps["A4"], 1.0 / len(self.target))
        self.pending_ca[module_name] = {n: x.reshape(1, 1, spatial, spatial).detach() for n,x in maps.items()}
        self.content_mass.append(float(mass.mean().item()))

    def flush_step(self, step, timestep, latent):
        super().flush_step(step, timestep, latent)
        if set(self.pending_ca) != set(self.ca_modules) or self.reference_query is None:
            raise ValueError("SG24 missed CA module or reference feature")
        ref = self.reference_query
        pseudo = torch.softmax(ref @ ref.T / math.sqrt(ref.shape[-1]), dim=-1)
        similarities, goca = {}, {}
        for module in self.ca_modules:
            if module not in self.pending_self:
                raise ValueError(f"missing paired self attention for {module}")
            q, k = self.pending_self[module]
            if self.dual_orientation:
                similarities[module], goca[module] = affinity_similarity(q, k, pseudo, both=True)
            else:
                similarities[module] = affinity_similarity(q, k, pseudo)
        if self.dual_orientation:
            selected = torch.stack([goca[m] for m in self.selected_modules]).clamp_min(0)
            weights = selected / selected.sum().clamp_min(1e-20)
            if float(selected.sum()) <= 1e-20:
                weights.fill_(1 / len(self.selected_modules))
            self.weights.append((step, "selected-goca", weights.cpu().numpy().copy()))
            for i, module in enumerate(self.selected_modules):
                up = F.interpolate(self.pending_ca[module]["A3"], size=(64,64),
                                   mode="bilinear", align_corners=False)[0,0]
                self.sums["selected/A3G"] += up * weights[i] / 31
        for modules, label in ((self.selected_modules, "selected"),
                               (self.ca_modules, "expanded")):
            if label == "expanded" and self.experiment != "b":
                continue
            selected = torch.stack([similarities[m] for m in modules]).clamp_min(0)
            weights = selected / selected.sum().clamp_min(1e-20)
            if float(selected.sum()) <= 1e-20:
                weights.fill_(1 / len(modules))
            self.weights.append((step, label, weights.cpu().numpy().copy()))
            for name in self.enabled:
                if self.experiment == "b" and name not in self.enabled:
                    continue
                use_weight = name in ("A3", "A4")
                for i, module in enumerate(modules):
                    source = self.pending_ca[module][name]
                    up = F.interpolate(source, size=(64,64), mode="bilinear", align_corners=False)[0,0]
                    self.sums[f"{label}/{name}"] += up * (weights[i] if use_weight else 1/len(modules)) / 31
        self.pending_ca.clear(); self.pending_self.clear(); self.reference_query = None
        self.steps[-1].clear()

    def seeds(self):
        return {name: cv.normalize_batch(tensor[None], "minmax")[0] for name,tensor in self.sums.items()}

    def clear(self):
        super().clear()
        self.pending_ca.clear(); self.pending_self.clear(); self.sa_qk.clear()
        self.sums.clear(); self.weights.clear(); self.content_mass.clear()
        self.content_mass_map = None; self.target_top_map = None
