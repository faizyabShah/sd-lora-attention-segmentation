"""Exercise full finalization on synthetic manifests built from live smoke schemas.

All artificial records live in a TemporaryDirectory and are never published.
This checks the report/selection code path, not scientific performance.
"""
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from src.segmentation import controlled_cv as cv
from src.segmentation.experiment_records import NumericSidecar
from scripts.run_coco_cap_transfer import voc_rows,coco_rows
import json


def main():
    with tempfile.TemporaryDirectory(prefix='locked_report_check_') as tmp:
        for stage in ('sg12_crf','sg14','sg15'):
            output=Path(tmp)/stage
            smoke=ROOT/'outputs/segmentation/controlled_attention_cv'/(stage+'_smoke_v1')
            for ds in ('voc','coco'):
                path=sorted((smoke/ds/'shards').glob('class_[0-9][0-9].npz'))[0]
                template=cv.load_results([path])
                diag=NumericSidecar(path.with_suffix('.diagnostics.npz'),template.settings)
                values=diag.records[template.image_ids[0]]
                for shard in range(20):
                    cls,rows=(voc_rows if ds=='voc' else coco_rows)(shard)
                    ids=[r['image_id'] for r in rows]
                    folds=cv.read_folds(ROOT/'configs/voc_sim_folds.csv') if ds=='voc' else {i:0 for i in ids}
                    target=output/ds/'shards'/f'class_{shard:02d}.npz'
                    writer=cv.ShardWriter(target,template.configurations,folds,template.settings,template.thresholds)
                    writer.image_ids=ids;writer.classes=[cls]*len(ids)
                    writer.counts=[template.counts[:,0]]*len(ids);writer.ap=[template.ap[:,0]]*len(ids)
                    writer.finish()
                    sidecar=NumericSidecar(target.with_suffix('.diagnostics.npz'),template.settings)
                    sidecar.records={i:values for i in ids};sidecar.add(ids[0],values)
            subprocess.run([sys.executable,str(ROOT/'scripts/finalize_locked_refinement_package.py'),'--stage',stage,'--output-dir',str(output)],check=True,stdout=subprocess.DEVNULL)
            summary=json.loads((output/'summary.json').read_text())
            assert summary['voc_images']==490 and summary['coco_images']==707
            assert summary['screens'] and summary['forced_cell_oof'] and (output/'CHATGPT_RESULTS.md').is_file()
            assert len(summary['cells'])==len(template.configurations)
            if stage=='sg15':
                shutil.copytree(smoke/'visual_cache',output/'visual_cache')
                subprocess.run(['/home/shafa/miniconda3/envs/ovam_sdxl/bin/python',str(ROOT/'scripts/plot_locked_region_refinements.py'),'--output-dir',str(output)],check=True)
                assert len(list((output/'visuals').glob('*.jpg')))>=2
                assert not list((output/'visual_cache').glob('*.npz'))
            print(f'{stage}: full synthetic finalization and required report fields passed',flush=True)

if __name__=='__main__':main()
