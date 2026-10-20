"""Explicit, evidence-bound recovery of complete cross-hardware result packages.

The default finalizers still enforce their historical comparison. Recovery
requires independently reproduced GPU samples, complete structural audits,
and SHA256 matches for every original metric/diagnostic shard.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare_recovery_evidence(audit_dir: Path):
    cpu_path=audit_dir/'count_audit.json'
    live_path=audit_dir/'v100/audit.json'
    cpu=json.loads(cpu_path.read_text());live=json.loads(live_path.read_text())
    assert 'V100' in live['environment']['gpu']
    samples=live['samples']
    assert len(samples)==4 and len({r['image_id'] for r in samples})==4
    assert sum('count_delta_dense_CRF_saved_max' in r for r in samples)==2
    for row in samples:
        assert row['count_delta_sg11_saved_max']==0
        assert row['count_delta_dense_raw_saved_max']==0
        assert max(row['old_new_SG2_max_abs_by_REF_P1_SG2'])<=2e-6
        if 'count_delta_dense_CRF_saved_max' in row:
            assert row['count_delta_dense_CRF_saved_max']==0
        assert (audit_dir/'v100'/row['small_arrays']).is_file()
    for ds,expected in [('voc',490),('coco',707)]:
        assert cpu['comparisons'][ds]['Dense_REF_vs_SG11']['max_fraction']<=.002
        assert cpu['sg11_sidecars'][ds]['images']==expected
        for family in ('SG11','Dense'):
            assert cpu['contracts'][f'{family}/{ds}']['images']==expected
            assert cpu['contracts'][f'{family}/{ds}']['complete_shards']==20
    proof={'kind':'cross_hardware_reference_recovery_v1',
           'status':'within-package V100 comparisons recovered; historical pixel reproduction remains failed',
           'cpu_audit':str(cpu_path.resolve().relative_to(ROOT)), 'cpu_audit_sha256':file_sha256(cpu_path),
           'live_v100_audit':str(live_path.resolve().relative_to(ROOT)), 'live_v100_audit_sha256':file_sha256(live_path),
           'input_sha256':cpu['input_sha256'],
           'scope':'No method, split, threshold-selection or promotion change. No raw measurement rewritten.',
           'review':'Claude independent review required before pipeline or paper promotion.'}
    path=audit_dir/'recovery_evidence.json'
    path.write_text(json.dumps(proof,indent=2)+'\n')
    return path


def verify_recovery_evidence(path: Path | None, directory: Path, family: str):
    if path is None:return None
    proof=json.loads(path.read_text())
    if proof.get('kind')!='cross_hardware_reference_recovery_v1':
        raise ValueError('unrecognized reference recovery evidence')
    for key in ('cpu_audit','live_v100_audit'):
        source=ROOT/proof[key]
        if file_sha256(source)!=proof[f'{key}_sha256']:
            raise ValueError('recovery audit changed; investigate before accepting the package')
    expected=proof['input_sha256'][family]
    actual={str(p.relative_to(directory)):file_sha256(p) for p in sorted(directory.glob('*/shards/*.npz'))}
    if actual!=expected:
        raise ValueError('metric/diagnostic shards differ from audited snapshot')
    return {'evidence':str(path.resolve().relative_to(ROOT)), 'status':proof['status'],
            'raw_shards_sha256_verified':len(actual), 'historical_pixel_reproduction':'failed and retained in report',
            'method_or_selection_changes':False, 'independent_review_required':True}
