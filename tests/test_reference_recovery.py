"""Recovery must bind both reproduction evidence and every original input."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import json
import tempfile
from src.segmentation import reference_recovery as recovery


def test_recovery_rejects_modified_measurements_and_evidence():
    with tempfile.TemporaryDirectory() as temporary:
        root=Path(temporary)
        old_root=recovery.ROOT;recovery.ROOT=root
        try:
            family='sg11_sg12_observation_reliability'
            directory=root/family
            payload=directory/'voc/shards/class_00.npz';payload.parent.mkdir(parents=True)
            payload.write_bytes(b'original immutable measurements')
            cpu=root/'count_audit.json';cpu.write_text('{}')
            live=root/'live.json';live.write_text('{}')
            proof={'kind':'cross_hardware_reference_recovery_v1','status':'review required',
                   'cpu_audit':'count_audit.json','cpu_audit_sha256':recovery.file_sha256(cpu),
                   'live_v100_audit':'live.json','live_v100_audit_sha256':recovery.file_sha256(live),
                   'input_sha256':{family:{'voc/shards/class_00.npz':recovery.file_sha256(payload)}}}
            path=root/'proof.json';path.write_text(json.dumps(proof))
            assert recovery.verify_recovery_evidence(None,directory,family) is None
            assert recovery.verify_recovery_evidence(path,directory,family)['raw_shards_sha256_verified']==1
            payload.write_bytes(b'changed measurements')
            try:recovery.verify_recovery_evidence(path,directory,family)
            except ValueError:pass
            else:raise AssertionError('changed measurements accepted')
            payload.write_bytes(b'original immutable measurements')
            live.write_text('{"changed":true}')
            try:recovery.verify_recovery_evidence(path,directory,family)
            except ValueError:pass
            else:raise AssertionError('changed live audit accepted')
        finally:recovery.ROOT=old_root
    print('Recovery evidence/input mutation checks passed')


if __name__=='__main__':test_recovery_rejects_modified_measurements_and_evidence()
