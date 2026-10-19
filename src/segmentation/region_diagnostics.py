"""Small atomic SG13 sidecars: no full maps or image archive."""
import json
import os
from pathlib import Path
import numpy as np


class RegionDiagnosticWriter:
    FIELDS = ('quantities', 'scores', 'area64', 'gt_fraction', 'oracle_counts', 'operator_audit_errors')

    def __init__(self, path: Path, spec):
        self.path, self.spec, self.records = path, spec, {}
        if path.exists():
            with np.load(path, allow_pickle=False) as data:
                if json.loads(str(data['spec'])) != spec:
                    raise ValueError('region diagnostic specification differs')
                self.records = {image_id: tuple(data[f][i] for f in self.FIELDS)
                                for i, image_id in enumerate(data['image_ids'].tolist())}

    def add(self, image_id, quantities, scores, area, fraction, oracle_counts, operator_audit_errors=None):
        if image_id in self.records:
            return
        q, s, a, f = np.zeros((32, 4), np.float32), np.zeros((3, 32), np.float32), np.zeros(32, np.int32), np.zeros(32, np.float32)
        k = len(area)
        q[:k], s[:, :k], a[:k], f[:k] = quantities.cpu().numpy(), scores.cpu().numpy(), area.cpu().numpy(), fraction.cpu().numpy()
        self.records[image_id] = (q, s, a, f, oracle_counts, np.zeros(2, np.float32) if operator_audit_errors is None else np.asarray(operator_audit_errors, np.float32))
        ids = sorted(self.records)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(self.path.name+'.tmp.npz')
        np.savez_compressed(temporary, image_ids=np.asarray(ids), spec=np.array(json.dumps(self.spec, sort_keys=True)),
                           **{field: np.stack([self.records[i][j] for i in ids]) for j, field in enumerate(self.FIELDS)})
        os.replace(temporary, self.path)
