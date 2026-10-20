"""Atomic small numeric per-image sidecars for streaming experiments."""
import json
import os
from pathlib import Path
import numpy as np


class NumericSidecar:
    def __init__(self, path: Path, settings):
        self.path,self.settings,self.records=path,settings,{}
        self.fields=None
        if path.exists():
            with np.load(path,allow_pickle=False) as data:
                if json.loads(str(data['settings']))!=settings:
                    raise ValueError('diagnostic sidecar settings changed')
                self.fields=tuple(data['fields'].tolist())
                self.records={name:{f:data[f][i] for f in self.fields} for i,name in enumerate(data['image_ids'].tolist())}

    def add(self, image_id, record):
        fields=tuple(sorted(record))
        if self.fields is not None and self.fields!=fields:
            raise ValueError('diagnostic fields changed')
        self.fields=fields
        self.records[image_id]={k:np.asarray(v) for k,v in record.items()}
        ids=sorted(self.records)
        self.path.parent.mkdir(parents=True,exist_ok=True)
        tmp=self.path.with_name(self.path.name+'.tmp.npz')
        np.savez_compressed(tmp,image_ids=np.asarray(ids),fields=np.asarray(fields),
                            settings=np.array(json.dumps(self.settings,sort_keys=True)),
                            **{f:np.stack([self.records[i][f] for i in ids]) for f in fields})
        os.replace(tmp,self.path)
