'''remove the _criterion_kd state dict from the trained checkpoint of student'''

import torch
import sys

src = sys.argv[1]
tgt = sys.argv[2]


ckpt = torch.load(src, map_location='cpu')
res = {}
for k, v in ckpt.items():
    if not k.startswith('_criterion_kd'):
        res[k] = v
torch.save(res, tgt)
