# ruff: noqa
import inspect

import numpy as np
import torch
from PIL import Image

dev = "cuda"
loaded = torch.hub.load("facebookresearch/vjepa2", "vjepa2_1_vit_base_384", trust_repo=True)
enc = loaded[0] if isinstance(loaded, (tuple, list)) else loaded
print(
    "loaded:",
    type(loaded).__name__,
    "| encoder:",
    type(enc).__name__,
    "| embed_dim:",
    getattr(enc, "embed_dim", None),
    flush=True,
)
enc = enc.to(dev).eval()
try:
    print("fwd sig:", str(inspect.signature(enc.forward))[:160], flush=True)
except:
    pass
pp = None
try:
    pp = torch.hub.load("facebookresearch/vjepa2", "vjepa2_preprocessor", trust_repo=True)
    print(
        "pp:",
        type(pp).__name__,
        "| sig:",
        str(inspect.signature(pp.__call__ if hasattr(pp, "__call__") else pp))[:120],
        flush=True,
    )
except Exception as e:
    print("pp err:", repr(e)[:120], flush=True)
img = (np.random.rand(480, 640, 3) * 255).astype(np.uint8)


def totensor(x):
    return x.to(dev) if torch.is_tensor(x) else x


attempts = []
if pp is not None:
    attempts += [
        ("pp([PIL])->enc", lambda: enc(totensor(pp([Image.fromarray(img)])))),
        ("pp(PIL)->enc", lambda: enc(totensor(pp(Image.fromarray(img))))),
    ]
attempts += [
    (
        "vid(1,3,1,384,384)",
        lambda: enc(
            torch.from_numpy(np.array(Image.fromarray(img).resize((384, 384))))
            .permute(2, 0, 1)[None, :, None]
            .float()
            .to(dev)
            / 255
        ),
    )
]
for desc, fn in attempts:
    try:
        with torch.no_grad():
            out = fn()
        o = out[0] if isinstance(out, (list, tuple)) else out
        print(f"OK [{desc}] -> {tuple(o.shape)}", flush=True)
        break
    except Exception as e:
        print(f"x[{desc}] {repr(e)[:150]}", flush=True)
