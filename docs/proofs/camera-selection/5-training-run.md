# Camera selection, measured on real training runs

Two `lerobot-train` ACT runs over the same dataset, identical except for
`--dataset.cameras`. The dataset is synthetic and disposable — three cameras
(`top_l`, `top_r`, `wrist`), 2 episodes, 80 frames, built for this check so no
recorded data is involved.

```
lerobot-train --dataset.repo_id=demo/camera_selection --policy.type=act \
              --steps=20 --batch_size=2 --num_workers=0 [--dataset.cameras='[top_l]']
```

## 1. What the run reports

```
restricted    INFO  datasets/factory.py:97  Cameras: using 1 of 3 (top_l)
unrestricted  INFO  datasets/factory.py:97  Cameras: using 3 of 3 (top_l, top_r, wrist)
```

Logged on every run, not only when restricted, so a log always answers "what did
this train on". Same wording as the line HVLA's own trainer prints.

## 2. What the checkpoint records

`checkpoints/000020/pretrained_model/config.json`, `input_features`:

| run          | visual input features                                                                  |
| ------------ | -------------------------------------------------------------------------------------- |
| restricted   | `['observation.images.top_l']`                                                         |
| unrestricted | `['observation.images.top_l', 'observation.images.top_r', 'observation.images.wrist']` |

This is the artifact inference reads, so a policy trained on one eye asks the
robot for one eye.

## 3. What the run actually opened

The claim that unselected cameras are _not decoded_ — rather than decoded and
discarded — measured with `strace -f -e trace=openat` on the real training
process, counting opens of each camera's `.mp4`:

| camera  | restricted | unrestricted |
| ------- | ---------- | ------------ |
| `top_l` | 1          | 1            |
| `top_r` | **0**      | 1            |
| `wrist` | **0**      | 1            |

One open per file rather than one per frame because the decoder is cached per
file; the point is the zero. The two unselected videos are never opened at all.

## 4. Both runs trained

Neither run is a no-op — 20 steps each, loss falling comparably, so the
restricted run is a real training run and not a silent early exit:

```
restricted    step:20 smpl:40 loss:10.446 grdn:404.710 l1_loss:0.641
unrestricted  step:20 smpl:40 loss:10.458 grdn:404.478 l1_loss:0.654
```
