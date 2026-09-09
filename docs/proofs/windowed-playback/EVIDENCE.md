# Windowed playback, driven on the rig

Every number here was taken from a browser on a workstation against a GUI
served by fc500t over the tailnet, at the branch tip, with one GUI server on
that host and an isolated config and window cache so nothing of the operator's
was read or written. The JSON under `measurements/` is what each claim is read
from; nothing here is quoted from a panel's own report.

Round-trip time to the rig measured 250 ms during these runs.

## Playback keeps time at every speed the transport offers

Four cameras, 30 fps, 174-frame episode, after the rung ladder had settled.
`clock` is the player's own clock sampled against the wall clock; `redraws` is
how often the canvases were repainted.

| speed | clock | asked | ratio | redraws/s | stalls |
| ----- | ----- | ----- | ----- | --------- | ------ |
| 0.25x | 7.6   | 7.5   | 1.01  | 7.6       | 0      |
| 0.5x  | 15.0  | 15.0  | 1.00  | 13.9      | 0      |
| 1x    | 29.8  | 30.0  | 0.99  | 15.8      | 0      |
| 1.5x  | 44.3  | 45.0  | 0.98  | 15.7      | 0      |
| 2x    | 58.0  | 60.0  | 0.97  | 15.6      | 0      |

First picture arrived 1.46 s after the episode was selected.

`recordings/speeds-0.25x-to-2x.mp4` is that run: four cameras and the
visualizer playing while the selector is moved through every setting.

The two columns differ on purpose. The player keeps time and skips frames when
it cannot repaint fast enough, so counting distinct frames drawn reads a
correct real-time playback as half speed — `measurements/clock-vs-redraw.json`
records the same run as 29.5 clock fps and 16.7 redraws per second, 1.77 source
frames per redraw. Above 0.5x the picture is drawing fewer frames than the
source holds; it is not running slow.

## The ladder settles rather than draining

Two minutes at 1x: the rung stepped down once, from 320 to 160, and then held.
The buffer stopped moving at 5.8 s, which is the whole episode, and no further
windows were fetched. `measurements/ladder-settles.json`.

## Saved masks and the effects on them

On a dataset with current-name mask columns, both mask lanes render, the
windows are fetched composited, and each of the four treatments changes the
picture on the camera carrying the mask. All three of tint, blur and random
differ from each other and from none, compared as canvas fingerprints rather
than by trusting the control. `measurements/masks-and-effects.json`, and the
`states/2..5` stills.

## SAM builds a mask column from nothing

A dataset with no mask column at all: the Inspector offered **Segment across
all episodes…** rather than Fill gaps, the job took all three cameras, and the
pass created the column and filled it.

| camera      | frames with masks |
| ----------- | ----------------- |
| top         | 448 / 528         |
| right_wrist | 270 / 528         |
| left_wrist  | 0 / 528           |

`left_wrist` is zero because the ball is not in that camera's view, which the
apply run below independently agrees with.

The batch worker spawned and the dataset was rebound after its in-place save
40.1 s later: 528 frames across 3 cameras is about 40 camera-frames a second.

## Apply-and-play, through the controls an operator uses

Armed the checkbox, pressed **Play**, let the run segment, pressed **Save
Changes**. `right_wrist` went from 270 to 272 frames on disk, and the server
logged the staging that produced it (`MASK_RUN_STAGE ep=0 flushed=1 frames=1
labels=['ball']`, then `frames=2`). The run fills only where the label is
absent, so a camera whose frames already carry the label gains nothing, which
is what two earlier runs on `top` and `left_wrist` showed.

## What has a recording and what does not

Speeds, the effects, the SAM first pass and apply-and-play each have a video
here. The episode switch, the dataset switch and trims and deletions are
covered by tests rather than by a rig recording -- `test_window_playback_combinations.py`
for the first two, `test_data_tab_edits_playback.py` for the third -- because
each asserts on pixels the eye cannot check anyway: which dataset's grey band
is on the tile, and whether a painted frame lies inside the trim.

## What is not established here

The live path apply-and-play rides is far slower than the batch path: about
0.76 frames a second on one camera against roughly 40 camera-frames a second
for the batch worker, on the same rig, model, dataset and session. The gap is
measured; its cause is not. The live path adds a serialized round trip per
frame — publish a frame, wait for that frame's masks — and does not batch, but
the split between model time and round-trip time has not been measured, and the
server records only the first overlay served per camera rather than per-frame
timings.
