# What the TrackEverything demo-map experiment actually did

The source was `~/.cache/huggingface/lerobot/demos/insertion/recording`, demo
episode 0. The replay was
`~/.cache/huggingface/lerobot/demos/flash_insertion_trial_20261020_020-20261020-203310-655401`,
episode 0. This experiment sampled **seven demo frames, 475–540** (a 2.2-second
span at 30 fps) and **ten replay frames, 472–515** (a 1.4-second span). This
was not a complete five-second insertion phase.

![Actual saved TrackEverything points on the seven demo frames](demo-tracked-points.jpg)

Cyan dots are TrackEverything query predictions that also passed the recorded
dowel mask and measured-depth check at that frame. A–E label five query IDs
that pass those checks at **every frame from 475 through 540**, including the
intervening frames not shown. Thirty query IDs pass this strict check; A–E
highlight five of them on the visible upper dowel. Their letters follow the
same saved IDs. These dots were **not** run on the replay.

Persistent ID does not mean exact rigid tracking: from demo frame 475 to 540,
26 of those 30 continuously supported points agree with one rigid transform
within the experiment's 5 mm diagnostic band. Their median change in pairwise
3D separation is 2.8 mm. The lower insertion end is often hidden while the
visible upper points carry this estimate.

```mermaid
flowchart LR
    A["Demo frame 500<br/>five measured reference points"] --> B["TrackEverything IDs in the demo<br/>rigid fit 500 to each demo view"]
    B --> C["Seven candidate demo views<br/>475, 490, 500, 510, 520, 530, 540"]
    C --> D["Independent masked RGB-D<br/>geometry fit for each replay frame"]
    E["Ten recorded replay frames<br/>472–515"] --> D
    D --> F["Choose highest surface score<br/>project five demo reference points"]
```

The five **numbered red points below are not A–E**. They are separate measured
surface reference points chosen in demo frame 500. TrackEverything demo IDs
were used to move those reference points to a candidate demo view. Masked
RGB-D registration (PCA candidates refined by trimmed ICP) then moved them
from that view to a replay frame. The yellow line is the saved replay dowel
mask. No replay TrackEverything tracks, object-to-ring motion trajectory, or
robot action was produced.

![Every tested replay frame and the demo view selected by its independent RGB-D fit](replay-frame-matches.jpg)

| Replay frame | Chosen demo frame |
| -----------: | ----------------: |
|          472 |               510 |
|          477 |               510 |
|          482 |               475 |
|          485 |               500 |
|          490 |               475 |
|          496 |               500 |
|          502 |               530 |
|          506 |               520 |
|          507 |               520 |
|          515 |               520 |

**The intended pipeline is a different, fuller experiment:**

```mermaid
flowchart LR
    A["Whole insertion interval in demo<br/>track dowel and ring frame by frame"] --> B["Extract desired dowel-to-ring<br/>motion versus task progress"]
    C["Initial live RGB-D view"] --> D["Geometry match to start of demo phase<br/>establish point and object mapping"]
    D --> E["Track dowel and ring with TrackEverything<br/>through adjacent live frames"]
    B --> F["Retarget desired relative motion<br/>using current live estimates"]
    E --> F
    F --> G["Gripper goal through current<br/>gripper-to-dowel attachment"]
```

That pipeline needs the ring as the moving reference and the current
gripper-to-dowel attachment to turn a desired object motion into a robot
command. The initial live view should first be registered to the beginning
of the relevant demo interval; another clear demo view can also serve as the
anchor if that first view is occluded. A one-time geometry anchor can be
ambiguous or drift later, so new depth observations and independent demo-view
matches should be able to correct or reject it. The live tracker must carry
point identity through adjacent frames. Choosing a fresh demo frame
independently for each replay frame, as done above, does not test temporal
consistency. Robot execution also needs
task progress rather than assuming demo frame number equals live time.
