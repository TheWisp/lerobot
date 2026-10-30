# FLASH visible RGB-D tracking baseline

Observed = accepted image/depth pose; predicted = fallback pose; lost = no pose.
Mask IoU is agreement with supplied masks, not segmentation truth. Pose error is conditional on a published pose.
These traces feed supplied masks into perception, so mask IoU 1.00 does not test SAM3.

## CaryxAI/everyday-manipulation-3d

| Case / role          | Visible frames | Observed / predicted / lost | Mask IoU median |    Labeled pose error |
| -------------------- | -------------: | --------------------------: | --------------: | --------------------: |
| caryx-000 / object_a |            455 |                 1 / 7 / 447 |            1.00 | no labeled pose score |
| caryx-002 / object_a |            389 |               1 / 147 / 241 |            1.00 | no labeled pose score |
| caryx-003 / object_a |            323 |                1 / 37 / 285 |            1.00 | no labeled pose score |
| caryx-007 / object_a |            119 |                 1 / 7 / 111 |            1.00 | no labeled pose score |
| caryx-007 / object_b |            200 |               25 / 50 / 125 |            1.00 | no labeled pose score |
| caryx-014 / object_a |            194 |               43 / 47 / 104 |            1.00 | no labeled pose score |
| caryx-015 / object_a |            318 |                1 / 67 / 250 |            1.00 | no labeled pose score |
| caryx-032 / object_a |            174 |                153 / 21 / 0 |            1.00 | no labeled pose score |
| caryx-032 / object_b |            174 |                1 / 49 / 124 |            1.00 | no labeled pose score |
| caryx-041 / object_a |            518 |               23 / 95 / 400 |            1.00 | no labeled pose score |
| caryx-053 / object_a |            139 |                 41 / 93 / 5 |            1.00 | no labeled pose score |
| caryx-053 / object_b |            139 |                 55 / 84 / 0 |            1.00 | no labeled pose score |
| caryx-090 / object_a |            245 |                72 / 166 / 7 |            1.00 | no labeled pose score |
| caryx-090 / object_b |            245 |                1 / 54 / 190 |            1.00 | no labeled pose score |

## YCBInEOAT

| Case / role                              | Visible frames | Observed / predicted / lost | Mask IoU median |               Labeled pose error |
| ---------------------------------------- | -------------: | --------------------------: | --------------: | -------------------------------: |
| ycb-bleach0 / object                     |            181 |                4 / 66 / 111 |            1.00 |   0.3 mm / 0.7° on 69/180 frames |
| ycb-bleach_hard_00_03_chaitanya / object |            181 |                1 / 23 / 157 |            1.00 |   0.1 mm / 0.1° on 23/180 frames |
| ycb-cracker_box_reorient / object        |            181 |                28 / 87 / 66 |            1.00 |  6.5 mm / 3.0° on 114/180 frames |
| ycb-cracker_box_yalehand0 / object       |            181 |                 1 / 0 / 180 |            1.00 |            no labeled pose score |
| ycb-mustard0 / object                    |            181 |               19 / 45 / 117 |            1.00 |  12.0 mm / 8.0° on 63/180 frames |
| ycb-mustard_easy_00_02 / object          |            181 |               22 / 34 / 125 |            1.00 |   2.7 mm / 6.8° on 55/180 frames |
| ycb-sugar_box1 / object                  |            181 |                 181 / 0 / 0 |            1.00 |  0.9 mm / 1.0° on 180/180 frames |
| ycb-sugar_box_yalehand0 / object         |            181 |                8 / 151 / 22 |            1.00 |  4.7 mm / 7.8° on 158/180 frames |
| ycb-tomato_soup_can_yalehand0 / object   |            181 |                 177 / 4 / 0 |            1.00 | 8.6 mm / 13.4° on 180/180 frames |

The suite intentionally has no cross-domain success percentage. See JSON for per-frame states, failures, and provenance.
