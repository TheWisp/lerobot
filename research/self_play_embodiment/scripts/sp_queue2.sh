#!/bin/bash
# Resume: redo 3 corrupt phase-1 arms (ctrl already saved OK), then phases 2-4 verbatim.
P=/home/feit/miniforge3/envs/lerobot/bin/python
CK=/tmp/selfplay_probe/s1/s1sp_24000.pt
cd /tmp/selfplay_probe
export MUJOCO_GL=egl
echo "=== QUEUE2 START $(date) ==="
rm -f s2_models/ood_b05.pt s2_models/ood_init.pt

echo "--- [1/4 resume] retrain b01/b05/init with save ---"
$P sp_vj_act_s2.py $CK 10 12000 0.1 24 ood_b01  1 "" ins > q_ood_b01.log 2>&1
$P sp_vj_act_s2.py $CK 10 12000 0.5 24 ood_b05  1 "" ins > q_ood_b05.log 2>&1
$P sp_vj_act_s2.py $CK 10 12000 0   24 ood_init 1 "" ins > q_ood_init.log 2>&1
grep -h RESULT q_ood_b01.log q_ood_b05.log q_ood_init.log

echo "--- [2/4] OOD degradation slopes (f=1/2/3, n=48, paired) ---"
rm -f q_oodslope.log
for m in ood_ctrl ood_b01 ood_b05 ood_init; do
  for f in 1 2 3; do
    $P sp_ood_eval.py s2_models/$m.pt $f 48 $m >> q_oodslope.log 2>&1
  done
done
grep -h "OOD" q_oodslope.log

echo "--- [3/4] universality: transfer_cube K=10 seed 1 ---"
$P sp_vj_act_s2.py ""  10 24000 0   24 cube_ctrl 1 12000 cube > q_cube_ctrl.log 2>&1
$P sp_vj_act_s2.py $CK 10 24000 0.1 24 cube_b01  1 12000 cube > q_cube_b01.log 2>&1
$P sp_vj_act_s2.py $CK 10 24000 0   24 cube_init 1 12000 cube > q_cube_init.log 2>&1
grep -h RESULT q_cube_*.log

echo "--- [4/4] K sweep (insertion, seed 1): K=3, K=25 ---"
$P sp_vj_act_s2.py ""  3  12000 0   24 k3_ctrl  1 "" ins > q_k3_ctrl.log 2>&1
$P sp_vj_act_s2.py $CK 3  12000 0.1 24 k3_b01   1 "" ins > q_k3_b01.log 2>&1
$P sp_vj_act_s2.py ""  25 12000 0   24 k25_ctrl 1 "" ins > q_k25_ctrl.log 2>&1
$P sp_vj_act_s2.py $CK 25 12000 0.1 24 k25_b01  1 "" ins > q_k25_b01.log 2>&1
grep -h RESULT q_k*.log
echo "=== QUEUE2 DONE $(date) ==="
