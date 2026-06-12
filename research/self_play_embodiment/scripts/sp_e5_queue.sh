#!/bin/bash
# E5: self-play-ONLY stage-1 corpus (teleop budget = exactly K) -> K=25 x3 seeds -> OOD ladders.
# Optimization applied: twin dropped (dead diagnostic, no grad path). Numerics UNCHANGED (fp32, batch 32)
# so stage-2/OOD numbers are directly comparable with E4 arms.
P=/home/feit/miniforge3/envs/lerobot/bin/python
cd /tmp/selfplay_probe  # nosec B108
export MUJOCO_GL=egl
echo "=== E5 START $(date) ==="
$P sp_vj_act.py 60000 2000 sponly notwin > e5_s1_sponly.log 2>&1 || { echo "S1 FAILED"; tail -5 e5_s1_sponly.log; exit 1; }
grep -E "plateau-stop|done corpus" e5_s1_sponly.log | tail -2
sleep 5; nvidia-smi --query-gpu=utilization.gpu,power.draw --format=csv,noheader | head -1
BEST=$(grep "plateau-stop" e5_s1_sponly.log | sed 's/.*best @ \([0-9]*\)).*/\1/')
[ -z "$BEST" ] && BEST=$(ls -t s1/s1sponly_*.pt | head -1 | sed 's/.*_\([0-9]*\).pt/\1/')
CK=/tmp/selfplay_probe/s1/s1sponly_${BEST}.pt  # nosec B108
echo "[ckpt] $CK"
for s in 1 2 3; do
  $P sp_vj_act_s2.py $CK 25 36000 0.1 24 e5_wmSP_s$s $s 12000,24000 ins > e5_wmSP_s$s.log 2>&1 || { echo "S2 s$s FAILED"; tail -3 e5_wmSP_s$s.log; exit 1; }
done
grep -h "RESULT" e5_wmSP_s*.log
for s in 1 2 3; do
  arm=e5_wmSP_s$s
  best=$(grep "RESULT $arm@" $arm.log | awk -F'SR>=1 ' '{split($2,a,"%"); print a[1]}' | paste -d' ' - <(grep "RESULT $arm@" $arm.log | sed 's/.*@\([0-9]*\)\].*/\1/') | sort -rn | head -1 | awk '{print $2}')
  [ -f s2_models/$arm@$best.pt ] && M=s2_models/$arm@$best.pt || M=s2_models/$arm.pt
  echo "[peak] $arm -> $M"
  for f in 1 1.25 1.5 2; do $P sp_ood_eval.py $M $f 96 ${arm}_pk >> e5_ood.log 2>&1; done
done
grep -h "OOD " e5_ood.log
echo "=== E5 DONE $(date) ==="
