# RLT v2 Running Log

## 2026-04-03: Initial implementation

### Branch setup
- Created `feature/rlt-v2` from `0592ce70` (pre-RLT, has intervention support)
- Copied rlt/ core modules from v1 with fixes:
  - Replay buffer: added threading.Lock on all public methods
  - Token reconstruction loss: changed from MSE (mean) to L2 sum (paper Eq. 2)
  - Actor BC penalty: L2 sum (paper Eq. 5) — carried from v1 fix
  - Config: sigma=0.01 (joint angles need lower noise than delta EE)

### RL Token v2 Training (Phase 1)
- S1 checkpoint: `outputs/flow_s1_no_s2_v1/checkpoints/last/pretrained_model`
- Dataset: `thewisp/cylinder_ring_assembly` (199K frames)
- Steps: 10,000, batch=64, lr=1e-4
- Loss: 451 → 114 (sum loss, not comparable to v1's 0.207 MSE loss)
- Per-element relative error: 60.4% (v1 was 66.7% on same test data)
- Compression ratio: 1025 tokens × 768 dim → 1 × 768 = 1025× compression
- Paper uses same ratio (~1025 × 2048 → 1 × 2048) and trains 2000-10000 steps
- Conclusion: bottleneck is inherently lossy at this ratio. Proceed to RL and see if it works.
- Checkpoint: `outputs/rlt_token_v2/checkpoint-10000`

### Integration (v2 architecture)
- Episode = RL boundary. Operator prepares scene during reset.
- InferenceThread: S1 encoder → z_rl → actor → refined chunk → replay buffer → inline gradient updates
- Actor NOT called during intervention. Human chunks accumulated by main thread.
- R key = success (+1). Collection paused during intervention/reset.
- Q values logged directly (mean/min/max per batch)
- GUI dashboard: success rate, Q values, actor delta, critic loss

### Commits on feature/rlt-v2:
1. `97e49406` — Core modules with infra fixes
2. `dd3e3a35` — Updated design doc
3. `a897f9cb` — Integration (s1_inference, s1_process, launch, GUI)

## 2026-04-03: Baseline measurement

### Setup
- S1 checkpoint: `flow_s1_no_s2_v1`
- Task: cylinder-ring assembly, critical phase only (operator teleops to pre-critical)
- Episode timeout: 60s
- No RLT, no intervention
- 20 episodes

### Results
| Metric | Value |
|--------|-------|
| Autonomous success rate | **65%** (13/20) |
| Throughput | **10.4 successes / 10 min** |
| Median success time | ~24s |
| Failed episodes | all timed out at 60s |
| Log file | `outputs/hvla_runs/run_20260403_155719.log` |

### Next steps:
- Write tests (Step 5)
- Robot testing with RLT v2 — target: beat 65% success rate and 10.4/10min throughput
