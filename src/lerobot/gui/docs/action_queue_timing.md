# Action queue timing experiment

The 2026-10-16 insertion act at `20261016_171152` executed 144 distinct recorded
insertion samples as 172 motor commands. Five target retargets each turned one
sample into six commands, adding 25 commands without any successful queue
blend. The arm log also shows that the motor loop waited a full 33 ms _after_
each write, then read the servos before the next write. Its insertion command
interval averaged about 45 ms despite the nominal 30 Hz setting.

LeRobot's regular `record_loop` sleeps for the control interval minus the work
already done in the iteration. Its RTC inference engine is a closer match for
FLASH's asynchronous planner: the previous action queue runs while a new chunk
is computed, and `ActionQueue` removes the actions corresponding to inference
delay before replacing the queue. FLASH also checks IK, joint continuity,
workspace and gripper events, so an RTC-style time trim must validate the
splice before exposing the new commands to the motor loop.

`benchmarks/flash_queue_sim.py` compares four scheduling choices. It uses the
insertion's per-sample solve durations and motor read/write costs, plus the last
logged joint command for each recorded phase. That phase series is smoothed to
remove the missing interpolated subcommands. Five synthetic target shifts occur
during the rollout. A producer spends the measured solve time on each replan
and subsequent queue refill; a 30 Hz consumer continues executing the old queue
during that work. The table below is for target shifts three times the default
simulation amplitude, with the motor read included in the 30 Hz period.

| Retarget method               | Commands | Duration | First response after each shift | Maximum joint step | Maximum joint acceleration | Empty-queue ticks |
| ----------------------------- | -------: | -------: | ------------------------------: | -----------------: | -------------------------: | ----------------: |
| Add six commands per retarget |      169 |   5.69 s |                       33–123 ms |              0.91° |                0.78°/tick² |                 0 |
| Correct in one tick           |      144 |   4.86 s |                       33–123 ms |              3.58° |                3.35°/tick² |                 0 |
| Replace a 30-slot timed chunk |      144 |   4.86 s |                      200–289 ms |              0.91° |                0.43°/tick² |                 0 |
| Replace an 8-slot timed chunk |      144 |   4.86 s |                       56–154 ms |              1.39° |                0.47°/tick² |                 0 |

The eight-slot chunk preserves the recorded sample count, keeps the old queue
running during computation, and reacts sooner than replanning the full one-second
buffer. The code now checks that a truncated replacement joins the last sent
command within the joint speed, acceleration and workspace limits. If fixed
slots cannot fit the correction or a gripper boundary is involved, it falls
back to the existing incremental checked replanner. The motor loop now budgets
the read within each 33 ms interval while still spacing writes after a slow
read, and records intentional pacing separately from command work.

This is a joint-command scheduling surrogate. It does not simulate servo
dynamics, collision, contact, camera delay or IK feasibility, and its target
shifts are synthetic. It establishes the queue timing tradeoff; a physical
rollout is still needed to measure achieved rate and insertion accuracy.

## Motor read profile in subsequent Act records

The `arm.npz` file now saves `pre_read_wait_ms`, `pre_read_late_ms`,
`read_budget_ms`, and `tick_gap_ms` in addition to the existing tick phases.
The first three reveal how much of each interval is deliberate pacing and how
late the read starts; `tick_gap_ms` measures the interval between loop command
endpoints, including a tick without a write.

`read_state_ms`, `read_check_ms`, and `read_fault_ms` split the checked read.
`bus_setup_ms`, `bus_tx_ms`, `bus_rx_ms`, and `bus_decode_ms` split the serial
block read; `bus_rx_motor_ms` and `bus_rx_motor_cpu_ms` have columns in the
saved `motors` order. `read_cpu_ms` and `bus_cpu_ms` are thread CPU time; the
matching `*_off_cpu_ms` values are wall time minus CPU time. Off-CPU time can
include serial waiting, thread descheduling, or lock waits, so it must be read
alongside the transmit/reply timings rather than called bus latency alone.
`send_action_ms` and `send_action_cpu_ms` isolate the motor write from the
queue, clamp, and IK work within `command_ms`; they are NaN on a tick with no
write.
Older Act records simply lack these optional arrays.
