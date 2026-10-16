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
back to the existing incremental checked replanner. The motor loop reads and
then writes at once in each slot of a fixed 30 Hz schedule (see
[Motor loop timing](#motor-loop-timing)).

This is a joint-command scheduling surrogate. It does not simulate servo
dynamics, collision, contact, camera delay or IK feasibility, and its target
shifts are synthetic. It establishes the queue timing tradeoff; a physical
rollout is still needed to measure achieved rate and insertion accuracy.

## Motor read profile in subsequent Act records

The `arm.npz` file now saves `pre_read_wait_ms`, `pre_read_late_ms`, and
`tick_gap_ms` in addition to the existing tick phases. The first two reveal how
much of each interval is deliberate pacing and how late the read starts after
its slot; `tick_gap_ms` measures the interval between loop command endpoints,
including a tick without a write. Records made before the fixed schedule also
have `read_budget_ms` and `pace_ms`, the second sleep that sat between read
and write.

`read_state_ms`, `read_check_ms`, and `read_fault_ms` split the checked read.
`bus_setup_ms`, `bus_tx_ms`, `bus_rx_ms`, and `bus_decode_ms` split the serial
block read, and `bus_rx_wait_ms` is the part of `bus_rx_ms` spent waiting for
the whole block of replies; `bus_rx_motor_ms` and `bus_rx_motor_cpu_ms` have columns in the
saved `motors` order. `read_cpu_ms` and `bus_cpu_ms` are thread CPU time; the
matching `*_off_cpu_ms` values are wall time minus CPU time. Off-CPU time can
include serial waiting, thread descheduling, or lock waits, so it must be read
alongside the transmit/reply timings rather than called bus latency alone.
`send_action_ms` and `send_action_cpu_ms` isolate the motor write from the
queue, clamp, and IK work within `command_ms`; they are NaN on a tick with no
write. `gc_ms` is the time the process spent in garbage collection since the
previous tick; a collection holds the GIL throughout.
Older Act records simply lack these optional arrays.

## Motor loop timing

The insertion Act of 2026-10-16 (`20261016_201209`) achieved 21.5 motor ticks/s.
Its checked reads took 3.1/26.7 ms at p50/p95 against 1.6/2.4 ms on the idle
arm, yet only 1.3 ms of thread CPU at p95; the worst took 395 ms of wall time on
0.7 ms of CPU. The SDK reads replies by polling the nonblocking port, and every
poll gives up the GIL and must take it back. Another thread running Python
keeps the GIL until its switch interval (5 ms) runs out, so each retake can
cost that long.

`benchmarks/motor_bus_gil.py` reproduces this without the arm: seven fake
servos answer on a pseudo-terminal from another process while the jog's tick
(block read, then goal write, at 30 Hz) runs beside other work in the same
process. "Two threads" are two threads of IK-like small-array numpy, each busy
for half of every 20 ms; "four processes" is the same work outside the process;
"long C call" encodes a 100k-item dict to JSON every 250 ms.

| Other work     | Replies  | Switch | Read p50/p95 ms | Read CPU p50 ms | Write p95 ms | Wake late p95 ms | Writes/s |
| -------------- | -------- | -----: | --------------: | --------------: | -----------: | ---------------: | -------: |
| none           | SDK poll |   5 ms |       1.42/1.51 |            1.38 |         0.04 |             0.07 |    30.00 |
| none           | wait     |   5 ms |       1.45/1.55 |            0.16 |         0.05 |             0.07 |    30.00 |
| four processes | SDK poll |   5 ms |       1.41/1.51 |            1.38 |         0.05 |             0.07 |    30.00 |
| two threads    | SDK poll |   5 ms |       27.3/65.5 |            0.27 |         15.0 |             5.52 |    21.67 |
| two threads    | wait     |   5 ms |       10.2/20.3 |            0.13 |         10.1 |             5.95 |    29.64 |
| two threads    | SDK poll | 0.5 ms |       2.89/6.20 |            0.16 |         1.15 |             0.64 |    29.99 |
| two threads    | wait     | 0.5 ms |       2.38/2.96 |            0.14 |         1.16 |             0.64 |    30.01 |
| long C call    | SDK poll |   5 ms |       1.41/1.62 |            1.37 |         0.04 |             23.5 |    29.78 |
| long C call    | wait     | 0.5 ms |       1.45/1.51 |            0.15 |         0.05 |             23.2 |    29.75 |

CPU load from other processes changes nothing; GIL contention inside the process
reproduces slow, CPU-idle reads. Sampling the motor thread's `wchan` from
another process put 99 % of such a read in `futex_do_wait`, a GIL wait. A
smaller switch interval bounds every such wait, including the transmit, the
goal write and the wake from sleep, which no change to the read alone reaches.
Waiting for the whole block of replies cuts the retakes per read and stops the
thread spinning on the CPU while they arrive. Encoding one large dict holds the
GIL for its whole C call: neither change helps, and the 45 ms stalls in that row
remain. One or three CPU-bound threads did the same work per second at a 5, 1
or 0.5 ms switch interval (259k–268k iterations/s with `--throughput`).

The changes, each kept by a test:

- `FeetechMotorsBus.sync_read_block` waits in the kernel for the whole block of
  replies, up to the SDK's packet timeout, before the SDK's `rxPacket` checks
  each one. A missing or bad reply fails as before, and a port without a file
  descriptor is read as before.
- The GUI server runs with a 0.5 ms switch interval.
- The jog loop reads and then writes at once on a fixed 30 Hz schedule. Timing
  each period from the previous write added the tick's own work to every period:
  28.5 ticks/s on the idle arm. A late tick is followed on schedule; a tick that
  overran the next slot restarts the schedule a period after its write, so a
  stall is never followed by two commands back to back.

### On the arm

Measured on 2026-10-16 with the camera and P2P tracking running (about 9.5
tracker frames/s), the SO-107 left arm at the ready pose, and py-spy sampling the
GIL holder from outside the server. The baseline profiles ran the build before
these changes with only the `gc_ms` timing added. The baseline Act,
`20261016_201209`, ran earlier the same day on the build before these changes,
without py-spy. The Act with the changes is `20261016_211332`, from the same
demo and settings (speed 1, 1 s horizon, grasp prediction allowed).

| Run                     | Ticks/s | Gap p50/p95/p99 ms  | Gaps > 2 ticks | Read p50/p95/max ms | Read CPU p95 ms | Write p95 ms | Queue empty |
| ----------------------- | ------: | ------------------- | -------------: | ------------------- | --------------: | -----------: | ----------: |
| stationary, baseline    |    26.1 | 35.2 / 52.5 / 63.9  |              3 | 1.8 / 12.3 / 27.7   |            1.35 |          7.6 |             |
| stationary, changed     |    29.9 | 33.1 / 44.4 / 49.4  |              0 | 2.0 / 8.1 / 22.7    |            0.43 |          4.0 |             |
| 10 mm moves, baseline   |    25.9 | 35.5 / 53.2 / 59.3  |              0 | 1.7 / 17.3 / 26.0   |            1.35 |          7.4 |             |
| 10 mm moves, changed    |    30.0 | 33.4 / 45.4 / 51.6  |              0 | 4.3 / 12.5 / 27.7   |            0.45 |          8.1 |             |
| insertion Act, baseline |    21.4 | 39.0 / 82.4 / 117.5 |             60 | 3.1 / 26.7 / 395.1  |            1.31 |          9.2 |      2.87 s |
| insertion Act, changed  |    28.6 | 33.5 / 47.4 / 65.0  |              6 | 2.0 / 12.5 / 79.5   |            0.39 |          6.4 |      2.43 s |

The changed loop also executes the demo closer to its recorded speed, so its
commands are faster per second (p95 21.2 to 26.0 °/s) while smoother per tick
(second difference p95 0.39° to 0.27°). Following error p95 rose from 2.7° to
3.4° accordingly.

Where the baseline's reads waited, sampled from the motor thread's `wchan` and
`schedstat`: 74–76 % in `futex_do_wait` (the GIL), about 20 % on the CPU (the
SDK's poll), about 1 % runnable but queued, and at most 0.2 % blocked in the
kernel (`__flush_work`). During reads over 5 ms, py-spy found the GIL held by the overlay
thread replanning the Act preview (`_render_live` → `_act_preview` →
`plan_key_frames`, 53–63 % of the samples), by the RealSense read loop
(25–29 %) and by the event loop (6–22 %).

What still stalls the loop after these changes:

- Garbage collection. Collections totalling 270 ms (stationary baseline) and
  316 ms (changed Act) within a single tick were recorded in `gc_ms`; the latter
  is the changed Act's longest gap, 340 ms.
- The RealSense spatial and temporal filters keep the GIL for each call, about
  4.4 ms per frame at 30 frames/s. A thread in the same process as the camera's
  read loop, sleeping at 30 Hz, woke up to 4.1 ms late with them and 0.08 ms
  without them, at a 0.5 ms switch interval. In the changed Act the camera's read loop
  held the GIL in 35 % of the samples taken during reads over 5 ms.
- The queue empties at key frame 7 of the insertion, where the producer keeps a
  held object's release at the head of the queue and supplies one sample per
  solve: 26 of the 26 empty spells there fell within ten ticks of a gripper
  command change. They total 1.18 s, the longest 135 ms (487 ms in the baseline).
