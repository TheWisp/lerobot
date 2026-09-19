---
name: remote-rig
description: Working against a remote machine that runs this code — a training rig, a robot host, any box you sync to and measure on. Use when running training, inference, benchmarks or the GUI somewhere other than the machine you are editing on.
---

# Working against a remote rig

A rig is a machine you sync code to and measure on. Everything here is about
one hazard: **the rig can be running something other than what you just
wrote**, and nothing tells you. Every gate below exists because skipping it
produced a confident, wrong answer that was believed for a while.

## Connecting

The transport is `ssh <alias> <command>` and `scp` for files. The alias is the
only thing you need, and you never need credentials — a configured alias
carries the hostname and identity.

If you do not know the alias, do not guess and do not hardcode one:

```bash
awk '/^Host /&&$2!~/\*/{print $2}' ~/.ssh/config   # candidates
ssh -o BatchMode=yes <alias> true && echo reachable
```

Show the candidates, ask which is the rig, and **record the answer in your
memory** so you ask once rather than every session. If `ssh` prompts for a
password or hangs, the alias is not set up — ask the user rather than working
around it.

## The gates

**1. Sync is not deploy.** A long-lived process keeps the code it started
with. After syncing, restart it, or you are testing what you already had.

```bash
scripts/gui/restart_gui.sh <alias>     # restarts and waits until it serves
```

This is not hypothetical: a fix was reported as verified twice while the
server was still running the previous revision.

**2. A rebuilt image is not a verified image.** Rebuild, then prove the
artifact contains your change before trusting anything it produces — grep the
file inside the image, or check the package version it installed. A stale
image once supplied a whole benchmark row that had to be retracted.

**3. Pin the revision before measuring.** Check the rig's checkout is at the
commit you think it is. A measurement of the wrong commit looks exactly like a
measurement of the right one.

```bash
ssh <alias> "git -C <checkout> rev-parse --short HEAD"
```

**4. The rig is a consumer of code, not a source.** Sync one way: commit and
push where you work, fetch on the rig. Do not edit or commit on the rig —
credentials and hooks usually live on your machine, so work committed there is
unlinted, unpushable, and destroyed by the next sync.

**5. Never mutate shared data to test.** Datasets, checkpoints and recordings
on a rig are the user's. Clone to a throwaway name and use that. This applies
to anything a job writes to, not just what it reads.

**6. Let a run prove its own configuration.** Before quoting a number, confirm
from the run's own log that it used the settings you intended — the mode it
selected, the data it read, the flags it received. Runs have been launched with
a flag silently dropped and measured as if it were applied.

## When a measurement disagrees with expectation

Suspect the gates before the code. In order: is the process restarted, is the
image current, is the checkout at the right commit, does the log say the run
used what you meant? Most surprising results come from measuring something
other than what you think you are measuring.
