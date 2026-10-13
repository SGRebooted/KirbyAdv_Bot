# Kirby’s Adventure Bot

A RAM-based reinforcement-learning project for **Kirby’s Adventure (USA, Rev 1)**
in **BizHawk / NesHawk**. Independent emulator workers collect experience for
one shared neural network; the model does not see screenshots.

**Current release: v9 — Double DQN and expanded RAM observations.**
Updated October 13, 2026. Training remains scoped to stage 1-1, rooms 43–45.

The deployed source files were compared with the prepared v9 package and matched.
The user subsequently reported more varied behavior in the first training round.
That is an early qualitative observation, not proof of better completion rates.
Offline correctness checks are described below; comparative learning results are
still pending.

## Quick start

Run from the project folder using the existing VS Code Parallel configuration,
or in PowerShell:

```powershell
Set-Location 'C:\VSCode_Projects_SSD\KA_Bot_SSD'
python KA_Parallel.py
```

For an existing v9 model, at `Parallel>`:

```text
workers 6
launch
loadmodel
verify
```

Use `newmodel` instead of `loadmodel` only when starting fresh. A v8 checkpoint
cannot initialize this encoder. `newmodel` activates the fresh model created at
console startup and refuses to replace an already active model. `loadmodel`
restores manual latest; `autosaves` then `loadauto` can recover newer autosaved
progress. Loading clears replay.

`launch` starts/reuses recognized managed workers and performs start verification;
it does not train. If emulators were already open while you replaced Lua, reload
each existing `worker_startup/worker_N.lua` script. `verify` resets selected workers
to their saved starts and checks both isolation and the v9 observation layout.

A short six-worker run with explicit exploration settings:

```text
ptrain 3 0.8 0.5 0.8 0.5 0.8 0.5
savemodel
eval
```

These are trial exploration values, not tuned recommendations. With six workers,
three attempts each means 18 attempts total. Omitting epsilon values retains the
alternating 0.1/0.3 defaults. Supply exactly one value per selected worker when
setting them manually. Attempt limits remain 1–1,000 per worker.

The current saved-start setup uses workers 1–4 in room 43 and 5–6 in room 44.
Worker count alone does not assign rooms. The frozen evaluation set is currently
workers 1–4: selecting six training workers does not silently change that baseline.
`eval` runs one greedy nonlearning attempt per frozen worker. Preserve that fixed
baseline for comparisons; extending it is a separate deliberate change.

During a run, Q requests a controlled stop, P toggles pause/resume, and S saves
when paused. At the prompt, `shutdown` saves the active model first and closes
managed emulators after verification. `quit` stops bridges but does not implicitly
save the model or close emulator windows.

## Requirements and files

- Python with PyTorch in the interpreter used by VS Code; CUDA is optional.
- BizHawk 2.11.1 and the matching ROM, configured in `launcher_settings.json`.
- Existing training starts and, for evaluation, frozen evaluation states.
- Project storage on the SSD; no change to `launch.json` is required for v9.

| File | Role |
|---|---|
| `KA_Bot.py` | Environment, rewards, replay, network, and Double DQN updates |
| `KA_Observation.py` | V9 feature encoding and action availability |
| `KA_Parallel.py` | Shared learner, workers, training/evaluation commands |
| `KA_Bridge.lua` | Shared RAM/control bridge; load through worker startup scripts |
| `KA_Workers.py`, `KA_Launcher.py` | Worker configuration and emulator launching |
| `KA_Starts.py` | Training/evaluation start management |
| `KA_Autosave.py` | Bounded automatic checkpoint history |
| `KA_ControlPanel.py`, `KA_Telemetry.py`, `KA_History.py` | Live dashboard and log browsing |
| `KA_Inspect.py` | Read-only display of the latest bridge response |
| `KA_Diagnostics.py` | Bounded worker-probe recording and analysis |
| `KA_GameRecorder.lua`, `KA_GameLog.py` | Separate manual gameplay RAM research tools |
| `KA_TerrainProbe.lua` | Earlier terrain research tool |

## Checkpoints and compatibility

| Output | V9 location |
|---|---|
| Manual latest | `models/kirby_parallel_v9_latest.pth` |
| Best evaluated | `models/kirby_parallel_v9_best.pth` |
| Automatic recovery | `models/autosave_v9/` |
| Attempt history | `logs/parallel_v9_attempts.csv` |
| Single-worker model | `models/kirby_v9.pth` |

Automatic checkpoints occur every 150 completed attempts across all workers or
30 minutes with new updates, and on handled training completion/stop. The newest
three v9 autosaves are retained. Manual latest and evaluated best remain separate.
Six workers × 25 attempts is 150 completed attempts; this is an aggregate counter,
not a synchronized round barrier. A crash or forced process kill is not a handled
stop and can lose changes since the last completed save.

V8 files remain archival/comparison checkpoints. Run them with matching v8 code.
V9 requires encoder version 4 and training version 9; old weight imports are
disabled. Checkpoints include online/target networks, optimizer, and update count,
but not replay memory or an exact live-emulator/session continuation.

Bridge control protocol stays at version 5; **observation schema is 2**. These
are different version numbers. Attempt CSV column schema remains 8 because the
columns did not change; the filename separates v9 training history.

## Observations: 4,456 inputs

| Component | Inputs | Content |
|---|---:|---|
| Player and context | 121 | HP/death/lives, X/Y and velocity, categorical ability/action/mouth, flags, script/pose, location IDs, sub-tile position, progress context, previous buttons |
| Non-player object slots | 1,445 | 17 slots × 85 inputs: presence, script, relative position, velocity, pose, eight scratch bytes |
| Wide terrain | 2,890 | 17×17 cells × availability, in-room flag, eight attribute bits |
| **Total** | **4,456** | Fixed layout; fresh weights required |

The bridge supplies all 18 slots; the encoder excludes slot 1 because Kirby has
dedicated inputs. Empty object slots are zeroed. Scratch variables are raw
observations, not verified health/ownership/hitbox labels, and can remain stale
during reuse of active slots. Effects are not automatically classified as hazards.

Position uses 256-pixel scaling and velocity uses a divisor of 1,024. Relative
object positions use the shortest signed displacement in 16-bit coordinate space.
This handles common wraparound, but does not explain every special animation
coordinate. Position within a 16-pixel tile is included explicitly.

The terrain view spans tile offsets **X −8…+8, Y −4…+12**, giving more information
beside and below Kirby. Unavailable data, out-of-room cells, and air are distinct.
Raw attributes preserve unknown terrain; there are no guessed semisolid rules.
The old 9×9 grid remains available for existing displays. Vertical layouts remain
unavailable until visually verified.

Observed action states distinguish walking (1), running (2), braking (3), jump
phase (4), falling (5), inhale (9), spit/exhale (10), copy/Mix (11), ability action
(12), and floating (13), with contextual/unknown values retained. Ground contact
does not imply normal standing: Kirby can remain inflated on the floor.
The observed candy protection bit is available, but its remaining timer is not.
Mix's changing ability value is exposed alongside its action state; the learner
does not receive a hard-coded explanation of roulette or a universal control lock.

Kirby receives level/stage/room and position. Progress records are separate per
room, so room transitions are not treated as simple backward travel in one map.
Previous buttons cover only the preceding action, not recurrent sequence memory.

### Actions

The learned action order is fixed:

```text
wait left right up down a b left_a right_a left_b right_b jump right_jump left_jump
```

The first eleven are held-button actions. `wait` advances with buttons released; it does not freeze momentum or enemies.

Fresh-jump actions release A for one frame, then press A for the remaining requested frames, optionally holding the horizontal direction. A normal four-frame `right_jump` is one Right-only frame followed by three Right+A frames. Held `right_a` can sustain A; repeated fresh-jump commands insert repeated release frames and can produce short hops. No action guarantees a jump when Kirby is airborne, hurt, or otherwise unable to jump.

Automatic fresh jumps retain transition/death guards and count as one learning experience. Manual controls intentionally bypass normal attempt constraints for debugging.

During established floating, automatic selection excludes the three fresh-jump
macros. Held A and horizontal+A remain available. Held A is not disabled during
jump phases, and Up is not disabled with a full mouth because it can enter doors.
Random exploration, greedy selection, and Double DQN target selection share this
restriction. Direct manual controls remain unrestricted.

### Trainer

| Setting | Current value |
|---|---|
| Network | 4,456 → 128 ReLU → 128 ReLU → 14 action values |
| Algorithm | Double DQN: online selects the next action; target evaluates it |
| Optimizer | Adam, learning rate 0.0001 |
| Loss | Smooth L1 / Huber |
| Batch size | 32 |
| Parallel replay capacity | 10,000 |
| Automated replay warmup | 128 stored experiences |
| Update cadence | One update per four stored experiences across all workers, subject to warmup/stop handling |
| Target refresh | Every 100 training updates |
| Gradient norm cap | 10 |
| Discount | 0.99 per four elapsed game frames; longer intervals adjust discount accordingly |

Action values are estimated future rewards, not probabilities. Delayed transition rewards and endpoint values account for elapsed frames; displayed reward totals are undiscounted, while replay uses the learning reward calculated for that interval.

### Reward rules

| Event | Reward / behavior |
|---|---|
| New rightward record in a room | +0.01 per new pixel; retracing old ground pays nothing |
| Normal health loss | −1 per visible health segment estimated from raw HP change |
| Lost life | −5; ends the attempt |
| Normal gameplay time | −0.0005 per frame, or −0.002 per four frames |
| First new supported gameplay room | +5, once per room per attempt; starting room is already visited |
| Confirmed stage 1-1 goal route | +10 for room 45 → 42 at level/stage 0/0; ends successfully |
| No-progress watchdog | Ends the attempt as a truncation; adds no separate punishment |

The time cost applies in supported rooms 43–45, without a grace period. Known loading/death/goal and other excluded intervals are handled conservatively; movement during scripted transitions is not rewarded as ordinary progress. Ordinary fighting and hurt recovery still consume gameplay time.

There is **no score reward, kill reward, healing reward, automatic door entry, repeated-action penalty, or leaderboard speed bonus**. Some were proposed historically; they were not shipped. Defeating an enemy can help indirectly by preventing damage or enabling progress.

### Attempt limits

| Limit | Current behavior |
|---|---|
| No progress | 600 normal game frames |
| Room budget | 3,600 frames; refreshed on first entry to a new supported room, not on revisits |
| Overall backstop | 18,000 frames |
| Loading wait | Bounded at 120 frames |

The watchdog resets on a new rightward record, a newly explored 16×16 cell, or first entry to another supported room. A new cell can be to the left or above: novelty is a timeout heuristic, not proof of useful route progress, and earns no separate reward.

Life loss and stage completion are terminal outcomes. Room/time/watchdog limits are truncations. Unsupported terrain/rooms and bridge errors stop automation for inspection rather than being interpreted as successful play.

The larger observations use about 340 MiB of float32 replay tensors at full
parallel capacity, plus Python/model/optimizer/batch overhead. Wider bridge replies
also add work. Measure live throughput rather than assuming unchanged speed.

## Inspecting and recording RAM

At an idle `Parallel>` prompt, `read 1` obtains a snapshot. In another terminal:

```powershell
python KA_Inspect.py --worker 1
```

The inspector reads the saved response; it does not send commands, train, or
continuously follow the emulator. Obtain another `read ID` for fresh data.

`step ID ACTION FRAMES` supports post-attempt debugging without replay/training
entries. `release ID` hands control back to normal emulator input; `reconnect ID`
pauses and regains bridge control. Verify before resuming automation. Saving a
training start is explicit; verification restores saved starts.

`probe ID FRAMES [STRIDE=4] [ACTION=wait]` records bounded worker snapshots into
`diagnostics/` without learning, for 0–600 frames at stride 1 or 4. Use
`KA_Diagnostics.py` for these `ram_...jsonl` files. Fresh-jump macros are excluded
from probes; held inputs are supported.

The standalone `KA_GameRecorder.lua` supports manual gameplay research without
the worker bridge. It records RAM ranges $0000–$07FF and $6000–$7FFF using full
snapshots plus deltas, sampled buttons, and emulator frame indices. Configure its
ROOT, LABEL, STRIDE, and MAX_FRAMES; do not run it alongside the worker bridge.
Use `KA_GameLog.py` for its `game_...jsonl` files. It does not measure wall-clock
frame timing or input-poll lag. Avoid savestate loads during comparisons and
record a clear description of the tested behavior.

## Command reference

All commands below are entered at **`Parallel>`**, after any active run has finished or stopped.

| Command | Purpose / important behavior |
|---|---|
| `help` | Print the current command list |
| `workers N` | Select IDs 1–N; remember count; prepare startup scripts; requires re-verification |
| `launch` | Launch/reuse managed workers, restore/verify training starts, optionally open panel |
| `verify` | Test independent stepping and restore selected training starts |
| `read ID` | Raw snapshot without requesting frame advancement; freeplay can still advance independently |
| `save ID` | Overwrite this worker’s training start with its current state |
| `reset ID` | Restore this worker’s training start |
| `step ID ACTION FRAMES` | Manual movement, no learning or attempt accounting |
| `release ID` | Unpause and return control to ordinary emulator input |
| `reconnect ID` | Pause and regain bridge control without saving/resetting |
| `clonestart SOURCE TARGET` | Copy a saved training start; existing target file is protected |
| `freezeeval` | Establish evaluation states/IDs from selected saved training starts |
| `freezeeval replace` | Explicitly replace that evaluation baseline |
| `saveeval ID` | Overwrite an existing evaluation worker’s evaluation start from live state |
| `loadmodel` | Load v9 latest; clear replay |
| `loadbest` | Load v9 best; clear replay |
| `newmodel` | Activate a fresh v9 model; refuses if a model is already active |
| `importv7` / `importprevious` | Disabled in v9; fresh weights required |
| `importsingle` | Strict import of the current single-worker checkpoint |
| `savemodel` | Save the loaded shared model to v9 manual latest |
| `autosaves` | List automatic recovery checkpoints, newest first |
| `loadauto [1/2/3]` | Load a retained automatic checkpoint; defaults to newest, clears replay |
| `ptrain [ATTEMPTS] [EPSILONS...]` | Training; defaults to 3 attempts each and alternating 0.1/0.3 |
| `eval` | One greedy, nonlearning attempt per frozen evaluation worker |
| `bench [STEPS]` | Collection-only benchmark; default 120, range 1–900 decisions per worker |
| `telemetry on` / `telemetry off` | Enable/disable optional telemetry; disabled panel data becomes stale |
| `probe ID FRAMES [STRIDE] [ACTION]` | Bounded raw RAM recording |
| `shutdown` | Save loaded model first, request managed emulator exits, retain console on failure |
| `quit` | Stop selected/released bridges and exit console; no implicit model save or emulator-window close |

With one worker, `bench` tests one worker. With more, it compares floor(N/2) against N: six workers means three versus six. It does not change weights or replay, but it does run game attempts and write attempt records.

Legacy single-worker commands such as `terrain`, `inspect`, `scores`, `play`, `random`, and `trainstep` are not parallel-console commands. The legacy single-worker code remains for reference/debugging; use the v9 parallel workflow for current training, including when selecting only one worker. Do not run both coordinators against the same emulator.

## Dashboard and logs

Open the dashboard automatically with `launch`, or separately:

```powershell
python KA_ControlPanel.py
```

Despite its name, the current panel is **read-only**. Closing it does not stop training. Commands and hotkeys remain in the coordinator terminal.

### Live workers

View all selected workers together or focus on one. Live status includes player position/state, action, attempt information, reward components, timers, throughput, shared update count, replay, and checkpoint information.

The loss graph is **shared model loss**, not a separate loss per Kirby. Smooth L1/Huber loss measures mismatch between predicted action values and the trainer’s current reward-based targets. Low loss can coexist with poor gameplay; use evaluations and behavior metrics to judge progress.

The graph can also show recorded attempt metrics. These change when attempts finish, unlike live player snapshots. The panel keeps a bounded displayed history rather than plotting unlimited points at once.

### History

The History view parses current and archived attempt logs, including compatible legacy logs. Filters distinguish session, run, worker, starting room, mode, and date range. Useful graphs include progress, rooms, damage, reward, duration, success, reward components, and ending reasons. Optional smoothing uses a trailing ten-sample mean over the displayed sequence.

Compare **training with training and evaluation with evaluation**. After introducing room-44 starts, filter training by starting room before interpreting progress totals: a shorter/later start changes the task. Room count includes the starting room, so “rooms 1” means no newly visited room during that attempt.

Legacy logs lack some identifiers and metrics. The parser preserves unknown values and file-based grouping rather than inventing historical session boundaries. Historical loss browsing is not the same as attempt-history browsing; the current History tab is built around attempt metrics.

### Model weights

The weights tab shows layer statistics such as mean, spread, minimum, and maximum. These help inspect the model but are not a direct measure of playing skill.

### Data files and refresh

- `logs/parallel_v9_attempts.csv`: one reported completed/interrupted attempt per row, with schema version 8, session/run IDs, UTC timing, worker, exploration, starts, outcomes, updates, and reward components.
- `logs/archive/`: after a complete evaluation, a CSV with at least **5,000 data rows** is rotated as a whole. Archives are not automatically deleted. Interrupted evaluations do not trigger rotation.
- `telemetry/status_*.json`: uniquely named completed snapshots. Normally eight are retained; locked older files can survive until later cleanup.
- `telemetry/session_*.csv`: sampled shared loss history, not every optimizer update. Separate process sessions get separate files; samples also carry run/mode metadata.
- `diagnostics/*.jsonl`: explicit manual RAM captures, excluded from attempt-history scanning.

Normal telemetry publication is about once per second, weight statistics about once per five seconds, and history scanning about once per ten seconds. A stale label indicates no recent snapshot; it cannot by itself distinguish idle, disabled telemetry, or a disconnected process.

The documented legacy scan includes CSVs under `logs/` (including archives), root `parallel*.csv` files, and the legacy `parallel_training_logs_v7` folder. Changing a file’s location/name can affect whether it is discovered.

## Troubleshooting and limits

- **Observation schema error:** reload every worker's startup Lua script so it
  loads the v9 shared bridge. Restart the Python coordinator after replacing code.
- **Checkpoint mismatch:** use a v9 checkpoint or `newmodel`; renaming v8 weights
  does not convert them.
- **No model active:** use `newmodel` or `loadmodel` before training/saving.
- **Lua error near quotes on line 1:** paste file contents without quote wrappers
  or Markdown fences, and ensure Python was not pasted into the Lua file.
- **Timeout:** inspect the emulator/Lua console and restart the affected bridge
  and coordinator to restore synchronization. Do not let two coordinators command
  the same worker.
- **Unsupported terrain/room:** the current baseline intentionally limits training
  to the verified stage-1 rooms. This is not a whole-game model yet.

Progress reward still favors rightward movement and is not a general route planner
for optional subareas. Boss-specific damage areas, attack coverage, a robust candy
timer location, swallowed-enemy counts, vertical terrain, and additional ability
locks remain unverified or deferred. Research findings do not automatically become
handwritten gameplay rules.

Planned next patch: a deduplicated, rate-limited unknown-value log with session,
worker, room and frame context. Separate unknown IDs from known families with
unverified subtypes; do not log every changing scratch byte as a new discovery.
**This logger is not implemented in v9 yet.**

Further learning experiments may add multi-step returns and prioritized replay
toward Rainbow. The current release is Double DQN, not full Rainbow. Minigames
and Whispy-specific attack rules are deferred, potentially to v10.

## Validation

Offline v9 checks executed the Lua serializer on seven recorded RAM snapshots,
verified unchanged legacy fields and wider terrain against a separate lookup,
and checked encoder shapes, empty slots, wrapped coordinates, action availability,
numeric Double DQN targets, termination/truncation, checkpoint round-trips, old
checkpoint rejection, and v8 autosave preservation. Five Python regression tests
passed; source syntax checks also passed. `test_v9.py` and `test_snapshots.json`
are optional files included in the release package for rerunning these checks.

These checks establish implementation behavior, not learning quality. The user
reported more complex behavior in the first v9 training round; compare frozen
evaluations, deaths, progress, and clears before claiming improvement. Since v9
changes both observations and learning, v8/v9 comparisons do not isolate either
change's contribution.

## Version history and changelog

Dates below follow **America/Chicago** conversation timestamps. Early work was incremental rather than tagged releases: “foundation” and “v1” group those milestones retrospectively. Later version labels are the names actually used in the chat, release notes, and checkpoint files. This is a feature/migration history, not a fabricated Git commit log.

### Foundation — October 4, 2026

- Started from an empty `KA_Bot.py`, the prior RAM-mapping work, and an object-logging Lua script.
- Added the file-based Python ↔ Lua request/response bridge, raw player/object snapshots, bounded button stepping, and game save/reset.
- Preserved unknown script IDs, slot data, poses, and eight raw scratch variables rather than stopping on unknown objects.
- Corrected script-root discovery after BizHawk identified loaded Lua as `main` instead of a normal source filename. Early bridges used an explicit root path.
- Fixed the initial paused-emulator stepping bug by unpausing for frame advancement and pausing afterward.
- Added the clearer `wait` alias for neutral stepping.
- Verified raw health scaling with observed 47 → 39 damage, then added health-segment penalties.
- Added per-room farthest-right progress, excluding rewards for retracing previously reached ground.
- Added life-loss handling, the special death-health value, bounded attempts, and the `KirbyEnvironment` class.
- Established the collaborative copy/paste workflow and commented-code preference.

### V1 — initial observation and learning pipeline, October 4, 2026

- Added a 237-input player/object encoder: 48 player/location features plus nine 21-feature slots, retaining binary IDs and zeroing empty slots.
- Added the two-hidden-layer PyTorch network and 11 original action outputs. Initial scores were untrained action values.
- Added bounded retries for Windows command-file replacement collisions.
- Investigated graphical PPU data, then prioritized collision/terrain attributes for nearby ground and walls.
- Built and expanded the standalone terrain probe. Checked local-grid alignment across block boundaries and mapped doors, platforms, slopes, water, breakable blocks, and ladders in observed scenes.
- Integrated 9×9 terrain with availability/inside-room flags and raw attribute bits, expanding observations **237 → 1,047**.
- Added sprite-state-based transition detection and neutral input after observed loading/travel begins; suppressed scripted movement reward. Midboss intro movement remained unresolved.
- Added replay collection, sampled DQN updates, a target network, checkpoints, epsilon-greedy play, and automatic multi-attempt training.
- Initial single-worker replay capacity: 5,000. Checkpoints used `models/kirby_latest.pth`.
- Fixed a missing `stored_count += 1` that had caused training on every eligible action instead of the intended every-four cadence. A 900-experience validation run then produced 194 new updates after warmup.
- Fixed the local VS Code interpreter mismatch so the established CUDA-enabled Python was used.

### V2 — progress timer and loading fixes, October 4–5, 2026

- Added one progress-timer input: **1,047 → 1,048**.
- Added a room-43-only stall cost of −0.01 per four frames after a no-new-rightward-progress grace period.
- Initial grace was 300 frames; it was later shortened to 120 during experiments.
- Started a separate fresh model/checkpoint, `models/kirby_progress_v2.pth`, preserving the prior model.
- Identified the one-new-pixel timer-reset loophole. A 16-pixel threshold was proposed; do not confuse that proposal with the later v6 novelty watchdog.
- Expanded the training command allowance to 200 attempts. Autosaving was proposed but declined; saving remained explicit.
- Fixed room-44 loading gaps where the room ID changed before terrain and player coordinates had settled. Added bounded neutral waiting and delayed progress-baseline establishment.
- At this stage, transition-containing learning records were conservatively skipped; v3 subsequently addressed credit for entering doors.
- Rejected score rewards after observing that score can rise when enemies die as a result of Kirby taking damage.

### V3 — room rewards and full-stage attempt structure, October 5, 2026

- Added +2 for first visits to validated gameplay rooms 43/44/45; the start counts as already visited.
- Confirmed the stage 1-1 route **45 → 42 → hub 0**. Added +10 and success termination for the validated 45 → 42 transition.
- Added a 3,600-frame room budget refreshed on first visits, plus an 18,000-frame overall cap. Revisiting a room does not renew time.
- Joined door-triggering actions to bounded loading intervals so delayed room/goal rewards reach replay, with elapsed-frame discount handling.
- Kept the room-43 stall rule at this stage; added no score reward.
- Added Q to stop automation and return to the console instead of needing a debugger save workaround.
- Saved to `models/kirby_rooms_v3.pth`. Historical `importv2` preserved learned policy weights while resetting target/training bookkeeping, optimizer, replay, and update count.
- Inputs/actions remained 1,048/11.

### V4 — progress context and previous buttons, October 5, 2026

- Added distance behind the current room’s farthest-right record and six previous-button flags: **1,048 → 1,055 inputs**.
- Deliberately left Kirby’s raw sprite state out of the neural inputs because ordinary animation/blinking changes it; transition logic still uses it.
- Kept terrain, actions, and reward rules unchanged for this experiment.
- Historical `importv3` copied policy/target weights and appended seven zero-weight input columns. Optimizer, replay, and update count restarted.
- Saved to `models/kirby_context_v4.pth`, preserving v3.
- Discussed full-room maps, exit coordinates, input scaling, and automatic door assistance. They were not included; door entry remained learned.

### Parallel foundation — October 5, 2026, between v4 and v5

- Added `KA_Parallel.py` and two numbered Lua bridges with independent communication folders and starts.
- Two environments shared one model, optimizer, and expanded **20,000-experience replay**.
- Added worker isolation verification, collection benchmarks, per-worker exploration, training CSVs, greedy evaluation, and distinct latest/best checkpoints.
- Preserved learning cadence across aggregate worker experience rather than multiplying updates per worker.
- Historical `importsingle` continued from the v4 single-worker checkpoint; parallel saves used `kirby_parallel_latest.pth` and `kirby_parallel_best.pth`.
- A reported collection benchmark increased from about 44 to 86 combined frames/second, approximately 1.96×. This was a particular historical measurement, not a current performance target.

### V5 — continuous gameplay time cost, October 6, 2026

- Replaced the old room-43 stall penalty with **−0.002 per four normal gameplay frames** across supported rooms 43–45.
- Removed the penalty grace period; movement, attacks, and waiting all cost time. Known loading/death and other excluded intervals are handled conservatively.
- Retained the legacy progress timer as an input with its old meaning for compatibility, even though it no longer controlled a penalty.
- Kept 1,055 inputs and 11 actions, +2 room reward, +10 goal reward, damage/progress rewards, and budgets.
- Used separate `kirby_parallel_time_v5_latest.pth` / `best.pth` and `parallel_time_v5_attempts.csv` files. The single-worker filename was `kirby_time_v5.pth`.
- Historical parallel `importprevious` preserved policy, target, optimizer, and update count from the earlier parallel checkpoint, clearing replay.

### V6 — four workers, dashboard, watchdog, and log archives, October 7, 2026

- Expanded parallel operation to four workers with numbered bridges.
- Added separate telemetry and a read-only control panel, including shared loss and weight statistics.
- Added P pause/resume and S save while fully paused; retained Q stop/drain behavior.
- Increased first-new-room reward **+2 → +5**.
- Added the **600-frame no-progress watchdog**, reset by new rightward progress, newly visited 16×16 cells, or first entry into a supported room.
- Added clear stop reasons and reward-component reporting. Novel cells remained a timeout heuristic, without extra reward.
- Added whole-file CSV rotation after complete evaluations at 5,000 rows, retaining archives indefinitely.
- Preserved the small continuous time cost, room/overall budgets, encoder, and 11 actions.
- Used `kirby_parallel_v6_latest.pth` / `best.pth` and `parallel_v6_attempts.csv`. Historical v5 migration retained model, target, optimizer, and update count; replay restarted.

### V7 — fresh-jump actions and Windows telemetry fix, October 8–9, 2026

- Diagnosed that holding A through a landing does not guarantee another jump. Pausing with neutral joypad state is not the same as simulating an A-release frame.
- Added `jump`, `right_jump`, and `left_jump`: **11 → 14 actions**, with one simulated release frame before the press. `left_jump` was included before installation, so it remained v7 rather than becoming another release.
- Preserved original held-A actions for sustained input. New macros did not force jumping at obstacles or add vertical terrain support.
- Migrated v6 policy and target outputs, seeding new rows from `a`, `right_a`, and `left_a`. Preserved historical update count; reset optimizer and replay because output shapes changed.
- Fixed telemetry’s Windows overwrite collision by publishing uniquely named completed snapshots instead of repeatedly replacing the file currently being read.
- Kept bounded snapshot retention, loss history, and nonfatal telemetry failure reporting.
- Used `kirby_parallel_v7_latest.pth` / `best.pth`, `parallel_v7_attempts.csv`, and single-worker `kirby_v7.pth`.
- Later reports showed greedy entry into room 44, followed by stalling. Mouth-full/hovering behavior and jump-hold sequences remained diagnostic questions, not confirmed fixes.

### V8 — scalable operation and debugging tools, October 9–11, 2026

V8 deliberately preserved the v7 network, observations, actions, rewards, and learning cadence while improving operation and visibility.

1. **Shared bridge:** replaced duplicated numbered Lua implementations with one `KA_Bridge.lua` plus generated per-worker startup files and explicit worker identities.
2. **Dynamic workers:** generalized selected worker count, persistent settings, per-worker epsilon arguments, verification, and benchmarks. Added distinct v8 model/log paths and a compatible v7 import.
3. **Dashboard/history:** added all-worker and focused views, clearer checkpoint status, attempt graphs, session/run/mode/worker/start-room/date filtering, and legacy/archive parsing.
4. **Manual debugging:** added raw movement after attempt completion, Start/Select for manual use, release/reconnect for normal gameplay, and separation from replay/reward/attempt accounting.
5. **Launcher and starts:** added managed emulator launch/reuse, per-worker configs/tokens, saved-start restoration, separate frozen evaluation states/IDs, and protected training-state cloning.
6. **Save-first shutdown:** save the loaded model before requesting managed emulator exits, include recognized deselected workers, ignore unrelated processes, and keep the console available on failure.
7. **RAM recording:** added bounded one-/four-frame probe sampling into separate JSONL files with metadata, raw snapshots, completion markers, and a summary tool.
8. **Integration review:** checked installed files, regression suites, Lua execution with simulated APIs, starts/settings, and read-only checkpoint compatibility. Corrected installation copy/paste mix-ups affecting bridge/panel files; these were file-placement repairs, not learning-algorithm changes.

9. **Overnight recovery follow-up (October 11):** added 150-attempt/30-minute automatic checkpoints, final saves on completion/Q/handled worker errors, newest-three retention in a separate folder, failure retries, and explicit `autosaves`/`loadauto` recovery commands. Preserved the user’s locally increased 1,000-attempt limit. No reward, action, encoder, or learning-cadence changes.

Operational changes and observations during v8:

- Moved the active project to SSD after the HDD bottleneck was confirmed separately.
- The pretraining review found v7/v8 latest policy and target weights identical at **787,933 updates**. This is a historical snapshot, not the expected current update count after subsequent training.
- The first recorded v8 baseline evaluation completed with zero updates: **0/4 clears**, average **241 new pixels**, **1 health segment lost**, and **794 frames**. It established the first v8 best file; it did not prove v8 regressed or improved against older tests with different conditions.
- User confirmed managed operation, dashboard, shutdown, and real diagnostic samples at frames 0/4/8.
- Training was subsequently run with six workers, initially all in room 43. The user then prepared room-44 starts for workers 5–6 while retaining the four-worker evaluation baseline.
- No PPU capture, attack-coverage mask, new combat label, or additional neural input was added in v8.

### V9 — richer observations and Double DQN, October 13, 2026

- Replaced the 1,055-input encoder with 4,456 inputs, including movement/contact
  state, categorical mouth/ability state, player pose, lower object slots and
  scratch/subtype information.
- Expanded terrain coverage to 17×17 while preserving the old display grid.
- Rescaled numeric observations and corrected wrapped relative displacements.
- Added Double DQN and a consistent floating restriction on fresh-jump macros.
- Added `newmodel`; required fresh weights and separated v9 checkpoints, logs,
  and autosave retention from v8.
- Reduced parallel replay capacity to 10,000 for the larger input vectors.
- Added a read-only inspector and offline regression checks. Existing launcher,
  debugging controls, dashboard history, and frozen evaluations remain available.
- Began live training; comparative performance is not yet established.

## Historical documentation sources

V1–v8 history below the current operational documentation is preserved from the
previous README, reconstructed from “Start coding the bot,” saved source/release
packages, and the v8 development conversation. V9 behavior was checked against
the prepared source and deployed file comparison. Historical commands and paths
describe their releases; use the current sections above for v9 operation.
