# Kirby’s Adventure Bot

A RAM-based reinforcement-learning project for **Kirby’s Adventure (USA, Rev 1)** in **BizHawk / NesHawk**. Python controls independent emulator workers through Lua, collects their experiences, and trains one shared neural network.

**Current project release: v8.** Documentation reconstructed on **October 10, 2026** from the complete available history of **“Start coding the bot”**, its saved release notes/source packages, the v8 development conversation, and the installed source files.

The current training target is **stage 1-1, rooms 43–45**. This is not yet a whole-game bot. The model does not see screenshots: its observations come from player RAM, nine object slots, and a local terrain grid.

## Contents

- [Quick start](#quick-start)
- [Requirements and setup](#requirements-and-setup)
- [Files and folders](#files-and-folders)
- [How the bot works](#how-the-bot-works)
- [Training and exploration](#training-and-exploration)
- [Evaluation and checkpoints](#evaluation-and-checkpoints)
- [Worker counts and saved starts](#worker-counts-and-saved-starts)
- [Manual control and RAM recording](#manual-control-and-ram-recording)
- [Command reference](#command-reference)
- [Dashboard and logs](#dashboard-and-logs)
- [Observations, actions, rewards, and limits](#observations-actions-rewards-and-limits)
- [RAM and terrain reference](#ram-and-terrain-reference)
- [Troubleshooting](#troubleshooting)
- [Version history and changelog](#version-history-and-changelog)
- [Known limitations and future work](#known-limitations-and-future-work)
- [Validation and historical sources](#validation-and-historical-sources)

## Quick start

These instructions assume the existing v8 installation, saved starts, launcher settings, and v8 checkpoint are already present.

In a PowerShell terminal:

```powershell
Set-Location 'C:\VSCode_Projects_SSD\KA_Bot_SSD'
python KA_Parallel.py
```

The prompt should be **`Parallel>`**. Enter the following there, not in PowerShell:

```text
workers 6
launch
loadmodel
```

`launch` opens or reuses recognized managed emulator processes, checks their bridge identity, restores training starts, verifies independent stepping, and opens the dashboard if configured. **It does not start training.** `loadmodel` restores the latest v8 checkpoint and clears replay memory.

For the current curriculum, workers 1–4 start in room 43 and workers 5–6 start in room 44. That split comes from their individual saved game states; selecting six workers alone does not assign rooms.

Run a short batch, save it, then evaluate:

```text
ptrain 3
savemodel
eval
```

With six selected workers, this requests **18 training attempts total**. Evaluation uses the separately frozen worker set, currently **1–4**, even when six workers are selected.

To finish and free resources:

```text
shutdown
```

Shutdown saves the loaded model first, then requests normal exit from recognized launcher-managed emulator processes. It leaves the dashboard open. If saving or closing fails, the console stays available for recovery.

## Requirements and setup

### Working environment

| Component | Project setup |
|---|---|
| Operating system | Windows; terminal hotkeys and launcher process inspection use Windows APIs |
| Emulator | BizHawk 2.11.1 with NesHawk |
| Game | Kirby’s Adventure (USA) (Rev 1), supplied separately |
| Python | The project has used Python 3.13 |
| Learning library | PyTorch; CUDA when available, otherwise CPU |
| Dashboard | Tkinter/ttk, supplied with the Windows Python installation used here |
| Active project | `C:\VSCode_Projects_SSD\KA_Bot_SSD` |
| Previous location | `D:\VS_Code_Projects\KirbyAdv_Bot` |

The project was moved to SSD after the user confirmed the HDD was a communication/logging bottleneck. Keep the active files together: most paths are relative to the Python source directory, not the terminal’s current directory.

The historical working hardware was an i5-10400F, 32 GB RAM, and RTX 3060 12 GB. That is a record of the tested machine, not a minimum requirement or a guarantee that more workers improve speed.

Check the interpreter used to launch the bot:

```powershell
python -c "import sys, torch; print(sys.executable); print(torch.__version__); print(torch.cuda.is_available())"
```

The bot also prints interpreter/PyTorch diagnostics at startup. An earlier CPU-only run came from a different Python installation. Selecting the right interpreter fixed it; the model did not need resetting.

For VS Code F5, choose the **Parallel** launch configuration. Opening `KA_Parallel.py` in an editor tab does not itself change the selected debug configuration. `Kirby>` means the legacy single-worker console; its commands differ from `Parallel>`.

### Launcher settings

`launcher_settings.json` contains:

| Key | Purpose |
|---|---|
| `emuhawk` | Full path to `EmuHawk.exe` |
| `rom` | Full path to the ROM |
| `base_config` | Existing BizHawk configuration used as the initial worker template |
| `startup_timeout_seconds` | How long to wait for a newly launched bridge; currently 45 seconds |
| `open_panel` | Whether launch should open the control panel |

The configured emulator and ROM locations are outside the project. Check these paths after moving either file or changing machines. JSON accepts forward slashes in Windows paths; literal backslashes must be doubled.

The launcher creates separate worker configurations in `emulator_configs/`. It prepares multiple-instance use, paused startup, and NesHawk, and disables automatic loading of recent files/slots. Existing worker configuration files are reused; changing the base template does not automatically rebuild every existing worker configuration.

### Starting from a fresh folder

1. Place the source files listed below together, configure the launcher paths, and make trained checkpoints available if continuing an existing model.
2. Run `python KA_Parallel.py`, then `workers N` to prepare the worker directories and startup scripts.
3. Each worker needs a nonempty `bridge_N/start.State` before automated launch. Copy a compatible known training state into a new worker directory, or manually open the emulator and matching startup script, reach the intended position, and use `save N`.
4. With manually started bridges, use `verify` once the starts exist. Do not use managed `launch` over unrelated/manual emulator windows; close those windows before switching to managed launch.
5. Freeze an evaluation baseline once, using the intended selected workers and already saved training starts.
6. Load a compatible existing model, or explicitly migrate one as described below.

The v8 parallel console intentionally requires a loaded/imported model before automated runs. It has **no dedicated new-model/reset-learning command**. Removing a checkpoint is not a supported fresh-training workflow.

## Files and folders

| File or folder | Role |
|---|---|
| `KA_Bot.py` | Neural network, encoder, trainer, replay, bridge client, environment/reward logic, and legacy single-worker console |
| `KA_Parallel.py` | Main v8 console; worker orchestration, shared learning, evaluation, model saving, manual commands |
| `KA_Bridge.lua` | Shared Lua implementation: RAM snapshots, controller requests, savestates, release/reconnect, shutdown |
| `KA_Workers.py` | Generates small worker-specific startup scripts |
| `KA_Launcher.py` | Managed emulator launch/reuse, configuration copies, process identification, shutdown coordination |
| `KA_Starts.py` | Training-state cloning and frozen evaluation-state management |
| `KA_Telemetry.py` | Optional status/loss publication, terminal hotkeys, attempt-log archiving |
| `KA_ControlPanel.py` | Separate, read-only Tk dashboard |
| `KA_History.py` | Legacy/v8 attempt-log parsing, filtering, metrics, deduplication |
| `KA_Diagnostics.py` | Raw diagnostic recording and command-line recording summaries |
| `KA_TerrainProbe.lua` | Separate terrain-research tool; not required for normal training |
| `parallel_settings.json` | Remembered worker count and frozen evaluation worker IDs |
| `launcher_settings.json` | Emulator/ROM/configuration paths and launcher preferences |
| `worker_startup/worker_N.lua` | Generated worker number, project path, and launch token; loads the shared bridge |
| `bridge_N/` | One worker’s command/response files, readiness metadata, training and evaluation states |
| `bridge/` | Legacy single-worker communication directory, corresponding to worker 0 |
| `emulator_configs/` | Separate BizHawk settings for managed workers |
| `models/` | Neural-network checkpoints and preserved older versions |
| `logs/` | Attempt CSVs |
| `logs/archive/` | Rotated attempt CSVs, retained indefinitely |
| `telemetry/` | Completed status snapshots and per-session sampled loss CSVs |
| `diagnostics/` | Optional probe JSONL files, separate from training history |
| `parallel.lock` | OS-backed guard against two coordinators using the same project |
| `.vscode/` | Optional editor/debugger configuration |

Old numbered implementation files such as `KA_Bridge_1.lua` are no longer the v8 source of truth. Load **`worker_startup/worker_N.lua`**, not `KA_Bridge.lua` directly: the shared file needs the configuration supplied by its startup script.

Generated startup scripts contain absolute project paths. Regenerate them after moving the project. Do not regenerate/reload them in the middle of a managed session: managed launch tokens identify the running emulator and its bridge.

Preserve `models/`, saved `.State` files, settings, and any logs/diagnostics you want to keep. Temporary communication files and generated scripts are not backups of learned weights or game starts. Perform cleanup while the system is stopped.

## How the bot works

1. Each emulator runs its own Lua bridge and uses its own `bridge_N` directory.
2. Python reads a RAM snapshot and encodes it into 1,055 numeric inputs.
3. One shared network scores 14 actions. Exploration may substitute a random action.
4. Python sends a uniquely identified request. Lua advances the requested game frames, pauses, and returns a matching observation.
5. Python measures rewards, transitions, and attempt limits. During training, eligible experiences enter the shared replay buffer.
6. The shared trainer samples replay batches and updates the network. All workers subsequently use that network.

**Six workers are six environments, not six separate AIs.** They have separate game states, exploration choices, and attempt accounting, but share weights, an optimizer, and replay.

Emulator requests overlap across workers. Network decisions and learning run on the coordinator’s main thread. Each worker has at most one outstanding request, and its next action waits for the previous reply.

The file protocol uses request IDs and completed-file publication to avoid accepting stale responses or half-written commands. A lost reply is not silently retried as another movement action: its outcome may be uncertain.

### Game time versus wall-clock time

Normal decisions request four emulated frames. A room’s 3,600-frame budget is roughly one minute of game time, regardless of how long Python computation, file exchange, or pauses take in real time. Loading can extend an action’s total elapsed interval.

Combined game frames/second measures throughput across active workers. It can fall as workers finish; it is not the speed of each remaining emulator. `bench` excludes learning costs and therefore does not directly predict training throughput.

## Training and exploration

At `Parallel>`:

```text
ptrain ATTEMPTS EPSILON_1 EPSILON_2 ... EPSILON_N
```

- Attempts are **per worker**, from 1 to 200. The default is 3.
- Supply one epsilon for every selected worker, or omit all epsilon values.
- Epsilon is between 0 and 1: probability of choosing a random action instead of the model’s highest-scoring action.
- Omitted values alternate **0.1, 0.3, 0.1, 0.3, ...** in worker order.
- Exploration is fixed for that command. There is no automatic epsilon decay.
- Changing epsilon does not require changing model dimensions or deleting the checkpoint.

Examples:

```text
ptrain 3
ptrain 5 0.05 0.1 0.15 0.3 0.1 0.3
ptrain 5 0.2 0.2 0.2 0.2 0.2 0.2
```

The last two examples require six workers. `ptrain 5` with six workers means 30 attempts, unless stopped or interrupted by an error.

### Hotkeys while automation is running

Focus the **Python terminal**, then press the key without Enter:

| Key | Effect |
|---|---|
| **P** | Request pause; outstanding requests finish first. Press again after the pause completes to resume the same attempts. |
| **S** | Save latest while fully paused; wait for confirmation. |
| **Q** | Stop after outstanding work drains and return to `Parallel>`. Does not automatically save latest. |

Ordinary console commands become available after automation returns to the prompt. Use Q rather than Ctrl+C when you want to preserve the live console and finish requests cleanly.

Manual movement during training is outside the intended workflow. For debugging, finish/stop the run first. To rearrange emulator windows during a run, pause through the coordinator and wait for confirmation first.

## Evaluation and checkpoints

### Three different things called “saving”

| Command | Saves | Does not save |
|---|---|---|
| `save ID` | That emulator’s current state to its **training start** | Model weights, replay |
| `saveeval ID` | That emulator’s current state to its **evaluation start**; explicit baseline change | Model weights, replay |
| `savemodel` | Shared network, target network, optimizer, update count, compatibility metadata | Replay, emulator state, an exact whole-session continuation |

Replay contains up to 20,000 experiences and lives only in memory. Restarting Python or loading a model clears it. A checkpoint is not an exact capture of worker positions, random-number state, and replay history.

### Latest versus best

| Path | Meaning |
|---|---|
| `models/kirby_parallel_v8_latest.pth` | Explicitly saved latest training state; also the save-first shutdown target |
| `models/kirby_parallel_v8_best.pth` | Best complete evaluation under a matching baseline signature |
| `models/kirby_parallel_v7_latest.pth` | Preserved v7 migration source |

`loadmodel` loads latest; `loadbest` loads best into memory. Both clear replay. Loading does not itself overwrite the latest file, but a later `savemodel` or successful save-first shutdown saves whichever model is currently loaded.

Finishing `ptrain` does **not** automatically save latest. A successful new-best evaluation saves **best**, not latest. Preserve important checkpoints with separate backup filenames before experiments.

### Fixed evaluation baseline

`freezeeval` copies the selected workers’ existing `start.State` files into separate `eval.State` files and records those worker IDs. It copies saved starts, not the current live positions. Existing evaluation states are protected unless `freezeeval replace` is explicitly used.

Current baseline: workers **1–4**. Leave that baseline fixed when changing the training starts of workers 5–6. All evaluation IDs must be among the currently selected workers; selecting six satisfies that requirement, selecting only one does not.

`eval` resets those workers to evaluation states, uses epsilon 0, adds no replay items, and makes **zero training updates**. It can write attempt logs, rotate the attempt CSV, and save a new best model.

Best comparison uses these averages in priority order:

1. Stage-clear rate.
2. Number of distinct visited rooms, including the starting room.
3. New rightward progress.
4. Negative health lost: less damage is better when earlier metrics tie.
5. Negative elapsed game frames: less time is better when all earlier metrics tie.

This is a lexicographic comparison, not a weighted sum. More progress can beat a safer run. “New best” does not mean a stage was cleared, nor does low loss guarantee a better evaluation.

The comparison signature includes evaluation savestate contents, model/reward version fields and budget settings, and the bot source. When it changes, the existing best is retained rather than compared as though conditions were identical. Preserve/rename the old best before intentionally establishing a new baseline.

### Migration versus normal restart

| Current v8 command | Source and effect |
|---|---|
| `loadmodel` | Normal restart: load v8 latest with strict compatibility checks |
| `importv7` | Import v7 parallel latest; preserve policy, target, optimizer, update count; clear replay. Use `savemodel` to create v8 latest. |
| `importprevious` | Import **v6** parallel latest into the 14-action network; preserve old policy/target outputs and update count, seed fresh-jump outputs, reset optimizer/replay |
| `importsingle` | Load the current single-worker checkpoint path, `models/kirby_v7.pth`, using strict compatibility checks |

Do not repeatedly migrate after v8 latest already exists: use `loadmodel`. Older chats used `importprevious` for different version pairs; its meaning depends on the code version running at the time.

Project release **v8** still uses `ENCODER_VERSION = 3` and `TRAINING_VERSION = 6`, because its quality-of-life changes did not alter the v7 encoder, action list, or reward rules. These internal compatibility numbers are not release names and should not be casually bumped.

## Worker counts and saved starts

`workers N` selects IDs 1 through N and remembers the count. It prepares directories/startup files and invalidates verification. It does **not** launch/close emulators or change their saved room positions. Previously selected emulator windows can remain open after reducing the count.

Each worker has:

```text
bridge_N/start.State   training reset point
bridge_N/eval.State    frozen evaluation reset point, if in the evaluation set
```

`verify` restores training starts, advances workers separately to check isolation, and restores the selected starts again. It changes live game positions, but does not train or replace saved starts.

### Put worker 5 in room 44

After automation has finished:

```text
release 5
```

Use normal emulator controls to enter room 44. Pause near the desired start, then:

```text
reconnect 5
read 5
save 5
```

Confirm the room and player state before saving. `save 5` replaces its previous training start. A stable state near entry with the opening Waddle Dee still present is suitable; the enemy need not be at its exact first-frame position. Its position, player health, ability, mouth contents, and momentum are all part of the saved situation.

### Copy worker 5’s start to worker 6

The command is **`clonestart 5 6`**, not `clonesave`.

```text
clonestart 5 6
verify
```

Cloning copies the **saved training state**, not the current live state, and refuses to overwrite an existing destination. If worker 6 already has `start.State`, rename it to a backup such as `start_room43_backup.State` while the coordinator is idle, then clone.

Both source and destination IDs must be selected. `verify` loads the cloned state into the live worker. Evaluation files are untouched.

Two identical starts are allowed. Different exploration values can diversify their training experience, but diversity or improved learning is not guaranteed by worker count alone.

## Manual control and RAM recording

### Post-attempt control

V8 manual commands work after an attempt ends:

```text
read 1
step 1 right 20
step 1 wait 4
step 1 right_jump 4
```

`step` accepts 1–60 frames; fresh-jump macros need at least two. These commands bypass the training environment’s rewards, watchdogs, replay, and attempt CSVs. The panel labels their snapshots as manual. They do not teach the model by demonstration.

Manual Start/Select inputs are also available, for example `step 1 select 1`. Neither is a learned model action.

### Normal gameplay between tests

```text
release 1
```

This gives normal emulator/controller control back and unpauses that worker. When finished:

```text
reconnect 1
```

Reconnect pauses and reads the current position; it does not save or reset it. Controlled movement requires reconnect first. Run `verify` before returning to automation. Verification restores saved starts, so save a deliberately changed training start before verifying if you want to retain it.

### Record raw RAM sequences

```text
probe ID FRAMES [STRIDE=4] [ACTION=wait]
```

| Example | Effect |
|---|---|
| `probe 1 0` | One snapshot without advancing |
| `probe 1 8 4 wait` | Snapshots at elapsed frames 0, 4, and 8 |
| `probe 1 60 1` | Initial snapshot plus a snapshot after each of 60 frames |
| `probe 1 60 4 b` | Hold B for 60 frames, sampling every four frames |

Frames must be 0–600; stride must be 1 or 4. A final shorter interval handles frame counts not divisible by four. A probe holds the same input throughout; fresh-jump macros are rejected so sampling does not repeatedly retrigger them. Use held `a`, `right_a`, or `left_a` when appropriate.

Each JSONL recording contains a header, raw samples, and a completion row. Headers identify the capture, coordinator session, worker, action, and timing. Samples preserve player state, all nine object slots and raw variables, terrain, and bridge metadata. A file missing its completion row is partial.

Recordings go into **`diagnostics/`**, outside training history. Probing requires controlled mode, leaves the worker paused after successful completion, and requires verification before automation. It adds no training/replay data and does not overwrite starts.

To summarize a recording, run in a separate PowerShell terminal:

```powershell
python KA_Diagnostics.py 'diagnostics\YOUR_RECORDING.jsonl'
```

Use the real filename printed by `probe`. Counts are script/slot observations, including empty slots, not unique enemy counts. Slots can be reused. Script/pose values are raw evidence, not confirmed hitboxes, attack preparation labels, or entity ownership.

There is no continuous recording during released gameplay, no PPU/OAM capture, and no dedicated diagnostic dashboard tab yet. Ctrl+C during a probe invalidates the bridge because a request may still be finishing; restart the bridge/coordinator before continuing.

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
| `loadmodel` | Load v8 latest; clear replay |
| `loadbest` | Load v8 best; clear replay |
| `importv7` | One-time compatible v7 parallel import |
| `importprevious` | Dedicated v6-to-current action migration |
| `importsingle` | Strict import of the current single-worker checkpoint |
| `savemodel` | Save the loaded shared model to v8 latest |
| `ptrain [ATTEMPTS] [EPSILONS...]` | Training; defaults to 3 attempts each and alternating 0.1/0.3 |
| `eval` | One greedy, nonlearning attempt per frozen evaluation worker |
| `bench [STEPS]` | Collection-only benchmark; default 120, range 1–900 decisions per worker |
| `telemetry on` / `telemetry off` | Enable/disable optional telemetry; disabled panel data becomes stale |
| `probe ID FRAMES [STRIDE] [ACTION]` | Bounded raw RAM recording |
| `shutdown` | Save loaded model first, request managed emulator exits, retain console on failure |
| `quit` | Stop selected/released bridges and exit console; no implicit model save or emulator-window close |

With one worker, `bench` tests one worker. With more, it compares floor(N/2) against N: six workers means three versus six. It does not change weights or replay, but it does run game attempts and write attempt records.

Legacy single-worker commands such as `terrain`, `inspect`, `scores`, `play`, `random`, and `trainstep` are not parallel-console commands. The legacy single-worker code remains for reference/debugging; use the v8 parallel workflow for current training, including when selecting only one worker. Do not run both coordinators against the same emulator.

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

- `logs/parallel_v8_attempts.csv`: one reported completed/interrupted attempt per row, with schema version 8, session/run IDs, UTC timing, worker, exploration, starts, outcomes, updates, and reward components.
- `logs/archive/`: after a complete evaluation, a CSV with at least **5,000 data rows** is rotated as a whole. Archives are not automatically deleted. Interrupted evaluations do not trigger rotation.
- `telemetry/status_*.json`: uniquely named completed snapshots. Normally eight are retained; locked older files can survive until later cleanup.
- `telemetry/session_*.csv`: sampled shared loss history, not every optimizer update. Separate process sessions get separate files; samples also carry run/mode metadata.
- `diagnostics/*.jsonl`: explicit manual RAM captures, excluded from attempt-history scanning.

Normal telemetry publication is about once per second, weight statistics about once per five seconds, and history scanning about once per ten seconds. A stale label indicates no recent snapshot; it cannot by itself distinguish idle, disabled telemetry, or a disconnected process.

The documented legacy scan includes CSVs under `logs/` (including archives), root `parallel*.csv` files, and the legacy `parallel_training_logs_v7` folder. Changing a file’s location/name can affect whether it is discovered.

## Observations, actions, rewards, and limits

### Observation: 1,055 inputs

| Component | Inputs | Information |
|---|---:|---|
| Player and location | 48 | Raw health/lives/mouth state, X/Y, signed velocities, death flag, ability and level/stage/room IDs |
| Nine object slots | 189 | 21 values per slot: presence, script ID bits, relative X/Y, velocities, pose ID bits |
| Local 9×9 terrain | 810 | 81 cells × availability, inside-room flag, and eight raw attribute bits |
| Legacy progress timer | 1 | Room-43 progress timer, retained for input compatibility |
| Distance behind progress record | 1 | Gap from the room’s farthest-right point, divided by 256 |
| Previous buttons | 6 | Left, Right, Up, Down, A, B from the preceding controlled interval |
| **Total** | **1,055** | Fixed ordering required by the checkpoint |

IDs are bit-encoded; unknown IDs are preserved. Empty slots are zeroed. Raw object scratch variables remain available in diagnostic snapshots but are excluded from neural inputs because they may retain stale values.

The model **does receive X/Y and room identity**. Python tracks progress separately for each `(level, stage, room)` and establishes landing baselines, so X resetting on entry to room 44 is not treated as ordinary backward travel across room 43. This provides context; it does not guarantee the network has learned to use it well.

Kirby’s raw sprite state is used in transition handling but is not a model feature. There is no recurrent memory, full-room map, explicit enemy hitbox/attack coverage, or explicit remaining-room-time input. Six previous buttons provide limited immediate history, not a sequence memory.

Current numeric scaling remains inherited from earlier versions: positions/object distances use storage-range divisors such as 65,535, velocities use 32,768. The concern that nearby enemy distances become numerically small was discussed but not changed in v8.

### Actions

The learned action order is fixed:

```text
wait left right up down a b left_a right_a left_b right_b jump right_jump left_jump
```

The first eleven are held-button actions. `wait` advances with buttons released; it does not freeze momentum or enemies.

Fresh-jump actions release A for one frame, then press A for the remaining requested frames, optionally holding the horizontal direction. A normal four-frame `right_jump` is one Right-only frame followed by three Right+A frames. Held `right_a` can sustain A; repeated fresh-jump commands insert repeated release frames and can produce short hops. No action guarantees a jump when Kirby is airborne, hurt, or otherwise unable to jump.

Automatic fresh jumps retain transition/death guards and count as one learning experience. Manual controls intentionally bypass normal attempt constraints for debugging.

### Trainer

| Setting | Current value |
|---|---|
| Network | 1,055 → 128 ReLU → 128 ReLU → 14 action values |
| Algorithm | DQN with a separate target network |
| Optimizer | Adam, learning rate 0.0001 |
| Loss | Smooth L1 / Huber |
| Batch size | 32 |
| Parallel replay capacity | 20,000 |
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

## RAM and terrain reference

Addresses and terrain attributes below are **hexadecimal**. Object script IDs in the next table are **decimal**. These are observations used by this project, not a complete verified map of the game.

### Player and room fields

| Field | Address(es) | Notes |
|---|---|---|
| Health | `$0597` | Full health observed as decimal 47; a normal segment loss decreases it by 8; 255 observed during death |
| Lives | `$0599` | Decrease used for life-loss detection |
| Ability | `$05E3` | 255 observed with no ability; not a cutscene flag |
| Mouth state | `$05E0` | Raw mouth-full field |
| Sprite state | `$62C7` | 255 observed during loading/Warp Star travel; does not cover every scripted movement |
| X low/high | `$0083` / `$0095` | Room-local position |
| Y low/high | `$00B9` / `$00CB` | Room-local position |
| X velocity low/high | `$05B9` / `$05BA` | Signed raw velocity |
| Y velocity low/high | `$05BD` / `$05BE` | Signed raw velocity |
| Level / stage | `$0558` / `$0559` | Raw IDs |
| Room low/high | `$055E` / `$055F` | Two-byte room ID |

For slots `$09` through `$11`, the bridge reads script from `$6128 + slot`, pose from `$62C6 + slot`, and eight variables from `$61CA + index × $12 + slot`. Positions and velocities also come from the game’s slot arrays. The nine slots do not constitute a proven complete inventory of every possible attack or boss object.

### Object labels presently used

| Decimal script ID | Project label / qualification |
|---|---|
| 2 | Death-star effect, observed |
| 4 | End-level bonus stars, provisional |
| 5 | Pickup/item family; exact item differentiation unresolved |
| 6 | Warp Star / travel-star family |
| 8 | Midboss damage explosion |
| 13 | Warp Star landing effect, provisional |
| 14 / 15 | Waddle Dee / Waddle Doo |
| 16 / 17 / 18 | Shotzo / Sparky / Poppy Bros. Jr. |
| 19 | Poppy Bros. Jr. apple |
| 24 | Blipper |
| 32 / 33 | Broom Hatter / Cappy |
| 34 | Poppy Bros. Sr. / bomb family; shared script is not a unique identity |
| 41 / 43 | Hot Head / Sword Knight |
| 56 / 57 | Bronto Burt / Gordo |
| 65 | Cutter/blade family |
| 80 / 81 / 82 | Beam attack / Shotzo projectile / Sparky attack |
| 84 / 86 | Hot Head pre-attack sparks / fire, as currently labeled |
| 87 / 89 | Sword strike / Broom Hatter dust |
| 103 | Crane Fever plush |
| 255 | Empty slot |

Labels are for humans; the encoder retains IDs rather than feeding these names to the model. Family IDs, recycled slots, and stale variables mean labels alone cannot establish attack ownership, collision coverage, or boss state. Seeing a projectile in an encoded slot does not mean the model knows what damage it does.

### Terrain

The reader uses room dimensions at `$67EE/$67EF`, candidate block order at `$67F6`, stored tile data beginning `$68C8`, and attributes at `$7F00 + tile_id`. Blocks contain 16 columns × 12 rows of 16-pixel cells. The neural view covers offsets −4 through +4 around Kirby’s reference cell.

The live-tested block-order offset differed from an early disassembly interpretation; the implementation follows the observed mapping. Horizontal rooms were checked; tall/vertical layouts remain unverified and are rejected by the training reader.

| Hex attribute | Observed meaning |
|---|---|
| `FF` | Air/passable background in tested scenes |
| `0D` | Solid terrain |
| `0F` | Platform/semisolid candidate; retain raw value |
| `12`, `13` | Rising slope pieces in room 44 |
| `15`, `14` | Falling slope pieces in room 44 |
| `1C` | Breakable block |
| `23` | Ladder |
| `28` | Entrance marker; ordinary and goal doors can share it |
| `30` | Water in tested scenes |
| `-1` in JSON | Outside-room cell, distinct from air |

Unknown attributes remain distinct raw values. Door destination and stage completion require separate state evidence. Collision attributes, visible artwork, and enemy attack coverage are different things; decorative waterfall graphics, for example, did not make every covered tile water.

## Troubleshooting

| Symptom | Check / response |
|---|---|
| Lua: `unexpected symbol near '""'` on line 1 | Python text was likely pasted into a Lua file. The shared bridge begins with `-- Shared v8 Kirby RAM/control bridge for BizHawk.` Restore the correct file, then reload the worker startup script. |
| Panel does not open or behaves like a coordinator | Check that `KA_ControlPanel.py` contains the Tk dashboard, not a pasted copy of `KA_Parallel.py`. Its opening docstring describes the read-only dashboard. |
| “Load a generated worker startup script first” | Load `worker_startup/worker_N.lua`, not the shared bridge directly. |
| `Kirby>` instead of `Parallel>` | Start `KA_Parallel.py` or select the correct VS Code debug configuration. |
| Missing latest checkpoint | Use the deliberate import appropriate to the available checkpoint once, then `savemodel`; do not repeatedly migrate ordinary restarts. |
| “Use verify first” | Reconnect released workers, then verify after changing worker count, manual movement, starts, or bridge setup. |
| Evaluation asks for more workers | Select enough workers to include every frozen evaluation ID. Current baseline needs 1–4. |
| Existing target blocks `clonestart` | Back up/rename the destination training state while idle, then clone. No overwrite option currently exists. |
| Evaluation signature changed | Preserve the previous best. Check whether eval states, bot source, or settings changed before establishing a new comparison. |
| Wrong worker identity or launch token | Verify window/startup assignments. Restart through the managed launcher after saving; do not force requests into an uncertain bridge. |
| No reply / frame-count mismatch | Save the in-memory model if the prompt is available, then restart coordinator and affected bridge/emulators. A timed-out command may already have executed. |
| Launcher finds unknown/duplicate emulators | Close conflicting windows or use the intended manual workflow. The launcher will not assume unrelated processes are its workers. |
| Another coordinator is using the project | Close the actual duplicate console. A leftover lock filename alone is not an active OS lock; deleting it is not the normal fix. |
| Telemetry stale while idle | It may simply have no new publication; check coordinator status and `telemetry on`. Stale is not proof of a training failure. |
| Persistent telemetry errors | Check filesystem access/space. V7+ avoids overwriting a snapshot the panel is reading, but disk/permission failures can still occur. |
| CPU instead of CUDA | Check the actual Python/PyTorch printed at startup and VS Code’s selected debugger interpreter. |
| Enemy/room behavior looks wrong after manual play | Use release/reconnect after runs, then verify before automation. Do not mix ordinary gameplay with active training requests. |
| No-progress timeout | It is an attempt truncation, not necessarily a death. Check progress, action choice, terrain, and enemy behavior before changing rewards. |

Dragging/resizing an emulator window can temporarily stall its processing. Short delays usually affect throughput rather than dropping queued actions; the bridge waits for a matching reply. An outstanding request has a 15-second reply timeout. Pause from the coordinator before rearranging windows, and do not discard an otherwise successful run solely because a window was briefly moved.

The game’s live music can sound choppy because the bridge deliberately alternates advancing and pausing. This does not imply game frames were omitted.

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

### V8 — scalable operation and debugging tools, October 9–10, 2026

V8 deliberately preserved the v7 network, observations, actions, rewards, and learning cadence while improving operation and visibility.

1. **Shared bridge:** replaced duplicated numbered Lua implementations with one `KA_Bridge.lua` plus generated per-worker startup files and explicit worker identities.
2. **Dynamic workers:** generalized selected worker count, persistent settings, per-worker epsilon arguments, verification, and benchmarks. Added distinct v8 model/log paths and a compatible v7 import.
3. **Dashboard/history:** added all-worker and focused views, clearer checkpoint status, attempt graphs, session/run/mode/worker/start-room/date filtering, and legacy/archive parsing.
4. **Manual debugging:** added raw movement after attempt completion, Start/Select for manual use, release/reconnect for normal gameplay, and separation from replay/reward/attempt accounting.
5. **Launcher and starts:** added managed emulator launch/reuse, per-worker configs/tokens, saved-start restoration, separate frozen evaluation states/IDs, and protected training-state cloning.
6. **Save-first shutdown:** save the loaded model before requesting managed emulator exits, include recognized deselected workers, ignore unrelated processes, and keep the console available on failure.
7. **RAM recording:** added bounded one-/four-frame probe sampling into separate JSONL files with metadata, raw snapshots, completion markers, and a summary tool.
8. **Integration review:** checked installed files, regression suites, Lua execution with simulated APIs, starts/settings, and read-only checkpoint compatibility. Corrected installation copy/paste mix-ups affecting bridge/panel files; these were file-placement repairs, not learning-algorithm changes.

Operational changes and observations during v8:

- Moved the active project to SSD after the HDD bottleneck was confirmed separately.
- The pretraining review found v7/v8 latest policy and target weights identical at **787,933 updates**. This is a historical snapshot, not the expected current update count after subsequent training.
- The first recorded v8 baseline evaluation completed with zero updates: **0/4 clears**, average **241 new pixels**, **1 health segment lost**, and **794 frames**. It established the first v8 best file; it did not prove v8 regressed or improved against older tests with different conditions.
- User confirmed managed operation, dashboard, shutdown, and real diagnostic samples at frames 0/4/8.
- Training was subsequently run with six workers, initially all in room 43. The user then prepared room-44 starts for workers 5–6 while retaining the four-worker evaluation baseline.
- No PPU capture, attack-coverage mask, new combat label, or additional neural input was added in v8.

## Known limitations and future work

- **Supported scope:** stage 1-1 rooms 43–45 and the observed goal route. Other stages, boss fights, and vertical-map layouts need validation.
- **Enemy interpretation:** shared script families do not fully distinguish Poppy Bros. Sr. from bombs, player items from related effects, or ownership/coverage of all attacks.
- **Attack research:** collect sequences for enemy preparation poses and projectiles, including Hot Head’s fireball, before designing coverage masks or applying findings to Whispy Woods.
- **PPU/OAM research:** optional visual-state sampling every four frames was discussed; acquisition, interpretation, and performance impact are unverified and unimplemented.
- **Representation:** full-room/exit information, grounded/facing/control-lock flags, improved scaling, and stronger temporal memory remain possible experiments requiring compatibility planning.
- **Rewards:** rightward progress is a limited proxy for route completion. New-cell watchdog behavior can allow unhelpful exploration; larger punishments are not automatically the right fix.
- **Evaluation:** a fixed set supports consistent comparisons but is small. Identical or nearly identical starts may produce similar greedy behavior; four workers are not automatically four independent difficulty tests.
- **Performance:** file exchange, emulator scheduling, and fresh-jump requests have overhead. Measure real training throughput before adding more workers or changing transport.
- **Diagnostic UI:** the recorder currently has console commands and a summary script, not a dedicated panel menu.
- **Persistence:** replay and a complete live-session snapshot are not saved. No periodic automatic latest save or fresh-model console command has been added.

Keep experiments separate and preserve working checkpoints. Do not change observation meaning/order, action order/count, or reward metadata without an explicit migration or retraining plan.

## Validation and historical sources

This README distinguishes implemented behavior, user-observed gameplay, simulated checks, and future proposals. It does not claim every game mechanic is mapped or that training quality is proven.

The final v8 review ran eight existing regression scripts covering orchestration, manual control, diagnostics, launcher/start management, shutdown, telemetry/history, and panel construction. The shared bridge also executed under the bundled Lua 5.4 runtime with simulated BizHawk APIs. These tests did not run GPU training or replace live emulator integration tests. User smoke tests supplied the later live confirmation described above.

Primary historical sources:

- **“Start coding the bot”**, October 4–9, 2026: available conversation paged back to its first turn, including pasted-code milestones and subsequent corrections.
- Original foundation files and README in the prior chat’s `outputs/KirbyAdv_Bot/` folder.
- `Reward-rework-notes.md` and `KA_Bot_reward_v3.py`.
- `Context-v4-notes.md` and `KA_Bot_context_v4.py`.
- `Parallel-setup.md` and the initial parallel source/bridges.
- `Time-cost-v5-notes.md` and v5 source files.
- `Kirby_v6/START_HERE.md`, `Kirby_v7/START_HERE.md`, and their source packages.
- **“Gather Kirby Bot Chat Context”**: v8 design, delivered steps, live checks, and six-worker curriculum discussion.
- Installed v8 Python/Lua source: authority for the current command and behavior reference when an older message differs.

Some early milestones were direct pasted edits without a separately named archived release. Their chronology is documented from the chat rather than claiming complete tagged source snapshots exist. Historical setup commands belong to their corresponding release; use the v8 sections above for current operation.
