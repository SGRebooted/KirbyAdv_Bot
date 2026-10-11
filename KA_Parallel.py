"""Configurable BizHawk workers sharing the existing Kirby model and replay.

Save beside KA_Bot.py. Startup files are prepared automatically; load each
worker_startup/worker_N.lua in its own emulator, or use the launch command.
Only environment/bridge calls use threads; all PyTorch work stays on main.
"""

from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from dataclasses import dataclass
from datetime import datetime, timezone
import csv
import hashlib
import json
import os
from pathlib import Path
import time
import uuid

from KA_Telemetry import Telemetry, archive_log, read_key
import KA_Bot as bot
from KA_Workers import prepare_worker
from KA_Launcher import Launcher
from KA_Autosave import AutoCheckpoint, checkpoints, checkpoint_at
from KA_Diagnostics import capture as capture_diagnostics
from KA_Starts import evaluation_workers, freeze_evaluation, clone_training_start

ROOT = Path(__file__).resolve().parent
LATEST_PATH = ROOT / "models" / "kirby_parallel_v8_latest.pth"
V7_PATH = ROOT / "models" / "kirby_parallel_v7_latest.pth"
BEST_PATH = ROOT / "models" / "kirby_parallel_v8_best.pth"
LOG_PATH = ROOT / "logs" / "parallel_v8_attempts.csv"
SETTINGS_PATH = ROOT / "parallel_settings.json"
LOG_FIELDS = [
    "schema_version", "session_id", "run_id", "run_started_utc", "utc",
    "mode", "worker", "episode", "epsilon", "steps", "frames", "reward",
    "health_lost", "new_progress", "rooms", "success", "reason", "updates",
    "run_start_updates", "start_level", "start_stage", "start_room",
    "start_x", "start_y", "end_room", "visited_rooms",
    "progress_reward", "damage_penalty", "death_penalty", "time_penalty",
    "room_bonus", "goal_bonus",
]


def configured_worker_count():
    """Remember the last selected count; a fresh setup defaults to four."""
    if not SETTINGS_PATH.exists():
        return 4
    data = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
    count = data.get("worker_count") if isinstance(data, dict) else None
    if type(count) is not int or count < 1:
        raise ValueError("parallel_settings.json needs a positive integer worker_count.")
    return count


def check_log_header():
    """Refuse to mix incompatible CSV layouts before moving any workers."""
    if LOG_PATH.exists() and LOG_PATH.stat().st_size:
        with LOG_PATH.open(newline="", encoding="utf-8") as stream:
            if next(csv.reader(stream), None) != LOG_FIELDS:
                raise ValueError("v8 log columns differ; archive that CSV before running.")

REPLAY_CAPACITY = 20000


class WorkerBridge(bot.KirbyBridge):
    """Reject replies from a wrongly assigned or old Lua bridge."""

    def __init__(self, worker_id):
        self.worker_id = worker_id
        self.start_kind = "train"
        super().__init__(ROOT / f"bridge_{worker_id}")

    def request(self, command, frames=0, buttons=()):
        # The same environment reset logic can load either start profile.
        if command == "reset" and self.start_kind == "eval":
            command = "reseteval"
        reply = super().request(command, frames, buttons)
        if reply.get("worker_id") != self.worker_id:
            self.failed = True
            raise RuntimeError(f"Worker {self.worker_id}: wrong bridge identity. Reload its worker_startup script.")
        return reply


def make_environment(worker_id):
    """Each worker gets all of its own timers, room records, and file paths."""
    environment = bot.KirbyEnvironment()
    environment.bridge = WorkerBridge(worker_id)
    return environment


def save_checkpoint(trainer, path=LATEST_PATH, evaluation=None):
    """Write a complete checkpoint without overwriting the single-worker file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "format_version": 1,
        "encoder_version": bot.ENCODER_VERSION,
        "training_version": bot.TRAINING_VERSION,
        "observation_size": bot.OBSERVATION_SIZE,
        "actions": list(bot.MODEL_ACTIONS),
        "model": trainer.model.state_dict(),
        "target": trainer.target.state_dict(),
        "optimizer": trainer.optimizer.state_dict(),
        "updates": trainer.updates,
        "evaluation": evaluation,
    }
    temporary = path.with_suffix(".tmp")
    bot.torch.save(checkpoint, temporary)
    os.replace(temporary, path)


def load_checkpoint(path):
    """Reuse the existing strict loader, preserving optimizer and update count."""
    # Only the main thread loads checkpoints, while workers are idle.
    previous = bot.MODEL_PATH
    try:
        bot.MODEL_PATH = path
        return bot.load_model()
    finally:
        bot.MODEL_PATH = previous


@dataclass
class Attempt:
    """Statistics owned by the coordinator, never mutated by bridge threads."""

    worker: int
    episode: int = 0
    state: object = None
    steps: int = 0
    frames: int = 0
    reward: float = 0.0
    health_lost: float = 0.0
    progress: int = 0
    reported: bool = True
    breakdown: object = None
    start_location: object = None


def write_result(result):
    """Append one completed or interrupted attempt for later comparisons."""
    check_log_header()
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not LOG_PATH.exists() or LOG_PATH.stat().st_size == 0
    with LOG_PATH.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=LOG_FIELDS)
        if needs_header:
            writer.writeheader()
        writer.writerow(result)


class Coordinator:
    """Own the one model, optimizer, replay buffer, and independent workers."""

    def __init__(self):
        self.environments = {}
        self.trainer = bot.DQNTrainer(bot.KirbyQNetwork().to(bot.DEVICE))
        self.replay = bot.ReplayBuffer(REPLAY_CAPACITY)
        self.stored_total = 0
        self.model_loaded = False
        self.verified = False
        self.worker_ids = ()
        self.released_workers = set()
        self.launcher = Launcher(ROOT)
        self.telemetry = Telemetry(ROOT)
        self.last_loss = None
        self.telemetry.checkpoint = {"loaded_from": "No model loaded", "saved_to": "None this session",
                                     "saved_updates": None, "loaded_updates": None}
        # Match telemetry's session ID so the panel can join both data sources.
        self.session_id = self.telemetry.session
        self.select_workers(configured_worker_count(), remember=False)

    def select_workers(self, count, remember=True):
        """Change the active set at the console, after run() has drained I/O.

        Previously created environments stay available internally, but only
        selected IDs can receive console actions or participate in a run.
        This prepares startup files; it does not launch or close emulators.
        """
        if type(count) is not int or count < 1:
            raise ValueError("Use workers N with a positive integer N.")
        selected = tuple(range(1, count + 1))
        for worker_id in selected:
            prepare_worker(worker_id, ROOT)
            if worker_id not in self.environments:
                self.environments[worker_id] = make_environment(worker_id)
        if remember:
            # Preserve future settings fields when changing only the count.
            settings = json.loads(SETTINGS_PATH.read_text(encoding="utf-8")) if SETTINGS_PATH.exists() else {}
            if not isinstance(settings, dict):
                raise ValueError("parallel_settings.json must contain a JSON object.")
            settings["worker_count"] = count
            temporary = SETTINGS_PATH.with_suffix(".tmp")
            temporary.write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
            os.replace(temporary, SETTINGS_PATH)
        self.worker_ids = selected
        self.verified = False
        self.telemetry.workers = {}
        print(f"Selected {count} workers. Startup scripts: {ROOT / 'worker_startup'}")
        print("Use one startup script per emulator. Existing starts are retained; verify before running.")

    def load(self, path):
        """Loading is explicit and clears experiences tied to the prior run."""
        restored = load_checkpoint(path)
        self.trainer = restored
        self.replay = bot.ReplayBuffer(REPLAY_CAPACITY)
        self.stored_total = 0
        self.model_loaded = True
        self.last_loss = None
        self.telemetry.checkpoint = {"loaded_from": str(path), "loaded_updates": restored.updates,
                                     "saved_to": "None since this load", "saved_updates": None}
        self.telemetry.publish(self.trainer, len(self.replay), self.last_loss, force=True)
        print(f"Loaded update {restored.updates} from {path.name}; replay cleared.")

    def save(self, path=LATEST_PATH, evaluation=None):
        """Update the panel only after the checkpoint write succeeds."""
        save_checkpoint(self.trainer, path, evaluation)
        self.telemetry.checkpoint.update(saved_to=str(path), saved_updates=self.trainer.updates)
        self.telemetry.publish(self.trainer, len(self.replay), self.last_loss, force=True)

    def update_worker(self, context, epsilon, action="", info=None):
        """Publish small display fields; never modify observations or rewards."""
        info = info or {}
        player = (context.state or {}).get("kirby", {})
        self.telemetry.workers[str(context.worker)] = {
            "attempt": context.episode, "status": "running",
            "room": player.get("room"), "hp": player.get("hp"),
            "lives": player.get("lives"), "ability": player.get("ability"),
            "x": player.get("x"), "y": player.get("y"),
            "action": action, "epsilon": epsilon,
            "reward": round(context.reward, 3), "frames": context.frames,
            "watchdog_seconds": info.get("watchdog_frames", 0) / 60,
            "watchdog_limit_seconds": bot.NO_PROGRESS_LIMIT / 60,
            "watchdog_reset": info.get("watchdog_reset", "reset"),
            "room_frames_remaining": info.get("room_frames_remaining"),
            "stop_reason": info.get("stop_reason", ""), "rewards": dict(context.breakdown or {}),
        }

    def launch_workers(self):
        """Start/reuse managed emulators, prove bridge identity, then reset starts."""
        self.verified = False
        ready = self.launcher.launch(self.worker_ids)
        for worker_id, identity in ready.items():
            if identity["new"]:
                # A newly created emulator gets fresh Python-side bookkeeping.
                self.environments[worker_id] = make_environment(worker_id)
                self.released_workers.discard(worker_id)
            bridge = self.environments[worker_id].bridge
            state = bridge.request("hello")
            if state.get("bridge_version") != 5 or state.get("launch_token") != identity["token"]:
                bridge.failed = True
                raise RuntimeError(f"W{worker_id}: launch identity mismatch; no automation started.")
            if state.get("control_mode") == "freeplay":
                self.released_workers.add(worker_id)
                raise ValueError(f"W{worker_id} is in freeplay. Use reconnect {worker_id}, then launch again.")
        self.verify()
        self.launcher.open_panel()
        print("All selected workers loaded their training starts and passed verification. No training started.")

    def shutdown(self):
        """Save first, close managed emulators, and exit only on complete success.

        The console accepts this command only after run() returns, so all of
        its outstanding requests have already drained. Q returns there safely.
        """
        if self.model_loaded:
            try:
                self.save()
            except (Exception, KeyboardInterrupt) as error:
                print(f"Shutdown cancelled: checkpoint save failed/interrupted ({error}).")
                print("No emulator close requests were sent. Model and replay remain in memory.")
                return False
            print(f"Saved update {self.trainer.updates} to {LATEST_PATH}.")
        else:
            print("No model loaded: skipping checkpoint save; existing checkpoints are untouched.")

        self.verified = False
        self.telemetry.status = "shutting down"
        self.telemetry.publish(self.trainer, len(self.replay), None, force=True)

        def close_worker(worker_id, token):
            # Include managed workers that have since been deselected.
            environment = self.environments.get(worker_id)
            bridge = environment.bridge if environment is not None else WorkerBridge(worker_id)
            state = bridge.request("hello")
            if state.get("launch_token") != token or not state.get("shutdown_supported"):
                raise RuntimeError("Shutdown identity/capability check failed; reload the updated bridge or close this window manually.")
            acknowledgement = bridge.request("shutdown", buttons=(token,))
            if (acknowledgement.get("launch_token") != token
                    or not acknowledgement.get("shutdown_requested")):
                raise RuntimeError("Bridge did not acknowledge shutdown.")

        try:
            result = self.launcher.shutdown(close_worker)
        except (Exception, KeyboardInterrupt) as error:
            print(f"Could not complete/verify emulator shutdown: {error or 'interrupted'}.")
            print("The console remains open with model/replay in memory. No force-kill was used.")
            self.telemetry.status = "shutdown incomplete"
            self.telemetry.publish(self.trainer, len(self.replay), None, force=True)
            return False
        for message in result["errors"]:
            print(message)
        for worker_id in result["closed"]:
            self.released_workers.discard(worker_id)
            self.telemetry.workers.setdefault(str(worker_id), {}).update(status="closed", stop_reason="Shutdown confirmed")
        if result["remaining"]:
            print("These managed workers are still open: " + ", ".join(map(str, result["remaining"])))
            print("The console remains open. Check their windows for a dialog or Lua error, then retry shutdown.")
            self.telemetry.status = "shutdown incomplete"
            self.telemetry.publish(self.trainer, len(self.replay), None, force=True)
            return False
        self.telemetry.status = "shutdown complete"
        self.telemetry.frames_per_second = 0.0
        self.telemetry.publish(self.trainer, len(self.replay), None, force=True)
        print("Managed emulators closed. Other emulator windows and the read-only panel were left alone.")
        print("Replay is memory-only and is not restored on the next launch.")
        return True

    def console_worker(self, worker_id):
        """Console commands are accepted only after run() drains its requests."""
        if worker_id not in self.worker_ids:
            raise ValueError("Choose one of the currently selected worker IDs.")
        return self.environments[worker_id]

    def require_controlled(self, worker_id):
        environment = self.console_worker(worker_id)
        if worker_id in self.released_workers:
            raise ValueError(f"Use reconnect {worker_id} before issuing controlled commands.")
        return environment

    def show_manual(self, worker_id, state, action="observe", frames=0):
        """A separate display snapshot, with no attempt/reward bookkeeping."""
        player = state["kirby"]
        free = state.get("control_mode") == "freeplay"
        if free:
            self.released_workers.add(worker_id)
            self.verified = False
        self.telemetry.status = "manual"
        self.telemetry.mode = "manual"
        # Manual snapshots must not look like new results from the last run.
        self.telemetry.run_id = ""
        self.telemetry.frames_per_second = 0.0
        self.telemetry.workers[str(worker_id)] = {
            "status": "freeplay" if free else "manual", "attempt": "—",
            "room": player.get("room"), "hp": player.get("hp"),
            "lives": player.get("lives"), "ability": player.get("ability"),
            "x": player.get("x"), "y": player.get("y"),
            "action": action, "frames": frames, "reward": 0.0, "rewards": {},
            "stop_reason": "Manual snapshot; no learning or attempt accounting",
        }
        # None prevents a manual command from creating a sampled loss row.
        self.telemetry.publish(self.trainer, len(self.replay), None, force=True)

    def manual_step(self, worker_id, action, frames):
        """Execute raw controls without calling KirbyEnvironment.step().

        This deliberately ignores attempt_finished, rewards, watchdogs, and
        supported-room checks. It never writes an attempt CSV or replay item.
        Jump macros release A for one frame, then press for the remaining frames.
        """
        actions = {**bot.ACTIONS, "start": ["Start"], "select": ["Select"]}
        if action not in actions:
            raise ValueError("Manual actions: " + " ".join(actions))
        if type(frames) is not int or not 1 <= frames <= 60:
            raise ValueError("Manual steps require 1–60 frames.")
        jumps = {"jump": "wait", "left_jump": "left", "right_jump": "right"}
        if action in jumps and frames < 2:
            raise ValueError("Fresh jumps require 2–60 frames; use a for a one-frame A press.")
        environment = self.require_controlled(worker_id)
        self.verified = False
        before = environment.bridge.observe()
        if before.get("control_mode") == "freeplay":
            self.show_manual(worker_id, before)
            raise ValueError(f"Use reconnect {worker_id} first.")
        if action in jumps:
            parts = [(1, actions[jumps[action]]), (frames - 1, actions[action])]
        else:
            parts = [(frames, actions[action])]
        previous = before
        for count, buttons in parts:
            state = environment.bridge.request("manual", count, buttons)
            if state["frame"] - previous["frame"] != count:
                environment.bridge.failed = True
                raise RuntimeError("Manual frame count mismatch; restart the bridge and coordinator before continuing.")
            previous = state
        self.show_manual(worker_id, state, action, frames)
        print(f"W{worker_id}: manual {action}, {frames} frames. No learning; verify before automation.")
        return state

    def probe_worker(self, worker_id, frames, stride=4, action="wait"):
        """A bounded manual recording; never calls environment.step or trainer."""
        actions = {**bot.ACTIONS, "start": ["Start"], "select": ["Select"]}
        # Macros would re-trigger a fresh jump at every sample. Use held A instead.
        for macro in ("jump", "left_jump", "right_jump"):
            actions.pop(macro, None)
        if action not in actions:
            raise ValueError("Probe holds one input. Actions: " + " ".join(actions))
        environment = self.require_controlled(worker_id)
        self.verified = False
        try:
            path, state = capture_diagnostics(ROOT, environment.bridge, worker_id,
                                             self.session_id, frames, stride,
                                             action, actions[action])
        except KeyboardInterrupt:
            environment.bridge.failed = True
            print("Probe interrupted; a request may still be finishing. Restart bridge/coordinator before continuing.")
            print("Any partial recording is in diagnostics/; no learning was performed.")
            return
        self.show_manual(worker_id, state, "probe " + action, frames)
        print(f"RAM recording: {path}")
        print("No learning or attempt log entries. Worker paused; verify before automation.")
        return path

    def release_worker(self, worker_id):
        """Let normal emulator/controller input run until explicit reconnect."""
        environment = self.console_worker(worker_id)
        self.verified = False
        state = environment.bridge.request("release")
        if state.get("control_mode") != "freeplay":
            raise RuntimeError("Reload the step-4 shared Lua bridge; freeplay was not acknowledged.")
        self.released_workers.add(worker_id)
        self.show_manual(worker_id, state, "released to controller")
        print(f"W{worker_id}: ordinary gameplay enabled. Use reconnect {worker_id} when finished.")
        return state

    def reconnect_worker(self, worker_id):
        """Pause and observe current RAM; do not save or reset the game."""
        environment = self.console_worker(worker_id)
        self.verified = False
        state = environment.bridge.request("reconnect")
        if state.get("control_mode") != "controlled":
            raise RuntimeError("Reload the step-4 shared Lua bridge; reconnect was not acknowledged.")
        self.released_workers.discard(worker_id)
        self.show_manual(worker_id, state, "reconnected")
        print(f"W{worker_id}: paused with a fresh snapshot. Saved start unchanged; verify before automation.")
        return state

    def verify(self):
        """Reset selected workers; advancing one must leave every other frozen."""
        self.verified = False
        released = self.released_workers.intersection(self.worker_ids)
        if released:
            raise ValueError("Reconnect released workers first: " + ", ".join(map(str, sorted(released))))
        for i in self.worker_ids:
            self.environments[i].bridge.start_kind = "train"
            self.environments[i].reset()
        for active in self.worker_ids:
            before = {i: e.observe()["frame"] for i, e in self.environments.items() if i in self.worker_ids}
            self.environments[active].step("wait", 4)
            after = {i: e.observe()["frame"] for i, e in self.environments.items() if i in self.worker_ids}
            if after[active] - before[active] != 4 or any(after[i] != before[i] for i in self.worker_ids if i != active):
                raise RuntimeError("Isolation check failed. Use separate emulator windows and one worker startup script per window.")
        for i in self.worker_ids:
            self.environments[i].bridge.start_kind = "train"
            self.environments[i].reset()
        self.telemetry.status = "idle"
        self.telemetry.mode = ""
        self.telemetry.run_id = ""
        self.telemetry.frames_per_second = 0.0
        for i in self.worker_ids:
            state = self.environments[i].observe()
            context = Attempt(i, state=state)
            self.update_worker(context, 0.0)
            self.telemetry.workers[str(i)]["status"] = "ready"
        self.telemetry.publish(self.trainer, len(self.replay), None, force=True)
        self.verified = True
        print("Verified: each worker advances independently. All selected workers restored to their saved starts.")

    def run(self, worker_ids=None, attempts=1, epsilons=None,
            training=False, step_limit=None, mode="play", start_kind="train"):
        """Overlap emulator requests, with one outstanding request per worker.

        The main thread selects actions, receives results, and performs updates.
        Threads never use the model or replay, avoiding inference/update races.
        """
        worker_ids = self.worker_ids if worker_ids is None else worker_ids
        if not self.model_loaded:
            raise ValueError("Use importv7 or loadmodel first.")
        if not self.verified:
            raise ValueError("Use verify first; every selected worker needs a saved starting state.")
        if type(attempts) is not int or not 1 <= attempts <= 1000:
            raise ValueError("Choose 1–1000 attempts per worker.")
        if not worker_ids or len(set(worker_ids)) != len(worker_ids) or any(i not in self.worker_ids for i in worker_ids):
            raise ValueError("Choose a nonempty subset of the selected workers.")
        epsilons = epsilons or {i: 0.0 for i in worker_ids}
        if set(epsilons) != set(worker_ids):
            raise ValueError("Supply one exploration value for every participating worker.")
        if any(not 0 <= epsilons[i] <= 1 for i in worker_ids):
            raise ValueError("Exploration must be between 0 and 1.")
        if step_limit is not None and (type(step_limit) is not int or not 1 <= step_limit <= bot.MAX_ATTEMPT_FRAMES):
            raise ValueError("Invalid step limit.")

        if start_kind not in {"train", "eval"} or (training and start_kind != "train"):
            raise ValueError("Training must use training starts; start_kind must be train or eval.")
        for worker_id in worker_ids:
            self.environments[worker_id].bridge.start_kind = start_kind
        check_log_header()
        # One ID per command/run, shared by all its workers and attempts.
        run_id = uuid.uuid4().hex
        run_started_utc = datetime.now(timezone.utc).isoformat()
        contexts = {i: Attempt(i) for i in worker_ids}
        paused = False
        self.telemetry.workers = {}
        self.telemetry.status = mode
        self.telemetry.mode = mode
        self.telemetry.run_id = run_id
        self.telemetry.frames_per_second = 0.0
        pending = {}
        results = []
        stopping = False
        failed = False
        total_frames = total_steps = 0
        paused_seconds = 0.0
        pause_started = None
        started = time.perf_counter()
        starting_updates = self.trainer.updates
        # Only training creates automatic checkpoints. Evaluation/bench/manual
        # work never triggers this writer or its retention cleanup.
        automatic = AutoCheckpoint(ROOT, self.save) if training else None

        def report(context, reason, success=False):
            """Write each episode once, including partial episodes on Q/errors."""
            if context.reported:
                return
            context.reported = True
            environment = self.environments[context.worker]
            start = context.start_location or {}
            visited = sorted(environment.visited_rooms) if context.start_location is not None else []
            result = {
                "schema_version": 8, "session_id": self.session_id,
                "run_id": run_id, "run_started_utc": run_started_utc,
                "run_start_updates": starting_updates,
                "start_level": start.get("level", ""), "start_stage": start.get("stage", ""),
                "start_room": start.get("room", ""),
                "start_x": start.get("x", ""), "start_y": start.get("y", ""),
                "end_room": context.state["kirby"]["room"] if context.state is not None else "",
                "visited_rooms": json.dumps(visited, separators=(",", ":")),
                "utc": datetime.now(timezone.utc).isoformat(),
                "mode": mode, "worker": context.worker,
                "episode": context.episode, "epsilon": epsilons[context.worker],
                "steps": context.steps, "frames": context.frames,
                "reward": round(context.reward, 6),
                "health_lost": context.health_lost,
                "new_progress": context.progress,
                "rooms": len(visited),
                "success": bool(success), "reason": reason,
                "updates": self.trainer.updates,
            }
            results.append(result)
            result.update(context.breakdown or {})
            write_result(result)
            display = self.telemetry.workers.setdefault(str(context.worker), {})
            display.update(status="error" if "error" in reason.lower() else "finished",
                           stop_reason=reason, attempt=context.episode)
            print(f"W{context.worker} attempt {context.episode}/{attempts}: {reason} | "
                  f"reward {context.reward:.2f} | rooms {result['rooms']} | "
                  f"damage {context.health_lost:g} | frames {context.frames}")

        print(f"{mode}: {attempts} attempt(s) per worker; {len(worker_ids)} worker(s). Q stops safely; P pauses/resumes; S saves while fully paused.")
        if automatic is not None:
            print("Autosave: every 150 completed attempts across all workers or 30 minutes; also on finish/Q. Keeps newest 3 in models/autosave/.")
        with ThreadPoolExecutor(max_workers=len(worker_ids)) as executor:
            def reset_worker(worker_id):
                """Reset in that worker's thread without blocking the other one."""
                old = contexts[worker_id]
                contexts[worker_id] = Attempt(worker_id, episode=old.episode + 1, reported=False, breakdown={k: 0.0 for k in ("progress_reward", "damage_penalty", "death_penalty", "time_penalty", "room_bonus", "goal_bonus")})
                self.telemetry.workers[str(worker_id)] = {"status": "resetting", "attempt": contexts[worker_id].episode}
                future = executor.submit(self.environments[worker_id].reset)
                pending[future] = (worker_id, "reset", None, None)

            def act(worker_id):
                """Take a snapshot and choose on main; submit only environment I/O."""
                context = contexts[worker_id]
                environment = self.environments[worker_id]
                if not environment.ready(context.state) or environment.room_key(context.state) not in bot.GAMEPLAY_ROOMS:
                    raise RuntimeError(f"W{worker_id}: unsupported room or unavailable terrain.")
                observation = bot.encode_observation(context.state)
                action, _ = bot.choose_action(self.trainer.model, observation, epsilons[worker_id])
                future = executor.submit(environment.step, action, 4)
                pending[future] = (worker_id, "step", observation, action)

            for worker_id in worker_ids:
                reset_worker(worker_id)

            while pending or (paused and not stopping):
                # Synchronous on main, never in the middle of an optimizer
                # update. Bridge threads can finish their outstanding replies.
                if automatic is not None and not stopping:
                    automatic.maybe_save(self.trainer.updates)
                key = read_key()
                if key == "q":
                    stopping = True
                    print("Stopping after outstanding requests finish.")
                elif key == "p" and not stopping:
                    if paused and pending:
                        print("Still draining requests; press P again once fully paused.")
                    else:
                        paused = not paused
                        print("Pausing after outstanding requests finish." if paused else "Resuming.")
                elif key == "s" and paused and not pending:
                    try:
                        self.save()
                        print(f"Saved update {self.trainer.updates} while paused.")
                    except OSError as error:
                        print(f"Save failed; still paused with model in memory: {error}")
                self.telemetry.status = "stopping" if stopping else ("pausing" if paused and pending else "paused" if paused else mode)
                self.telemetry.publish(self.trainer, len(self.replay), self.last_loss)
                # No request is submitted while paused. A finished attempt is
                # held until resume, then reset only if another was requested.
                if not pending:
                    if stopping:
                        break
                    if paused:
                        if pause_started is None:
                            pause_started = time.perf_counter()
                        time.sleep(0.05)
                        continue
                    if pause_started is not None:
                        paused_seconds += time.perf_counter() - pause_started
                        pause_started = None
                    for wid, ctx in contexts.items():
                        try:
                            if ctx.reported:
                                if ctx.episode < attempts:
                                    reset_worker(wid)
                            else:
                                act(wid)
                        except Exception as error:
                            stopping = failed = True
                            self.verified = False
                            report(ctx, f"error on resume: {error}")
                            break
                    if not pending:
                        break
                done, _ = wait(pending, timeout=0.05, return_when=FIRST_COMPLETED)
                for future in done:
                    worker_id, operation, observation, action = pending.pop(future)
                    context = contexts[worker_id]
                    try:
                        response = future.result()
                        if operation == "reset":
                            context.state = response
                            context.start_location = dict(response["kirby"])
                            self.update_worker(context, epsilons[worker_id])
                        else:
                            state, reward, terminated, truncated, info = response
                            context.state = state
                            context.steps += 1
                            context.frames += info["elapsed_frames"]
                            context.reward += reward
                            context.health_lost += info["health_lost"]
                            context.progress += info["new_progress"]
                            for name in context.breakdown:
                                context.breakdown[name] += info.get(name, 0.0)
                            self.update_worker(context, epsilons[worker_id], action, info)
                            total_frames += info["elapsed_frames"]
                            total_steps += 1
                            self.telemetry.frames_per_second = total_frames / max(time.perf_counter() - started - paused_seconds, 0.001)

                            # Train only from training runs. Evaluation and
                            # benchmarks never alter replay, weights, or counts.
                            if training and info["replay_ok"]:
                                next_inputs = [0.0] * bot.OBSERVATION_SIZE if terminated else bot.encode_observation(state)
                                self.replay.add(
                                    observation=observation,
                                    action_index=bot.MODEL_ACTIONS.index(action),
                                    reward=info["learning_reward"],
                                    next_observation=next_inputs,
                                    terminated=terminated, truncated=truncated,
                                    frames=info["elapsed_frames"],
                                )
                                self.stored_total += 1
                                # Count across ALL workers: do not multiply the
                                # update frequency when adding an emulator.
                                if not stopping and len(self.replay) >= 128 and self.stored_total % 4 == 0:
                                    self.last_loss = self.trainer.train_step(self.replay)
                            if info["first_visit"] or info["success"]:
                                print(f"W{worker_id}: room {state['kirby']['room']} | "
                                      f"room bonus +{info['room_bonus']:g} | goal +{info['goal_bonus']:g}")
                            if total_steps % 100 == 0:
                                seconds = max(time.perf_counter() - started - paused_seconds, 0.001)
                                print(f"Combined: {total_steps / seconds:.1f} decisions/s | "
                                      f"{total_frames / seconds:.0f} game frames/s | "
                                      f"updates {self.trainer.updates} | replay {len(self.replay)}")

                            if info["stop_reason"]:
                                report(context, info["stop_reason"], info["success"])
                                if automatic is not None and (terminated or truncated):
                                    automatic.completed_attempt()
                                if not (terminated or truncated):
                                    stopping = failed = True
                            elif step_limit is not None and context.steps >= step_limit:
                                report(context, "requested step limit")
                                if automatic is not None:
                                    automatic.completed_attempt()

                        if not stopping and not paused:
                            if context.reported:
                                if context.episode < attempts:
                                    reset_worker(worker_id)
                            else:
                                act(worker_id)
                    except Exception as error:
                        # Drain other workers before accepting new commands.
                        # A timeout is not retried: its game outcome is uncertain.
                        stopping = failed = True
                        self.verified = False
                        if isinstance(error, (TimeoutError, OSError)):
                            self.environments[worker_id].bridge.failed = True
                        report(context, f"error: {error}")
                        print(f"W{worker_id}: {error}. Save the model before restarting a failed bridge.")

            for context in contexts.values():
                report(context, "stopped by Q" if not failed else "stopped after worker error")

        # All outstanding requests have drained before this final checkpoint.
        # Worker errors return here too; partial/Q attempts do not increment
        # the periodic completion counter but learned weights are still saved.
        if automatic is not None:
            reason = "worker error stop" if failed else "stopped by Q" if stopping else "training finished"
            automatic.maybe_save(self.trainer.updates, final_reason=reason)
        if pause_started is not None:
            paused_seconds += time.perf_counter() - pause_started
        self.telemetry.status = "idle"
        self.telemetry.publish(self.trainer, len(self.replay), self.last_loss, force=True)
        seconds = max(time.perf_counter() - started - paused_seconds, 0.001)
        summary = {
            "session_id": self.session_id, "run_id": run_id,
            "seconds": seconds, "steps": total_steps, "frames": total_frames,
            "decisions_per_second": total_steps / seconds,
            "frames_per_second": total_frames / seconds,
            "new_updates": self.trainer.updates - starting_updates,
            "interrupted": stopping, "results": results,
        }
        print(f"Finished in {seconds:.1f}s | {total_frames / seconds:.0f} combined game frames/s | "
              f"{summary['new_updates']} new updates. No latest checkpoint saved automatically.")
        return summary

    def evaluation_signature(self):
        """Only compare best scores across identical starts and reward settings."""
        material = {
            "states": [hashlib.sha256((ROOT / f"bridge_{i}" / "eval.State").read_bytes()).hexdigest()
                       for i in evaluation_workers(ROOT)],
            "encoder": bot.ENCODER_VERSION, "training": bot.TRAINING_VERSION,
            "room_budget": bot.ROOM_FRAME_BUDGET, "overall_budget": bot.MAX_ATTEMPT_FRAMES,
            "grace": bot.PROGRESS_GRACE_FRAMES,
            # A code change starts a fresh comparison, even without a version bump.
            "bot_code": hashlib.sha256(Path(bot.__file__).read_bytes()).hexdigest(),
        }
        return hashlib.sha256(json.dumps(material, sort_keys=True).encode()).hexdigest()

    def evaluate(self):
        """Use the fixed evaluation worker set and evaluation savestates."""
        ids = evaluation_workers(ROOT)
        if any(i not in self.worker_ids for i in ids):
            raise ValueError("Select enough workers to include the frozen evaluation set: " + str(ids))
        signature = self.evaluation_signature()
        summary = self.run(worker_ids=ids, epsilons={i: 0.0 for i in ids}, mode="eval", start_kind="eval")
        if summary["interrupted"] or len(summary["results"]) != len(ids):
            print("Incomplete evaluation; best checkpoint unchanged.")
            return
        # Rotate only after a complete evaluation, never between its workers.
        try:
            archived = archive_log(LOG_PATH)
            if archived:
                print(f"Archived attempt log: {archived}")
        except OSError as error:
            print(f"Log archive deferred: {error}")
        results = summary["results"]
        count = len(results)
        # Favor clear rate, then rooms reached, new progress, less damage, speed.
        # These are comparison metrics, not extra training rewards.
        rank = [
            sum(r["success"] for r in results) / count,
            sum(r["rooms"] for r in results) / count,
            sum(r["new_progress"] for r in results) / count,
            -sum(r["health_lost"] for r in results) / count,
            -sum(r["frames"] for r in results) / count,
        ]
        previous = None
        if BEST_PATH.exists():
            previous = bot.torch.load(BEST_PATH, map_location="cpu", weights_only=True).get("evaluation")
        if previous and previous["signature"] != signature:
            print("Evaluation starts/settings changed. Existing best kept; archive/rename it before establishing a new best.")
            return
        if previous is None or tuple(rank) > tuple(previous["rank"]):
            self.save(BEST_PATH, {"signature": signature, "rank": rank})
            print(f"Saved new best evaluated checkpoint: {BEST_PATH.name}")
        else:
            print("Best evaluated checkpoint retained.")
        print(f"Evaluation: {sum(r['success'] for r in results)}/{count} clears; comparison rank {rank}")

    def benchmark(self, steps=120):
        """Compare collection throughput only; use ptrain to measure learning."""
        if not 1 <= steps <= 900:
            raise ValueError("Choose 1–900 benchmark decisions per worker.")
        full_count = len(self.worker_ids)
        counts = (1,) if full_count == 1 else (max(1, full_count // 2), full_count)
        rates = []
        for count in counts:
            ids = self.worker_ids[:count]
            result = self.run(ids, epsilons={i: 0.1 for i in ids}, step_limit=steps, mode=f"bench-{count}")
            if result["interrupted"]:
                return
            rates.append(result["frames_per_second"])
        if len(rates) == 2:
            print(f"Collection speedup ({counts[0]} vs {counts[1]} workers): {rates[1] / max(rates[0], 1e-9):.2f}x. Compare ptrain throughput too.")
        else:
            print(f"Single-worker collection: {rates[0]:.0f} game frames/s.")


def main():
    """A separate console keeps the single-worker bot and checkpoint intact."""
    # A second coordinator would collide with command files. Use an OS lock
    # released automatically when the process closes (including after a crash).
    import msvcrt
    lock = (ROOT / "parallel.lock").open("a+b")
    lock.seek(0, 2)
    if lock.tell() == 0:
        lock.write(b"0")
        lock.flush()
    lock.seek(0)
    try:
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError:
        lock.close()
        print("Another parallel console is already using this project.")
        return
    try:
        app = Coordinator()
        print(f"Parallel device: {bot.DEVICE}; shared replay capacity {REPLAY_CAPACITY}.")
        print(f"Session: {app.session_id}. Type help.")
        while True:
            try:
                parts = input("Parallel> ").lower().split()
                if not parts:
                    continue
                command = parts[0]
                if command == "help":
                    print("workers N | read ID | save ID | reset ID | step ID ACTION FRAMES | verify")
                    print("step is manual-only (1–60 frames, no learning). release ID | reconnect ID")
                    print("launch | freezeeval [replace] | saveeval ID | clonestart SOURCE TARGET")
                    print("probe ID FRAMES [STRIDE=4] [ACTION=wait] (0–600 frames; stride 1 or 4)")
                    print("telemetry on/off | importv7 | importprevious | importsingle | loadmodel | loadbest | savemodel | bench [steps]")
                    print("ptrain [attempts PER WORKER] [one epsilon per selected worker] | eval | shutdown | quit")
                    print("autosaves | loadauto [1/2/3] (1=newest; loading clears replay)")
                    print("Defaults: 3 attempts each; alternating epsilons 0.1/0.3 for any worker count. Q stop; P pause/resume; S save when paused.")
                elif command == "probe":
                    if not 3 <= len(parts) <= 5:
                        raise ValueError("Use probe ID FRAMES [STRIDE=4] [ACTION=wait].")
                    app.probe_worker(int(parts[1]), int(parts[2]),
                                     int(parts[3]) if len(parts) >= 4 else 4,
                                     parts[4] if len(parts) == 5 else "wait")
                elif command == "launch" and len(parts) == 1:
                    app.launch_workers()
                elif command == "freezeeval" and (len(parts) == 1 or parts == ["freezeeval", "replace"]):
                    freeze_evaluation(ROOT, app.worker_ids, replace=len(parts) == 2)
                    print(f"Frozen evaluation workers {app.worker_ids}; future training start changes leave these untouched.")
                elif command == "clonestart" and len(parts) == 3:
                    source_id, target_id = int(parts[1]), int(parts[2])
                    app.console_worker(source_id)
                    app.console_worker(target_id)
                    app.verified = False
                    path = clone_training_start(ROOT, source_id, target_id)
                    print(f"Copied training start to {path}. Existing starts are never overwritten by clonestart.")
                elif command in {"read", "save", "saveeval", "reset", "step", "release", "reconnect"}:
                    expected = 4 if command == "step" else 2
                    if len(parts) != expected:
                        raise ValueError("Use read/save/reset/release/reconnect ID or step ID ACTION FRAMES.")
                    worker_id = int(parts[1])
                    environment = app.console_worker(worker_id)
                    if command == "read":
                        # Raw RAM only: do not attach the old attempt's context.
                        state = environment.bridge.observe()
                        app.show_manual(worker_id, state)
                    elif command == "release":
                        state = app.release_worker(worker_id)
                    elif command == "reconnect":
                        state = app.reconnect_worker(worker_id)
                    elif command == "saveeval":
                        app.require_controlled(worker_id)
                        if worker_id not in evaluation_workers(ROOT):
                            raise ValueError("This worker is not in the frozen evaluation set.")
                        state = environment.bridge.request("saveeval")
                        app.show_manual(worker_id, state, "saved evaluation start")
                        print(f"W{worker_id}: evaluation start replaced; future scores have a new comparison signature.")
                    elif command == "save":
                        app.require_controlled(worker_id)
                        app.verified = False
                        state = environment.bridge.request("save")
                        app.show_manual(worker_id, state, "saved start")
                        print(f"W{worker_id} starting state overwritten with current position. Run verify before automation.")
                    elif command == "reset":
                        app.require_controlled(worker_id)
                        app.verified = False
                        environment.bridge.start_kind = "train"
                        state = environment.reset()
                        app.show_manual(worker_id, state, "reset to saved start")
                    else:
                        state = app.manual_step(worker_id, parts[2], int(parts[3]))
                    bot.show_state(state)
                elif command == "telemetry" and len(parts) == 2 and parts[1] in {"on", "off"}:
                    app.telemetry.enabled = parts[1] == "on"
                    app.telemetry.publish(app.trainer, len(app.replay), app.last_loss, force=True)
                    print(f"Telemetry {parts[1]}. The panel becomes stale when disabled.")
                elif command == "workers" and len(parts) == 2:
                    app.select_workers(int(parts[1]))
                elif command == "verify" and len(parts) == 1:
                    app.verify()
                elif command == "importv7" and len(parts) == 1:
                    # Only orchestration/logging changed: preserve the v7 model,
                    # target network, optimizer, and update count via strict load.
                    app.load(V7_PATH)
                    print("Use savemodel to create the separate v8 latest checkpoint.")
                elif command == "importprevious" and len(parts) == 1:
                    # Import the v6 parallel weights into the 14-action network.
                    # Keep update history; reset optimizer state and replay.
                    restored = bot.import_time_model(ROOT / "models" / "kirby_parallel_v6_latest.pth")
                    app.trainer = restored
                    app.replay = bot.ReplayBuffer(REPLAY_CAPACITY)
                    app.stored_total = 0
                    app.model_loaded = True
                    print(f"Imported previous parallel trainer at update {restored.updates}; optimizer and replay reset.")
                    app.last_loss = None
                    app.telemetry.checkpoint = {"loaded_from": "v6 import (optimizer reset)",
                                                "loaded_updates": restored.updates,
                                                "saved_to": "None since import", "saved_updates": None}
                    print("Use savemodel to create the separate v8 checkpoint.")
                elif command in {"importsingle", "loadmodel", "loadbest"} and len(parts) == 1:
                    path = {"importsingle": bot.MODEL_PATH, "loadmodel": LATEST_PATH, "loadbest": BEST_PATH}[command]
                    app.load(path)
                elif command == "autosaves" and len(parts) == 1:
                    paths = checkpoints(ROOT)
                    if not paths:
                        print("No automatic checkpoints yet.")
                    for index, path in enumerate(paths, 1):
                        print(f"{index}: {path}")
                elif command == "loadauto" and len(parts) <= 2:
                    # Explicit recovery: never silently prefer auto over latest.
                    app.load(checkpoint_at(ROOT, int(parts[1]) if len(parts) == 2 else 1))
                elif command == "savemodel" and len(parts) == 1:
                    if not app.model_loaded:
                        raise ValueError("Load your trained model first.")
                    app.save()
                    print(f"Saved update {app.trainer.updates} to {LATEST_PATH}")
                elif command == "ptrain":
                    attempts = int(parts[1]) if len(parts) > 1 else 3
                    supplied = [float(x) for x in parts[2:]]
                    if supplied and len(supplied) != len(app.worker_ids):
                        raise ValueError("Give one epsilon per selected worker, or omit all epsilons.")
                    values = supplied or [0.1 if index % 2 == 0 else 0.3 for index in range(len(app.worker_ids))]
                    app.run(attempts=attempts, epsilons=dict(zip(app.worker_ids, values)), training=True, mode="train")
                elif command == "eval" and len(parts) == 1:
                    app.evaluate()
                elif command == "bench" and len(parts) <= 2:
                    app.benchmark(int(parts[1]) if len(parts) > 1 else 120)
                elif command == "shutdown" and len(parts) == 1:
                    if app.shutdown():
                        break
                elif command == "quit" and len(parts) == 1:
                    # Leave selected emulators paused; no implicit model save.
                    # Released workers may have been deselected since release.
                    for worker_id in sorted(set(app.worker_ids) | app.released_workers):
                        environment = app.environments[worker_id]
                        try:
                            environment.close()
                        except (OSError, RuntimeError) as error:
                            print(f"W{worker_id}: {error}")
                    break
                else:
                    print("Unknown command or arguments. Type help.")
            except (ValueError, RuntimeError, OSError) as error:
                print(error)
                print("The model is still in memory. Use savemodel if needed.")
            except (KeyboardInterrupt, EOFError):
                print("Console exiting. Use Q during automation next time to return without exiting.")
                break
    finally:
        lock.close()


if __name__ == "__main__":
    main()
