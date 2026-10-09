"""Up to four BizHawk workers sharing the existing Kirby bot's model and replay.

Save beside KA_Bot.py, not in a separate folder. Start this console first,
then load KA_Bridge_1.lua and KA_Bridge_2.lua in separate BizHawk windows.
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

from KA_Telemetry import Telemetry, archive_log, read_key
import KA_Bot as bot

ROOT = Path(__file__).resolve().parent
LATEST_PATH = ROOT / "models" / "kirby_parallel_v7_latest.pth"
BEST_PATH = ROOT / "models" / "kirby_parallel_v7_best.pth"
LOG_PATH = ROOT / "logs" / "parallel_v7_attempts.csv"
REPLAY_CAPACITY = 20000


class WorkerBridge(bot.KirbyBridge):
    """Reject replies from a wrongly assigned or old Lua bridge."""

    def __init__(self, worker_id):
        self.worker_id = worker_id
        super().__init__(ROOT / f"bridge_{worker_id}")

    def request(self, command, frames=0, buttons=()):
        reply = super().request(command, frames, buttons)
        if reply.get("worker_id") != self.worker_id:
            self.failed = True
            raise RuntimeError(f"Worker {self.worker_id}: wrong bridge identity. Reload its numbered Lua script.")
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


def write_result(result):
    """Append one completed or interrupted attempt for later comparisons."""
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not LOG_PATH.exists() or LOG_PATH.stat().st_size == 0
    with LOG_PATH.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(result))
        if needs_header:
            writer.writeheader()
        writer.writerow(result)


class Coordinator:
    """Own the one model, optimizer, replay buffer, and independent workers."""

    def __init__(self):
        self.environments = {i: make_environment(i) for i in range(1, 5)}
        self.trainer = bot.DQNTrainer(bot.KirbyQNetwork().to(bot.DEVICE))
        self.replay = bot.ReplayBuffer(REPLAY_CAPACITY)
        self.stored_total = 0
        self.model_loaded = False
        self.verified = False
        self.worker_ids = (1, 2)
        self.telemetry = Telemetry(ROOT)
        self.last_loss = None

    def load(self, path):
        """Loading is explicit and clears experiences tied to the prior run."""
        restored = load_checkpoint(path)
        self.trainer = restored
        self.replay = bot.ReplayBuffer(REPLAY_CAPACITY)
        self.stored_total = 0
        self.model_loaded = True
        print(f"Loaded update {restored.updates} from {path.name}; replay cleared.")

    def verify(self):
        """Reset selected workers; advancing one must leave every other frozen."""
        self.verified = False
        for i in self.worker_ids:
            self.environments[i].reset()
        for active in self.worker_ids:
            before = {i: e.observe()["frame"] for i, e in self.environments.items() if i in self.worker_ids}
            self.environments[active].step("wait", 4)
            after = {i: e.observe()["frame"] for i, e in self.environments.items() if i in self.worker_ids}
            if after[active] - before[active] != 4 or any(after[i] != before[i] for i in self.worker_ids if i != active):
                raise RuntimeError("Isolation check failed. Use separate emulator windows and one numbered script per window.")
        for i in self.worker_ids:
            self.environments[i].reset()
        self.verified = True
        print("Verified: each worker advances independently. All selected workers restored to their saved starts.")

    def run(self, worker_ids=None, attempts=1, epsilons=None,
            training=False, step_limit=None, mode="play"):
        """Overlap emulator requests, with one outstanding request per worker.

        The main thread selects actions, receives results, and performs updates.
        Threads never use the model or replay, avoiding inference/update races.
        """
        worker_ids = self.worker_ids if worker_ids is None else worker_ids
        if not self.model_loaded:
            raise ValueError("Use importprevious or loadmodel first.")
        if not self.verified:
            raise ValueError("Use verify first, after saving all selected starting states.")
        if type(attempts) is not int or not 1 <= attempts <= 200:
            raise ValueError("Choose 1–200 attempts per worker.")
        if not worker_ids or len(set(worker_ids)) != len(worker_ids) or any(i not in self.worker_ids for i in worker_ids):
            raise ValueError("Choose a nonempty subset of the selected workers.")
        epsilons = epsilons or {i: 0.0 for i in worker_ids}
        if any(not 0 <= epsilons[i] <= 1 for i in worker_ids):
            raise ValueError("Exploration must be between 0 and 1.")
        if step_limit is not None and (type(step_limit) is not int or not 1 <= step_limit <= bot.MAX_ATTEMPT_FRAMES):
            raise ValueError("Invalid step limit.")

        contexts = {i: Attempt(i) for i in worker_ids}
        paused = False
        self.telemetry.workers = {}
        self.telemetry.status = mode
        pending = {}
        results = []
        stopping = False
        failed = False
        total_frames = total_steps = 0
        paused_seconds = 0.0
        pause_started = None
        started = time.perf_counter()
        starting_updates = self.trainer.updates

        def report(context, reason, success=False):
            """Write each episode once, including partial episodes on Q/errors."""
            if context.reported:
                return
            context.reported = True
            environment = self.environments[context.worker]
            result = {
                "utc": datetime.now(timezone.utc).isoformat(),
                "mode": mode, "worker": context.worker,
                "episode": context.episode, "epsilon": epsilons[context.worker],
                "steps": context.steps, "frames": context.frames,
                "reward": round(context.reward, 6),
                "health_lost": context.health_lost,
                "new_progress": context.progress,
                "rooms": len(environment.visited_rooms),
                "success": bool(success), "reason": reason,
                "updates": self.trainer.updates,
            }
            results.append(result)
            result.update(context.breakdown or {})
            write_result(result)
            print(f"W{context.worker} attempt {context.episode}/{attempts}: {reason} | "
                  f"reward {context.reward:.2f} | rooms {result['rooms']} | "
                  f"damage {context.health_lost:g} | frames {context.frames}")

        print(f"{mode}: {attempts} attempt(s) per worker; {len(worker_ids)} worker(s). Q stops safely; P pauses/resumes; S saves while fully paused.")
        with ThreadPoolExecutor(max_workers=len(worker_ids)) as executor:
            def reset_worker(worker_id):
                """Reset in that worker's thread without blocking the other one."""
                old = contexts[worker_id]
                contexts[worker_id] = Attempt(worker_id, episode=old.episode + 1, reported=False, breakdown={k: 0.0 for k in ("progress_reward", "damage_penalty", "death_penalty", "time_penalty", "room_bonus", "goal_bonus")})
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
                        save_checkpoint(self.trainer)
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
                            self.telemetry.workers[str(worker_id)] = {
                                "attempt": context.episode, "room": state["kirby"]["room"],
                                "action": action, "epsilon": epsilons[worker_id],
                                "reward": round(context.reward, 3), "frames": context.frames,
                                "watchdog_seconds": info.get("watchdog_frames", 0) / 60,
                                "watchdog_reset": info.get("watchdog_reset", ""),
                                "stop_reason": info["stop_reason"], "rewards": dict(context.breakdown),
                            }
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
                                if not (terminated or truncated):
                                    stopping = failed = True
                            elif step_limit is not None and context.steps >= step_limit:
                                report(context, "requested step limit")

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

        if pause_started is not None:
            paused_seconds += time.perf_counter() - pause_started
        self.telemetry.status = "idle"
        self.telemetry.publish(self.trainer, len(self.replay), self.last_loss, force=True)
        seconds = max(time.perf_counter() - started - paused_seconds, 0.001)
        summary = {
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
            "states": [hashlib.sha256((e.bridge.directory / "start.State").read_bytes()).hexdigest()
                       for i, e in self.environments.items() if i in self.worker_ids],
            "encoder": bot.ENCODER_VERSION, "training": bot.TRAINING_VERSION,
            "room_budget": bot.ROOM_FRAME_BUDGET, "overall_budget": bot.MAX_ATTEMPT_FRAMES,
            "grace": bot.PROGRESS_GRACE_FRAMES,
            # A code change starts a fresh comparison, even without a version bump.
            "bot_code": hashlib.sha256(Path(bot.__file__).read_bytes()).hexdigest(),
        }
        return hashlib.sha256(json.dumps(material, sort_keys=True).encode()).hexdigest()

    def evaluate(self):
        """Freeze the model and evaluate each saved starting state once."""
        signature = self.evaluation_signature()
        summary = self.run(epsilons={i: 0.0 for i in self.worker_ids}, mode="eval")
        if summary["interrupted"] or len(summary["results"]) != len(self.worker_ids):
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
        if previous is None or tuple(rank) > tuple(previous["rank"]): #seve best or new tie
            save_checkpoint(self.trainer, BEST_PATH, {"signature": signature, "rank": rank})
            print(f"Saved new best evaluated checkpoint: {BEST_PATH.name}")
        else:
            print("Best evaluated checkpoint retained.")
        print(f"Evaluation: {sum(r['success'] for r in results)}/{count} clears; comparison rank {rank}")

    def benchmark(self, steps=120):
        """Compare collection throughput only; use ptrain to measure learning."""
        if not 1 <= steps <= 900:
            raise ValueError("Choose 1–900 benchmark decisions per worker.")
        counts = (2, 4) if len(self.worker_ids) == 4 else (1, 2)
        rates = []
        for count in counts:
            ids = self.worker_ids[:count]
            result = self.run(ids, epsilons={i: 0.1 for i in ids}, step_limit=steps, mode=f"bench-{count}")
            if result["interrupted"]:
                return
            rates.append(result["frames_per_second"])
        print(f"Collection speedup: {rates[1] / max(rates[0], 1e-9):.2f}x. Compare ptrain throughput too.")


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
        print("Load each numbered Lua bridge in a different BizHawk window. Type help.")
        while True:
            try:
                parts = input("Parallel> ").lower().split()
                if not parts:
                    continue
                command = parts[0]
                if command == "help":
                    print("workers 2/4 | read ID | save ID | reset ID | step ID ACTION FRAMES | verify")
                    print("telemetry on/off | importprevious | importsingle | loadmodel | loadbest | savemodel | bench [steps]")
                    print("ptrain [attempts PER WORKER] [one epsilon per selected worker] | eval | quit")
                    print("Defaults: 3 attempts each; epsilons 0.1 0.3 (0.1 0.3 for W3/W4). Q stop; P pause/resume; S save when paused.")
                elif command in {"read", "save", "reset", "step"}:
                    expected = 4 if command == "step" else 2
                    if len(parts) != expected or parts[1] not in {"1", "2", "3", "4"}:
                        raise ValueError("Use read/save/reset with ID 1–4; step WORKER ACTION FRAMES.")
                    worker_id = int(parts[1])
                    environment = app.environments[worker_id]
                    if command == "read":
                        state = environment.observe()
                    elif command == "save":
                        state = environment.save()
                        app.verified = False
                        print(f"W{worker_id} starting state saved. Run verify before automation.")
                    elif command == "reset":
                        state = environment.reset()
                    else:
                        state, reward, _, _, info = environment.step(parts[2], int(parts[3]))
                        print(f"W{worker_id} reward {reward:.3f}; time cost {info['time_penalty']:.3f}; {info['stop_reason']}")
                    bot.show_state(state)
                elif command == "telemetry" and len(parts) == 2 and parts[1] in {"on", "off"}:
                    app.telemetry.enabled = parts[1] == "on"
                    print(f"Telemetry {parts[1]}. The panel becomes stale when disabled.")
                elif command == "workers" and len(parts) == 2 and parts[1] in {"2", "4"}:
                    app.worker_ids = tuple(range(1, int(parts[1]) + 1))
                    app.verified = False
                    print(f"Selected {len(app.worker_ids)} workers. Save each start, then verify.")
                elif command == "verify" and len(parts) == 1:
                    app.verify()
                elif command == "importprevious" and len(parts) == 1:
                    # Import the v6 parallel weights into the 14-action network.
                    # Keep update history; reset optimizer state and replay.
                    restored = bot.import_time_model(ROOT / "models" / "kirby_parallel_v6_latest.pth")
                    app.trainer = restored
                    app.replay = bot.ReplayBuffer(REPLAY_CAPACITY)
                    app.stored_total = 0
                    app.model_loaded = True
                    print(f"Imported previous parallel trainer at update {restored.updates}; optimizer and replay reset.")
                    print("Use savemodel to create the separate v7 checkpoint.")
                elif command in {"importsingle", "loadmodel", "loadbest"} and len(parts) == 1:
                    path = {"importsingle": bot.MODEL_PATH, "loadmodel": LATEST_PATH, "loadbest": BEST_PATH}[command]
                    app.load(path)
                elif command == "savemodel" and len(parts) == 1:
                    if not app.model_loaded:
                        raise ValueError("Load your trained model first.")
                    save_checkpoint(app.trainer)
                    print(f"Saved update {app.trainer.updates} to {LATEST_PATH}")
                elif command == "ptrain":
                    attempts = int(parts[1]) if len(parts) > 1 else 3
                    supplied = [float(x) for x in parts[2:]]
                    if supplied and len(supplied) != len(app.worker_ids):
                        raise ValueError("Give one epsilon per selected worker, or omit all epsilons.")
                    values = supplied or ([0.1, 0.3] * 2)[:len(app.worker_ids)]
                    app.run(attempts=attempts, epsilons=dict(zip(app.worker_ids, values)), training=True, mode="train")
                elif command == "eval" and len(parts) == 1:
                    app.evaluate()
                elif command == "bench" and len(parts) <= 2:
                    app.benchmark(int(parts[1]) if len(parts) > 1 else 120)
                elif command == "quit" and len(parts) == 1:
                    # Leave selected emulators paused; no implicit model save.
                    for worker_id in app.worker_ids:
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
