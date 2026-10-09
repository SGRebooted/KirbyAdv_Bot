"""Small, optional telemetry files; failures must never stop training."""
from datetime import datetime, timezone
from pathlib import Path
import csv
import json
import os
import time
import uuid

ARCHIVE_ROWS = 5000  # Completed/interrupted attempts, not individual actions.


def read_key():
    """Read hotkeys from the focused Windows terminal without waiting for Enter."""
    import msvcrt
    if not msvcrt.kbhit():
        return ""
    key = msvcrt.getwch()
    if key in ("\x00", "\xe0"):
        if msvcrt.kbhit():
            msvcrt.getwch()
        return ""
    return key.lower()


def archive_log(path, threshold=ARCHIVE_ROWS):
    """Rename the full CSV after evaluation; preserve its header and all rows.

    Only the coordinator calls this, with workers drained and the writer closed.
    On failure, the caller reports it and retries after a later evaluation.
    Existing archives are never deleted or overwritten.
    """
    path = Path(path)
    if not path.exists():
        return None
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.reader(stream)
        header = next(reader, None)
        count = sum(1 for row in reader if row)
    if not header or count < threshold:
        return None
    directory = path.parent / "archive"
    directory.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    target = directory / f"{path.stem}_{stamp}_{uuid.uuid4().hex[:6]}.csv"
    path.rename(target)
    # If this write fails, the archive still contains every original row;
    # the normal append routine creates the next active file when needed.
    with path.open("w", newline="", encoding="utf-8") as stream:
        csv.writer(stream).writerow(header)
    return target


class Telemetry:
    """Publish once per second, and weight statistics once per five seconds.

    Each snapshot has a unique name and is published by atomic rename. The
    panel reads completed JSON files, never the temporary file being written.
    History persists across rounds, in a separate file for each process session.
    No PyTorch import is needed in the panel process.
    """

    def __init__(self, root):
        self.directory = Path(root) / "telemetry"
        self.session = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        self.history = self.directory / f"session_{self.session}.csv"
        self.enabled = True
        self.workers = {}
        self.status = "idle"
        self.frames_per_second = 0.0
        self.last_publish = self.last_weights = 0.0
        self.weights = []
        self.last_update = None
        self.warned = False
        self.snapshot_failures = 0
        self.snapshot_sequence = 0
        self.started = time.monotonic()

    def publish(self, trainer, replay_size, loss, force=False):
        """Use bounded summaries, never copy full matrices to the status file."""
        if not self.enabled:
            return
        now = time.monotonic()
        if not force and now - self.last_publish < 1.0:
            return
        self.last_publish = now
        try:
            self.directory.mkdir(exist_ok=True)
            if now - self.last_weights >= 5.0:
                self.last_weights = now
                self.weights = []
                # Transferring weights can synchronize the GPU. At this small
                # network size, sample infrequently and benchmark the overhead.
                for name, parameter in trainer.model.named_parameters():
                    values = parameter.detach().float().cpu()
                    self.weights.append({"layer": name, "mean": values.mean().item(),
                                         "std": values.std(unbiased=False).item(),
                                         "min": values.min().item(), "max": values.max().item()})
            payload = {"session": self.session, "timestamp": time.time(),
                       "status": self.status, "updates": trainer.updates, "fps": self.frames_per_second,
                       "replay": replay_size, "loss": loss, "workers": self.workers,
                       "weights": self.weights, "history": self.history.name}
            # Publish a unique completed file. Never overwrite a snapshot
            # that the panel (or another reader) may currently have open.
            self.snapshot_sequence += 1
            destination = self.directory / f"status_{self.session}_{self.snapshot_sequence:012d}.json"
            temporary = destination.with_suffix(".tmp")
            try:
                temporary.write_text(json.dumps(payload, allow_nan=False), encoding="utf-8")
                os.replace(temporary, destination)
            except OSError as error:
                # No sleep/retry loop in the training thread. Next refresh
                # publishes the newest state; history below is still written.
                self.snapshot_failures += 1
                if self.snapshot_failures == 5 or self.snapshot_failures % 60 == 0:
                    print(f"Telemetry snapshot unavailable for {self.snapshot_failures} refreshes: {error}. Training continues.")
            else:
                if self.snapshot_failures >= 5:
                    print("Telemetry snapshot publication recovered.")
                self.snapshot_failures = 0
                # Retain eight snapshots. Locked old files can wait until a
                # later refresh; cleanup must never interrupt training.
                for old in sorted(self.directory.glob("status_*.json"))[:-8]:
                    try:
                        old.unlink()
                    except OSError:
                        pass
            finally:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    pass

            # Sample the latest loss only after new training updates; evaluation
            # must not appear as repeated learning points in the graph.
            if loss is not None and trainer.updates != self.last_update:
                new = not self.history.exists() or self.history.stat().st_size == 0
                with self.history.open("a", newline="", encoding="utf-8") as stream:
                    writer = csv.writer(stream)
                    if new:
                        writer.writerow(["seconds", "updates", "loss", "replay"])
                    writer.writerow([round(now-self.started, 3), trainer.updates, loss, replay_size])
                self.last_update = trainer.updates
        except (OSError, ValueError) as error:
            if not self.warned:
                print(f"Telemetry refresh skipped ({error}); training continues.")
                self.warned = True
