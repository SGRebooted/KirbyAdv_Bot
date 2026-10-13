"""Bounded automatic checkpoint history, separate from manual latest/best.

The coordinator calls this only on its main thread between training updates.
The existing checkpoint writer must publish a completed .pth atomically.
"""
from datetime import datetime, timezone
from pathlib import Path
import re
import time
import uuid

AUTO_ATTEMPTS = 150       # Completed attempts across all selected workers.
AUTO_SECONDS = 30 * 60    # Wall-clock time; independent of game-frame budgets.
AUTO_KEEP = 3
RETRY_SECONDS = 60
_NAME = re.compile(r"kirby_parallel_v9_auto_\d{8}T\d{12}Z_[0-9a-f]{32}\.pth")


def checkpoints(root):
    """Return only this feature's completed files, newest timestamp first.

    Ignore temporary files, manually named backups, and latest/best files.
    """
    directory = Path(root) / "models" / "autosave_v9"
    return sorted((p for p in directory.glob("*.pth")
                   if p.is_file() and _NAME.fullmatch(p.name)),
                  key=lambda p: p.name, reverse=True)


def checkpoint_at(root, index=1):
    if type(index) is not int or not 1 <= index <= AUTO_KEEP:
        raise ValueError("Use loadauto or loadauto 1/2/3 (1 is newest).")
    paths = checkpoints(root)
    if len(paths) < index:
        raise ValueError(f"Automatic checkpoint {index} does not exist. Use autosaves to list them.")
    return paths[index - 1]


class AutoCheckpoint:
    """One training command's attempt/time counters and retry cooldown."""

    def __init__(self, root, save, clock=None, announce=print):
        self.root = Path(root)
        self.save = save
        self.clock = clock or time.monotonic
        self.announce = announce
        self.completed = 0
        self.saved_completed = 0
        self.saved_at = self.clock()
        self.saved_updates = None
        self.retry_at = 0.0

    def completed_attempt(self):
        self.completed += 1

    def maybe_save(self, updates, final_reason=None):
        """Save on either threshold; always try once on normal/controlled exit.

        Failed writes do not reset counters or remove previous checkpoints.
        While paused with no newer model updates, avoid repeated timed copies.
        Final saves still happen even if their update count matches a prior save.
        """
        now = self.clock()
        if final_reason is None:
            if now < self.retry_at or updates == self.saved_updates:
                return None
            if self.completed - self.saved_completed >= AUTO_ATTEMPTS:
                reason = f"{AUTO_ATTEMPTS} completed attempts"
            elif now - self.saved_at >= AUTO_SECONDS:
                reason = "30-minute timer"
            else:
                return None
        else:
            reason = final_reason

        directory = self.root / "models" / "autosave_v9"
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        path = directory / f"kirby_parallel_v9_auto_{stamp}_{uuid.uuid4().hex}.pth"
        try:
            directory.mkdir(parents=True, exist_ok=True)
            self.save(path)
            if not path.is_file():
                raise OSError("Checkpoint writer did not publish the completed file.")
        except Exception as error:
            self.retry_at = self.clock() + RETRY_SECONDS
            # A failed serializer can leave its unique temporary file behind.
            # Remove only that file, never an older checkpoint.
            try:
                path.with_suffix(".tmp").unlink(missing_ok=True)
            except OSError:
                pass
            self.announce(f"AUTOSAVE FAILED ({reason}): {error}")
            self.announce("Model remains in memory; previous saves retained. Retry in 60s during training, or use savemodel at the prompt.")
            return None

        self.saved_at = self.clock()
        self.saved_completed = self.completed
        self.saved_updates = updates
        self.retry_at = 0.0
        self.announce(f"Autosaved update {updates} ({reason}): {path}")
        try:
            for old in checkpoints(self.root)[AUTO_KEEP:]:
                old.unlink()
        except OSError as error:
            # A locked old file is harmless; prune again on the next success.
            self.announce(f"Autosave succeeded; old-file cleanup deferred: {error}")
        return path
