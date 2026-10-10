"""Read attempt logs for the panel. No emulator, trainer, or PyTorch imports."""
import csv
from datetime import datetime, timezone
import math
from pathlib import Path


METRICS = {
    "Rooms visited": ("rooms", "Distinct rooms per attempt"),
    "New progress": ("new_progress", "New rightward progress (pixels)"),
    "Damage taken": ("health_lost", "Health segments lost"),
    "Total reward": ("reward", "Reward per attempt"),
    "Attempt duration": ("frames", "Game frames per attempt"),
    "Success rate": ("success", "Successful attempts (%)"),
    "Progress reward": ("progress_reward", "Reward contribution"),
    "Damage penalty": ("damage_penalty", "Reward contribution"),
    "Death penalty": ("death_penalty", "Reward contribution"),
    "Time penalty": ("time_penalty", "Reward contribution"),
    "Room bonus": ("room_bonus", "Reward contribution"),
    "Goal bonus": ("goal_bonus", "Reward contribution"),
}
REQUIRED = {"utc", "mode", "worker", "episode", "frames", "reward", "success", "reason"}


def complete_lines(stream):
    """Ignore a trailing line still being written by the coordinator."""
    for line in stream:
        if not line.endswith("\n"):
            return
        yield line


def number(value):
    """Missing legacy measurements stay missing instead of becoming zero."""
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def load_attempts(root):
    """Scan active/archive CSVs; return records and readable diagnostics.

    Legacy logs have no trustworthy session/run identity. Keep their source
    filename as a grouping label rather than inventing session boundaries.
    This function runs on the panel's background reader thread.
    """
    root = Path(root)
    paths = set((root / "logs").rglob("*.csv"))
    paths.update(root.glob("parallel*.csv"))
    old_folder = root / "parallel_training_logs_v7"
    if old_folder.is_dir():
        paths.update(old_folder.rglob("*.csv"))
    records, warnings, seen = [], [], set()
    for path in sorted(paths):
        try:
            with path.open(encoding="utf-8-sig", newline="") as stream:
                reader = csv.DictReader(complete_lines(stream), strict=True)
                if not REQUIRED.issubset(reader.fieldnames or ()):
                    continue
                for row in reader:
                    if None in row or any(row.get(key) is None for key in REQUIRED):
                        warnings.append(f"Skipped incomplete row in {path.name}")
                        continue
                    try:
                        stamp = datetime.fromisoformat(row["utc"].replace("Z", "+00:00"))
                        if stamp.tzinfo is None:
                            stamp = stamp.replace(tzinfo=timezone.utc)
                        row["_stamp"] = stamp.timestamp()
                        row["_date"] = stamp.astimezone(timezone.utc).date().isoformat()
                        row["worker"] = int(row["worker"])
                        row["episode"] = int(row["episode"])
                    except (ValueError, TypeError, OverflowError):
                        warnings.append(f"Skipped invalid date/worker/attempt in {path.name}")
                        continue
                    session, run = row.get("session_id"), row.get("run_id")
                    if session and run:
                        identity = (session, run, row["worker"], row["episode"])
                        if identity in seen:
                            continue
                        seen.add(identity)
                    row["_session"] = session or f"Legacy (session unknown): {path.relative_to(root)}"
                    row["_run"] = run or "Legacy (run unknown)"
                    row["_source"] = str(path.relative_to(root))
                    success = row["success"].strip().lower()
                    row["success"] = 100.0 if success in {"true", "1"} else 0.0 if success in {"false", "0"} else None
                    for key in {field for field, _ in METRICS.values()} - {"success"}:
                        row[key] = number(row.get(key))
                    records.append(row)
        except (OSError, csv.Error, UnicodeError) as error:
            warnings.append(f"{path.name}: {error}")
    records.sort(key=lambda row: (row["_stamp"], row["worker"], row["episode"]))
    return records, warnings


def rolling_mean(points, window=10):
    """Trailing average; the first points use the observations available."""
    result, values = [], []
    for x, y in points:
        values.append(y)
        if len(values) > window:
            values.pop(0)
        result.append((x, sum(values) / len(values)))
    return result


def metric_points(records, metric):
    field = METRICS[metric][0]
    return [(index, row[field]) for index, row in enumerate(records, 1)
            if row.get(field) is not None]
