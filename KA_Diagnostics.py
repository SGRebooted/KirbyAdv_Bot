"""Optional raw RAM recordings. No trainer, replay, or emulator imports."""
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import uuid


def capture(root, bridge, worker_id, session_id, frames, stride, action, buttons):
    """Capture held input in controlled mode; each request ends paused.

    Missing end records identify interrupted/failed captures. Flush each sample
    before requesting more movement, so disk errors stop subsequent requests.
    """
    if type(frames) is not int or not 0 <= frames <= 600:
        raise ValueError("Probe frames must be 0–600 (0 takes one snapshot).")
    if type(stride) is not int or stride not in (1, 4):
        raise ValueError("Probe stride must be 1 or 4 frames.")
    state = bridge.observe()
    if state.get("control_mode") != "controlled":
        raise ValueError(f"Use reconnect {worker_id} before probing.")
    directory = Path(root) / "diagnostics"
    directory.mkdir(parents=True, exist_ok=True)
    capture_id = uuid.uuid4().hex
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    path = directory / f"ram_{stamp}_w{worker_id}_{capture_id}.jsonl"
    with path.open("x", encoding="utf-8") as stream:
        def write(record):
            record["utc"] = datetime.now(timezone.utc).isoformat()
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()
        write({"type": "header", "schema_version": 1, "capture_id": capture_id,
               "session_id": session_id, "worker": worker_id, "mode": "diagnostic",
               "requested_frames": frames, "stride": stride, "action": action,
               "buttons": list(buttons), "source": "bridge RAM snapshot",
               "note": "Slot reuse is possible; script/pose values are raw, not confirmed entity labels."})
        elapsed = 0
        write({"type": "sample", "elapsed_frames": elapsed, "state": state})
        while elapsed < frames:
            count = min(stride, frames - elapsed)
            previous = state
            state = bridge.request("manual", count, buttons)
            if state["frame"] - previous["frame"] != count:
                bridge.failed = True
                write({"type": "error", "message": "Frame mismatch; bridge restart required.",
                       "state": state})
                raise RuntimeError(f"Probe frame mismatch. Restart bridge/coordinator. Partial recording: {path}")
            elapsed += count
            write({"type": "sample", "elapsed_frames": elapsed, "state": state})
        write({"type": "end", "complete": True, "elapsed_frames": elapsed})
    return path, state


def summarize(path):
    """Stream recordings without treating object slots as stable identities."""
    samples = 0
    complete = False
    scripts = Counter()
    poses = set()
    rooms = set()
    malformed = 0
    header = {}
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue
            if row.get("type") == "header":
                header = row
            elif row.get("type") == "end":
                complete = row.get("complete") is True
            elif row.get("type") == "sample":
                samples += 1
                state = row["state"]
                player = state["kirby"]
                rooms.add((player.get("level"), player.get("stage"), player.get("room")))
                for obj in state.get("objects", []):
                    scripts[obj.get("script")] += 1
                    poses.add((obj.get("script"), obj.get("pose")))
    return {"worker": header.get("worker"), "action": header.get("action"),
            "requested_frames": header.get("requested_frames"), "samples": samples,
            "complete": complete and not malformed, "malformed_lines": malformed,
            "rooms": sorted(rooms, key=str),
            "script_slot_observations": dict(scripts),
            "script_pose_pairs": sorted(poses, key=str)}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Summarize a diagnostic JSONL recording.")
    parser.add_argument("recording", type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize(args.recording), indent=2))
    print("Counts include empty slots; these are observations, not enemy counts or confirmed identities.")
