"""Offline analysis of KA_GameRecorder.lua recordings; standard library only.

No bridge, trainer, emulator control, or checkpoint imports.
"""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path


def samples(path, info):
    ram = None
    segment = None
    last_index = -1
    info.update(header={}, end=None, errors=[])
    with Path(path).open(encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                row = json.loads(line)
                kind = row["type"]
                if kind == "header":
                    if row.get("schema") != "ka-game-ram-v1":
                        raise ValueError("Expected a KA_GameRecorder log (use KA_Diagnostics.py for old probes)")
                    info["header"] = row
                elif kind == "boundary":
                    ram = None
                elif kind == "end":
                    info["end"] = row
                elif kind == "sample":
                    if not info["header"]:
                        raise ValueError("Missing supported header")
                    if row["sample"] != last_index + 1:
                        ram = None
                    last_index = row["sample"]
                    if row["segment"] != segment:
                        ram = None
                    segment = row["segment"]
                    if "full" in row:
                        ram = {}
                        for start, block in row["full"].items():
                            for offset, value in enumerate(bytes.fromhex(block)):
                                ram[int(start, 16) + offset] = value
                        expected = sum(b - a + 1 for a, b in info["header"]["ranges"])
                        if len(ram) != expected or any(addr not in ram for a, b in info["header"]["ranges"] for addr in range(a, b + 1)):
                            raise ValueError("Incomplete full RAM snapshot")
                    else:
                        if ram is None:
                            raise ValueError("Delta has no intact baseline; waiting for a full snapshot")
                        for address, value in row["delta"].items():
                            addr = int(address, 16)
                            if addr not in ram or type(value) is not int or not 0 <= value <= 255:
                                raise ValueError("Invalid RAM delta")
                            ram[addr] = value
                    yield row, dict(ram)
            except (ValueError, KeyError, TypeError) as exc:
                info["errors"].append(f"Line {line_number}: {exc}")
                ram = None


def decode(ram):
    def word(low, high, signed=False):
        value = ram[low] + 256 * ram[high]
        return value - 65536 if signed and value >= 32768 else value
    player = {key: ram[addr] for key, addr in {
        "hp": 0x597, "lives": 0x599, "ability": 0x5E3, "mouth_full": 0x5E0,
        "sprite_state": 0x62C7, "level": 0x558, "stage": 0x559}.items()}
    player.update(room=word(0x55E, 0x55F), x=word(0x83, 0x95), y=word(0xB9, 0xCB),
                  vx=word(0x5B9, 0x5BA, True), vy=word(0x5BD, 0x5BE, True))
    objects = []
    for slot in range(9, 18):
        obj = dict(slot=slot, script=ram[0x6128 + slot], pose=ram[0x62C6 + slot],
                   x=word(0x82 + slot, 0x94 + slot), y=word(0xB8 + slot, 0xCA + slot),
                   vx=word(0x60E0 + slot, 0x60F2 + slot, True),
                   vy=word(0x6104 + slot, 0x6116 + slot, True))
        obj.update({f"var{i}": ram[0x61CA + i * 0x12 + slot] for i in range(8)})
        objects.append(obj)
    return player, objects


def analyze(path, output, compare=None):
    # A new directory prevents accidental replacement of an earlier report.
    output.mkdir(parents=True, exist_ok=False)
    info, changes, events, chosen = {}, Counter(), Counter(), {}
    previous, previous_row, previous_player = None, None, None
    total = 0
    with (output / "player.csv").open("w", newline="", encoding="utf-8") as pf, \
         (output / "objects.csv").open("w", newline="", encoding="utf-8") as of, \
         (output / "events.csv").open("w", newline="", encoding="utf-8") as ef:
        pw = ow = None
        ew = csv.writer(ef)
        ew.writerow(["sample", "frame", "segment", "field", "before", "after"])
        for row, ram in samples(path, info):
            player, objects = decode(ram)
            context = {key: row[key] for key in ("sample", "frame", "elapsed_frames", "segment")}
            pr = dict(context, buttons="+".join(row["buttons"]), **player)
            if pw is None:
                pw = csv.DictWriter(pf, fieldnames=list(pr)); pw.writeheader()
                ow = csv.DictWriter(of, fieldnames=list(context) + list(objects[0])); ow.writeheader()
            pw.writerow(pr)
            for obj in objects:
                ow.writerow(dict(context, **obj))
            continuous = previous is not None and row["segment"] == previous_row["segment"] and row["sample"] == previous_row["sample"] + 1 and row["frame"] >= previous_row["frame"]
            if continuous:
                changes.update(addr for addr, value in ram.items() if previous[addr] != value)
                for key in ("hp", "lives", "ability", "mouth_full", "sprite_state", "level", "stage", "room"):
                    if player[key] != previous_player[key]:
                        ew.writerow([row["sample"], row["frame"], row["segment"], key, previous_player[key], player[key]])
                        events[key] += 1
                for slot in range(9, 18):
                    for name, addr in (("script", 0x6128 + slot), ("pose", 0x62C6 + slot)):
                        if ram[addr] != previous[addr]:
                            ew.writerow([row["sample"], row["frame"], row["segment"], f"slot_{slot}.{name}", previous[addr], ram[addr]])
            if compare and row["sample"] in compare:
                chosen[row["sample"]] = (row, ram)
            previous, previous_row, previous_player = ram, row, player
            total += 1
    with (output / "changed_addresses.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream); writer.writerow(["address_hex", "changes_between_adjacent_samples"])
        writer.writerows((f"{addr:04X}", count) for addr, count in changes.most_common())
    if compare:
        if any(index not in chosen for index in compare):
            info["errors"].append("Comparison sample missing or damaged")
        else:
            (a, before), (b, after) = (chosen[index] for index in compare)
            if a["segment"] != b["segment"]:
                info["errors"].append("Comparison refused across savestate/reset segments")
            else:
                with (output / "comparison.csv").open("w", newline="", encoding="utf-8") as stream:
                    writer = csv.writer(stream)
                    writer.writerow(["address_hex", "before_decimal", "after_decimal", "before_hex", "after_hex"])
                    writer.writerows((f"{addr:04X}", before[addr], value, f"{before[addr]:02X}", f"{value:02X}") for addr, value in after.items() if before[addr] != value)
    end = info["end"] or {}
    complete = end.get("reason") in ("frame_limit", "script_stopped") and end.get("samples") == total and not info["errors"]
    report = dict(recording=str(path.resolve()), header=info["header"], samples=total,
                  clean_end=complete, end=info["end"], player_changes=dict(events), errors=info["errors"],
                  note="Raw values and correlations only. Slots can be reused; script 255 is currently treated as empty. Pose is not a confirmed attack/hitbox. Input is sampled, not an input-poll trace.")
    (output / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", type=Path)
    parser.add_argument("--compare", nargs=2, type=int, metavar=("BEFORE_SAMPLE", "AFTER_SAMPLE"))
    parser.add_argument("--out", type=Path, help="New report directory; defaults to recording name + _report")
    args = parser.parse_args()
    output = args.out or args.recording.with_name(args.recording.stem + "_report")
    report = analyze(args.recording, output, args.compare)
    print(f"Report: {output.resolve()}\nSamples: {report['samples']} | Clean end: {report['clean_end']}")
    if report["errors"]:
        print("Some data could not be used; see summary.json.")


if __name__ == "__main__":
    main()
