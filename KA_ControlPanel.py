"""Run separately with Python 3.13; uses only standard-library Tkinter.

This panel only reads files. Use P, S and Q in the trainer's terminal to control
training; closing this window cannot stop or modify the trainer.
"""
from collections import deque
import csv
import json
from pathlib import Path
import time
import tkinter as tk
from tkinter import ttk

ROOT = Path(__file__).resolve().parent


def read_snapshot(directory):
    """Read the newest complete snapshot; tolerate cleanup/read races.

    Writers never replace these files. If cleanup removes a candidate before
    we open it, try the next recent file and retry normally next second.
    """
    candidates = sorted(directory.glob("status_*.json"), reverse=True)
    for path in candidates[:8]:
        try:
            text = path.read_text(encoding="utf-8")
            json.loads(text)  # Reject a damaged snapshot before returning it.
            return text
        except (OSError, ValueError):
            continue
    raise FileNotFoundError("No completed v7 telemetry snapshot yet.")


class Panel:
    """A read-only dashboard with a bounded loss graph and current worker data."""

    def __init__(self, window):
        self.window = window
        window.title("Kirby Training Panel")
        window.geometry("1080x780")
        self.status = tk.StringVar(value="Waiting for the trainer's first run…")
        ttk.Label(window, textvariable=self.status, padding=10).pack(fill="x")
        ttk.Label(window, text="Trainer terminal: P pause/resume • S save when paused • Q stop run. Panel is read-only.").pack()
        self.worker_text = tk.Text(window, height=15, wrap="word", font=("Consolas", 10))
        self.worker_text.pack(fill="x", padx=10, pady=8)
        self.canvas = tk.Canvas(window, height=230, background="#17212b", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True, padx=10)
        self.weight_text = tk.Text(window, height=9, font=("Consolas", 10))
        self.weight_text.pack(fill="x", padx=10, pady=8)
        self.points = deque(maxlen=1000)
        self.session = None
        self.offset = 0
        self.header = None
        self.refresh()

    @staticmethod
    def replace_text(widget, text):
        """Keep display text selectable but protect it against accidental edits."""
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", text)
        widget.configure(state="disabled")

    def read_history(self, name):
        """Read new complete CSV lines only; never rescan a growing file."""
        path = ROOT / "telemetry" / Path(name).name
        if not path.exists():
            return
        with path.open("r", encoding="utf-8", newline="") as stream:
            stream.seek(self.offset)
            while True:
                start = stream.tell()
                line = stream.readline()
                if not line or not line.endswith("\n"):
                    self.offset = start
                    break
                self.offset = stream.tell()
                row = next(csv.reader([line]))
                if self.header is None:
                    self.header = row
                else:
                    self.points.append((int(row[1]), float(row[2])))

    def draw_graph(self):
        """Show the latest 1,000 sampled losses; the full CSV remains on disk."""
        c = self.canvas
        c.delete("all")
        width, height = max(c.winfo_width(), 200), max(c.winfo_height(), 120)
        c.create_text(12, 12, anchor="nw", fill="white", text="Loss (sampled about once/sec; latest 1,000 points)")
        if len(self.points) < 2:
            return
        low = min(v for _, v in self.points)
        high = max(v for _, v in self.points)
        first, last = self.points[0][0], self.points[-1][0]
        coords = []
        for update, value in self.points:
            coords.extend([70+(width-90)*(update-first)/max(last-first, 1),
                           height-30-(height-65)*(value-low)/max(high-low, 1e-12)])
        c.create_line(*coords, fill="#61d9a8", width=2)
        c.create_text(5, 36, anchor="nw", fill="white", text=f"{high:.5f}")
        c.create_text(5, height-32, anchor="nw", fill="white", text=f"{low:.5f}")
        c.create_text(70, height-8, fill="white", text=str(first))
        c.create_text(width-50, height-8, fill="white", text=f"{last}")

    def refresh(self):
        """Retry on missing, locked, or temporarily unreadable files."""
        try:
            data = json.loads(read_snapshot(ROOT/"telemetry"))
            if data["session"] != self.session:
                self.session = data["session"]
                self.points.clear()
                self.offset = 0
                self.header = None
            age = time.time()-data["timestamp"]
            suffix = f" • last update {age:.0f}s ago" if age > 3 else ""
            loss = "—" if data["loss"] is None else f"{data['loss']:.6f}"
            self.status.set(f"{data['status'].upper()} • Updates {data['updates']:,} • Replay {data['replay']:,} • Loss {loss}{suffix}")
            lines = []
            for worker, state in sorted(data["workers"].items()):
                lines.append(f"W{worker} | attempt {state['attempt']} | room {state['room']} | {state['action']} | exploration {state['epsilon']:.0%} | reward {state['reward']:.3f}")
                lines.append(f"  Frames {state['frames']} | no progress {state['watchdog_seconds']:.2f}/10s | reset: {state['watchdog_reset']} | {state['stop_reason']}")
                lines.append("  " + "  ".join(f"{key}={value:.2f}" for key,value in state['rewards'].items()))
            self.replace_text(self.worker_text, "\n".join(lines) or "Waiting for completed actions…")
            weights = ["Layer                         Mean          Std          Min          Max"]
            for item in data["weights"]:
                weights.append(f"{item['layer']:<26} {item['mean']:>12.5f} {item['std']:>12.5f} {item['min']:>12.5f} {item['max']:>12.5f}")
            self.replace_text(self.weight_text, "\n".join(weights))
            self.read_history(data["history"])
            self.draw_graph()
        except (OSError, ValueError, KeyError, IndexError) as error:
            self.status.set(f"Waiting for telemetry: {error}")
        self.window.after(1000, self.refresh)


if __name__ == "__main__":
    window = tk.Tk()
    Panel(window)
    window.mainloop()
