"""Read-only v8 dashboard. Run separately; closing it does not stop training."""
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import date
import json
from pathlib import Path
import time
import tkinter as tk
from tkinter import ttk

from KA_History import METRICS, load_attempts, metric_points, number, rolling_mean

ROOT = Path(__file__).resolve().parent
CURRENT = "Current/latest session"
LOSS = "Shared model loss"
ENDINGS = "Attempt ending reasons"


def read_snapshot(directory):
    for path in sorted(directory.glob("status_*.json"), reverse=True)[:8]:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict) and {"session", "timestamp", "workers"} <= data.keys():
                return data
        except (OSError, ValueError):
            pass
    raise FileNotFoundError("No readable telemetry yet")


class LossReader:
    """Incremental reader: do not reread a long loss CSV every second."""
    def __init__(self):
        self.path = None
        self.offset = 0
        self.header = None
        self.points = deque(maxlen=1000)

    def read(self, path):
        if self.path != path or (path.exists() and path.stat().st_size < self.offset):
            self.path, self.offset, self.header = path, 0, None
            self.points.clear()
        if not path.exists():
            return
        with path.open(encoding="utf-8", newline="") as stream:
            stream.seek(self.offset)
            while True:
                start = stream.tell()
                line = stream.readline()
                if not line or not line.endswith("\n"):
                    self.offset = start
                    break
                self.offset = stream.tell()
                values = next(csv.reader([line]))
                if self.header is None:
                    self.header = values
                    continue
                row = dict(zip(self.header, values))
                update, loss = number(row.get("updates")), number(row.get("loss"))
                if update is not None and loss is not None:
                    # Loading an older checkpoint can move the update counter back.
                    if self.points and update < self.points[-1][0]:
                        self.points.clear()
                    self.points.append((update, loss))


def draw_plot(canvas, points, title, xlabel, ylabel, smooth=True, fixed_percent=False):
    """Show raw samples and an optional trailing ten-sample average."""
    canvas.delete("all")
    width, height = max(canvas.winfo_width(), 400), max(canvas.winfo_height(), 220)
    canvas.create_text(16, 12, anchor="nw", fill="white", text=title)
    if not points:
        canvas.create_text(width / 2, height / 2, fill="#bdcbd6", text="No matching measurements yet")
        return
    average = rolling_mean(points)[-2000:]
    points = points[-2000:]
    left, right, top, bottom = 90, width - 25, 58, height - 52
    first, last = points[0][0], points[-1][0]
    low, high = (0, 100) if fixed_percent else (min(0, min(y for _, y in points)), max(y for _, y in points))
    if high <= low:
        high = low + 1
    def xy(point):
        x, y = point
        return (left + (right-left) * (x-first) / max(last-first, 1),
                bottom - (bottom-top) * (y-low) / (high-low))
    for tick in range(5):
        y = bottom - (bottom-top) * tick / 4
        canvas.create_line(left, y, right, y, fill="#304152")
        canvas.create_text(left-8, y, anchor="e", fill="#d0dce6", text=f"{low+(high-low)*tick/4:.3g}")
    for fraction in (0, 0.5, 1):
        x = left + (right-left) * fraction
        canvas.create_text(x, bottom+13, fill="#d0dce6", text=f"{first+(last-first)*fraction:,.0f}")
    canvas.create_text(left, top-18, anchor="w", fill="#d0dce6", text=ylabel)
    canvas.create_text((left+right)/2, height-10, fill="#d0dce6", text=xlabel)
    for series, color, thickness in [(points, "#718494" if smooth else "#68d9b3", 1),
                                      (average if smooth else [], "#68d9b3", 2)]:
        coords = [v for point in series for v in xy(point)]
        if len(coords) >= 4:
            canvas.create_line(*coords, fill=color, width=thickness)
        elif coords:
            x, y = coords
            canvas.create_oval(x-3, y-3, x+3, y+3, fill=color, outline=color)


def draw_endings(canvas, records):
    canvas.delete("all")
    width, height = max(canvas.winfo_width(), 400), max(canvas.winfo_height(), 220)
    counts = Counter(row["reason"] for row in records).most_common(8)
    canvas.create_text(16, 12, anchor="nw", fill="white", text="Most common attempt endings (up to 8)")
    if not counts:
        canvas.create_text(width/2, height/2, fill="white", text="No matching attempts")
        return
    gap = (height-50) / len(counts)
    label_width = min(310, width*0.48)
    for index, (reason, count) in enumerate(counts):
        y = 43 + index*gap
        canvas.create_text(12, y, anchor="w", fill="#d0dce6", text=reason[:48])
        right = label_width + (width-label_width-65)*count/counts[0][1]
        canvas.create_rectangle(label_width, y-6, right, y+6, fill="#68d9b3", outline="")
        canvas.create_text(right+8, y, anchor="w", fill="white", text=str(count))


class Panel:
    def __init__(self, window):
        self.window = window
        self.data = {}
        self.records = []
        self.warnings = []
        self.session_lookup = {}
        self.run_lookup = {}
        self.loss = LossReader()
        self.reader = ThreadPoolExecutor(max_workers=1)
        self.pending = None
        self.last_scan = 0
        window.title("Kirby v8 Control Panel")
        window.geometry("1180x880")
        self.status = tk.StringVar(value="Waiting for telemetry…")
        ttk.Label(window, textvariable=self.status, padding=8).pack(fill="x")
        ttk.Label(window, text="Read-only • Trainer terminal: P pause/resume, S save while paused, Q end run").pack(anchor="w", padx=8)
        tabs = ttk.Notebook(window)
        tabs.pack(fill="both", expand=True, padx=8, pady=8)
        live, history, weights = (ttk.Frame(tabs) for _ in range(3))
        for frame, title in [(live, "Live workers"), (history, "History"), (weights, "Model weights")]:
            tabs.add(frame, text=title)
        self.smooth = tk.BooleanVar(value=True)
        tools = ttk.Frame(live)
        tools.pack(fill="x", pady=4)
        self.worker = self.combo(tools, "View", ["All workers"], "All workers", self.draw_live)
        self.live_metric = self.combo(tools, "Graph", [LOSS, *METRICS], LOSS, self.draw_live)
        ttk.Checkbutton(tools, text="10-sample average (raw shown faintly)", variable=self.smooth,
                        command=self.redraw).pack(side="left", padx=8)
        self.workers = self.table(live, ["worker", "status", "room", "health", "action", "exploration", "frames", "reward"], height=7)
        self.details = tk.Text(live, height=7, wrap="word", font=("Consolas", 10))
        self.details.pack(fill="x", pady=6)
        self.canvas = tk.Canvas(live, background="#17212b", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        ttk.Label(live, text="Loss is a learning-error signal shared by all workers; lower loss alone does not mean better play.").pack(anchor="w", pady=4)
        filters = ttk.Frame(history)
        filters.pack(fill="x", pady=4)
        self.session = self.combo(filters, "Session", [CURRENT, "All sessions"], CURRENT, self.session_changed, width=30)
        self.run = self.combo(filters, "Run", ["All runs"], "All runs", self.draw_history, width=22)
        self.mode = self.combo(filters, "Mode", ["train", "eval"], "train", self.draw_history, width=12)
        self.history_worker = self.combo(filters, "Worker", ["All workers"], "All workers", self.draw_history, width=12)
        filters2 = ttk.Frame(history)
        filters2.pack(fill="x", pady=4)
        self.metric = self.combo(filters2, "Graph", [*METRICS, ENDINGS], "Rooms visited", self.draw_history)
        self.start_room = self.combo(filters2, "Start room", ["Any start"], "Any start", self.draw_history, width=12)
        self.from_date, self.to_date = tk.StringVar(), tk.StringVar()
        for title, value in [("UTC from", self.from_date), ("to", self.to_date)]:
            ttk.Label(filters2, text=title).pack(side="left", padx=4)
            entry = ttk.Entry(filters2, textvariable=value, width=11)
            entry.pack(side="left")
            entry.bind("<Return>", lambda event: self.draw_history())
        ttk.Button(filters2, text="Apply", command=self.draw_history).pack(side="left", padx=4)
        ttk.Button(filters2, text="Refresh files", command=self.scan).pack(side="left", padx=4)
        ttk.Checkbutton(history, text="10-sample average (raw samples shown faintly)", variable=self.smooth, command=self.redraw).pack(anchor="w", padx=4)
        self.history_status = tk.StringVar(value="Reading logs…")
        ttk.Label(history, textvariable=self.history_status, wraplength=1100).pack(fill="x", pady=4)
        self.history_canvas = tk.Canvas(history, background="#17212b", highlightthickness=0)
        self.history_canvas.pack(fill="both", expand=True)
        ttk.Label(history, text="UTC dates: YYYY-MM-DD • Graphs show up to 2,000 recent points. Modes stay separate. Mixed starts can affect comparisons.").pack(anchor="w", pady=4)
        self.attempts = self.table(history, ["utc", "worker", "episode", "start", "rooms", "damage", "reward", "reason"], height=8)
        self.weight_text = tk.Text(weights, font=("Consolas", 11), wrap="none")
        self.weight_text.pack(fill="both", expand=True)
        for canvas in (self.canvas, self.history_canvas):
            canvas.bind("<Configure>", lambda event: self.redraw())
        window.protocol("WM_DELETE_WINDOW", self.close)
        self.scan()
        self.refresh()

    @staticmethod
    def table(parent, columns, height):
        frame = ttk.Frame(parent)
        frame.pack(fill="x")
        tree = ttk.Treeview(frame, columns=columns, show="headings", height=height)
        for column in columns:
            tree.heading(column, text=column.title())
            tree.column(column, width=120, minwidth=65)
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        tree.pack(side="left", fill="x", expand=True)
        scrollbar.pack(side="right", fill="y")
        return tree

    @staticmethod
    def combo(parent, label, values, default, callback, width=22):
        ttk.Label(parent, text=label).pack(side="left", padx=4)
        widget = ttk.Combobox(parent, values=values, state="readonly", width=width)
        widget.set(default)
        widget.pack(side="left", padx=4)
        widget.bind("<<ComboboxSelected>>", lambda event: callback())
        return widget

    @staticmethod
    def options(widget, values):
        widget.configure(values=values)
        if widget.get() not in values:
            widget.set(values[0])

    @staticmethod
    def text(widget, content):
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", content)
        widget.configure(state="disabled")

    def current_session(self):
        return self.data.get("session") or (self.records[-1]["_session"] if self.records else "")

    def session_records(self):
        selected = self.session.get()
        selected = self.current_session() if selected == CURRENT else self.session_lookup.get(selected, selected)
        return [r for r in self.records if selected == "All sessions" or r["_session"] == selected]

    def session_changed(self):
        self.run.set("All runs")
        self.update_filters()
        self.draw_history()

    def update_filters(self):
        sessions = list(dict.fromkeys(r["_session"] for r in reversed(self.records)))
        first = {}
        for row in self.records:
            first.setdefault(row["_session"], row)
        self.session_lookup = {
            (session if session.startswith("Legacy") else f"{first[session]['utc'][:19].replace('T', ' ')} UTC [{session[-6:]}]"): session
            for session in sessions
        }
        self.options(self.session, [CURRENT, "All sessions", *self.session_lookup])
        rows = self.session_records()
        runs = {}
        for row in rows:
            runs.setdefault(row["_run"], row)
        self.run_lookup = {
            (run if run.startswith("Legacy") else f"{(row.get('run_started_utc') or row['utc'])[:19].replace('T', ' ')} UTC {row['mode']} [{run[:8]}]"): run
            for run, row in reversed(list(runs.items()))
        }
        self.options(self.run, ["All runs", *self.run_lookup])
        self.options(self.mode, list(dict.fromkeys(["train", "eval", *sorted({r["mode"] for r in rows})])))
        self.options(self.history_worker, ["All workers", *[str(i) for i in sorted({r["worker"] for r in rows})]])
        self.options(self.start_room, ["Any start", *sorted({r.get("start_room") or "Unknown" for r in rows})])

    def scan(self):
        if self.pending is None:
            self.pending = self.reader.submit(load_attempts, ROOT)
            self.last_scan = time.monotonic()

    def draw_live(self):
        self.workers.delete(*self.workers.get_children())
        workers = self.data.get("workers", {})
        self.options(self.worker, ["All workers", *sorted(workers, key=int)])
        selected = self.worker.get()
        lines = []
        age = max(0, time.time()-self.data.get("timestamp", 0))
        for wid, state in sorted(workers.items(), key=lambda item: int(item[0])):
            if selected != "All workers" and wid != selected:
                continue
            status = state.get("status", "observed")
            if age > 3:
                status += " (stale)"
            elif self.data.get("status") in {"paused", "pausing", "stopping"} and status not in {"finished", "error"}:
                status = self.data["status"]
            hp = state.get("hp")
            health = "—" if hp is None else f"{hp} raw"
            epsilon = state.get("epsilon")
            self.workers.insert("", "end", values=(f"W{wid}", status, state.get("room", "—"), health,
                state.get("action", "—"), "—" if epsilon is None else f"{epsilon:.0%}",
                state.get("frames", 0), f"{state.get('reward', 0):.3f}"))
            lines.append(f"W{wid}: attempt {state.get('attempt', '—')} | {state.get('stop_reason') or status}")
            if selected != "All workers":
                lines.append(f"Position ({state.get('x', '—')}, {state.get('y', '—')}) | lives {state.get('lives', '—')} | ability {state.get('ability', '—')}")
                lines.append(f"No-progress timer: {state.get('watchdog_seconds', 0):.2f}s / {state.get('watchdog_limit_seconds', 10):g}s; reset: {state.get('watchdog_reset', '—')}")
                lines.append(f"Remaining room/overall budget: {state.get('room_frames_remaining', '—')} game frames")
                lines.append("Reward contributions: " + ", ".join(f"{key}={value:.3f}" for key, value in state.get("rewards", {}).items()))
        checkpoint = self.data.get("checkpoint", {})
        note = f"Loaded: {checkpoint.get('loaded_from', 'unknown')} | last saved: {checkpoint.get('saved_to', 'unknown')} (update {checkpoint.get('saved_updates', '—')})"
        baseline = checkpoint.get("saved_updates")
        if baseline is None:
            baseline = checkpoint.get("loaded_updates")
        if baseline is not None:
            note += f"\nUpdates since loaded/saved checkpoint: {max(0, self.data.get('updates', baseline)-baseline):,}"
        self.text(self.details, note + "\n" + ("\n".join(lines) or "Waiting for worker observations…"))
        metric = self.live_metric.get()
        if metric == LOSS:
            draw_plot(self.canvas, list(self.loss.points), "Shared model loss • latest 1,000 samples • all workers", "Training update", "Smooth L1 loss", self.smooth.get())
        else:
            rows = [r for r in self.records if r["_session"] == self.data.get("session") and r["_run"] == self.data.get("run_id")
                    and (selected == "All workers" or str(r["worker"]) == selected)]
            draw_plot(self.canvas, metric_points(rows, metric), f"{metric} • recorded attempts in current/last run",
                      "Attempt completion order", METRICS[metric][1], self.smooth.get(), metric == "Success rate")

    def draw_history(self):
        try:
            lower, upper = self.from_date.get().strip(), self.to_date.get().strip()
            for value in (lower, upper):
                if value:
                    date.fromisoformat(value)
            if lower and upper and lower > upper:
                raise ValueError("From date is later than to date")
        except ValueError:
            self.history_status.set("Use valid YYYY-MM-DD dates, with From no later than To.")
            return
        rows = [r for r in self.session_records() if r["mode"] == self.mode.get()
                and (self.run.get() == "All runs" or r["_run"] == self.run_lookup.get(self.run.get(), self.run.get()))
                and (self.history_worker.get() == "All workers" or str(r["worker"]) == self.history_worker.get())
                and (self.start_room.get() == "Any start" or (r.get("start_room") or "Unknown") == self.start_room.get())
                and (not lower or r["_date"] >= lower) and (not upper or r["_date"] <= upper)]
        known = [r["success"] for r in rows if r["success"] is not None]
        success = f"{sum(known)/len(known):.1f}%" if known else "—"
        legacy = " • Legacy session/run boundaries are unknown." if any(r["_run"].startswith("Legacy") for r in rows) else ""
        warnings = f" • {len(self.warnings)} file/row warnings; first: {self.warnings[0]}" if self.warnings else ""
        self.history_status.set(f"{len(rows):,} attempts • success {success} • {self.mode.get()} only • refresh every 10s{legacy}{warnings}")
        metric = self.metric.get()
        if metric == ENDINGS:
            draw_endings(self.history_canvas, rows)
        else:
            draw_plot(self.history_canvas, metric_points(rows, metric), f"{metric} • {self.mode.get()} • raw + optional 10-sample average",
                      "Filtered attempt completion order", METRICS[metric][1], self.smooth.get(), metric == "Success rate")
        self.attempts.delete(*self.attempts.get_children())
        for row in reversed(rows[-100:]):
            self.attempts.insert("", "end", values=(row["utc"], row["worker"], row["episode"], row.get("start_room") or "?",
                row.get("rooms"), row.get("health_lost"), row.get("reward"), row["reason"]))

    def redraw(self):
        self.draw_live()
        self.draw_history()

    def refresh(self):
        try:
            self.data = read_snapshot(ROOT / "telemetry")
            age = max(0, time.time()-self.data["timestamp"])
            stale = f" • STALE ({age:.0f}s since update; idle/off/disconnected is not distinguishable)" if age > 3 else ""
            loss = self.data.get("loss")
            self.status.set(f"{self.data.get('status', 'unknown').upper()} • Updates {self.data.get('updates', 0):,} • Replay {self.data.get('replay', 0):,} • {self.data.get('fps', 0):.0f} game frames/s • Loss {loss if loss is not None else '—'}{stale}")
            self.loss.read(ROOT / "telemetry" / Path(self.data["history"]).name)
            weights = ["Layer                           Mean          Std          Min          Max"]
            for item in self.data.get("weights", []):
                weights.append(f"{item['layer']:<28} {item['mean']:>12.5f} {item['std']:>12.5f} {item['min']:>12.5f} {item['max']:>12.5f}")
            self.text(self.weight_text, "\n".join(weights))
        except (OSError, ValueError, KeyError, TypeError, csv.Error) as error:
            self.status.set(f"Waiting for telemetry: {error}")
        if self.pending is not None and self.pending.done():
            try:
                self.records, self.warnings = self.pending.result()
            except Exception as error:
                self.warnings = [str(error)]
            self.pending = None
        if time.monotonic() - self.last_scan >= 10:
            self.scan()
        self.update_filters()
        self.redraw()
        self.timer = self.window.after(1000, self.refresh)

    def close(self):
        self.window.after_cancel(self.timer)
        self.reader.shutdown(wait=False, cancel_futures=True)
        self.window.destroy()


if __name__ == "__main__":
    window = tk.Tk()
    Panel(window)
    window.mainloop()
