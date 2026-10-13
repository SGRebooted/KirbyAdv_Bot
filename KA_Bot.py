"""First Kirby bot milestone: inspect RAM and control BizHawk one step at a time.

Run this file with PyTorch installed, then load KA_Bridge.lua in BizHawk.
The bridge uses small local files so its messages are easy to inspect and debug.
"""

import json
import os
from pathlib import Path
import time
import uuid
import random
import torch
from torch import nn
from collections import deque
from dataclasses import dataclass
import sys

# Diagnose the environment used by this actual bot process.
print("Python executable:", sys.executable)
print("PyTorch version:", torch.__version__)
print("PyTorch location:", torch.__file__)
print("PyTorch CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("CUDA visibility:", os.environ.get("CUDA_VISIBLE_DEVICES", "(not set)"))

# Both programs use a folder beside these source files to exchange messages.
BRIDGE_DIR = Path(__file__).resolve().parent / "bridge"
ACTIONS = {
    "none": [], "wait": [], "left": ["Left"], "right": ["Right"],
    "up": ["Up"], "down": ["Down"], "a": ["A"], "b": ["B"],
    "left_a": ["Left", "A"], "right_a": ["Right", "A"],
    "left_b": ["Left", "B"], "right_b": ["Right", "B"],
    # These names are macros intercepted by KirbyEnvironment.step below.
    "jump": ["A"], "right_jump": ["Right", "A"], "left_jump": ["Left", "A"],
}

# Script IDs describe behavior families, not necessarily unique enemy species.
# These labels include the later findings from the RAM-mapping conversation.
OBJECT_NAMES = {
    2: "Effect family (not necessarily a kill)",
    4: "End-level bonus stars?", 5: "Pickup / item family",
    6: "Warp Star / travel star family", 8: "Midboss damage explosion",
    13: "Warp Star landing effect?", 14: "Waddle Dee", 15: "Waddle Doo",
    16: "Shotzo", 17: "Sparky", 18: "Poppy Bros. Jr.",
    19: "Poppy Bros. Jr. apple", 24: "Blipper", 32: "Broom Hatter",
    33: "Cappy", 34: "Poppy Bros. Sr. / bomb family", 41: "Hot Head",
    43: "Sword Knight", 56: "Bronto Burt", 57: "Gordo",
    65: "Cutter / blade family", 80: "Beam attack", 81: "Shotzo projectile",
    82: "Sparky attack", 84: "Hot Head associated effect (timing unverified)", 86: "Hot Head fire",
    87: "Sword strike", 89: "Broom Hatter dust", 103: "Crane Fever plush",
    255: "Empty",
}

# Keep this order fixed: each network output corresponds to one action.
# Include Up and Down for doors and other gameplay interactions.
MODEL_ACTIONS = (
    "wait", "left", "right", "up", "down", "a", "b",
    "left_a", "right_a", "left_b", "right_b",
    "jump", "right_jump", "left_jump",
)

# Legacy constants are retained for old console helpers; the imported v9
# encoder below defines the active input layout. Legacy imports are disabled.
LEGACY_OBSERVATION_SIZE = 237 + 81 * 10 + 1
PREVIOUS_BUTTONS = ("Left", "Right", "Up", "Down", "A", "B")
OBSERVATION_SIZE = LEGACY_OBSERVATION_SIZE + 1 + len(PREVIOUS_BUTTONS)
from KA_Observation import (encode_observation, OBSERVATION_SIZE, ENCODER_VERSION,
                            MODE_OFFSET, allowed_indices)

# One unit of progress gap means one screen-width (256 pixels) to recover.
# Do not clamp: a gap larger than one screen still carries useful information.
PROGRESS_GAP_SCALE = 256.0
V3_MODEL_PATH = Path(__file__).resolve().parent / "models" / "kirby_rooms_v3.pth"
TRAINING_VERSION = 9

# Tested stage-1 gameplay rooms; hub and bonus rooms are not ordinary visits.
GAMEPLAY_ROOMS = {(0, 0, 43), (0, 0, 44), (0, 0, 45)}
NEW_ROOM_REWARD = 5.0
GOAL_REWARD = 10.0
ROOM_FRAME_BUDGET = 3600
# Ten seconds without a new 16-pixel cell or new rightward record ends an
# attempt. This is an external truncation, not a death or extra punishment.
NO_PROGRESS_LIMIT = 600
LOADING_FRAME_LIMIT = 120
# Preserve the existing v2 checkpoint when importing learned weights.
LEGACY_MODEL_PATH = Path(__file__).resolve().parent / "models" / "kirby_progress_v2.pth"

# Keep the existing room-43 progress timer scaling as a legacy input only.
PROGRESS_GRACE_FRAMES = 120

# Charge 0.002 per four normal gameplay frames in every supported room.
# This is 0.03 per game second, or 1.8 for a full minute without loading.
# No grace period: moving, fighting, and waiting all use the same time cost.
TIME_COST_PER_FRAME = 0.002 / 4.0

# Retain the old room-43 timer feature; this set does not control time cost.
# Time cost applies to all GAMEPLAY_ROOMS, independent of direction.
PROGRESS_PENALTY_ROOMS = {43}

# Preserve the old checkpoint while testing the new inputs and reward.
MODEL_PATH = (
    Path(__file__).resolve().parent
    / "models"
    / "kirby_v9.pth"
)

# Use the GPU when available; otherwise use the CPU.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class KirbyQNetwork(nn.Module):
    """Estimate a future-reward score for each available action."""

    def __init__(self):
        """Create two intermediate layers and an action-scoring layer."""
        super().__init__()

        # The 128-unit layers combine input features into learned patterns.
        # ReLU adds nonlinear behavior, allowing more complex relationships.
        self.layers = nn.Sequential(
            nn.Linear(OBSERVATION_SIZE, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),

            # One unrestricted score per action: scores can be negative.
            nn.Linear(128, len(MODEL_ACTIONS)),
        )

    def forward(self, observations):
        """Calculate action scores for a batch of encoded observations."""
        return self.layers(observations)

class KirbyBridge:
    """Send one command at a time and wait for its matching observation."""

    def __init__(self, directory=BRIDGE_DIR, timeout=15.0):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        self.failed = False

    def request(self, command, frames=0, buttons=()):
        """Publish a complete command; never accept a previous command's reply."""
        if self.failed:
            raise RuntimeError("Restart both scripts after a timeout to resynchronize.")
        request_id = uuid.uuid4().hex
        message = f"{request_id}|{command}|{frames}|{','.join(buttons)}"
        temporary = self.directory / "command.tmp"
        temporary.write_text(message, encoding="ascii")
        # Renaming prevents Lua from seeing a half-written command.
        # Publish the complete command by replacing the previous file.
        # Lua may briefly have that file open, so allow short retries.
        publish_deadline = time.monotonic() + 2.0

        while True:
            try:
                os.replace(temporary, self.directory / "command.txt")
                break  # Published successfully; now we can await the reply.

            except PermissionError as error:
                # A persistent access problem needs investigation.
                # Stop after two seconds instead of retrying forever.
                if time.monotonic() >= publish_deadline:
                    self.failed = True
                    raise TimeoutError(
                        "Could not publish command.txt within two seconds. "
                        "The file may be locked or access may be restricted."
                    ) from error

                # Give Lua time to finish reading and close the file.
                time.sleep(0.01)
        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            try:
                reply = json.loads((self.directory / "response.json").read_text())
                if reply.get("id") == request_id:
                    if "error" in reply:
                        raise RuntimeError(reply["error"])
                    if reply.get("observation_schema") != 2:
                        raise RuntimeError("V9 requires the new KA_Bridge.lua (observation schema 2). Reload all workers.")
                    return reply
            except (OSError, ValueError):
                # Lua may still be writing its response; try again shortly.
                pass
            time.sleep(0.01)
        self.failed = True
        raise TimeoutError("No reply. Check that KA_Bridge.lua is running in BizHawk.")

    def observe(self):
        """Read memory without advancing any game frames."""
        return self.request("observe")

    def step(self, action="none", frames=4):
        """Hold one named action for 1–60 frames and return the resulting state."""
        if action not in ACTIONS:
            raise ValueError("Unknown action. Type 'help' for available actions.")
        if not 1 <= frames <= 60:
            raise ValueError("Choose between 1 and 60 frames.")
        return self.request("step", frames, ACTIONS[action])

@dataclass(frozen=True)
class Experience:
    """One action and its observed outcome."""
    observation: torch.Tensor
    action_index: int
    reward: float
    next_observation: torch.Tensor
    terminated: bool
    truncated: bool
    frames: int

class ReplayBuffer:
    """Keep recent experiences and sample them for future training."""

    def __init__(self, capacity=5000):
        # When full, deque automatically discards the oldest experience.
        self.items = deque(maxlen=capacity)

    def __len__(self):
        """Allow len(replay) to report how many experiences are stored."""
        return len(self.items)

    def add(
        self, observation, action_index, reward,
        next_observation, terminated, truncated, frames,
    ):
        """Store independent CPU copies of both encoded observations."""
        if (
            len(observation) != OBSERVATION_SIZE
            or len(next_observation) != OBSERVATION_SIZE
        ):
            raise ValueError("Experience has the wrong observation size.")

        if not 0 <= action_index < len(MODEL_ACTIONS):
            raise ValueError("Experience has an invalid action index.")

        if frames <= 0:
            raise ValueError("Experience must include at least one frame.")

        # CPU float32 tensors keep storage compact and off GPU memory.
        # torch.tensor copies the lists, preserving these snapshots.
        self.items.append(Experience(
            observation=torch.tensor(
                observation, dtype=torch.float32, device="cpu"
            ),
            action_index=action_index,
            reward=float(reward),
            next_observation=torch.tensor(
                next_observation, dtype=torch.float32, device="cpu"
            ),
            terminated=bool(terminated),
            truncated=bool(truncated),
            frames=frames,
        ))

    def sample(self, batch_size):
        """Choose a batch without removing experiences from storage."""
        if not 1 <= batch_size <= len(self.items):
            raise ValueError("Not enough experiences for that batch size.")

        # Mix experiences instead of always learning from consecutive frames.
        return random.sample(list(self.items), batch_size)

class KirbyEnvironment:
    """Track rewards and join a door action to its loading frames."""

    def __init__(self):
        self.bridge = KirbyBridge()
        self.farthest_by_room = {}
        self.visited_rooms = set()
        self.last_ready_room = None
        self.attempt_frames = 0
        self.room_deadline = ROOM_FRAME_BUDGET
        self.attempt_finished = False
        self.no_progress_frames = 0
        self.watchdog_frames = 0
        self.explored_cells = set()
        self.watchdog_reason = "reset"
        # History describes the preceding simulated action, not keys held on
        # the keyboard while the emulator is paused.
        self.previous_buttons = (0.0,) * len(PREVIOUS_BUTTONS)

    @staticmethod
    def room_key(state):
        """Room numbers are interpreted together with their level and stage."""
        player = state["kirby"]
        return (player["level"], player["stage"], player["room"])

    @staticmethod
    def ready(state):
        """Only normal terrain and a controllable sprite can end loading."""
        return (
            state["terrain"]["status"] == "candidate"
            and state["kirby"]["sprite_state"] != 255
        )

    def add_progress_context(self, state):
        """Attach independent snapshots of the bot's progress and input history."""
        state["no_progress_frames"] = self.no_progress_frames
        key = self.room_key(state)
        current_x = state["kirby"]["x"]
        # An unseen room starts with zero gap. Loading coordinates are not
        # meaningful, so they also receive zero until the room is ready.
        record = self.farthest_by_room.get(key, current_x)
        state["progress_gap_pixels"] = (
            max(0, record - current_x) if self.ready(state) else 0
        )
        # Make a new list so later actions cannot alter earlier observations.
        state["previous_buttons"] = list(self.previous_buttons)
        return state

    def observe(self):
        return self.add_progress_context(self.bridge.observe())

    def save(self):
        """Save the reset point, not the Python attempt bookkeeping."""
        return self.add_progress_context(self.bridge.request("save"))

    def reset(self):
        """The starting room is already visited and earns no entry bonus."""
        state = self.bridge.request("reset")
        if not self.ready(state) or self.room_key(state) not in GAMEPLAY_ROOMS:
            raise ValueError("Save a settled starting point in stage 1-1, room 43, 44, or 45.")
        self.farthest_by_room.clear()
        self.visited_rooms = {self.room_key(state)}
        self.last_ready_room = self.room_key(state)
        self.farthest_by_room[self.last_ready_room] = state["kirby"]["x"]
        self.attempt_frames = 0
        self.room_deadline = ROOM_FRAME_BUDGET
        self.attempt_finished = False
        self.no_progress_frames = 0
        self.watchdog_frames = 0
        self.explored_cells = set()
        self.watchdog_reason = "reset"
        # Reset starts a new history; do not carry inputs across attempts.
        self.previous_buttons = (0.0,) * len(PREVIOUS_BUTTONS)
        self.explored_cells.add(self.exploration_cell(state))
        return self.add_progress_context(state)

    @staticmethod
    def exploration_cell(state):
        """Include the room so each room has its own two-dimensional coverage."""
        p = state["kirby"]
        return (p["level"], p["stage"], p["room"], p["x"] // 16, p["y"] // 16)

    def close(self):
        self.bridge.request("stop")

    def step(self, action, frames=4):
        """A fresh jump releases A for one simulated frame, then presses it.

        Preserve the four-frame decision duration and store one replay item for
        the whole macro. Loading, death or a timeout during release cancels the
        press so an action cannot spill into a newly loaded room.
        """
        if action not in {"jump", "right_jump", "left_jump"}:
            return self._step_held(action, frames)
        if type(frames) is not int or not 2 <= frames <= 60:
            raise ValueError("Fresh jumps need 2–60 frames, including one release frame.")
        release = {"jump": "wait", "right_jump": "right", "left_jump": "left"}[action]
        press = {"jump": "a", "right_jump": "right_a", "left_jump": "left_a"}[action]
        first = self._step_held(release, 1)
        state, reward, terminated, truncated, info = first
        if terminated or truncated or info["stop_reason"] or info["transition_seen"] or state["kirby"]["hp"] == 255:
            return first
        second = self._step_held(press, frames - 1)
        state2, reward2, terminated2, truncated2, info2 = second
        merged = dict(info2)
        # These fields describe totals; all other fields describe the endpoint.
        for name in ("health_lost", "new_progress", "time_penalty", "room_bonus",
                     "goal_bonus", "progress_reward", "damage_penalty",
                     "death_penalty", "elapsed_frames"):
            merged[name] = info[name] + info2[name]
        merged["learning_reward"] = (info["learning_reward"]
            + 0.99 ** (info["elapsed_frames"] / 4.0) * info2["learning_reward"])
        merged["replay_ok"] = info["replay_ok"] and info2["replay_ok"]
        for name in ("transition_seen", "first_visit", "success", "lost_life"):
            merged[name] = info[name] or info2[name]
        merged["x_change"] = (None if info["x_change"] is None or info2["x_change"] is None
                              else info["x_change"] + info2["x_change"])
        return state2, reward + reward2, terminated2, truncated2, merged

    def _step_held(self, action, frames=4):
        """Perform an action, then finish detected loading with neutral input.

        Replay stores this entire interval under the original action. Rewards
        during loading are discounted to their time within that interval.
        """
        if self.attempt_finished:
            raise RuntimeError("Attempt finished. Type reset before continuing.")
        if action not in ACTIONS or type(frames) is not int or not 1 <= frames <= 60:
            raise ValueError("Use a known action and 1–60 frames.")

        before = self.observe()
        state = before
        elapsed_total = 0
        loading_frames = 0
        loading = not self.ready(before)
        transition_seen = loading
        reward_total = 0.0
        learning_reward = 0.0
        health_total = 0.0
        progress_total = 0
        time_total = 0.0
        room_bonus = 0.0
        goal_bonus = 0.0
        first_visit = False
        success = False
        lost_life = False
        truncated = False
        stop_reason = ""
        initial_ready = self.ready(before)

        while True:
            # Both budgets include loading frames. A newly settled room may
            # refresh its deadline, but the overall backstop never moves.
            remaining = min(self.room_deadline, MAX_ATTEMPT_FRAMES) - self.attempt_frames
            if remaining <= 0:
                truncated = True
                stop_reason = "time budget reached"
                break
            requested = min(4 if loading else frames, remaining)
            if loading:
                requested = min(requested, LOADING_FRAME_LIMIT - loading_frames)
                if requested <= 0:
                    stop_reason = "room data did not recover within the loading limit"
                    break

            previous_state = state
            previous = previous_state["kirby"]
            state = self.bridge.step("wait" if loading else action, requested)
            elapsed = state["frame"] - previous_state["frame"]
            if elapsed != requested:
                self.attempt_finished = True
                raise RuntimeError(f"Expected {requested} frames, received {elapsed}. Reset before continuing.")
            if loading:
                loading_frames += elapsed
            self.attempt_frames += elapsed
            current = state["kirby"]
            key = self.room_key(state)
            previous_key = self.room_key(previous_state)
            raw_transition = state.get("transition_seen")
            if type(raw_transition) is not bool:
                self.attempt_finished = True
                raise RuntimeError("Missing transition flag. Reload Lua, then reset.")

            # Record the buttons used during the preceding simulated interval.
            # Loading uses neutral input. Lua also releases inputs when it
            # detects a transition; use neutral history for that mixed interval.
            previous_action = "wait" if loading or raw_transition else action
            self.previous_buttons = tuple(
                float(button in ACTIONS[previous_action])
                for button in PREVIOUS_BUTTONS
            )

            # Room changes are transitions even if Lua never sampled state 255.
            chunk_transition = (
                loading or raw_transition or key != previous_key
                or not self.ready(previous_state) or not self.ready(state)
            )
            transition_seen = transition_seen or chunk_transition
            lost_life = current["lives"] < previous["lives"]

            # Check the observed goal route BEFORE checking bonus-game terrain.
            # A direct reset into room 42 cannot earn this reward.
            success = (
                not lost_life
                and self.last_ready_room == (0, 0, 45)
                and key == (0, 0, 42)
            )
            settled = (
                self.ready(state) and not raw_transition
                and key == previous_key
            )
            new_progress = 0
            entry_reward = 0.0

            if settled and not lost_life and not success and current["hp"] != 255:
                # Only the three validated gameplay rooms earn visit rewards.
                if key in GAMEPLAY_ROOMS:
                    if key not in self.visited_rooms:
                        self.visited_rooms.add(key)
                        entry_reward = NEW_ROOM_REWARD
                        first_visit = True
                        self.room_deadline = self.attempt_frames + ROOM_FRAME_BUDGET
                    self.last_ready_room = key

                if chunk_transition:
                    # Establish the landing baseline only after loading ends.
                    self.farthest_by_room[key] = max(
                        self.farthest_by_room.get(key, current["x"]), current["x"]
                    )
                else:
                    new_progress = measure_new_progress(previous, current, self.farthest_by_room)

            health_lost = calculate_health_lost(previous["hp"], current["hp"])
            time_penalty = 0.0
            timer_active = (
                not chunk_transition and not lost_life and not success
                and key == (0, 0, 43)
                and previous["hp"] != 255 and current["hp"] != 255
            )
            if not timer_active or new_progress > 0:
                self.no_progress_frames = 0
            else:
                self.no_progress_frames += elapsed

            # Preserve the old timer INPUT above so imported weights keep
            # their input meanings. It no longer determines any punishment.
            # Exclude entire mixed loading/death intervals conservatively.
            if (
                not chunk_transition and not lost_life and not success
                and key in GAMEPLAY_ROOMS
                and previous["hp"] != 255 and current["hp"] != 255
            ):
                time_penalty = elapsed * TIME_COST_PER_FRAME

            # Loading/death intervals do not consume watchdog time. Only a
            # settled room change resets it; a transient flag cannot refresh it.
            self.watchdog_reason = "none"
            if settled and not lost_life and not success and current["hp"] != 255 and key in GAMEPLAY_ROOMS:
                cell = self.exploration_cell(state)
                # Ignore off-map/death-animation coordinates as exploration.
                in_bounds = (0 <= current["x"] < state["terrain"].get("width", 4) * 256
                             and 0 <= current["y"] < state["terrain"].get("height", 1) * 192)
                novel = in_bounds and cell not in self.explored_cells
                if in_bounds:
                    self.explored_cells.add(cell)
                if first_visit or novel or new_progress > 0:
                    self.watchdog_frames = 0
                    self.watchdog_reason = "new room" if first_visit else ("new cell" if novel else "new rightward record")
                elif not chunk_transition:
                    self.watchdog_frames += elapsed

            # Stage completion gets +10, not +10 plus an ordinary room bonus.
            completion_reward = GOAL_REWARD if success else 0.0
            chunk_reward = (
                calculate_reward(health_lost, new_progress, lost_life=lost_life)
                + entry_reward + completion_reward - time_penalty
            )
            reward_total += chunk_reward
            learning_reward += (0.99 ** (elapsed_total / 4.0)) * chunk_reward
            elapsed_total += elapsed
            health_total += health_lost
            progress_total += new_progress
            time_total += time_penalty
            room_bonus += entry_reward
            goal_bonus += completion_reward

            # Death and success are terminal; deadlines are external truncations.
            if lost_life or success:
                stop_reason = "stage cleared" if success else "lost a life"
                break
            if self.attempt_frames >= min(self.room_deadline, MAX_ATTEMPT_FRAMES):
                truncated = True
                stop_reason = "overall time limit" if self.attempt_frames >= MAX_ATTEMPT_FRAMES else "room time limit"
                break

            if self.watchdog_frames >= NO_PROGRESS_LIMIT:
                truncated = True
                stop_reason = "no progress timeout"
                break

            # Require a neutral follow-up after a detected transition. This
            # avoids trusting the first frame that merely changes the room ID.
            if chunk_transition and not (loading and settled):
                loading = True
                if state["terrain"]["status"] == "vertical_layout_unverified":
                    stop_reason = "vertical terrain layout is still unverified"
                    break
                continue
            break

        terminated = lost_life or success
        self.attempt_finished = terminated or truncated
        self.add_progress_context(state)
        # A failed transition is not a learnable endpoint. Terminal endpoints
        # can be represented by zeros because the trainer never bootstraps them.
        replay_ok = initial_ready and not (
            stop_reason and not self.attempt_finished
        ) and (terminated or self.ready(state))
        info = {
            "attempt_frames": self.attempt_frames,
            "room_frames_remaining": max(0, min(self.room_deadline, MAX_ATTEMPT_FRAMES) - self.attempt_frames),
            "health_lost": health_total, "new_progress": progress_total,
            "x_change": state["kirby"]["x"] - before["kirby"]["x"] if not transition_seen and not lost_life else None,
            "transition_seen": transition_seen,
            "no_progress_frames": self.no_progress_frames,
            "watchdog_frames": self.watchdog_frames,
            "watchdog_reset": self.watchdog_reason,
            "progress_reward": progress_total * 0.01,
            "damage_penalty": health_total,
            "death_penalty": 5.0 if lost_life else 0.0,
            "time_penalty": time_total,
            "room_bonus": room_bonus, "goal_bonus": goal_bonus,
            "first_visit": first_visit, "success": success,
            "lost_life": lost_life, "stop_reason": stop_reason,
            "elapsed_frames": elapsed_total, "learning_reward": learning_reward,
            "replay_ok": replay_ok,
        }
        return state, reward_total, terminated, truncated, info

class DQNTrainer:
    """Update the action-scoring network using stored experiences."""

    def __init__(self, model):
        # Keep the same model used by the console's scores command.
        self.model = model
        self.device = next(model.parameters()).device

        # Start the target network as an identical, independent copy.
        self.target = KirbyQNetwork().to(self.device)
        self.target.load_state_dict(model.state_dict())
        self.target.eval()
        self.target.requires_grad_(False)

        # Adam adjusts the model's weights using the calculated gradients.
        # This learning rate is an initial tuning choice.
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=0.0001
        )

        # Huber loss is less sensitive to large errors than squared error.
        self.loss_function = nn.SmoothL1Loss()
        self.updates = 0

    def train_step(self, replay, batch_size=32):
        """Learn from one sampled batch without advancing the emulator."""
        if len(replay) < batch_size:
            raise ValueError(
                f"Need {batch_size} experiences; currently have {len(replay)}."
            )

        batch = replay.sample(batch_size)

        # Combine individual observations into batches on the model's device.
        observations = torch.stack(
            [item.observation for item in batch]
        ).to(self.device)

        next_observations = torch.stack(
            [item.next_observation for item in batch]
        ).to(self.device)

        actions = torch.tensor(
            [item.action_index for item in batch],
            dtype=torch.long,
            device=self.device,
        )
        rewards = torch.tensor(
            [item.reward for item in batch],
            dtype=torch.float32,
            device=self.device,
        )
        terminated = torch.tensor(
            [item.terminated for item in batch],
            dtype=torch.bool,
            device=self.device,
        )
        frames = torch.tensor(
            [item.frames for item in batch],
            dtype=torch.float32,
            device=self.device,
        )

        self.model.train()

        # The model produces one score per available action.
        # Select only the score for the action actually taken.
        all_scores = self.model(observations)
        predicted = all_scores.gather(
            1, actions.unsqueeze(1)
        ).squeeze(1)

        # Targets are fixed reference values for this update.
        with torch.no_grad():
            future = torch.zeros_like(rewards)
            continuing = ~terminated

            # Terminal observations need no future-value estimate.
            if continuing.any().item():
                upcoming = next_observations[continuing]
                # Double DQN: online network selects; target network evaluates.
                online_scores = self.model(upcoming)
                floating = upcoming[:, MODE_OFFSET + 13] == 1.0
                for name in ("jump", "right_jump", "left_jump"):
                    online_scores[floating, MODEL_ACTIONS.index(name)] = -torch.inf
                selected = online_scores.argmax(dim=1, keepdim=True)
                future[continuing] = self.target(upcoming).gather(1, selected).squeeze(1)

            # Discount by 0.99 per four-frame action.
            # Adjust for shorter actions at the attempt's time limit.
            discount = torch.full_like(frames, 0.99).pow(frames / 4.0)

            # Truncation is our external time limit, not an in-game death.
            # Therefore it retains a future-value estimate.
            targets = rewards + discount * future

        loss = self.loss_function(predicted, targets)
        if not torch.isfinite(loss).item():
            raise RuntimeError("Non-finite loss; update cancelled.")

        # Clear old gradients, calculate new ones, then update the weights.
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # Limit unusually large gradients before applying the update.
        nn.utils.clip_grad_norm_(
            self.model.parameters(),
            max_norm=10.0,
            error_if_nonfinite=True,
        )
        self.optimizer.step()
        self.updates += 1

        # Refresh the target periodically instead of after every update.
        if self.updates % 100 == 0:
            self.target.load_state_dict(self.model.state_dict())

        return loss.item()

def show_state(state):
    """Display raw RAM values without guessing at undocumented game meanings."""
    player = state["kirby"]
    print(f"\nFrame {state['frame']} | Kirby: {player}")
    for obj in state["objects"]:
        if obj["script"] == 255:
            continue
        name = OBJECT_NAMES.get(obj["script"], "UNKNOWN")
        print(f"  Slot {obj['slot']:02X}: {name} ({obj['script']}) "
              f"dx={obj['x'] - player['x']} dy={obj['y'] - player['y']} "
              f"vx={obj['vx']} vy={obj['vy']} pose={obj['pose']}")

def calculate_health_lost(previous_hp_raw, current_hp_raw):
    """Estimate damage during normal play, excluding the observed death value."""
    # We observed 255 during death. Comparing it with normal health
    # could produce false damage when Kirby respawns.
    # Attempt-ending logic handles losing a life separately.
    if previous_hp_raw == 255 or current_hp_raw == 255:
        return 0.0

    # One visible segment corresponded to 8 raw units in our tests.
    raw_decrease = previous_hp_raw - current_hp_raw
    return max(0, raw_decrease) / 8

def calculate_reward(health_lost, new_progress, lost_life=False):
    """Score progress, normal damage, and losing a life."""
    # Reward only ground beyond the previous farthest position.
    progress_reward = 0.01 * new_progress

    # Penalize ordinary damage using our observed health conversion.
    damage_penalty = 1.0 * health_lost

    # Losing a life adds a separate penalty.
    # Five is an initial tuning choice, not a value from game RAM.
    death_penalty = 5.0 if lost_life else 0.0

    return progress_reward - damage_penalty - death_penalty

def measure_new_progress(previous, current, farthest_by_room):
    """Track new rightward ground separately for each room."""
    # Include level and stage so different locations stay separate.
    previous_room = (
        previous["level"], previous["stage"], previous["room"]
    )
    current_room = (
        current["level"], current["stage"], current["room"]
    )

    # On the first observation of a room, use Kirby's position as the baseline.
    # setdefault keeps an existing record when Kirby revisits that room.
    farthest_by_room.setdefault(previous_room, previous["x"])
    farthest_by_room.setdefault(current_room, current["x"])

    # Room transitions and losing a life can relocate Kirby.
    # Don't count those position changes as forward movement.
    if previous_room != current_room or current["lives"] != previous["lives"]:
        return 0

    # Include the before-position so unobserved movement isn't rewarded.
    old_record = max(farthest_by_room[current_room], previous["x"])
    new_progress = max(0, current["x"] - old_record)

    # Moving backward never lowers the record.
    farthest_by_room[current_room] = max(old_record, current["x"])

    return new_progress

def choose_action(model, observation, epsilon):
    """Choose randomly for exploration, or use the model's highest score."""
    allowed = allowed_indices(observation, MODEL_ACTIONS)
    if model is None or random.random() < epsilon:
        return MODEL_ACTIONS[random.choice(allowed)], "random"

    # Make a batch containing one observation on the model's device.
    device = next(model.parameters()).device
    inputs = torch.tensor(
        [observation], dtype=torch.float32, device=device
    )

    # Choosing an action does not update the network.
    with torch.no_grad():
        scores = model(inputs)[0]

        if not torch.isfinite(scores).all().item():
            raise RuntimeError("Non-finite action scores; play stopped.")

        blocked = [i for i in range(len(MODEL_ACTIONS)) if i not in allowed]
        scores[blocked] = -torch.inf
        action_index = scores.argmax().item()

    return MODEL_ACTIONS[action_index], "model"

def run_attempt(environment, replay, max_steps=None, model=None, epsilon=1.0, trainer=None):
    """Play one attempt; transitions stay attached to the triggering action."""
    if not 0.0 <= epsilon <= 1.0:
        raise ValueError("Exploration must be between 0 and 1.")
    if max_steps is not None and (type(max_steps) is not int or not 1 <= max_steps <= MAX_ATTEMPT_FRAMES):
        raise ValueError(f"Choose 1–{MAX_ATTEMPT_FRAMES} steps, or omit the step limit.")
    state = environment.reset()
    stored_count = skipped_count = 0
    total_reward = 0.0
    last_loss = None
    # At least one frame advances per iteration; this is an additional bound.
    step_limit = max_steps if max_steps is not None else MAX_ATTEMPT_FRAMES
    print("Press Q in this terminal to stop between actions and return to Kirby>.")
    for step_number in range(1, step_limit + 1):
        # Poll between complete bridge requests, avoiding Ctrl+C halfway through
        # a request or optimizer update. No model is saved automatically.
        if stop_requested():
            print("Stopped by Q. Use savemodel to keep these updates.")
            break
        if not environment.ready(state) or environment.room_key(state) not in GAMEPLAY_ROOMS:
            print("Play stopped: current room/terrain is not validated for this training setup.")
            break
        observation = encode_observation(state)
        action, action_source = choose_action(model, observation, epsilon)
        state, reward, terminated, truncated, info = environment.step(action, frames=4)
        total_reward += reward

        if info["replay_ok"]:
            # A terminal state has no future value, including the goal game.
            next_inputs = [0.0] * OBSERVATION_SIZE if terminated else encode_observation(state)
            replay.add(
                observation=observation, action_index=MODEL_ACTIONS.index(action),
                reward=info["learning_reward"], next_observation=next_inputs,
                terminated=terminated, truncated=truncated,
                frames=info["elapsed_frames"],
            )
            stored_count += 1
            if trainer is not None and len(replay) >= 128 and stored_count % 4 == 0:
                last_loss = trainer.train_step(replay)
        else:
            skipped_count += 1

        if step_number % 10 == 0 or info["health_lost"] or info["transition_seen"] or terminated or truncated:
            print(
                f"Step {step_number} | {action} ({action_source}) | Reward: {reward:.2f} | "
                f"Total: {total_reward:.2f} | Room: {state['kirby']['room']} | "
                f"Room time left: {info['room_frames_remaining']}f | "
                f"Time cost: {info['time_penalty']:.3f}"
            )
        if info["first_visit"]:
            print(f"First room visit: +{info['room_bonus']:g}; fresh room budget.")
        if info["success"]:
            print(f"Stage 1-1 cleared! Goal reward: +{info['goal_bonus']:g}.")
        if info["stop_reason"]:
            print(f"Automated attempt stopped: {info['stop_reason']}.")
            break
    else:
        print("Automated test stopped: requested step limit reached.")

    print(f"Total reward: {total_reward:.2f}")
    print(f"Stored this run: {stored_count} | Skipped: {skipped_count}")
    print(f"Replay buffer: {len(replay)} / {replay.items.maxlen}")
    if trainer is not None:
        print(f"Training updates: {trainer.updates}" + (f" | Latest loss: {last_loss:.6f}" if last_loss is not None else " | No new update this attempt."))
    return state


def stop_requested():
    """Windows terminal: Q stops automation without closing Python."""
    import msvcrt
    while msvcrt.kbhit():
        key = msvcrt.getwch()
        # Consume both bytes of special keys such as arrows.
        if key in ("\x00", "\xe0"):
            if msvcrt.kbhit():
                msvcrt.getwch()
        elif key.lower() == "q":
            return True
    return False

def encode_id(value, bit_count):
    """Represent an ID as a fixed number of binary flags."""
    # Eight flags can represent any byte value, including unknown IDs.
    # The lowest-value bit comes first; we always use the same ordering.
    return [
        float((value >> bit) & 1)
        for bit in range(bit_count)
    ]

def encode_terrain(state):
    """Encode 81 terrain cells, keeping missing data separate from air."""
    terrain = state.get("terrain")

    # Missing terrain usually means the old Lua bridge is still running.
    if terrain is None:
        raise ValueError("No terrain received. Reload the updated Lua bridge.")

    # These are the unavailable states currently reported by our bridge.
    unavailable_states = {
        "transition",
        "invalid_dimensions",
        "vertical_layout_unverified",
        "invalid_block_order",
    }

    if terrain["status"] in unavailable_states:
        # Each cell has ten zeros: its availability flag is false.
        return [0.0] * (81 * 10)

    if terrain["status"] != "candidate":
        raise ValueError("Unrecognized terrain status.")

    cells = terrain["cells"]

    # Validate the entire grid before passing anything to the model.
    if not isinstance(cells, list) or len(cells) != 81:
        raise ValueError("Expected 81 terrain cells.")
    if any(type(value) is not int or not -1 <= value <= 255 for value in cells):
        raise ValueError("Invalid terrain attribute received.")

    features = []

    for attribute in cells:
        if attribute == -1:
            # Available data, but this cell lies outside the room.
            # Attribute bits are unused, so keep them zero.
            features.extend([1.0, 0.0] + [0.0] * 8)
        else:
            # Available data for a cell inside the room.
            features.extend([1.0, 1.0])

            # Preserve every raw attribute, including unknown terrain types.
            features.extend(encode_id(attribute, 8))

    return features

def show_terrain(state):
    """Display the terrain snapshot received from Lua."""
    terrain = state.get("terrain")

    # Explain an old bridge version instead of crashing on a missing field.
    if terrain is None:
        print("No terrain received. Load the updated KA_Bridge.lua.")
        return

    # Unavailable terrain is different from a grid of empty cells.
    if terrain["status"] != "candidate":
        print(f"Terrain unavailable: {terrain['status']}")
        return

    cells = terrain["cells"]

    # Check the message before trying to display it.
    if not isinstance(cells, list) or len(cells) != 81:
        raise ValueError("Expected 81 terrain cells.")
    if any(type(value) is not int or not -1 <= value <= 255 for value in cells):
        raise ValueError("Invalid terrain attribute received.")

    print(
        f"Terrain center: ({terrain['center_x']}, "
        f"{terrain['center_y']}) | "
        f"Room layout: {terrain['width']}x{terrain['height']} blocks"
    )
    print("Raw hex attributes | [XX] = Kirby's cell | -- = outside room")

    # Reconstruct nine rows from the flat list sent by Lua.
    for row in range(9):
        displayed = []

        for column in range(9):
            value = cells[row * 9 + column]
            text = "--" if value == -1 else f"{value:02X}"

            # Preserve the center cell's attribute while highlighting it.
            if row == 4 and column == 4:
                text = f"[{text}]"
            else:
                text = f" {text} "

            displayed.append(text)

        print(f"{row - 4:+d}: " + " ".join(displayed))

def save_model(trainer):
    """Save the networks and optimizer without changing the game."""
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "format_version": 1,
        "encoder_version": ENCODER_VERSION,
        "training_version": TRAINING_VERSION,
        "observation_size": OBSERVATION_SIZE,
        "actions": list(MODEL_ACTIONS),
        "model": trainer.model.state_dict(),
        "target": trainer.target.state_dict(),
        "optimizer": trainer.optimizer.state_dict(),
        "updates": trainer.updates,
    }

    # Finish writing a temporary file before replacing the previous checkpoint.
    # If writing fails, the previous checkpoint remains available.
    temporary = MODEL_PATH.with_suffix(".tmp")
    torch.save(checkpoint, temporary)
    os.replace(temporary, MODEL_PATH)

def load_model():
    """Build a restored trainer before replacing the active one."""
    # Load tensors onto the CPU first, then copy them to the selected device.
    # weights_only supports the tensors and basic metadata saved above.
    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=True,
    )

    if not isinstance(checkpoint, dict):
        raise ValueError("Expected a training checkpoint dictionary.")

    # Equal input counts alone aren't enough: their meanings and the action
    # ordering must match too. Version numbers track intentional changes.
    expected = {
        "format_version": 1,
        "encoder_version": ENCODER_VERSION,
        "training_version": TRAINING_VERSION,
        "observation_size": OBSERVATION_SIZE,
        "actions": list(MODEL_ACTIONS),
    }

    for field, value in expected.items():
        if checkpoint.get(field) != value:
            raise ValueError(f"Checkpoint mismatch: {field}.")

    required = ("model", "target", "optimizer", "updates")
    if any(field not in checkpoint for field in required):
        raise ValueError("Checkpoint is missing training data.")

    updates = checkpoint["updates"]
    if type(updates) is not int or updates < 0:
        raise ValueError("Checkpoint has an invalid update count.")

    # Restore into new objects. If restoration fails, the active trainer
    # and model in main() have not been modified.
    restored_model = KirbyQNetwork().to(DEVICE)
    restored_trainer = DQNTrainer(restored_model)

    restored_trainer.model.load_state_dict(checkpoint["model"])
    restored_trainer.target.load_state_dict(checkpoint["target"])
    restored_trainer.optimizer.load_state_dict(checkpoint["optimizer"])
    restored_trainer.updates = updates

    return restored_trainer


def expand_legacy_weights(weights, model):
    raise ValueError("V9 uses a new encoder. Start fresh; legacy weight imports are disabled.")
    """Add zero-weight columns for the seven new inputs; preserve old weights."""
    expected = model.state_dict()
    if set(weights) != set(expected):
        raise ValueError("Checkpoint network layers do not match this model.")
    expanded = {}
    for name, template in expected.items():
        old = weights[name]
        if name == "layers.0.weight":
            if tuple(old.shape) != (template.shape[0], LEGACY_OBSERVATION_SIZE):
                raise ValueError("Checkpoint has an unexpected input-layer shape.")
            # Zero new connections preserve predictions on migration, even
            # when new context inputs are nonzero. Training can learn them later.
            value = torch.zeros_like(template)
            value[:, :LEGACY_OBSERVATION_SIZE] = old.to(
                device=value.device, dtype=value.dtype
            )
            expanded[name] = value
        else:
            if tuple(old.shape) != tuple(template.shape):
                raise ValueError(f"Checkpoint shape mismatch: {name}.")
            expanded[name] = old.to(device=template.device, dtype=template.dtype)
    return expanded


def import_legacy_model(path, training_version):
    raise ValueError("V9 uses a new encoder. Start fresh; legacy weight imports are disabled.")
    """Migrate a 1048-input model without modifying its checkpoint on disk."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("Expected a checkpoint dictionary.")
    expected = {
        "format_version": 1, "encoder_version": 2,
        "training_version": training_version,
        "observation_size": LEGACY_OBSERVATION_SIZE,
        "actions": list(MODEL_ACTIONS),
    }
    for field, value in expected.items():
        if checkpoint.get(field) != value:
            raise ValueError(f"Cannot import checkpoint: {field} mismatch.")
    model = KirbyQNetwork().to(DEVICE)
    model.load_state_dict(expand_legacy_weights(checkpoint["model"], model))
    restored = DQNTrainer(model)
    # V3 already uses our current rewards, so preserve its target estimates too.
    # V2 used different rewards; keep the fresh target copied from the policy.
    if training_version == TRAINING_VERSION:
        restored.target.load_state_dict(
            expand_legacy_weights(checkpoint["target"], restored.target)
        )
    # The input layer changed shape: start fresh optimizer moments and count.
    # The policy weights have NOT been reset to random values.
    return restored


def import_v2_model():
    """Keep the older explicit import command available if needed."""
    return import_legacy_model(LEGACY_MODEL_PATH, training_version=2)


def import_v3_model():
    """Import the current room-reward model into the expanded network."""
    return import_legacy_model(V3_MODEL_PATH, training_version=3)


def import_time_model(path):
    raise ValueError("V9 uses a new encoder. Start fresh; legacy weight imports are disabled.")
    """Import a v6 checkpoint, retaining all learned policy/target weights.

    Append three output rows, seeded from their held-A counterparts. Adam state
    is deliberately reset because the output tensors changed shape; replay is
    cleared by the caller. The historical update count is retained.
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    expected = {
        "format_version": 1, "encoder_version": ENCODER_VERSION,
        "training_version": 5, "observation_size": OBSERVATION_SIZE,
        "actions": list(MODEL_ACTIONS[:-3]),
    }
    if not isinstance(checkpoint, dict):
        raise ValueError("Expected a checkpoint dictionary.")
    for field, value in expected.items():
        if checkpoint.get(field) != value:
            raise ValueError(f"v6 checkpoint mismatch: {field}.")
    if type(checkpoint.get("updates")) is not int or checkpoint["updates"] < 0:
        raise ValueError("Invalid checkpoint update count.")
    trainer = DQNTrainer(KirbyQNetwork().to(DEVICE))
    for name, network in (("model", trainer.model), ("target", trainer.target)):
        old = checkpoint[name]
        expanded = network.state_dict()
        if set(old) != set(expanded):
            raise ValueError(f"Unexpected {name} parameter names.")
        for key, value in old.items():
            if key in {"layers.4.weight", "layers.4.bias"}:
                expected_shape = (len(MODEL_ACTIONS) - 3,) + tuple(expanded[key].shape[1:])
                if tuple(value.shape) != expected_shape:
                    raise ValueError(f"Unexpected shape for {name}.{key}.")
                expanded[key][:-3].copy_(value)
                # Similar actions provide a starting estimate, not proven values.
                expanded[key][-3].copy_(value[MODEL_ACTIONS.index("a")])
                expanded[key][-2].copy_(value[MODEL_ACTIONS.index("right_a")])
                expanded[key][-1].copy_(value[MODEL_ACTIONS.index("left_a")])
            else:
                if value.shape != expanded[key].shape:
                    raise ValueError(f"Unexpected shape for {name}.{key}.")
                expanded[key].copy_(value)
        network.load_state_dict(expanded)
    trainer.updates = checkpoint["updates"]
    return trainer

def train_attempts(environment, replay, trainer, attempts=3, epsilon=0.5):
    """Train across resets, letting room deadlines decide each attempt's length."""
    if type(attempts) is not int or not 1 <= attempts <= 200:
        raise ValueError("Choose between 1 and 200 attempts.")
    if not 0.0 <= epsilon <= 1.0:
        raise ValueError("Exploration must be between 0 and 1.")
    starting_updates = trainer.updates
    for attempt_number in range(1, attempts + 1):
        print(f"\n=== Training attempt {attempt_number}/{attempts} | Exploration: {epsilon:.0%} ===")
        state = run_attempt(environment, replay, model=trainer.model, epsilon=epsilon, trainer=trainer)
        # Q, unsupported terrain, and loading failures stop the entire batch.
        if not environment.attempt_finished:
            print("Training stopped early; inspect the last messages.")
            break
    print(f"\nTraining session finished: {trainer.updates - starting_updates} new updates.")
    print("Use savemodel to keep these weights; nothing was saved automatically.")
    return state

# Five-minute overall backstop; each newly visited room gets one minute.
MAX_ATTEMPT_FRAMES = 18000

def main():
    """Translate typed commands into environment calls and display results."""
    environment = KirbyEnvironment()

    # Keep experiences across random attempts within this Python session.
    replay = ReplayBuffer()

    # Create one network for this console session.
    # No saved weights are loaded yet, so it starts untrained.
    model = KirbyQNetwork().to(DEVICE)
    # Create the optimizer and target network once per session.
    trainer = DQNTrainer(model)
    print(f"Network device: {DEVICE}\n")

    print("Load KA_Bridge.lua in BizHawk, then type read here.")
    print("Commands: read, save, reset, quit, or an action and frame count.")
    print("Example: right_a 4. Type help to list actions.")

    while True:
        try:
            # Split something like 'right 20' into a command and frame count.
            parts = input("Kirby> ").lower().split()
            if not parts:
                continue

            command = parts[0]

            if command == "help":
                print("Actions:", ", ".join(ACTIONS))
                print("train [attempts] [exploration] | play [exploration] [optional steps]")
                print("save/reset: game | savemodel/loadmodel: time-cost model | importv6: migrate old v4 trainer")
                print("Press Q in the terminal during automation to return here safely.")
                continue

            if command == "quit":
                environment.close()
                break

            # These commands don't perform a gameplay action.
            if command == "read":
                state = environment.observe()

            elif command == "terrain":
                # Read and display without advancing the game.
                state = environment.observe()
                show_terrain(state)

            elif command == "inspect":
                # Read and encode without advancing the game.
                state = environment.observe()
                features = encode_observation(state)

                # Show the size and a small, readable sample.
                print(f"Encoded input count: {len(features)}")
                print(f"Distance behind rightward record: {state['progress_gap_pixels']} pixels")
                held_names = [
                    name for name, held in zip(PREVIOUS_BUTTONS, state["previous_buttons"])
                    if held
                ]
                print("Previous simulated buttons:", ", ".join(held_names) or "none")
                print(
                    "First 8 player inputs:",
                    [round(value, 4) for value in features[:8]],
                )

                # Each object block starts with its presence flag.
                active_slots = sum(
                    features[48 + index * 21]
                    for index in range(9)
                )
                print(f"Active object slots: {int(active_slots)}")

            elif command == "save":
                state = environment.save()
                print("Starting point saved.")

            elif command == "reset":
                state = environment.reset()
                print("New attempt started.")

            elif command == "scores":
                # Read the current game without advancing any frames.
                state = environment.observe()
                features = encode_observation(state)

                # A tensor is PyTorch's numeric array.
                # The outer list makes a batch containing one observation.
                inputs = torch.tensor(
                    [features],
                    dtype=torch.float32,
                    device=DEVICE,
                )

                # Inspect predictions without tracking training calculations.
                with torch.no_grad():
                    scores = model(inputs)[0].cpu().tolist()

                # Display the connection between output positions and actions.
                for action, score in zip(MODEL_ACTIONS, scores):
                    print(f"{action:>8}: {score:+.4f}")

                # Scores can now reflect updates performed in this session.
                print(
                    f"Training updates: {trainer.updates}. "
                    "No action was performed."
                )

            elif command == "savemodel":
                # Save training progress independently of the game savestate.
                try:
                    save_model(trainer)
                except OSError as error:
                    print(f"Could not save checkpoint: {error}")
                else:
                    print(
                        f"Saved update {trainer.updates} to {MODEL_PATH}"
                    )
                continue

            elif command == "loadmodel":
                # Replace both references only after restoration succeeds.
                try:
                    restored = load_model()
                except OSError as error:
                    print(f"Could not read checkpoint: {error}")
                else:
                    trainer = restored
                    model = trainer.model
                    print(
                        f"Loaded checkpoint at update {trainer.updates}."
                    )
                    replay = ReplayBuffer()
                    print("Game state unchanged; replay cleared for the restored model.")
                continue

            elif command == "importv6":
                # Use once for the old single-worker v6 checkpoint.
                trainer = import_time_model(
                    Path(__file__).resolve().parent / "models" / "kirby_v6.pth"
                )
                model = trainer.model
                replay = ReplayBuffer()
                print("Imported v6 weights; optimizer and replay reset. Save to create the v7 checkpoint.")
                continue

            elif command == "importv3":
                # Import once when upgrading; later use loadmodel for v4 saves.
                trainer = import_v3_model()
                model = trainer.model
                replay = ReplayBuffer()
                print("Imported v3 policy and target weights; seven new input columns start at zero.")
                print("Optimizer, replay, and update count reset. Use savemodel for a separate time-cost file.")
                continue

            elif command == "importv2":
                # Explicit warm start: preserve weights but reset old training targets.
                restored = import_v2_model()
                trainer = restored
                model = trainer.model
                replay = ReplayBuffer()
                print("Imported v2 weights. Optimizer, target network, replay, and update count reset.")
                print("Use savemodel to save a separate time-cost checkpoint.")
                continue

            elif command == "train":
                # Examples:
                # train       -> three attempts, 50% random exploration
                # train 5 0.3 -> five attempts, 30% random exploration
                if len(parts) > 3:
                    raise ValueError("Usage: train [attempts] [exploration]")

                attempts = int(parts[1]) if len(parts) > 1 else 3
                epsilon = float(parts[2]) if len(parts) > 2 else 0.5

                state = train_attempts(
                    environment,
                    replay,
                    trainer,
                    attempts=attempts,
                    epsilon=epsilon,
                )

            elif command == "trainstep":
                # One update uses existing experiences; no game input is sent.
                loss = trainer.train_step(replay)
                print(
                    f"Training update {trainer.updates} | "
                    f"Loss: {loss:.6f}"
                )
                continue

            elif command == "memory":
                # Inspect storage without changing the game or the model.
                print(f"Stored experiences: {len(replay)}")

                if len(replay) > 0:
                    example = replay.sample(1)[0]
                    print(
                        f"Sample: {MODEL_ACTIONS[example.action_index]} | "
                        f"Reward: {example.reward:g} | "
                        f"Frames: {example.frames}"
                    )
                    print(
                        f"Inputs: {example.observation.numel()} → "
                        f"{example.next_observation.numel()} | "
                        f"Terminated: {example.terminated} | "
                        f"Truncated: {example.truncated}"
                    )

                continue

            elif command == "random":
                # Collect experiences using entirely random actions.
                state = run_attempt(environment, replay)

            elif command == "play":
                # Examples:
                # play         -> 10% exploration, room-based deadlines
                # play 0       -> model choices only, room-based deadlines
                # play 0.1 300 -> 10% exploration, 300 steps
                if len(parts) > 3:
                    raise ValueError("Usage: play [exploration] [steps]")

                epsilon = float(parts[1]) if len(parts) > 1 else 0.1
                steps = int(parts[2]) if len(parts) > 2 else None

                state = run_attempt(
                    environment,
                    replay,
                    max_steps=steps,
                    model=model,
                    epsilon=epsilon,
                )

            else:
                # The environment now handles all gameplay bookkeeping.
                frames = int(parts[1]) if len(parts) > 1 else 4
                state, reward, terminated, truncated, info = (
                    environment.step(command, frames)
                )

                # The console's job is to display the returned information.
                print(
                    f"Attempt frames: "
                    f"{info['attempt_frames']}/{MAX_ATTEMPT_FRAMES}"
                )
                print(
                    f"Health segments lost during action: "
                    f"{info['health_lost']:g}"
                )
                print(f"New rightward progress: {info['new_progress']}")
                # Make the new reward rule visible during manual testing.
                print(f"No-progress timer: {info['no_progress_frames']} frames")
                print(f"Time penalty: {info['time_penalty']:.3f}")
                print(f"Reward: {reward:.3f}")
                if info["transition_seen"]:
                    print(
                        "Transition handled with neutral input; loading included in this action."
                    )

                if info["x_change"] is not None:
                    print(
                        f"Horizontal movement: {info['x_change']:+d}"
                    )
                else:
                    print("Location or lives changed; movement not compared.")

                print(f"Room time left: {info['room_frames_remaining']} frames")
                print(f"Room bonus: {info['room_bonus']:g} | Goal bonus: {info['goal_bonus']:g}")
                if info["stop_reason"]:
                    print(f"Attempt stopped: {info['stop_reason']}.")

            # All successful commands display their resulting observation.
            show_state(state)

        except (ValueError, RuntimeError) as error:
            # Explain invalid commands while keeping the console available.
            print(error)

        except TimeoutError as error:
            # A missing reply means we no longer know the command's outcome.
            environment.bridge.failed = True
            print(error, "Save your model, then restart both scripts before playing again.")

        except OSError as error:
            print(f"File/device error: {error}")

        except (KeyboardInterrupt, EOFError):
            # Ctrl+C or closing input exits the Python console.
            print("\nConsole closed. Stop the Lua script in BizHawk too.")
            break

if __name__ == "__main__":
    main()
