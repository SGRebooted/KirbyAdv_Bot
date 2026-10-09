"""Check the transport without needing a running emulator or a game ROM."""

import json
from pathlib import Path
import tempfile
import threading
import time
import unittest

from KA_Bot import KirbyBridge


class BridgeTests(unittest.TestCase):
    """Use an isolated temporary folder to simulate the Lua side of the bridge."""

    def test_step_ignores_stale_reply_and_sends_buttons(self):
        """An old response must not be mistaken for the current step's result."""
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / "response.json").write_text('{"id":"old"}')
            received = []

            def emulator():
                """Stand in for Lua: receive one command and acknowledge its ID."""
                deadline = time.monotonic() + 2
                while time.monotonic() < deadline:
                    try:
                        message = (folder / "command.txt").read_text()
                        break
                    except FileNotFoundError:
                        time.sleep(0.005)
                else:
                    return
                received.append(message.split("|"))
                (folder / "response.json").write_text(json.dumps({
                    "id": received[0][0], "frame": 4,
                }))

            worker = threading.Thread(target=emulator)
            worker.start()
            try:
                result = KirbyBridge(folder, timeout=2).step("right_a", 4)
                self.assertEqual(result["frame"], 4)
                self.assertEqual(received[0][1:], ["step", "4", "Right,A"])
            finally:
                worker.join()

    def test_invalid_steps_never_publish_commands(self):
        """Reject typos and oversized inputs before sending anything to Lua."""
        with tempfile.TemporaryDirectory() as directory:
            bridge = KirbyBridge(directory)
            for action, frames in [("fly", 4), ("right", 0), ("right", 61)]:
                with self.assertRaises(ValueError):
                    bridge.step(action, frames)
            self.assertFalse((Path(directory) / "command.txt").exists())

    def test_timeout_requires_restart(self):
        """Avoid sending further moves when the previous command's fate is unknown."""
        with tempfile.TemporaryDirectory() as directory:
            bridge = KirbyBridge(directory, timeout=0.02)
            with self.assertRaises(TimeoutError):
                bridge.observe()
            with self.assertRaises(RuntimeError):
                bridge.step("right")


if __name__ == "__main__":
    unittest.main()
