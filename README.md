# Kirby bot — first connection

This is the foundation for the planned general gameplay model. It reads your
mapped RAM and gives Python manual, frame-by-frame control. It does not train a
neural network yet. No Python packages need to be installed.

## Start

1. Keep `KA_Bot.py` and `KA_Bridge.lua` together in the project folder.
2. Run `python KA_Bot.py` in the VS Code terminal. Leave its prompt open.
3. In BizHawk, open Kirby's Adventure and reach a playable scene.
4. Stop the original object logger in the Lua Console. Load `KA_Bridge.lua`
   from this project folder. The emulator pauses while waiting for commands.
5. At the Python prompt, type `read`. Compare HP, position, and room with RAM Watch.
6. Type `save` to record this training starting point. Type `right 4` to advance
   four frames holding Right, or `right_a 4` to hold Right and A.
7. Type `reset` to return to that starting point. Type `quit` to stop the bridge.

Use `help` to list actions. Button names A/B are literal NES controller buttons.
`none 4` advances four frames with every controller button released.
Disable autofire and avoid manual controller input during the control check.

## What to expect

Python displays Kirby and nonempty object slots. Full observations include all
nine slots, raw script IDs, position, signed raw velocity, pose, and eight scratch
variables. Unknown objects are retained and displayed as UNKNOWN; they do not
pause processing. Question-mark labels remain provisional.

The addresses come from your existing Lua logger and saved RAM Watch file.
Level, stage, room, ability, and mouth-full values are raw RAM values; they are
not yet decoded into user-facing stage names or normalized training inputs.
Scratch variables may retain values from prior objects. Script IDs 5, 34, and 65
identify families rather than a guaranteed unique object.

Only run one Python console and one bridge script at a time. Start Python before
loading Lua, then issue commands after Lua reports ready. After a timeout, stop
and restart both scripts: a timed-out command may still have executed. A reset
requires `save` in the current Lua session. Saving again replaces only the
bridge's own `bridge/start.State` file.

## Validation and next steps

Automated checks cover the Python message exchange, stale replies, frame limits,
and timeouts. Live RAM readings, input timing, and save/reset still need the
in-emulator check above. Terrain, rewards, episode endings, and neural-network
training are not implemented yet. This readable file transport is for initial
validation; measure its speed before choosing a transport for long training runs.

BizHawk API reference: https://tasvideos.org/Bizhawk/LuaFunctions
