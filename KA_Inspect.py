"""Read a saved bridge response without sending commands or importing the trainer.

Usage: python KA_Inspect.py --worker 1
       python KA_Inspect.py --file path/to/response.json
The output describes the last saved response, not a live connection.
"""
import argparse
import json
from pathlib import Path

MODES = {0: 'neutral / context dependent', 1: 'walking', 2: 'running',
         3: 'braking / reversal', 4: 'jump phase', 5: 'falling',
         6: 'crouch / recovery', 9: 'inhale', 10: 'spit / exhale',
         11: 'swallow / copy / Mix', 12: 'ability action', 13: 'floating'}


def describe(state):
    if 'error' in state:
        raise ValueError(f"Bridge response is an error: {state['error']}")
    if state.get('observation_schema') != 2:
        raise ValueError('This response lacks v9 observations. Reload the new bridge and obtain a fresh response.')
    p = state['kirby']
    objects = state['all_objects']
    if [o['slot'] for o in objects] != list(range(18)):
        raise ValueError('Expected all 18 object slots in order.')
    mode, flags = p['action_state'], p['contact_flags']
    lines = [f"Saved snapshot: worker {state.get('worker_id', '?')}, frame {state['frame']}",
             f"Location: level {p['level']}, stage {p['stage']}, room {p['room']}; X {p['x']}, Y {p['y']}",
             f"Player script {p['script']}; action {mode}: {MODES.get(mode, 'unknown')}",
             f"Flags 0x{flags:02X}: contact bit={bool(flags & 4)}, candy protection bit={bool(flags & 64)}",
             f"Ability {p['ability']}; mouth field {p['mouth_full']}; HP {p['hp']}; lives {p['lives']}",
             f"Raw velocity: X {p['vx']}, Y {p['vy']}",
             f"Terrain: {state.get('terrain', {}).get('status', 'missing')}",
             f"Wide terrain: {state.get('terrain_wide', {}).get('status', 'missing')} (17 by 17 when available)"]
    if p['script'] != 1:
        lines.append('Nonstandard player script: ordinary movement labels may not apply.')
    if mode == 11:
        lines.append('Swallow/copy/Mix phase: ability value may be transitional.')
    lines.append('Active object slots (raw fields; effects are not automatically hazards):')
    for o in objects:
        if o['script'] == 255:
            continue
        lines.append(f"  slot {o['slot']:2} script {o['script']:3} pose {o['pose']:3} "
                     f"XY=({o['x']},{o['y']}) vars={o['vars']}")
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--worker', type=int, default=1)
    group.add_argument('--file', type=Path)
    args = parser.parse_args()
    if args.worker < 0:
        parser.error('Worker must be zero or a positive number.')
    directory = 'bridge' if args.worker == 0 else f'bridge_{args.worker}'
    path = args.file or Path(__file__).resolve().parent / directory / 'response.json'
    try:
        state = json.loads(path.read_text(encoding='utf-8-sig'))
        print(describe(state))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(1, f'Cannot inspect {path}: {exc}\nIf a worker was writing the response, retry after it finishes.\n')


if __name__ == '__main__':
    main()
