"""V9 RAM encoder. No emulator control, rewards, or checkpoint I/O."""
import math

ENCODER_VERSION = 4
SCHEMA = 2
MODES = tuple(range(14))
MOUTHS = (0, 1, 2, 3, 9)
ABILITIES = tuple(range(24)) + (255,)
MODE_OFFSET = 8 + len(ABILITIES) + 1
OBJECT_SIZE = 85
TERRAIN_CELLS = 17 * 17
PLAYER_SIZE = 121
OBSERVATION_SIZE = PLAYER_SIZE + 17 * OBJECT_SIZE + TERRAIN_CELLS * 10


def bits(value, count=8):
    if type(value) is not int or not 0 <= value < 2 ** count:
        raise ValueError('Invalid RAM identifier')
    return [float((value >> i) & 1) for i in range(count)]


def category(value, known):
    return [float(value == k) for k in known] + [float(value not in known)]


def relative(value, origin):
    # Shortest displacement in a 16-bit coordinate space: 65520 near X0 is -16.
    # This fixes wraparound, not special animation/inhale coordinate semantics.
    return ((value - origin + 32768) % 65536 - 32768) / 256.0


def encode_observation(state):
    if state.get('observation_schema') != SCHEMA:
        raise ValueError('V9 requires observation schema 2. Reload the v9 worker Lua scripts.')
    p = state['kirby']
    f = [p['hp'] / 47.0 if p['hp'] != 255 else 0.0, float(p['hp'] == 255),
         p['lives'] / 10.0, p['x'] / 256.0, p['y'] / 256.0,
         p['vx'] / 1024.0, p['vy'] / 1024.0, float(p['script'] == 1)]
    f += category(p['ability'], ABILITIES)
    f += category(p['action_state'], MODES)
    f += category(p['mouth_full'], MOUTHS)
    f += bits(p['contact_flags']) + bits(p['script']) + bits(p['sprite_state'])
    f += bits(p['level']) + bits(p['stage']) + bits(p['room'], 16)
    # Sub-tile position helps distinguish approaching an edge from its center.
    f += [(p['x'] % 16) / 16.0, (p['y'] % 16) / 16.0]
    previous = state['previous_buttons']
    if len(previous) != 6 or any(x not in (0, 1) for x in previous):
        raise ValueError('Expected six preceding button flags')
    f += [state['progress_gap_pixels'] / 256.0, min(state['no_progress_frames'] / 600.0, 1.0)] + list(previous)
    # Keep the constant derived from the actual layout, not legacy offsets.
    if len(f) != PLAYER_SIZE:
        raise ValueError(f'Player layout mismatch: {len(f)}')
    objects = state['all_objects']
    by_slot = {o['slot']: o for o in objects}
    if len(objects) != 18 or set(by_slot) != set(range(18)):
        raise ValueError('Expected exactly 18 object slots')
    for slot in range(18):
        if slot == 1:  # Kirby already has dedicated inputs.
            continue
        o = by_slot[slot]
        if o['script'] == 255:
            f += [0.0] * OBJECT_SIZE
            continue
        if len(o['vars']) != 8:
            raise ValueError('Expected eight object scratch bytes')
        f += [1.0] + bits(o['script'])
        f += [relative(o['x'], p['x']), relative(o['y'], p['y']), o['vx']/1024.0, o['vy']/1024.0]
        f += bits(o['pose'])
        for value in o['vars']:
            f += bits(value)
    terrain = state['terrain_wide']
    if terrain['status'] == 'candidate':
        cells = terrain['cells']
        if len(cells) != TERRAIN_CELLS:
            raise ValueError('Expected 17 by 17 terrain cells')
        for v in cells:
            if type(v) is not int or not -1 <= v <= 255:
                raise ValueError('Invalid terrain attribute')
            f += [1.0, float(v != -1)] + (bits(v) if v != -1 else [0.0]*8)
    elif terrain['status'] in {'transition', 'invalid_dimensions', 'vertical_layout_unverified', 'invalid_block_order'}:
        f += [0.0] * (TERRAIN_CELLS * 10)
    else:
        raise ValueError('Unknown terrain status')
    if len(f) != OBSERVATION_SIZE or not all(math.isfinite(x) for x in f):
        raise ValueError('Invalid v9 observation')
    return f


def allowed_indices(observation, actions):
    """Only remove release/repress macros during established floating.

    Held A stays available in all states: cutting it off shortens jumps.
    Up stays available with a full mouth: it can enter a door.
    """
    floating = observation[MODE_OFFSET + 13] == 1.0
    return [i for i, name in enumerate(actions)
            if not (floating and name in {'jump', 'left_jump', 'right_jump'})]
