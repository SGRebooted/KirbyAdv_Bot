"""Explicit saved-start management; never changes model or replay state."""
import json
import os
from pathlib import Path


def settings(root):
    path = Path(root) / 'parallel_settings.json'
    data = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
    if not isinstance(data, dict):
        raise ValueError('parallel_settings.json must contain an object.')
    return data


def save_settings(root, data):
    path = Path(root) / 'parallel_settings.json'
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8')
    os.replace(temporary, path)


def evaluation_workers(root):
    workers = settings(root).get('evaluation_workers')
    if (not isinstance(workers, list) or not workers
            or any(type(i) is not int or i < 1 for i in workers)
            or len(set(workers)) != len(workers)):
        raise ValueError('Use freezeeval once to establish the evaluation starts and worker set.')
    return tuple(workers)


def freeze_evaluation(root, worker_ids, replace=False):
    """Copy training starts into fixed evaluation starts only on explicit command."""
    root = Path(root)
    data = settings(root)
    prepared = []
    for worker_id in worker_ids:
        source = root / f'bridge_{worker_id}' / 'start.State'
        target = source.with_name('eval.State')
        if target.exists() and not replace:
            raise ValueError('Evaluation starts already exist. Use freezeeval replace only to intentionally replace the baseline.')
        content = source.read_bytes()
        if not content:
            raise ValueError(f'Worker {worker_id} has an empty training start.')
        prepared.append((target, content))
    # Read/validate all source files before writing any destinations.
    for target, content in prepared:
        temporary = target.with_suffix('.tmp')
        temporary.write_bytes(content)
        os.replace(temporary, target)
    data['evaluation_workers'] = list(worker_ids)
    save_settings(root, data)


def clone_training_start(root, source_worker, target_worker):
    """Seed a new worker; refuse to overwrite any existing starting state."""
    if any(type(i) is not int or i < 1 for i in (source_worker, target_worker)):
        raise ValueError('Worker IDs must be positive integers.')
    root = Path(root)
    source = root / f'bridge_{source_worker}' / 'start.State'
    target = root / f'bridge_{target_worker}' / 'start.State'
    content = source.read_bytes()
    if not content:
        raise ValueError('Source starting state is empty.')
    target.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents an accidental replacement of a real start.
    with target.open('xb') as stream:
        stream.write(content)
    return target
