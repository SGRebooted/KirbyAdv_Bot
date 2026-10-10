"""Launch managed BizHawk workers. Called by the idle parallel coordinator.

No emulator is started when this module is imported. Existing states and the
user's original BizHawk configuration are never overwritten by the launcher.
"""
import ctypes
from ctypes import wintypes
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

from KA_Workers import prepare_worker


def normalized(path):
    return os.path.normcase(os.path.abspath(path))


def command_arguments(command):
    """Use Windows' argument parser, not whitespace splitting or shell quoting."""
    if not command:
        return []
    shell = ctypes.WinDLL('shell32', use_last_error=True)
    shell.CommandLineToArgvW.argtypes = [wintypes.LPCWSTR, ctypes.POINTER(ctypes.c_int)]
    shell.CommandLineToArgvW.restype = ctypes.POINTER(wintypes.LPWSTR)
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.LocalFree.argtypes = [ctypes.c_void_p]
    kernel.LocalFree.restype = ctypes.c_void_p
    count = ctypes.c_int()
    arguments = shell.CommandLineToArgvW(command, ctypes.byref(count))
    if not arguments:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        return [arguments[i] for i in range(count.value)]
    finally:
        kernel.LocalFree(arguments)


def option(arguments, name):
    for index, argument in enumerate(arguments):
        if argument == name and index + 1 < len(arguments):
            return arguments[index + 1]
        if argument.startswith(name + '='):
            return argument[len(name) + 1:]
    return None


def process_inventory():
    """Read only relevant process identities; fail closed if inspection fails."""
    script = '''$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
@(Get-CimInstance Win32_Process -Filter "Name = 'EmuHawk.exe' OR Name LIKE 'python%.exe'" |
    Select-Object ProcessId,Name,ExecutablePath,CommandLine) | ConvertTo-Json -Compress
'''
    result = subprocess.run(['powershell.exe', '-NoProfile', '-NonInteractive', '-Command', script],
                            capture_output=True, text=True, encoding='utf-8-sig',
                            timeout=20, creationflags=subprocess.CREATE_NO_WINDOW)
    if result.returncode:
        raise RuntimeError('Could not inspect running emulators; launch cancelled to avoid duplicates.')
    data = json.loads(result.stdout or '[]')
    return [data] if isinstance(data, dict) else (data or [])


class Launcher:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.children = {}  # Popen handles owned by this coordinator session.
        self.panel = None

    def settings(self):
        path = self.root / 'launcher_settings.json'
        if not path.is_file():
            raise ValueError('Copy launcher_settings.json beside KA_Parallel.py first.')
        data = json.loads(path.read_text(encoding='utf-8-sig'))
        if not isinstance(data, dict):
            raise ValueError('launcher_settings.json must contain a JSON object.')
        for key in ('emuhawk', 'rom', 'base_config'):
            if not isinstance(data.get(key), str) or not Path(data[key]).is_file():
                raise ValueError(f'launcher_settings.json: {key} must name an existing file.')
        if Path(data['emuhawk']).name.lower() != 'emuhawk.exe':
            raise ValueError('The emulator executable must be EmuHawk.exe.')
        timeout = data.get('startup_timeout_seconds', 45)
        if type(timeout) not in (int, float) or not 5 <= timeout <= 180:
            raise ValueError('Startup timeout must be 5–180 seconds.')
        return data

    def inspect(self, processes, settings, allow_unmanaged=False):
        """Recognize our workers by exact config/script paths and launch token.

        Unidentified EmuHawk windows may have a bridge loaded manually. Refuse
        to launch while those exist rather than guessing which worker they use.
        """
        owned, unknown = {}, []
        for process in processes:
            if str(process.get('Name', '')).lower() != 'emuhawk.exe':
                continue
            args = command_arguments(process.get('CommandLine'))
            config = option(args, '--config')
            script = option(args, '--lua')
            token = option(args, '--userdata') or ''
            worker_id = None
            if config and script and token.startswith('ka_launcher:'):
                config_path = Path(config)
                stem = config_path.stem
                if stem.startswith('worker_') and stem[7:].isdigit():
                    candidate = int(stem[7:])
                    if (candidate > 0 and normalized(config_path) == normalized(self.config_path(candidate))
                            and normalized(script) == normalized(self.root / 'worker_startup' / f'worker_{candidate}.lua')
                            and normalized(process.get('ExecutablePath') or '') == normalized(settings['emuhawk'])
                            and len(token.removeprefix('ka_launcher:')) == 32):
                        worker_id = candidate
            if worker_id is None:
                unknown.append(process['ProcessId'])
            elif worker_id in owned:
                raise RuntimeError(f'Multiple processes claim worker {worker_id}; close the duplicate windows first.')
            else:
                owned[worker_id] = {'pid': process['ProcessId'], 'token': token.removeprefix('ka_launcher:')}
        if unknown and not allow_unmanaged:
            raise RuntimeError('Unmanaged BizHawk windows are open (PIDs ' + ', '.join(map(str, unknown))
                               + '). Close those windows before launch; none were closed automatically.')
        return owned

    def config_path(self, worker_id):
        return self.root / 'emulator_configs' / f'worker_{worker_id}.json'

    def prepare_config(self, worker_id, settings):
        """Clone the user's settings once; keep each worker's later settings."""
        destination = self.config_path(worker_id)
        source = destination if destination.exists() else Path(settings['base_config'])
        data = json.loads(source.read_text(encoding='utf-8-sig'))
        data['SingleInstanceMode'] = False
        data['StartPaused'] = True
        data['AutoLoadLastSaveSlot'] = False
        data['RunLuaDuringTurbo'] = True
        data.setdefault('PreferredCores', {})['NES'] = 'NesHawk'
        # Prevent old scripts/sessions from loading alongside our shared bridge.
        for key, value in data.items():
            if key.startswith('Recent') and isinstance(value, dict) and 'AutoLoad' in value:
                value['AutoLoad'] = False
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix('.tmp')
        temporary.write_text(json.dumps(data, indent=2), encoding='utf-8')
        os.replace(temporary, destination)
        return destination

    def ready_path(self, worker_id):
        return self.root / f'bridge_{worker_id}' / 'ready.json'

    def wait_ready(self, worker_id, token, timeout, process=None):
        """A token prevents an old readiness file from satisfying a new launch."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if process is not None and process.poll() is not None:
                raise RuntimeError(f'Worker {worker_id} exited during startup (code {process.returncode}).')
            try:
                data = json.loads(self.ready_path(worker_id).read_text(encoding='utf-8'))
                if (data.get('worker_id') == worker_id and data.get('launch_token') == token
                        and data.get('bridge_version') == 5):
                    return
            except (OSError, ValueError):
                pass
            time.sleep(0.1)
        raise RuntimeError(f'Worker {worker_id} did not become ready. Check its Lua console; no duplicate was launched.')

    def launch(self, worker_ids):
        settings = self.settings()
        processes = process_inventory()
        existing = self.inspect(processes, settings)
        # Validate every selected saved start before starting any process.
        for worker_id in worker_ids:
            state = self.root / f'bridge_{worker_id}' / 'start.State'
            if not state.is_file() or not state.stat().st_size:
                raise ValueError(f'Worker {worker_id} needs a training start.State before launch. Use clonestart SOURCE TARGET for a new worker.')
        result = {}
        for worker_id in worker_ids:
            if worker_id in existing:
                item = dict(existing[worker_id], new=False)
                self.wait_ready(worker_id, item['token'], 2)
                print(f"W{worker_id}: reusing managed process {item['pid']}.")
            else:
                token = uuid.uuid4().hex
                startup = prepare_worker(worker_id, self.root, launch_token=token)
                config = self.prepare_config(worker_id, settings)
                self.ready_path(worker_id).unlink(missing_ok=True)
                command = [settings['emuhawk'], '--config', str(config), '--lua', str(startup),
                           '--userdata', f'ka_launcher:{token}', settings['rom']]
                # ROM is deliberately last. Popen uses an argument list, never a shell.
                process = subprocess.Popen(command, cwd=str(Path(settings['emuhawk']).parent))
                self.children[worker_id] = process
                print(f'W{worker_id}: opened process {process.pid}; waiting for Lua…')
                self.wait_ready(worker_id, token, settings.get('startup_timeout_seconds', 45), process)
                item = {'pid': process.pid, 'token': token, 'new': True}
            result[worker_id] = item
        return result

    def shutdown(self, send_shutdown, timeout=10):
        """Ask only this project's managed bridges to close their emulators.

        The caller must save the model successfully BEFORE calling this method.
        Ownership is checked from full startup paths and the unique launch token.
        Lua also checks that token before acting. No process is killed by name
        or PID, and an unresponsive bridge is left open for manual diagnosis.
        """
        owned = self.inspect(process_inventory(), self.settings(), allow_unmanaged=True)
        errors = []
        for worker_id, identity in owned.items():
            try:
                send_shutdown(worker_id, identity["token"])
            except (Exception, KeyboardInterrupt) as error:
                errors.append(f"W{worker_id}: {error or 'close request interrupted'}")
        deadline = time.monotonic() + timeout
        remaining = dict(owned)
        while remaining:
            # Use process presence, not disappearance of Lua/readiness files,
            # to determine whether a window really exited.
            running = {item["ProcessId"] for item in process_inventory()}
            remaining = {wid: identity for wid, identity in remaining.items() if identity["pid"] in running}
            if not remaining or time.monotonic() >= deadline:
                break
            time.sleep(0.25)
        closed = sorted(set(owned) - set(remaining))
        for worker_id in closed:
            self.children.pop(worker_id, None)
        return {"closed": closed, "remaining": sorted(remaining), "errors": errors}

    def open_panel(self):
        if not self.settings().get('open_panel', True):
            return
        path = self.root / 'KA_ControlPanel.py'
        if not path.is_file():
            print('Panel file missing; workers are ready. Copy KA_ControlPanel.py to open the panel.')
            return
        if self.panel is not None and self.panel.poll() is None:
            return
        for process in process_inventory():
            args = command_arguments(process.get('CommandLine'))
            if any(normalized(arg) == normalized(path) for arg in args[1:] if arg and not arg.startswith('-')):
                print('Control panel is already open.')
                return
        self.panel = subprocess.Popen([sys.executable, str(path)], cwd=str(self.root),
                                      creationflags=subprocess.CREATE_NO_WINDOW)
