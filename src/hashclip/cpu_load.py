"""Linux load snapshots outside measured retrieval calls."""
from pathlib import Path
import os
import time


def snapshot():
    cpus = {}
    for line in Path('/proc/stat').read_text().splitlines():
        fields = line.split()
        if fields and fields[0].startswith('cpu'):
            v = list(map(int, fields[1:]))
            cpus[fields[0]] = {'total': sum(v[:8]), 'idle': v[3] + v[4]}
    processes = {}
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():
            continue
        try:
            stat = (p / 'stat').read_text()
            fields = stat[stat.rfind(')') + 2:].split()
            processes[p.name] = {'ticks': int(fields[11]) + int(fields[12]),
                                 'name': stat[stat.find('(') + 1:stat.rfind(')')],
                                 'affinity': sorted(os.sched_getaffinity(int(p.name)))}
        except (OSError, ValueError):
            pass
    return {'monotonic': time.monotonic(), 'utc': time.time(), 'cpus': cpus, 'processes': processes}


def difference(before, after):
    elapsed = after['monotonic'] - before['monotonic']
    load = {}
    for cpu, b in before['cpus'].items():
        a = after['cpus'][cpu]
        total = a['total'] - b['total']
        load[cpu] = 100 * (1 - (a['idle'] - b['idle']) / total) if total else 0.0
    processes = []
    for pid, a in after['processes'].items():
        if pid in before['processes']:
            percent = (a['ticks'] - before['processes'][pid]['ticks']) / os.sysconf('SC_CLK_TCK') / elapsed * 100
            if percent >= 5:
                processes.append({'pid': int(pid), 'cpu_percent': percent, 'name': a['name'], 'affinity': a['affinity']})
    return {'start_utc': before['utc'], 'end_utc': after['utc'], 'seconds': elapsed,
            'cpu_busy_percent': load, 'processes_above_5_percent': sorted(processes, key=lambda p: -p['cpu_percent'])}
