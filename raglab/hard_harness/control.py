"""Cross-job quota signal; no model key or inference is used here."""
import json
import os
import subprocess
from pathlib import Path

from artifacts import write_json
from hard_harness.common import OUTPUT, now


def gate(phase, shard):
    repository = os.environ['GITHUB_REPOSITORY']
    run_id = os.environ['GITHUB_RUN_ID']
    data = json.loads(subprocess.check_output(['gh','api',
        f'repos/{repository}/actions/runs/{run_id}/artifacts?per_page=100'],text=True))
    paused = any(a['name'].startswith('hard-harness-pause-') for a in data.get('artifacts',[]))
    directory = f'author_{shard:02d}' if phase == 'author' else f'predictions_{shard:02d}'
    if paused:
        write_json(OUTPUT/directory/'manifest.json', {'status':'deferred_after_quota', 'phase':phase,
            'shard':shard,'timestamp':now(),'model_calls':0,
            'action':'Another shard reported a provider/credential pause. Await explicit user switch/resume.'})
    if os.environ.get('GITHUB_OUTPUT'):
        with open(os.environ['GITHUB_OUTPUT'],'a') as stream:
            stream.write('run='+('false' if paused else 'true')+'\n')
    print('Deferred after quota signal; no model calls' if paused else 'No quota signal; shard may run')
    return not paused


def record_pause(phase, shard):
    directory = f'author_{shard:02d}' if phase == 'author' else f'predictions_{shard:02d}'
    path = OUTPUT/directory/'manifest.json'
    data = json.loads(path.read_text()) if path.exists() else {}
    paused = data.get('status') == 'paused'
    if os.environ.get('GITHUB_OUTPUT'):
        with open(os.environ['GITHUB_OUTPUT'],'a') as stream:
            stream.write('paused='+str(paused).lower()+'\n')
    return paused
