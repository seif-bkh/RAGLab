"""Byte-bounded, credential-free checkpoints readable through GitHub Checks."""
import json
import os
import subprocess
import tempfile
from pathlib import Path

from artifacts import fingerprint, write_json
from hard_harness.common import OUTPUT, read_json, read_jsonl

MAX_BYTES = 48000


def file_parts(path, root):
    name = path.relative_to(root).as_posix()
    if path.suffix == '.jsonl':
        value, kind = read_jsonl(path), 'jsonl'
    else:
        value, kind = read_json(path), 'json'
    digest = fingerprint(value)
    if isinstance(value, list):
        batches, batch = [], []
        for row in value:
            candidate = batch + [row]
            if batch and len(json.dumps(candidate, ensure_ascii=False).encode()) > MAX_BYTES:
                batches.append(batch); batch = []
            batch.append(row)
            if len(json.dumps(batch, ensure_ascii=False).encode()) > MAX_BYTES:
                raise ValueError('Single checkpoint record too large: ' + name)
        batches.append(batch)
    else:
        batches = [value]
    for number, data in enumerate(batches):
        yield {'file': name, 'kind': kind, 'list': isinstance(value, list), 'fingerprint': digest,
               'part': number, 'parts': len(batches), 'data': data}


def publish(phase):
    directory = OUTPUT / phase
    if not directory.exists():
        print('No phase output to publish')
        return
    # Runtime corpus bodies remain in full artifacts; reproducible from docs.
    paths = [p for p in sorted(directory.rglob('*')) if p.suffix in {'.json', '.jsonl'}
             and p.name not in {'runtime_chunks.json', 'runtime_documents.json'}]
    for path in paths:
        for part in file_parts(path, OUTPUT):
            text = '```json\n' + json.dumps(part, ensure_ascii=False, separators=(',', ':')) + '\n```'
            if len(text.encode()) > 60000:
                raise ValueError('Checkpoint part exceeds Checks budget: ' + part['file'])
            payload = {'name': f"Hard harness / {phase} / {part['file']} / {part['part']}",
                       'head_sha': os.environ['GITHUB_SHA'], 'status': 'completed', 'conclusion': 'neutral',
                       'output': {'title': 'Hard harness checkpoint — not a quality certification',
                                  'summary': 'Recorded source/data/results. No API credentials. Full files are in workflow artifacts.',
                                  'text': text}}
            with tempfile.TemporaryDirectory() as temporary:
                target = Path(temporary) / 'check.json'
                write_json(target, payload)
                subprocess.run(['gh', 'api', '--method', 'POST', f"repos/{os.environ['GITHUB_REPOSITORY']}/check-runs",
                                '--input', str(target), '--jq', '.id'], check=True)


def collect(repository, sha, destination):
    """Download all checkpoint parts, verify each file, then write local copies."""
    files = {}
    page = 1
    while True:
        response = json.loads(subprocess.check_output(['gh', 'api',
            f'repos/{repository}/commits/{sha}/check-runs?per_page=100&page={page}'], text=True))
        checks = response['check_runs']
        for check in checks:
            if not check['name'].startswith('Hard harness / '):
                continue
            text = check['output'].get('text') or ''
            part = json.loads(text.split('```json\n', 1)[1].rsplit('\n```', 1)[0])
            entry = files.setdefault(part['file'], {'meta': part, 'chunks': {}})
            if entry['meta']['fingerprint'] != part['fingerprint']:
                raise ValueError('Mixed checkpoint versions for ' + part['file'])
            entry['chunks'][part['part']] = part['data']
        if len(checks) < 100:
            break
        page += 1
    root = Path(destination).resolve()
    for name, entry in files.items():
        meta = entry['meta']
        if set(entry['chunks']) != set(range(meta['parts'])):
            raise ValueError('Incomplete checkpoint file: ' + name)
        value = ([r for i in range(meta['parts']) for r in entry['chunks'][i]]
                 if meta['list'] else entry['chunks'][0])
        if fingerprint(value) != meta['fingerprint']:
            raise ValueError('Checkpoint fingerprint mismatch: ' + name)
        path = (root / name).resolve()
        if not path.is_relative_to(root) or path.suffix not in {'.json', '.jsonl'}:
            raise ValueError('Unsafe checkpoint path')
        if meta['kind'] == 'jsonl':
            from hard_harness.common import write_jsonl
            write_jsonl(path, value)
        else:
            write_json(path, value)
    print(f'Collected and verified {len(files)} files into {root}')
