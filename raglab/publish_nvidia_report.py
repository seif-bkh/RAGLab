"""Publish SMALL, credential-free result records to GitHub Checks.

Full JSON/chunks/caches stay in workflow artifacts. The Checks API is readable
in environments where GitHub's redirected blob/log downloads are blocked.
This publishes measurements, not an independent quality certification.
"""
import json
import os
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent / "results/nvidia"


def report_parts(report):
    summary = {k: v for k, v in report.items() if k not in {'generation', 'translation_quality'}}
    parts = [('summary', summary)]
    for name, quality in report.get('translation_quality', {}).items():
        parts.append(('translations-' + name.split('/')[1], {'model_prompt': name, **quality}))
    for run in report.get('generation', []):
        rows = []
        for q in run['questions']:
            # Keep actual answer, citations and quoted evidence; omit unused source
            # bodies (already in full artifact) and never include reasoning text.
            result = {k: v for k, v in q['result'].items() if k != 'sources'}
            result['source_ids'] = [{k: v for k, v in s.items() if k != 'text'} for s in q['result']['sources']]
            rows.append({**q, 'result': result})
        for start in range(0, len(rows), 8):
            parts.append((f"{run['label']}-{start//8+1}", {
                **{k: v for k, v in run.items() if k != 'questions'}, 'questions': rows[start:start+8]}))
    return parts


def main():
    path = ROOT / 'benchmark.json'
    if not path.exists():
        print('No benchmark result to publish (artifact upload still runs).')
        return
    report = json.loads(path.read_text())
    markdown = (ROOT / 'REPORT.md').read_text()
    if os.environ.get('GITHUB_STEP_SUMMARY'):
        with open(os.environ['GITHUB_STEP_SUMMARY'], 'a') as stream:
            stream.write(markdown)
    repository = os.environ.get('GITHUB_REPOSITORY')
    sha = os.environ.get('GITHUB_SHA')
    if not repository or not sha:
        print(markdown)
        return
    for name, data in report_parts(report):
        text = '```json\n' + json.dumps(data, ensure_ascii=False, separators=(',', ':')) + '\n```'
        if len(text.encode('utf-8')) > 60000:
            raise ValueError(f'Report part {name} is too large for Checks; inspect full artifact instead')
        payload = {
            'name': 'NVIDIA results / ' + name,
            'head_sha': sha, 'status': 'completed', 'conclusion': 'neutral',
            'output': {'title': 'Measured results: ' + name,
                       'summary': markdown[:50000] if name == 'summary' else 'Detailed measurements; full sources and raw JSON are in raglab-nvidia-results.',
                       'text': text},
        }
        with tempfile.TemporaryDirectory() as temporary:
            p = Path(temporary) / 'check.json'
            p.write_text(json.dumps(payload, ensure_ascii=False))
            subprocess.run(['gh', 'api', '--method', 'POST', f'repos/{repository}/check-runs',
                            '--input', str(p), '--jq', '.html_url'], check=True)


if __name__ == '__main__':
    main()
