"""Read-only xKiro catalog diagnostic for the selected Qwen model; no inference.

Only documented HTTPS endpoints below receive their own environment key.
Catalog presence (or HTTP 200 on a public catalog) proves neither authenticated
inference availability nor upstream model identity. Never substitute aliases.
"""
import argparse
import json
import os
import subprocess
import tempfile
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from artifacts import write_json
from pipeline_policy import ANSWER_MODEL

PROVIDERS = {
    'xkiro': {'base_url': 'https://api.xkiro.com/v1', 'key_env': 'XKIRO_API_KEY',
              'documentation': 'https://docs.xkiro.com/',
              'identity_note': 'Docs say the response reports the requested model across routing; not upstream identity proof.'},

}
EXACT_MODELS = [ANSWER_MODEL]
OUTPUT = Path(__file__).resolve().parent / 'results/provider_catalog/catalog.json'


class NoCredentialRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise urllib.error.HTTPError(req.full_url, code, 'Credentialed redirects are disabled', headers, fp)


def inspect_catalog(provider, *, opener=None):
    settings = PROVIDERS[provider]
    key = os.environ.get(settings['key_env'], '').strip()
    row = {'provider': provider, **settings, 'credential_configured': bool(key),
           'inference_calls': 0, 'catalog_requests': 0}
    if not key:
        return {**row, 'status': 'missing_key'}
    client = opener or urllib.request.build_opener(NoCredentialRedirects())
    request = urllib.request.Request(settings['base_url'] + '/models',
        headers={'Authorization': 'Bearer ' + key, 'Accept': 'application/json',
                 'User-Agent': 'RAGLab-readonly-catalog/1.0'})
    row['catalog_requests'] = 1
    try:
        with client.open(request, timeout=30) as response:
            if getattr(response, 'status', 200) != 200:
                return {**row, 'status': 'http_error', 'http_status': response.status}
            raw = response.read(2_000_001)
        if len(raw) > 2_000_000:
            raise ValueError('Oversized model catalog')
        data = json.loads(raw)
        if not isinstance(data, dict) or not isinstance(data.get('data'), list):
            raise ValueError('Expected an OpenAI-style data array')
        ids = sorted({m['id'] for m in data['data']
                      if isinstance(m, dict) and isinstance(m.get('id'), str)})
        related = [m.replace(key, '[REDACTED]') for m in ids
                   if any(term in m.lower() for term in ('qwen3.8-max',))]
        return {**row, 'status': 'catalog_listed', 'advertised_model_count': len(ids),
                'listed_exact_ids': [m for m in EXACT_MODELS if m in ids],
                'absent_exact_ids': [m for m in EXACT_MODELS if m not in ids],
                'related_ids': related[:100], 'related_ids_truncated': len(related) > 100}
    except urllib.error.HTTPError as exc:
        # Do not export raw error bodies/headers: a gateway may echo a key.
        exc.close()
        return {**row, 'status': 'http_error', 'http_status': exc.code}
    except Exception as exc:
        return {**row, 'status': 'error', 'error_type': type(exc).__name__}


def collect():
    plan_path = Path(__file__).resolve().parent / 'benchmarks/provider_catalog_plan.json'
    plan = json.loads(plan_path.read_text())
    if (plan.get('mode') != 'catalog_only' or plan.get('inference') is not False or
            plan.get('providers') != list(PROVIDERS)):
        raise ValueError('This probe only supports the declared read-only provider catalogs')
    report = {'generated_at': datetime.now(timezone.utc).isoformat(), 'mode': 'catalog_only',
              'inference_calls': 0, 'documents_sent': False, 'models_substituted': False,
              'limitation': 'A catalog listing, including HTTP 200, does not prove key validity, inference availability, exact upstream identity, or quality.',
              'providers': [inspect_catalog(name) for name in PROVIDERS]}
    write_json(OUTPUT, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def publish():
    report = json.loads(OUTPUT.read_text())
    text = '```json\n' + json.dumps(report, ensure_ascii=False) + '\n```'
    if len(text.encode()) > 60000:
        raise ValueError('Catalog summary exceeds Checks budget; use the artifact')
    repository, sha = os.environ['GITHUB_REPOSITORY'], os.environ['GITHUB_SHA']
    payload = {'name': 'Additional providers / read-only catalogs', 'head_sha': sha,
               'status': 'completed', 'conclusion': 'neutral',
               'output': {'title': 'Catalogs only — no inference or quality claim',
                          'summary': report['limitation'], 'text': text}}
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'check.json'
        write_json(path, payload)
        subprocess.run(['gh', 'api', '--method', 'POST', f'repos/{repository}/check-runs',
                        '--input', str(path), '--jq', '.html_url'], check=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--publish', action='store_true')
    args = parser.parse_args()
    if args.publish:
        publish()
    else:
        try:
            from dotenv import load_dotenv
            load_dotenv(Path(__file__).resolve().parent / '.env')
        except ImportError:
            pass  # GitHub's stdlib-only catalog job supplies environment secrets.
        result = collect()
        raise SystemExit(0 if all(p['status'] == 'catalog_listed' for p in result['providers']) else 2)
