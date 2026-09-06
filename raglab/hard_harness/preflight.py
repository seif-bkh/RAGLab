"""A zero-spend check that this machine is wired the way CI is before quota is touched.

Why this exists: every phase here has failed at least once for an environment reason rather than a
measurement reason - a missing .env, a fallback token estimator silently changing chunk sizes so
local numbers cannot be compared with CI, a credential alias that CI injects but a laptop does not
have, a cache directory that no longer exists because it is git-ignored. Each of those wastes a
provider's daily allowance to discover itself. This prints what is true first, and it makes no
completion call: the only network traffic is a provider's own model-listing endpoint, which costs
nothing and proves a key resolves.
"""
import json
import os
import urllib.error
import urllib.request

from hard_harness.common import OUTPUT, PLAN_PATH, read_json

# Free listing endpoints, per credential the harness may read. Nothing here is a completion.
LIST_ENDPOINTS = {
    'NVIDIA_API_KEY': ('https://integrate.api.nvidia.com/v1/models', 'Authorization'),
    'XKIRO_API_KEY': ('https://api.xkiro.com/v1/models', 'Authorization'),
    'XKIRO_API_KEY_JINKO': ('https://api.xkiro.com/v1/models', 'Authorization'),
    'EXPERIENTIAL_API_KEY': (None, 'Authorization'),          # base URL comes from the plan/env
    'GOOGLE_API_KEY': ('https://generativelanguage.googleapis.com/v1beta/models', 'x-goog-api-key'),
    'GEMINI_API_KEY': ('https://generativelanguage.googleapis.com/v1beta/models', 'x-goog-api-key'),
}
# Which profile a phase needs a key for, so 'doctor' can say what is blocking *this* step.
PHASE_ROLES = {'sources': ['llm'], 'author': ['author_llm'], 'compile': ['llm'],
               'retrieve': [], 'evaluate': ['candidate_llm'], 'grade': ['grader_llm']}
ROLE_LABEL = {'llm': 'reference/audit', 'author_llm': 'author', 'candidate_llm': 'answerer',
              'grader_llm': 'judge'}


def read_env_file(path):
    """Parse raglab/.env by hand. Not a second loader - a witness. The phases get their values from
    config.py's load_dotenv, and a value that exists in the file yet never reaches the environment is
    the failure mode that costs a whole session to spot (no dotenv installed, a name typed in lower
    case, an empty assignment). So doctor reads the file itself and compares, instead of calling every
    absent key 'missing'.
    """
    from pathlib import Path
    values = {}
    path = Path(path)
    if not path.exists():
        return values
    for line in path.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        name, _, value = line.partition('=')
        name = name.strip()[len('export '):].strip() if name.strip().startswith('export ') else name.strip()
        value = value.strip()
        if len(value) > 1 and value[0] == value[-1] and value[0] in ('"', "'"):
            value = value[1:-1]
        else:                                            # a key may contain '#', so only ' #' ends it
            value = value.split(' #', 1)[0].strip()
        if name:
            values[name] = value
    return values


def load_project_env(env_path=None, apply_load=True):
    """Run the load every phase runs, and report what it did. Importing config is what reads .env; the
    report says so, rather than trusting that some other module happened to import it first."""
    import importlib.util
    from pathlib import Path
    env_path = Path(env_path) if env_path else Path(__file__).resolve().parents[1] / '.env'
    state = {'path': str(env_path), 'exists': env_path.exists(), 'dotenv_installed':
             importlib.util.find_spec('dotenv') is not None, 'loaded': False, 'added': [], 'error': None,
             'assignments': read_env_file(env_path)}
    if not apply_load:
        state['loaded'] = 'not attempted'
        return state
    before = set(os.environ)
    try:
        import config                                                      # noqa: F401
    except Exception as exc:                                               # noqa: BLE001
        state['error'] = f'{type(exc).__name__}: {exc}'
        return state
    state['loaded'] = True
    state['added'] = sorted(name for name in os.environ if name not in before
                            and name.endswith(('_API_KEY', '_KEY', '_ALIAS')))
    return state


def masked(value):
    """The gateway's own rule: never echo more than the first eight characters of a key."""
    text = str(value or '')
    return text[:8] + '…' if len(text) > 8 else ('set' if text else '')


def _listing(base_url, key, header, timeout=10.0):
    request = urllib.request.Request(base_url, headers={header: ('Bearer ' + key) if header
                                                         == 'Authorization' else key,
                                                         'User-Agent': 'raglab-doctor/1'}, method='GET')
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:      # noqa: S310
            body = response.read().decode('utf-8', errors='replace')
    except urllib.error.HTTPError as exc:
        return 'rejected', f'HTTP {exc.code}'
    except Exception as exc:                                                     # noqa: BLE001
        return 'unreachable', type(exc).__name__
    try:
        data = json.loads(body)
    except ValueError:
        return 'answered', 'non-JSON listing'
    rows = data.get('data') or data.get('models') or []
    slugs = sorted({str(row.get('id') or row.get('name') or '') for row in rows if isinstance(row, dict)})
    slugs = [slug for slug in slugs if slug]
    return 'ok', (f'{len(slugs)} models listed' if slugs else 'no model rows'), slugs


def credential_status(alias, plan, probe=True, env_values=None):
    key = os.environ.get(alias, '').strip()
    if alias == 'EXPERIENTIAL_API_KEY':
        from hard_harness.experiential_client import gateway_base_url
        base_url = gateway_base_url((plan.get('grader_llm') or {}))
    else:
        base_url = LIST_ENDPOINTS[alias][0]
    header = LIST_ENDPOINTS[alias][1]
    row = {'present': bool(key), 'masked': masked(key), 'source': 'environment',
           'in_env_file': alias in (env_values or {})}
    if not row['present'] and row['in_env_file']:
        row['note'] = ('named in raglab/.env but empty there' if not (env_values or {}).get(alias)
                       else 'named in raglab/.env but never loaded - python-dotenv missing?')
    elif not row['present']:
        near = [name for name in (env_values or {}) if name.lower() == alias.lower()
                or alias.lower().rstrip('_KEY') in name.lower()]
        if near:
            row['note'] = f'raglab/.env has {", ".join(sorted(near))}, which is not spelled {alias}'
    # A copied .env.example still looks 'present' to a presence check, so the template's own
    # placeholders are reported as unfilled instead of being sent to a provider to fail with a 401.
    if key and any(mark in key.lower() for mark in ('paste-', 'your-key', 'placeholder', 'xxxx')):
        row.update({'present': False, 'note': 'still the placeholder from .env.example', 'probe': 'skipped'})
        return row
    if not os.environ.get('dotenv_loaded') and os.environ.get('HARNESS_API_KEY', '').strip() \
            and os.environ.get('HARNESS_CREDENTIAL_ALIAS', '').strip() == alias:
        row['source'] = 'HARNESS_API_KEY (CI-style injection)'
        key = os.environ['HARNESS_API_KEY'].strip()
    if not key or not base_url:
        row['probe'] = 'skipped' if not key else 'no base url'
        return row
    if not probe:
        row['probe'] = 'not attempted'
        return row
    outcome = _listing(base_url, key, header)
    row['probe'], row['detail'] = outcome[0], outcome[1]
    if outcome[0] == 'ok' and len(outcome) > 2:
        row['models'] = outcome[2][:12]
    return row


def tokenizer_status():
    """The quiet one: a fallback estimator makes local chunk sizes incomparable with CI."""
    try:
        import tiktoken
        tiktoken.get_encoding('cl100k_base')
        return {'identity': 'cl100k_base', 'comparable_with_ci': True}
    except Exception as exc:                                                     # noqa: BLE001
        return {'identity': f'estimator fallback ({type(exc).__name__})', 'comparable_with_ci': False,
                'fix': 'set TIKTOKEN_CACHE_DIR to a warmed cache, or allow one download of cl100k_base'}


def corpus_status(root=None):
    from pathlib import Path
    base = Path(root).resolve() if root else Path(__file__).resolve().parents[2]
    docs = sorted(path for path in base.joinpath('docs').glob('*') if path.is_file())
    return {'documents': len(docs), 'names': [doc.name for doc in docs]}


def state_status():
    """What is already on this machine, since results/ is git-ignored and starts empty."""
    from pathlib import Path
    out = Path(OUTPUT)
    found = {}
    for name, relative in (('sources', 'sources/manifest.json'), ('dataset', 'dataset/manifest.json'),
                           ('retrieval', 'retrieval/manifest.json'), ('grading', 'grading/manifest.json')):
        path = out / relative
        found[name] = path.exists()
    found['author_shards'] = sum(1 for shard in range(9) if (out / f'author_{shard:02d}' / 'families.jsonl').exists())
    found['prediction_shards'] = sum(1 for shard in range(20)
                                     if (out / f'predictions_{shard:02d}' / 'predictions.jsonl').exists())
    return found


def local_report(plan_path=None, root=None, probe=True, env_path=None, load_env=True):
    from pathlib import Path
    plan_path = Path(plan_path or PLAN_PATH)
    env = load_project_env(env_path, apply_load=load_env)
    plan = json.loads(plan_path.read_text())
    phase = plan.get('phase')
    profiles = {name: plan.get(name) or plan.get('llm') or {} for name in
                ('llm', 'author_llm', 'candidate_llm', 'grader_llm')}
    aliases = sorted({profile.get('credential_secret') for profile in profiles.values() if profile}
                     | set(LIST_ENDPOINTS))
    keys = {alias: credential_status(alias, plan, probe=probe, env_values=env['assignments'])
            for alias in aliases if alias}
    needed = {(profiles.get(role) or {}).get('credential_secret') for role in PHASE_ROLES.get(phase, [])}
    missing = sorted(alias for alias in needed if alias and not keys.get(alias, {}).get('present'))
    report = {
        'phase': phase, 'revision': plan.get('revision'),
        'profiles': {role: {'provider': profile.get('provider'), 'model': profile.get('model'),
                            'alias': profile.get('credential_secret'),
                            'label': ROLE_LABEL.get(role, role)}
                     for role, profile in profiles.items()},
        'chunking': plan.get('chunking') or {}, 'dataset_limit_per_language': plan.get('dataset_limit_per_language'),
        'keys': keys, 'blocking': missing, 'env': env,
        'tokenizer': tokenizer_status(), 'corpus': corpus_status(root),
        'local_state': state_status(),
    }
    report['summary'] = ('ready' if not missing and report['tokenizer']['comparable_with_ci']
                         else 'blocked' if missing else 'comparable-with-ci-no')
    return report


def format_report(report):
    lines = ['# Local hard-harness readiness', '',
             f"- Plan: `{report['revision']}` at phase **{report['phase']}**",
             f"- Corpus pin: {report['chunking'].get('chunk_size_tokens', 'unset')} token chunks, "
             f"overlap {report['chunking'].get('chunk_overlap_tokens', '?')}; "
             f"{report['dataset_limit_per_language'] or 'all'} families per language",
             f"- Documents in docs/: {report['corpus']['documents']} ({', '.join(report['corpus']['names'])})",
             f"- Tokenizer: {report['tokenizer']['identity']} "
             f"(comparable with CI: {report['tokenizer']['comparable_with_ci']})",
             '', '## Credentials (checked by listing models, which spends nothing)', '']
    env = report['env']
    if not env['exists']:
        lines.append(f"- `raglab/.env`: **not found** at {env['path']} - copy .env.example to it. "
                     "Values set in your shell count too, and win.")
    else:
        loaded = 'read by config.py' if env['loaded'] is True else (
            'not read: ' + (env['error'] or 'unknown'))
        lines.append(f"- `raglab/.env`: {len(env['assignments'])} assignment(s) "
                     f"({', '.join(sorted(env['assignments'])) or 'none'}); python-dotenv "
                     f"{'installed' if env['dotenv_installed'] else 'NOT installed -> the file is ignored'}; "
                     f"{loaded}"
                     + (f"; it put {', '.join(env['added'])} in the environment" if env['added'] else ''))
    for alias, row in sorted(report['keys'].items()):
        state = 'missing' if not row['present'] else f"{row['probe']}: {row.get('detail', '')}".strip(': ')
        if row.get('note'):
            state += f" - {row['note']}"
        lines.append(f"- **{alias}** — {state}"
                     + (f" ({row['masked']})" if row['present'] else '')
                     + (f"; source {row['source']}" if row.get('source') != 'environment' else ''))
    lines += ['', '## Profiles this phase uses', '']
    for role, profile in report['profiles'].items():
        lines.append(f"- {profile['label']} (`{role}`): {profile['provider']}/{profile['model']} "
                     f"via `{profile['alias']}`")
    state = report['local_state']
    lines += ['', '## What is already on this machine', '',
              f"- sources {state['sources']} · dataset {state['dataset']} · retrieval {state['retrieval']} "
              f"· grading {state['grading']} · author shards {state['author_shards']}/9 · prediction shards "
              f"{state['prediction_shards']}",
              "- `results/hard_harness/` is git-ignored, so a fresh clone starts empty. Pull the published "
              "checkpoints instead of re-running paid phases: "
              "`python hard_harness_main.py collect --sha <run-head-sha> --destination results/hard_harness`",
              '']
    if report['blocking']:
        lines += [f"- **Blocked:** add {', '.join(report['blocking'])} to `raglab/.env`."
                  + ('' if env['dotenv_installed'] else
                     ' python-dotenv is missing, so the file itself is ignored: `pip install python-dotenv`'
                     ' or re-run `./local.sh setup`.'), '']
        shadowed = [alias for alias in report['blocking']
                    if env['assignments'].get(alias) and env['dotenv_installed']]
        if shadowed:
            lines += [f"- Note: values already exported in your shell win over `raglab/.env`, so a stale "
                      f"export can also explain {', '.join(shadowed)} being unread.", '']
    elif report['summary'] == 'ready':
        lines += ['- **Ready.** Nothing above costs anything yet. To reproduce the published score: '
                  '`./local.sh restore && ./local.sh report` - that reads CI\'s checkpoints and makes no '
                  'provider request at all.', '']
    if not report['tokenizer']['comparable_with_ci']:
        lines += [f"- **Warning:** {report['tokenizer']['fix']}. Without it, chunk boundaries differ "
                  "from CI and every recall number means something else.", '']
    return '\n'.join(lines)
