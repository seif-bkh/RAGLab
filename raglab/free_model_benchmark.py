"""Free gateway answer experiments on a frozen native-Nemotron retrieval snapshot.

No embedding or translation API calls. Provider results are reported separately
from the exact NVIDIA comparison. Selection sees development labels only.
"""
import argparse
import json
import os
import re
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import nvidia_benchmark as nb
from answer import AnswerGenerator
from artifacts import fingerprint, write_json
from chunker import tokenizer_identity
from evaluate import load_question_set
from free_gateway import FreeGatewayClient, free_eligibility, load_pricing
from nvidia_api import EMBED_MODEL, safe_error
from publish_nvidia_report import check_text, report_parts

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / 'results/free_models'
FROZEN = ROOT / 'results/frozen_native'
PLAN = ROOT / 'benchmarks/free_provider_plan.json'


def label_for(prefix, provider, model):
    return re.sub(r'[^A-Za-z0-9_.-]', '_', f'{prefix}_{provider}_{model}')


def selection_key(row):
    m = row['metrics']
    latency = m.get('uncached_api_latency_mean_s')
    return (m.get('correct_refusal_rate') or 0, m.get('answer_rubric_pass') or 0,
            m.get('validation_rate_all') or 0, -(latency if latency is not None else float('inf')))


def eligible_generation(row):
    return row['status'] == 'completed' and row['metrics']['provider_success_rate'] == 1


def markdown(report):
    lines = ['# Free gateway grounded-answer comparison', '',
             f"Status: **{report['status']}**", '',
             'Gateway request-SKU results, not independently verified upstream model identities.',
             'Frozen original-query Nemotron retrieval; no new embedding/translation calls.', '',
             '| Arm | Provider | Requested model | Cases | Rubric | Refusal | Validation |',
             '|---|---|---|---:|---:|---:|---:|']
    for row in report.get('generation', []):
        m = row['metrics']
        lines.append(f"| {row['label']} | {row['provider']} | {row['model']} | {m['n']} | "
                     f"{m['answer_rubric_pass']} | {m['correct_refusal_rate']} | {m['validation_rate_all']} |")
    lines += ['', '## Selection, exclusions, and errors', '', '```json',
              json.dumps({k: report.get(k) for k in ('selected', 'gates', 'excluded_candidates', 'errors')},
                         ensure_ascii=False, indent=2), '```', '',
              'Free eligibility is checked from live pricing, not inferred from an open-weight license or model name. '
              'KiosAPI models must be exclusive to its zero-multiplier Free group; model_price=0 alone is insufficient. '
              'Free SKUs can be promotional and rate-limited. No paid fallback is allowed.', '',
              'The held-out set has only six independent facts in three languages plus nine refusal variants. '
              'Substring rubrics and quote membership are proxies, not legal/semantic certification. '
              'The screen is a small development subset; a screen winner is not a universal best model.']
    return '\n'.join(lines) + '\n'


def run():
    previous_output = nb.OUTPUT
    try:
        return _run()
    finally:
        nb.OUTPUT = previous_output


def _run():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    nb.OUTPUT = OUTPUT  # reuse tested answer scoring/output, not NVIDIA API setup
    plan = json.loads(PLAN.read_text())
    if plan.get('development_mode', 'screen') not in {'screen', 'all'} or not isinstance(plan['max_logical_calls'], int) or not 1 <= plan['max_logical_calls'] <= 100:
        raise ValueError('Invalid development mode or logical-call budget')
    full_comparison = plan.get('development_mode', 'screen') == 'all'
    report = {'generated_at': datetime.now(timezone.utc).isoformat(), 'status': 'running',
              'protocol': 'free-gateways-v2-full-development' if full_comparison else 'free-gateways-v1-original-retrieval', 'plan': plan,
              'embedding_model': EMBED_MODEL, 'embedding_api_calls': 0, 'translation_api_calls': 0,
              'model_identity': 'gateway_reported_not_independently_verified',
              'latency_scope': 'Uncached client calls, including invalid output and failed calls; excludes local guards and cached results.',
              'generation': [], 'pricing_checks': [], 'excluded_candidates': [], 'errors': [], 'gates': {},
              'production_ready': False}
    budget = {'used': 0, 'limit': plan['max_logical_calls']}
    generators = {}

    def save():
        report['logical_calls'] = dict(budget)
        report['http_requests'] = {f'{p}/{m}': gen.client.calls for (p, m), gen in generators.items()}
        write_json(OUTPUT / 'benchmark.json', report)
        (OUTPUT / 'REPORT.md').write_text(markdown(report))

    def evaluate(provider, model, cases, retrieval, prefix):
        cfg, generator = configs[(provider, model)], generators[(provider, model)]
        generator.client.recheck()
        fresh = prefix == 'dev' and plan.get('fresh_development', False)
        row = nb.generate_arm(cfg, None, cases, retrieval, label_for(prefix, provider, model),
                              generator=generator, use_cache=not fresh)
        row['provider'] = provider
        row['fresh_development'] = fresh
        report['generation'].append(row)
        save()
        return row

    save()
    try:
        if tokenizer_identity() != 'cl100k_base':
            raise ValueError('Real cl100k_base tokenization required')
        source = json.loads((FROZEN / 'benchmark.json').read_text())
        if source['commit'] != plan['source_commit'] or source['embedding_model'] != EMBED_MODEL or source['embedding_dimension'] != 2048:
            raise ValueError('Frozen retrieval source/model/dimension mismatch')
        if plan['retrieval_profile'] != 'original' or plan['answer_profile'] != 'grounded-v1':
            raise ValueError('This protocol only supports original retrieval and grounded-v1')
        chunks = json.loads((FROZEN / 'chunks.json').read_text())
        if fingerprint(chunks) != source['corpus']['chunk_manifest_sha256']:
            raise ValueError('Frozen chunk manifest changed')
        chunk_map = {f"{c['source']}::chunk_{c['index']:04d}": c for c in chunks}
        dev = load_question_set(ROOT / 'benchmarks/retrieval_dev.json')
        holdout = load_question_set(ROOT / 'benchmarks/retrieval_holdout.json')
        for file in ('retrieval_dev.json', 'retrieval_holdout.json'):
            if source['datasets'][file] != fingerprint(json.loads((ROOT / 'benchmarks' / file).read_text())):
                raise ValueError('Frozen question fixture changed: ' + file)
        dev_run = json.loads((FROZEN / 'dev_original.json').read_text())
        held_run = json.loads((FROZEN / 'holdout_original.json').read_text())
        for cases, retrieved in ((dev, dev_run), (holdout, held_run)):
            if {c['id'] for c in cases} != {c['id'] for c in retrieved['questions']}:
                raise ValueError('Incomplete frozen retrieval cases')
            for question in retrieved['questions']:
                for hit in question['hits']:
                    chunk = chunk_map.get(hit['id'])
                    if not chunk or hit['text'] != chunk['text'] or hit['metadata']['document'] != chunk['source']:
                        raise ValueError('Frozen retrieval text/source does not match its chunk manifest')
        report['frozen_retrieval'] = {'source_run': plan['source_run'], 'source_commit': source['commit'],
                                       'profile': 'original', 'corpus': source['corpus'],
                                       'dev_sha256': fingerprint(dev_run), 'holdout_sha256': fingerprint(held_run)}
        screen = [c for c in dev if c['id'] in plan['screen_ids']]
        if len(screen) != len(plan['screen_ids']):
            raise ValueError('Invalid development screen IDs')
        configs, finalists = {}, []
        for provider, models in plan['models'].items():
            if provider in plan.get('skip_providers', {}):
                report['errors'].append({'stage': 'provider_deferred', 'provider': provider,
                                        'error': plan['skip_providers'][provider]})
                save()
                continue
            try:
                pricing = load_pricing(provider)
                write_json(OUTPUT / f'pricing_{provider}.json', pricing)
            except Exception as exc:
                report['errors'].append({'stage': 'pricing', 'provider': provider, 'error': safe_error(exc)})
                save()
                continue
            screens = []
            for model in models:
                allowed, reason, evidence = free_eligibility(provider, model, pricing['catalog'])
                report['pricing_checks'].append({'provider': provider, 'model': model, 'eligible': allowed,
                    'reason': reason, 'checked_at': pricing['checked_at'], 'url': pricing['url'], 'evidence': evidence})
                if not allowed:
                    report['excluded_candidates'].append({'provider': provider, 'model': model, 'reason': reason})
                    save()
                    continue
                try:
                    cfg = nb.make_config(ANSWER_MODEL=model, ANSWER_PROVIDER=provider, ANSWER_WORKERS=1,
                        ANSWER_PROMPT_VERSION='grounded-v1', ANSWER_NEIGHBOR_RADIUS=0,
                        ANSWER_CACHE_PATH=ROOT / 'benchmark_cache/free_gateway_answers.json')
                    client = FreeGatewayClient(provider, model, pricing, budget=budget)
                    gen = AnswerGenerator(cfg, client, approved_models=(model,))
                    configs[(provider, model)], generators[(provider, model)] = cfg, gen
                    row = evaluate(provider, model, dev if full_comparison else screen,
                                   dev_run, 'dev' if full_comparison else 'screen')
                    if eligible_generation(row):
                        screens.append(row)
                    else:
                        report['errors'].append({'stage': 'development' if full_comparison else 'screen', 'provider': provider, 'model': model,
                                                  'error': 'Provider failures or incomplete candidate evaluation'})
                    if any(q['result'].get('http_status') == 401 for q in row['questions']):
                        report['errors'].append({'stage': 'provider_auth', 'provider': provider,
                                                'error': '401: remaining models skipped; check this provider key'})
                        break
                except Exception as exc:
                    report['errors'].append({'stage': 'development' if full_comparison else 'screen', 'provider': provider, 'model': model, 'error': safe_error(exc)})
                save()
            if full_comparison:
                finalists.extend(screens)
            elif screens:
                chosen = max(screens, key=selection_key)
                try:
                    full = evaluate(provider, chosen['model'], dev, dev_run, 'dev')
                    if eligible_generation(full):
                        finalists.append(full)
                except Exception as exc:
                    report['errors'].append({'stage': 'full_development', 'provider': provider, 'error': safe_error(exc)})
                save()
        if finalists:
            winner = max(finalists, key=selection_key)
            provider, model = winner['provider'], winner['model']
            report['selected'] = {'provider': provider, 'model': model, 'prompt': 'grounded-v1', 'retrieval': 'original'}
            metrics = winner['metrics']
            report['gates']['development_answers'] = bool((metrics['answer_rubric_pass'] or 0) >= .9
                and metrics['correct_refusal_rate'] == 1 and metrics['validation_rate_all'] == 1)
            save()  # selection frozen before any held-out answer calls
            result = evaluate(provider, model, holdout, held_run, 'holdout_selected')
            m = result['metrics']
            report['gates']['holdout_answers'] = bool(eligible_generation(result)
                and (m['answer_rubric_pass'] or 0) >= .9 and m['correct_refusal_rate'] == 1
                and m['citation_validity'] == 1 and m['validation_rate_all'] == 1)
            gen = generators[(provider, model)]
            gen.client.recheck()
            report['adversarial_context'] = nb.adversarial_context_checks(configs[(provider, model)], dev_run, generator=gen)
            report['gates']['adversarial_context'] = all(r['safe'] for r in report['adversarial_context'])
            if not eligible_generation(result) or any(r['status'] == 'error' for r in report['adversarial_context']):
                report['errors'].append({'stage': 'holdout_or_security', 'error': 'Provider failure left final evaluation incomplete'})
        else:
            report['errors'].append({'stage': 'selection', 'error': 'No complete full-development candidate'})
        report['gates']['both_providers_full_development'] = all(
            any(r['provider'] == p for r in finalists) for p in plan['models'])
        for provider in plan['models']:
            if not any(r['provider'] == provider for r in finalists):
                report['errors'].append({'stage': 'provider_completeness', 'provider': provider,
                                        'error': 'No complete full-development candidate; this provider is not fully measured'})
        report['status'] = 'completed' if not report['errors'] else 'incomplete'
    except Exception as exc:
        report['status'] = 'blocked'
        report['errors'].append({'stage': 'pipeline', 'error': safe_error(exc)})
    save()
    return report


def publish():
    report = json.loads((OUTPUT / 'benchmark.json').read_text())
    summary = markdown(report)
    if os.environ.get('GITHUB_STEP_SUMMARY'):
        with open(os.environ['GITHUB_STEP_SUMMARY'], 'a') as stream:
            stream.write(summary)
    for name, data in report_parts(report):
        text = check_text(data)
        if len(text.encode()) > 60000:
            raise ValueError('Gateway check exceeds size budget: ' + name)
        payload = {'name': 'Free gateway results / ' + name, 'head_sha': os.environ['GITHUB_SHA'],
                   'status': 'completed', 'conclusion': 'neutral',
                   'output': {'title': 'Measured gateway results — ' + name,
                              'summary': summary if name == 'summary' else 'Gateway outputs; upstream identities are unverified.', 'text': text}}
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'check.json'
            write_json(path, payload)
            subprocess.run(['gh', 'api', '--method', 'POST', f"repos/{os.environ['GITHUB_REPOSITORY']}/check-runs",
                            '--input', str(path), '--jq', '.html_url'], check=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--publish', action='store_true')
    args = parser.parse_args()
    if args.publish:
        publish()
    else:
        report = run()
        raise SystemExit(0 if report['status'] == 'completed' else 2)
