"""Coverage-aware reports: missing/provider/reference cases are never invented scores."""
import random
import statistics
from collections import Counter, defaultdict

from hard_harness.common import OUTPUT, now, read_json, read_jsonl
from artifacts import write_json


def clustered_interval(rows, replicates=500, seed=20260905):
    groups = defaultdict(list)
    for row in rows:
        if row.get('correct') is not None:
            groups[row['family_id']].append(int(row['correct']))
    keys = sorted(groups)
    if len(keys) < 2:
        return None
    rng = random.Random(seed)
    values = []
    for _ in range(replicates):
        sample = [groups[rng.choice(keys)] for __ in keys]
        values.append(sum(sum(g) for g in sample)/sum(len(g) for g in sample))
    values.sort()
    return {'low':values[int(.025*replicates)],'high':values[min(replicates-1,int(.975*replicates))],
            'unit':'paired scenario family','groups':len(keys),'replicates':replicates,
            'limitation':'Shared facts/attack templates add further dependence; this is not a production guarantee.'}


def build_report():
    root = OUTPUT/'grading'
    manifest = read_json(root/'manifest.json')
    rows = read_jsonl(root/'judgments.jsonl') if (root/'judgments.jsonl').exists() else []
    manifest['scenario_clustered_interval'] = clustered_interval(rows)
    manifest['by_category'] = {}
    for category in sorted({r['category'] for r in rows}):
        group = [r for r in rows if r['category']==category]
        scored = [r for r in group if r.get('correct') is not None]
        manifest['by_category'][category] = {'observed':len(group),'scored':len(scored),
            'correct':sum(r['correct'] for r in scored),'grades':dict(Counter(r['grade'] for r in group))}
    manifest['paired_language_disagreements'] = []
    families = defaultdict(dict)
    for row in rows:
        families[row['family_id']][row['language']] = row.get('correct')
    for family, values in sorted(families.items()):
        if len(values)==3 and len(set(values.values()))>1:
            manifest['paired_language_disagreements'].append({'family_id':family,'outcomes':values})
    lines = ['# Hard multilingual answer-agent harness', '',
             f"Status: **{manifest['status']}**; generated {now()}.", '',
             '**Target: 1,000 paired scenarios × Arabic/French/English = 3,000 question records.**',
             'Questions and anticipated answers are separate frozen files. The answering process receives no answer-key artifact.', '',
             '| Language | Target | Observed judgments | Scored | Correct | Correct / scored |',
             '|---|---:|---:|---:|---:|---:|']
    for lang in ('ar','fr','en'):
        row = manifest.get('by_language',{}).get(lang,{})
        score = row.get('score')
        lines.append(f"| {lang} | 1000 | {row.get('observed',0)} | {row.get('scored',0)} | {row.get('correct',0)} | {score if score is not None else 'unscored'} |")
    lines += ['', '## Interpretation', '',
              '- Missing predictions and quota/provider failures are not successful answers and are not fabricated zeros.',
              '- A reference_issue is a possible oracle defect, not automatically a model failure. It needs review.',
              '- Local private/live-data guards are counted separately from genuine model abstentions.',
              '- Semantic grading accepts faithful paraphrases; exact string identity is not the criterion.',
              '- Source membership is not entailment. Original PDF references are separate from the noisy runtime extraction.',
              '- Model-authored/audited keys and same-family judges are not independent expert validation.',
              '- Every provider/model/credential alias is recorded. Mixed-provider coverage is not reported as a Qwen-only result.',
              '- Correlated language/fact/attack families mean 3,000 records are not 3,000 independent facts.', '',
              '## Provider/model coverage', '', '```json',
              __import__('json').dumps(manifest.get('by_provider_model',{}),ensure_ascii=False,indent=2), '```', '',
              '## Category coverage', '', '```json',
              __import__('json').dumps(manifest['by_category'],ensure_ascii=False,indent=2), '```', '',
              'Full per-case outcomes: `judgments.jsonl`. Expected answers: the separate `hard-harness-answer-keys` artifact. '
              'Candidate outputs: `hard-harness-predictions-*`. Source and reference audit trails are retained.', '',
              '**No production certification is implied by this report.**']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    write_json(root/'manifest.json',manifest)
    return manifest
