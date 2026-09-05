"""Pre-freeze reference repairs. Never uses candidate answers or changes a scored key."""
import copy

from artifacts import fingerprint, write_json
from answer import normalized_quote
from hard_harness.common import OUTPUT, WORK, CheckpointClient, read_json, write_jsonl
from hard_harness.authoring import make_specs, author_messages, audit_messages, validate_family


def duplicate_ids(families):
    seen = {lang:{} for lang in ('ar','fr','en')}
    conflicts = {}
    for family in sorted(families,key=lambda x:x['id']):
        for lang in seen:
            text = normalized_quote(family['languages'][lang]['question'])
            if text in seen[lang]:
                conflicts.setdefault(family['id'],set()).add(seen[lang][text])
            else:
                seen[lang][text]=family['id']
    return {k:sorted(v) for k,v in conflicts.items()}


def repair_before_freeze(families, units, problems, *, purpose):
    if (OUTPUT/'dataset/manifest.json').exists() and read_json(OUTPUT/'dataset/manifest.json').get('status') == 'frozen':
        raise ValueError('Frozen reference keys cannot be repaired in place')
    if not problems:
        return families, []
    specs,_ = make_specs()
    spec_map = {s['id']:s for s in specs}
    by_id = {f['id']:f for f in families}
    author = CheckpointClient('pre_freeze_reference_repair',call_limit=200)
    reviewer = CheckpointClient('pre_freeze_repair_audit',call_limit=200)
    records = []
    for identifier, reason in sorted(problems.items()):
        spec = spec_map[identifier]
        original = by_id[identifier]
        peers = [f for f in by_id.values() if f['id']!=identifier and
                 (set(f.get('source_unit_ids',[])) & set(spec['source_unit_ids']))]
        peer_questions = [{'id':p['id'],'questions':{l:r['question'] for l,r in p['languages'].items()}} for p in peers]
        cache_key = fingerprint({'version':'pre-freeze-repair-v1','purpose':purpose,'original':original,
                                 'spec':spec,'reason':reason,'peers':peer_questions,
                                 'source_texts':{uid:units[uid]['text'] for uid in spec['source_unit_ids']}})
        cache_path = WORK/'reference_repairs'/f'{cache_key}.json'
        if cache_path.exists():
            repaired = read_json(cache_path)
        else:
            mapping = {f'U{i+1}':uid for i,uid in enumerate(sorted(spec['source_unit_ids']))}
            previous = copy.deepcopy(original)
            feedback = ('PRE-FREEZE REFERENCE REPAIR ONLY. '+reason+'\n'
                        'Produce a meaningfully distinct, source-grounded scenario, not a punctuation/name-only variation. '
                        'Do not require facts the question does not request. Keep the assigned category and ID. '
                        'Previously accepted peer questions: '+str(peer_questions))
            repaired = None
            for attempt in range(3):
                data, provenance = author.object(author_messages([spec],units,feedback,previous),max_tokens=6000)
                values = data.get('families',[])
                try:
                    if len(values)!=1 or values[0].get('id')!=identifier:
                        raise ValueError('Repair must return exactly the assigned ID')
                    previous = values[0]
                    value = copy.deepcopy(previous)
                    for ev in value.get('evidence',[]):
                        ev['unit_id'] = mapping.get(ev.get('unit_id'),ev.get('unit_id'))
                    value = validate_family(value,spec,units)
                    for lang in ('ar','fr','en'):
                        if any(normalized_quote(value['languages'][lang]['question'])==normalized_quote(f['languages'][lang]['question'])
                               for other,f in by_id.items() if other!=identifier):
                            raise ValueError('Repaired question still duplicates another family')
                    messages = audit_messages([value],[spec],units)
                    messages[0]['content'] += ' Also reject mere surface rewordings of the peer questions supplied in the repair feedback when a distinct scenario is required.'
                    messages[1]['content'] += '\nRepair reason and peer-question context: '+feedback[:16000]
                    audit,audit_provenance = reviewer.object(messages,max_tokens=3500)
                    decisions = audit.get('reviews',[])
                    if len(decisions)!=1 or decisions[0].get('id')!=identifier or decisions[0].get('approved') is not True or decisions[0].get('issues'):
                        raise ValueError('Repair audit rejected: '+str(decisions))
                    repaired = {'family':{**value,'author_provenance':provenance,'audit_provenance':audit_provenance},
                                'original_fingerprint':fingerprint(original),'reason':reason,'purpose':purpose,
                                'audit':decisions[0]}
                    write_json(cache_path,repaired)
                    break
                except Exception as exc:
                    author.check_pause();reviewer.check_pause()
                    feedback += f'\nRepair attempt {attempt+1} failed: {str(exc)[:1600]}'
            if repaired is None:
                raise ValueError('Reference still needs review before freeze: '+identifier)
        by_id[identifier] = repaired['family']
        records.append(repaired)
        write_jsonl(OUTPUT/'dataset'/f'{purpose}_repairs.jsonl',records)
    return [by_id[f['id']] for f in families], records
