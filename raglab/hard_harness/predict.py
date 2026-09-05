"""Candidate execution. This module never opens answer-key/reference files."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import config
from answer import AnswerGenerator, needs_private_or_live_data
from artifacts import fingerprint, write_json
from chunker import chunk_all
from embedder import build_embedder
from evaluate import prepare_query_text
from loader import load_all
from retrieval import retrieve
from store import get_collection, store_chunks
from hard_harness.common import (ROOT, WORK, OUTPUT, PLAN_PATH, LANGUAGES, CheckpointClient,
                                 now, read_json, read_jsonl, write_jsonl)


def load_public_questions(directory):
    directory = Path(directory)
    manifest = read_json(directory/'manifest.json')
    rows = []
    for lang in LANGUAGES:
        name = f'questions.{lang}.jsonl'
        part = read_jsonl(directory/name)
        if len(part) != 1000 or fingerprint(part) != manifest['public_files'][name]['fingerprint']:
            raise ValueError('Public question manifest mismatch: ' + name)
        for row in part:
            if set(row) - {'id','family_id','language','question','context_injections'}:
                raise ValueError('Non-public answer/label fields found in candidate inputs')
            if row['language'] != lang:
                raise ValueError('Language mismatch in public questions')
        rows.extend(part)
    if len({r['id'] for r in rows}) != 3000:
        raise ValueError('Candidate input must contain 3000 unique IDs')
    return manifest, rows


def runtime_config():
    cfg = SimpleNamespace(**{k:getattr(config,k) for k in dir(config) if k.isupper()})
    cfg.NVIDIA_EMBEDDING_CACHE_PATH = WORK/'runtime_embeddings_cache.json'
    cfg.EMBEDDING_CACHE_PATH = cfg.NVIDIA_EMBEDDING_CACHE_PATH
    cfg.CHROMA_DIR = WORK/'runtime_chroma'
    cfg.CHROMA_COLLECTION_NAME = 'hard_harness_runtime'
    cfg.ANSWER_CACHE_PATH = WORK/'unused_candidate_success_cache.json'
    cfg.QUERY_TRANSLATION_ENABLED = False
    cfg.QUERY_VARIANT_STRATEGY = 'original'
    policy = read_json(PLAN_PATH)['answer_policy']
    if policy.get('translation') is not False:
        raise ValueError('Hard-harness candidate profile cannot enable a translation model')
    cfg.ANSWER_TOP_K = policy['top_k']
    cfg.ANSWER_CONTEXT_TOKENS = policy['context_tokens']
    cfg.ANSWER_MAX_TOKENS = policy['max_tokens']
    cfg.ANSWER_PROMPT_VERSION = policy['prompt']
    return cfg


def prepare_retrieval():
    from dataclasses import asdict
    plan = read_json(PLAN_PATH)
    public_dir = OUTPUT/'dataset/public'
    manifest, questions = load_public_questions(public_dir)
    cfg = runtime_config()
    out = OUTPUT/'retrieval'
    out.mkdir(parents=True, exist_ok=True)
    report = {'status':'running','created_at':now(),'questions':len(questions),
              'public_files':manifest['public_files'],'embedding_model':cfg.NVIDIA_EMBEDDING_MODEL,
              'dimensions':2048,'reference_files_loaded':False,'completed':0}
    try:
        docs = load_all(ROOT.parent/'docs')
        chunks = chunk_all(docs,cfg)
        if fingerprint([asdict(c) for c in chunks]) != plan['source_runtime_manifest']:
            raise ValueError('Candidate corpus changed after reference design; version the dataset instead of mixing corpora')
        embedder = build_embedder(cfg)
        vectors = embedder.embed_texts([c.text for c in chunks], input_type='search_document')
        collection = get_collection(cfg, reset=True)
        if store_chunks(collection,list(zip(chunks,vectors)),cfg) != len(chunks):
            raise ValueError('Incomplete candidate corpus index')
        # Batch new queries. Private/live guards must not trigger model/index use.
        queries = [prepare_query_text(q['question']) for q in questions if not needs_private_or_live_data(q['question'])]
        embedder.embed_texts(queries,input_type='search_query')
        by_id = {}
        for question in questions:
            if needs_private_or_live_data(question['question']):
                hits, variants, reason = [], [], 'local_private_or_live_guard'
            else:
                hits, variants = retrieve(cfg, embedder, collection, prepare_query_text(question['question']),
                    language=question['language'], translator=None, top_k=cfg.ANSWER_TOP_K, variant_strategy='original')
                reason = None
            by_id[question['id']] = {**question, 'hits':hits, 'query_variants':variants,'retrieval_skipped':reason}
        for shard in range(plan['answer_shards']):
            ids = [r['id'] for r in read_jsonl(public_dir/'shards'/f'{shard:02d}.jsonl')]
            write_jsonl(out/f'{shard:02d}.jsonl',[by_id[i] for i in ids])
        report.update(status='retrieval_complete',completed=len(by_id),embedding_api_calls=embedder.api_calls,
                      runtime_manifest=plan['source_runtime_manifest'])
    except BaseException as exc:
        if isinstance(exc, KeyboardInterrupt):
            raise
        from nvidia_api import safe_error
        report.update(status='paused',error=safe_error(exc),
                      action='Embedding cache is checkpointed. Resume with the same native model; do not substitute another embedding space.')
    write_json(out/'manifest.json',report)
    return report


def terminal_result(result):
    # A completed but invalid model output is a measured failure, not a retry
    # opportunity to cherry-pick a later successful answer.
    return result.get('provider_ok') is True or result.get('http_status') == 422


def case_identity(row, answer_policy):
    return fingerprint({'public_input':{k:row.get(k) for k in ('id','question','language','context_injections')},
                        'retrieved_context':row['hits'], 'answer_profile':answer_policy})


def predict_shard(shard):
    plan = read_json(PLAN_PATH)
    metadata = read_json(OUTPUT/'retrieval/manifest.json')
    if metadata['status'] != 'retrieval_complete':
        raise ValueError('Candidate retrieval is not complete')
    rows = read_jsonl(OUTPUT/'retrieval'/f'{shard:02d}.jsonl')
    if len(rows) != 100 or len({r['id'] for r in rows}) != 100:
        raise ValueError('Prediction shard must contain 100 unique public inputs')
    out = OUTPUT/f'predictions_{shard:02d}'
    out.mkdir(parents=True,exist_ok=True)
    checkpoint = WORK/'prediction_shards'/fingerprint(metadata['public_files'])[:16]/f'{shard:02d}.jsonl'
    prior = read_jsonl(checkpoint) if checkpoint.exists() else []
    by_id = {r['id']:r for r in prior if r.get('terminal')}
    if set(by_id)-{r['id'] for r in rows}:
        raise ValueError('Checkpoint contains IDs outside this shard')
    for row in rows:
        if row['id'] in by_id and by_id[row['id']]['case_hash'] != case_identity(row,plan['answer_policy']):
            raise ValueError('Changed input/context cannot reuse an old prediction')
    if len(by_id)==len(rows):
        report={'status':'predictions_complete','shard':shard,'target':100,'completed':100,
                'reference_files_loaded':False,'dataset_public_files':metadata['public_files'],
                'completed_checkpoint_reused':True,'new_model_calls':0,
                'provider_models':sorted({r['provider']+'/'+r['model'] for r in by_id.values()})}
        write_jsonl(out/'predictions.jsonl',[by_id[r['id']] for r in rows])
        write_jsonl(out/'attempts.jsonl',[])
        write_json(out/'manifest.json',report)
        return report
    client = CheckpointClient('candidate',call_limit=120)
    cfg = runtime_config()
    cfg.ANSWER_MODEL = client.model
    cfg.ANSWER_PROVIDER = client.provider
    generator = AnswerGenerator(cfg,client,approved_models=(client.model,))
    report = {'status':'running','shard':shard,'target':100,'completed':len(by_id),
              'reference_files_loaded':False,'dataset_public_files':metadata['public_files'],
              'model':client.model,'provider':client.provider,'credential_alias':client.credential_alias,
              'fresh_success_only_cache_used':False}
    attempts = []
    try:
        for row in rows:
            case_hash = case_identity(row,plan['answer_policy'])
            if row['id'] in by_id:
                if by_id[row['id']]['case_hash'] != case_hash:
                    raise ValueError('Changed input/context cannot reuse an old prediction')
                continue
            hits = copy.deepcopy(row['hits'])
            additions = row.get('context_injections',[])
            # Deliberately untrusted input fixture; not part of the oracle/key.
            if additions:
                hits = copy.deepcopy(additions) + hits
            cached_before = client.cached_calls
            client.last_provenance = None
            result = generator.answer(row['question'],hits,row['language'],use_cache=False)
            prediction = {'id':row['id'],'family_id':row['family_id'],'language':row['language'],
                          'question':row['question'],'case_hash':case_hash,'result':result,
                          'provider':client.provider,'model':client.model,
                          'credential_alias':(client.last_provenance or {}).get('credential_alias', client.credential_alias),
                          'inference_provenance':client.last_provenance,
                          'model_inference':client.last_provenance is not None,
                          'response_checkpoint_replayed':client.cached_calls>cached_before,
                          'retrieval_skipped':row['retrieval_skipped'],
                          'retrieved_ids':[h['id'] for h in row['hits']],
                          'context_injections':additions,'terminal':terminal_result(result)}
            attempts.append(prediction)
            if prediction['terminal']:
                by_id[row['id']] = prediction
            write_jsonl(checkpoint,list(by_id.values()))
            write_jsonl(out/'predictions.jsonl',list(by_id.values()))
            write_jsonl(out/'attempts.jsonl',attempts)
            report['completed'] = len(by_id)
            write_json(out/'manifest.json',report)
            client.check_pause()
            print(f'[predict] shard {shard} {row["id"]}: {result["status"]}/{result["reason"]} {len(by_id)}/100',flush=True)
        report['status'] = 'predictions_complete' if len(by_id)==100 else 'incomplete'
    except Exception as exc:
        from nvidia_api import safe_error
        report.update(status='paused' if client.pause else 'blocked',error=safe_error(exc))
    report['client'] = client.summary()
    report['completed'] = len(by_id)
    write_jsonl(out/'predictions.jsonl',list(by_id.values()))
    write_jsonl(out/'attempts.jsonl',attempts)
    write_json(out/'manifest.json',report)
    return report
