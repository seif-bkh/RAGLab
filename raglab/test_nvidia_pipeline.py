"""Offline contract, provenance, safety, and regression tests; no API access."""
import inspect

import chunker
import chat
import semantic_chunking as sc
import io
import json
import os
import re
import tempfile
import unittest
import urllib.error
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from answer import (AnswerGenerator, answer_messages, build_sources,
                    needs_private_or_live_data, validate_answer)
from artifacts import write_json
from embedder import build_embedder, embedding_fingerprint
from evaluate import is_correct_hit, save_run, run_evaluation
from retrieval import retrieve
from nvidia_api import (DEEPSEEK_MODEL, EMBED_MODEL, KIMI_MODEL, RIVA_MODEL,
                        NvidiaAPIError, NvidiaClient, chat_payload,
                        final_content, read_event_stream, retry_after_seconds)
from nvidia_benchmark import make_config, selection_key

DOCS_DIRS = [Path(__file__).resolve().parent.parent / "docs",
             Path(__file__).resolve().parent / "data"]
from pipeline_policy import ANSWER_MODEL as QWEN_MODEL
from store import ensure_fresh_chunks, get_collection, store_chunks, chunk_fp
from translate import QueryTranslator, translation_issues


class Response:
    def __init__(self, data):
        self.data = data
        self.status = 200
    def __enter__(self):
        return self
    def __exit__(self, *args):
        return False
    def read(self):
        return json.dumps(self.data).encode()


class Contracts(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        self.cfg = make_config(NVIDIA_MIN_INTERVAL=0, NVIDIA_CHAT_STREAM=False,
                              QUERY_TRANSLATION_PROMPT='basic-v1',
                              NVIDIA_EMBEDDING_CACHE_PATH=self.path / 'embeddings.json',
                              EMBEDDING_CACHE_PATH=self.path / 'embeddings.json',
                              QUERY_TRANSLATION_CACHE_PATH=self.path / 'translations.json',
                              ANSWER_CACHE_PATH=self.path / 'answers.json',
                              CHROMA_DIR=self.path / 'chroma')
        self.env = patch.dict(os.environ, {'NVIDIA_API_KEY': 'test-key'})
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_explicit_empty_key_does_not_use_environment_credential(self):
        client = NvidiaClient(api_key='')
        with patch('urllib.request.urlopen') as request:
            with self.assertRaises(NvidiaAPIError):
                client.request('models')
            request.assert_not_called()

    def test_multiple_translation_cache_instances_merge_only_new_entries(self):
        from translate import TranslationCache
        path = self.path / 'shared-translations.json'
        a, b = TranslationCache(path), TranslationCache(path)
        a.put('model-a', 'ar', 'one', 'القديم')
        a.save()
        b.put('model-b', 'ar', 'two', 'الثاني')
        b.put('model-a', 'ar', 'one', 'الجديد')
        b.save()
        a.put('model-a', 'ar', 'three', 'الثالث')
        a.save()  # must not restore a stale copy of "one" or drop "two"
        warm = TranslationCache(path)
        self.assertEqual(len(warm.entries), 3)
        self.assertEqual(warm.get('model-a', 'ar', 'one')['translation'], 'الجديد')
        self.assertEqual(a.get('model-b', 'ar', 'two')['translation'], 'الثاني')
        self.assertEqual(b.get('model-a', 'ar', 'three')['translation'], 'الثالث')
        b.save()  # no dirty entries: must not overwrite another instance
        self.assertEqual(len(TranslationCache(path).entries), 3)

    def test_native_dimensions_only(self):
        self.cfg.NVIDIA_EMBEDDING_DIM = 512
        with self.assertRaisesRegex(ValueError, '2048'):
            build_embedder(self.cfg)

    def test_embedding_dimension_identity(self):
        self.cfg.NVIDIA_EMBEDDING_DIM = 0
        a = embedding_fingerprint(self.cfg)
        self.cfg.NVIDIA_EMBEDDING_DIM = 2048
        self.assertEqual(a, embedding_fingerprint(self.cfg))
        self.cfg.NVIDIA_EMBEDDING_MODEL = 'another-2048-dim-model'
        self.assertNotEqual(a, embedding_fingerprint(self.cfg))

    def test_endpoint_identity(self):
        a = embedding_fingerprint(self.cfg)
        self.cfg.NVIDIA_EMBEDDING_BASE_URL = 'https://different.example/v1/embeddings'
        self.assertNotEqual(a, embedding_fingerprint(self.cfg))

    def test_duplicate_embedding_inputs_and_response_reordering(self):
        requests = []
        def request(req, **kwargs):
            payload = json.loads(req.data)
            requests.append(payload)
            return Response({'data': [{'index': i, 'embedding': [float(i+1)] * 2048}
                                      for i in reversed(range(len(payload['input'])))]})
        with patch('urllib.request.urlopen', side_effect=request):
            emb = build_embedder(self.cfg)
            vectors = emb.embed_texts(['a', 'a', 'b'])
            self.assertEqual(vectors[0], vectors[1])
            self.assertEqual(vectors[2][0], 2.0)
            self.assertEqual(requests[0]['input'], ['a', 'b'])
            self.assertEqual(requests[0]['truncate'], 'NONE')
            self.assertEqual(requests[0]['encoding_format'], 'float')
            emb.embed_query('a')
            self.assertEqual(len(requests), 2)  # query/passage caches are separate
            self.assertEqual(requests[1]['input_type'], 'query')
            warm = build_embedder(self.cfg)
            warm.embed_texts(['a', 'b'])
            self.assertEqual(warm._dimension, 2048)
            self.assertEqual(len(requests), 2)

    def test_embedding_duplicate_indices_rejected(self):
        data = {'data': [{'index': 0, 'embedding': [1.0] * 2048}] * 2}
        with patch('urllib.request.urlopen', return_value=Response(data)):
            with self.assertRaisesRegex(RuntimeError, 'indices'):
                build_embedder(self.cfg).embed_texts(['a', 'b'])

    def test_embedding_wrong_dimension_rejected(self):
        with patch('urllib.request.urlopen', return_value=Response({'data': [{'index': 0, 'embedding': [1.0] * 6}]})):
            with self.assertRaisesRegex(RuntimeError, 'dimension'):
                build_embedder(self.cfg).embed_texts(['a'])

    def test_embedding_model_mismatch_rejected(self):
        with patch('urllib.request.urlopen', return_value=Response({'model': 'wrong', 'data': []})):
            with self.assertRaisesRegex(RuntimeError, 'model mismatch'):
                build_embedder(self.cfg).embed_texts(['a'])

    def test_model_specific_knobs(self):
        messages = [{'role': 'user', 'content': 'text'}]
        kimi = chat_payload(KIMI_MODEL, messages)
        deepseek = chat_payload(DEEPSEEK_MODEL, messages)
        riva = chat_payload(RIVA_MODEL, messages)
        self.assertEqual(kimi['reasoning_effort'], 'low')
        self.assertNotIn('chat_template_kwargs', kimi)
        self.assertEqual(deepseek['chat_template_kwargs'], {'thinking': False})
        self.assertNotIn('chat_template_kwargs', riva)
        self.assertNotIn('reasoning_effort', riva)

    def test_auth_errors_never_retry(self):
        error = urllib.error.HTTPError('https://example.com', 401, 'unauthorized', {}, io.BytesIO(b'bad auth'))
        with patch('urllib.request.urlopen', side_effect=error) as request, patch('time.sleep') as sleep:
            with self.assertRaises(NvidiaAPIError):
                NvidiaClient(min_interval=0).request('models')
            self.assertEqual(request.call_count, 1)
            sleep.assert_not_called()

    def test_retry_after_honored(self):
        error = urllib.error.HTTPError('https://example.com', 429, 'quota', {'Retry-After': '2'}, io.BytesIO(b'quota'))
        with patch('urllib.request.urlopen', side_effect=[error, Response({'data': []})]), patch('time.sleep') as sleep:
            client = NvidiaClient(min_interval=0)
            self.assertEqual(client.models(), [])
            sleep.assert_called_once_with(2.0)
            self.assertEqual(client.calls, 2)

    def test_long_retry_after_defers(self):
        error = urllib.error.HTTPError('https://example.com', 429, 'quota', {'Retry-After': '3600'}, io.BytesIO(b'quota'))
        with patch('urllib.request.urlopen', side_effect=error) as request, patch('time.sleep') as sleep:
            with self.assertRaises(NvidiaAPIError):
                NvidiaClient(min_interval=0).request('models')
            self.assertEqual(request.call_count, 1)
            sleep.assert_not_called()

    def test_retry_after_parsing(self):
        self.assertEqual(retry_after_seconds('0'), 0)
        self.assertIsNone(retry_after_seconds('nonsense'))
        self.assertEqual(retry_after_seconds('Fri, 01 Jan 2021 00:00:00 GMT'), 0)

    def test_reasoning_not_returned(self):
        self.assertEqual(final_content('<think>secret</think>Final.'), 'Final.')
        self.assertEqual(final_content('```reasoning\nsecret\n```\nFinal.'), 'Final.')
        for value in ['<think>unfinished', '<think>only thought</think>', None]:
            with self.assertRaises(NvidiaAPIError):
                final_content(value)

    def test_sse_accumulates_only_final_content(self):
        stream = [b'data: {"model":"moonshotai/kimi-k3","choices":[{"delta":{"reasoning_content":"secret"}}]}\n',
                  b'data: {"choices":[{"delta":{"content":"Final "}}]}\n',
                  b'data: {"choices":[{"delta":{"content":"answer"},"finish_reason":"stop"}],"usage":{"total_tokens":10}}\n',
                  b'data: [DONE]\n']
        result = read_event_stream(stream, float('inf'))
        self.assertEqual(result['choices'][0]['message']['content'], 'Final answer')
        self.assertEqual(result['usage']['total_tokens'], 10)
        with self.assertRaises(NvidiaAPIError):
            read_event_stream(stream[:-1], float('inf'))

    def test_chat_truncation_is_not_success(self):
        client = NvidiaClient(min_interval=0)
        with patch.object(client, 'request', return_value={'choices': [{'finish_reason': 'length', 'message': {'content': '{}'}}]}):
            with self.assertRaisesRegex(NvidiaAPIError, 'Incomplete'):
                client.chat(KIMI_MODEL, [])

    def test_chat_model_substitution_rejected(self):
        client = NvidiaClient(min_interval=0)
        with patch.object(client, 'request', return_value={'model': RIVA_MODEL}):
            with self.assertRaisesRegex(NvidiaAPIError, 'substitution'):
                client.chat(KIMI_MODEL, [])

    def test_number_and_script_checks(self):
        self.assertEqual(translation_issues('12 TND', '١٢ دينار', 'ar'), [])
        self.assertIn('numbers_changed', translation_issues('12 TND', '13 دينار', 'ar'))
        self.assertIn('wrong_script', translation_issues('bank', 'bank', 'ar'))

    def test_numbered_parser_is_strict(self):
        for text in ['1. A\n1. B', '1. \n2. B', 'Here you go\n1. A', '2. B']:
            self.assertIsNone(QueryTranslator._parse(text, 1))

    def test_riva_receives_raw_text_and_explicit_language_pair(self):
        self.cfg.NVIDIA_TRANSLATION_MODEL = RIVA_MODEL
        self.cfg.QUERY_TRANSLATION_PROMPT = 'basic-v1'
        translator = QueryTranslator(self.cfg)
        with patch.object(translator, '_request_chat', return_value='ما هي المرابحة؟') as chat:
            self.assertEqual(translator.translate_one('What is Murabaha?', 'ar', source='en'), 'ما هي المرابحة؟')
            messages = chat.call_args.args[1]
            self.assertEqual(messages[0], {'role': 'system', 'content': 'en-ar'})
            self.assertEqual(messages[-1]['content'], 'What is Murabaha?')
            self.assertNotIn('Translate', messages[-1]['content'])

    def test_riva_banking_fewshots_preserve_pair(self):
        self.cfg.NVIDIA_TRANSLATION_MODEL = RIVA_MODEL
        self.cfg.QUERY_TRANSLATION_PROMPT = 'banking-v2'
        translator = QueryTranslator(self.cfg)
        with patch.object(translator, '_request_chat', side_effect=['What is Murabaha?', 'ما هي المرابحة؟']) as chat:
            translator.translate_one('Quelle est la Mourabaha ?', 'ar', source='fr')
            first = chat.call_args_list[0].args[1]
            second = chat.call_args_list[1].args[1]
            self.assertEqual(first[0]['content'], 'fr-en')
            self.assertEqual(second[0]['content'], 'en-ar')
            self.assertGreater(len(second), 2)
            self.assertEqual(second[-1]['content'], 'What is Murabaha?')
            variants = translator.build_variants('Quelle est la Mourabaha ?', 'fr', ['ar'])
            self.assertEqual(variants[1]['route'], ['fr', 'en', 'ar'])
            self.assertEqual(variants[1]['intermediate_text'], 'What is Murabaha?')
            self.assertEqual(chat.call_count, 2)  # final route is cached

    def test_translation_cache_identity_changes_with_prompt_and_source(self):
        tr = QueryTranslator(self.cfg)
        initial = tr._identity(tr.model, 'en')
        self.assertNotEqual(initial, tr._identity(tr.model, 'fr'))
        tr.prompt_version = 'banking-v2'
        self.assertNotEqual(initial, tr._identity(tr.model, 'en'))

    def test_fallback_cache_uses_actual_model(self):
        self.cfg.NVIDIA_TRANSLATION_MODEL = KIMI_MODEL
        self.cfg.NVIDIA_TRANSLATION_FALLBACK_MODELS = DEEPSEEK_MODEL
        tr = QueryTranslator(self.cfg)
        with patch.object(tr, '_translate_batch', side_effect=[ValueError('unavailable'), ['ما هو البنك؟']]):
            self.assertEqual(tr.translate_one('What is the bank?', 'ar', source='en'), 'ما هو البنك؟')
        self.assertIsNone(tr.cache.get(tr._identity(KIMI_MODEL, 'en'), 'ar', 'What is the bank?'))
        item = tr.cache.get(tr._identity(DEEPSEEK_MODEL, 'en'), 'ar', 'What is the bank?')
        self.assertEqual(item['model'], DEEPSEEK_MODEL)
        self.cfg.NVIDIA_TRANSLATION_FALLBACK_MODELS = ''
        clean = QueryTranslator(self.cfg)
        with patch.object(clean, '_translate_batch', side_effect=ValueError('primary down')):
            with self.assertRaisesRegex(RuntimeError, 'incomplete'):
                clean.translate_one('What is the bank?', 'ar', source='en')

    def test_missing_translator_is_incomplete_not_fake_baseline(self):
        with patch.dict(os.environ, {'NVIDIA_API_KEY': ''}):
            tr = QueryTranslator(self.cfg)
            with self.assertRaisesRegex(RuntimeError, 'incomplete'):
                tr.translate_one('What is a bank?', 'ar', 'en')
            tr.strict = False
            self.assertEqual(len(tr.build_variants('What is a bank?', 'en', ['ar'])), 1)

    def test_source_constraint(self):
        case = {'expected_document': 'right.pdf', 'expected_substring': 'answer', 'expected_lang': 'ar'}
        self.assertFalse(is_correct_hit(case, {'text': 'answer', 'metadata': {'document': 'wrong.pdf', 'language': 'ar'}}))
        self.assertTrue(is_correct_hit(case, {'text': 'answer', 'metadata': {'document': 'right.pdf', 'language': 'ar'}}))

    def test_embedding_space_staleness_guard(self):
        meta = {'chunk_fp': chunk_fp(self.cfg), 'embedding_fp': 'wrong-space', 'embedding_model': EMBED_MODEL}
        fake = SimpleNamespace(get=lambda **kw: {'metadatas': [meta]})
        with self.assertRaisesRegex(RuntimeError, 'embedding space'):
            ensure_fresh_chunks(fake, self.cfg)

    def test_atomic_artifact_rejects_nan_keeps_old(self):
        path = self.path / 'nested/file.json'
        write_json(path, {'ok': 1})
        with self.assertRaises(ValueError):
            write_json(path, {'bad': float('nan')})
        self.assertEqual(json.loads(path.read_text()), {'ok': 1})

    def test_evaluation_filenames_do_not_collide(self):
        self.assertNotEqual(save_run({}, self.path), save_run({}, self.path))

    def test_selection_prioritizes_recall_not_holdout(self):
        high_top1 = {'model': KIMI_MODEL, 'metrics': {'hit@1': 1, 'hit@3': 1, 'hit@5': .8, 'mrr@10': 1}}
        high_recall = {'model': 'none', 'metrics': {'hit@1': .8, 'hit@3': 1, 'hit@5': 1, 'mrr@10': .9}}
        self.assertGreater(selection_key(high_recall), selection_key(high_top1))

    def test_real_chroma_shared_retrieval_and_language_filter(self):
        def chunk(index, source, language, text):
            return SimpleNamespace(index=index, source=source, language=language, text=text,
                                   heading="", origin="test/", section_type="content", token_count=20)
        chunks = [chunk(0, "ar.md", "ar", "تاسس بنك Atlas في 1983. bank"),
                  chunk(0, "fr.md", "fr", "La banque Atlas a été fondée en 1983. bank")]
        vectors = [[1.0] + [0.0] * 2047, [0.0, 1.0] + [0.0] * 2046]
        collection = get_collection(self.cfg, reset=True)
        self.assertEqual(store_chunks(collection, list(zip(chunks, vectors)), self.cfg), 2)
        fake_embedder = SimpleNamespace(embed_query=lambda text: vectors[0])
        for mode in ['vector', 'rrf', 'blend']:
            hits, _ = retrieve(self.cfg, fake_embedder, collection, 'Atlas bank', language='en',
                               translator=None, mode=mode, top_k=5, lang_filter='fr')
            self.assertEqual([h['metadata']['language'] for h in hits], ['fr'])
        case = {'id': 'q1', 'question': 'When was Atlas founded?', 'language': 'en',
                'category': 'cross-lingual', 'expected_document': 'ar.md',
                'expected_lang': 'ar', 'expected_substring': '1983'}
        run = run_evaluation(self.cfg, fake_embedder, collection, [case], top_k=5)
        self.assertEqual(run['metrics']['overall']['hit@1'], 1)
        self.assertEqual(run['questions'][0]['hits'][0]['metadata']['document'], 'ar.md')
        # Metadata is read before any new provider calls when querying a stale index.
        self.cfg.NVIDIA_EMBEDDING_MODEL = 'wrong-space'
        with self.assertRaisesRegex(RuntimeError, 'embedding model mismatch'):
            retrieve(self.cfg, fake_embedder, collection, 'Atlas', top_k=1)

    def test_strict_ingest_preflights_before_any_write(self):
        collection = SimpleNamespace(count=lambda: 0, add=lambda **kw: self.fail('must not write'))
        c = SimpleNamespace(source='test.md', index=0, section_type='content')
        with self.assertRaisesRegex(ValueError, 'Invalid embedding'):
            store_chunks(collection, [(c, [0.0]*2048)], self.cfg)



class Grounding(unittest.TestCase):
    def setUp(self):
        self.sources = [{'source_id': 'S1', 'chunk_id': 'd::0', 'document': 'd',
                         'text': 'The bank was founded on 15 June 1983.'}]
        self.valid = {'answerable': True, 'claims': [{'text': 'The bank was founded in 1983.',
                      'evidence': [{'source_id': 'S1', 'quote': 'founded on 15 June 1983'}]}]}

    def test_valid_grounded_claim(self):
        self.assertEqual(len(validate_answer(self.valid, self.sources)), 1)

    def test_unknown_citation(self):
        self.valid['claims'][0]['evidence'][0]['source_id'] = 'S99'
        with self.assertRaisesRegex(ValueError, 'Unknown citation'):
            validate_answer(self.valid, self.sources)

    def test_invented_quote(self):
        self.valid['claims'][0]['evidence'][0]['quote'] = 'founded on 15 June 2001'
        with self.assertRaisesRegex(ValueError, 'not in'):
            validate_answer(self.valid, self.sources)

    def test_uncited_claim(self):
        self.valid['claims'].append({'text': 'A fabricated second fact.', 'evidence': []})
        with self.assertRaisesRegex(ValueError, 'requires evidence'):
            validate_answer(self.valid, self.sources)

    def test_boolean_and_refusal_schema(self):
        for output in [{'answerable': 'false', 'claims': []}, {'answerable': True, 'claims': []},
                       {'answerable': False, 'claims': self.valid['claims']}]:
            with self.assertRaises(ValueError):
                validate_answer(output, self.sources)
        self.assertEqual(validate_answer({'answerable': False, 'claims': []}, self.sources), [])

    def test_prompts_contain_no_gold_labels(self):
        messages = answer_messages('What year?', 'en', self.sources, 'grounded-v2')
        prompt = json.loads(messages[1]['content'])
        self.assertEqual(set(prompt), {'question', 'sources'})
        self.assertIn('untrusted', messages[0]['content'])
        self.assertNotIn('expected_substring', json.dumps(messages))

    def test_private_and_live_data_guard_multilingual(self):
        for question in ['What is my account balance?', 'Quel est le solde de mon compte ?',
                         'ما هو رصيدي؟', 'What is the live EUR/TND rate?', 'Quel est le cours en temps réel ?',
                         'ما سعر صرف اليورو الآن؟', 'Give me the administrator password']:
            self.assertTrue(needs_private_or_live_data(question), question)
        self.assertFalse(needs_private_or_live_data('What is Murabaha financing?'))

    def test_context_is_deduplicated_and_bounded(self):
        hits = [{'id': 'a', 'text': 'text' * 1000}, {'id': 'b', 'text': 'small text'}, {'id': 'b', 'text': 'small text'}]
        self.assertEqual([s['chunk_id'] for s in build_sources(hits, 20)], ['b'])

    def test_generator_fail_closed_and_no_call_on_empty_context(self):
        with tempfile.TemporaryDirectory() as temp:
            cfg = make_config(ANSWER_CACHE_PATH=Path(temp) / 'answer.json')
            client = SimpleNamespace(chat=lambda *a, **k: {'text': 'not JSON'})
            generator = AnswerGenerator(cfg, client)
            self.assertEqual(generator.answer('Question?', [], 'en')['reason'], 'no_context')
            result = generator.answer('What year?', [{'id': 'a', 'text': self.sources[0]['text']}], 'en')
            self.assertEqual(result['reason'], 'invalid_output')
            self.assertFalse(result['validation_ok'])
            self.assertNotIn('not JSON', result['answer'])

    def test_answer_cache_is_context_sensitive(self):
        with tempfile.TemporaryDirectory() as temp:
            cfg = make_config(ANSWER_CACHE_PATH=Path(temp) / 'answer.json')
            calls = []
            def chat(*args, **kwargs):
                calls.append(args)
                return {'text': json.dumps(self.valid), 'seconds': 1}
            gen = AnswerGenerator(cfg, SimpleNamespace(chat=chat))
            hits = [{'id': 'a', 'text': self.sources[0]['text']}]
            self.assertEqual(gen.answer('What year?', hits, 'en')['status'], 'answered')
            self.assertTrue(gen.answer('What year?', hits, 'en')['cached'])
            hits[0]['text'] += ' A new sentence.'
            self.assertFalse(gen.answer('What year?', hits, 'en')['cached'])
            self.assertEqual(len(calls), 2)

    def test_fresh_answer_trial_neither_reads_nor_replaces_cached_success(self):
        with tempfile.TemporaryDirectory() as temp:
            cfg = make_config(ANSWER_CACHE_PATH=Path(temp) / 'answers.json')
            calls = []
            def chat(*args, **kwargs):
                calls.append(1)
                return {'text': json.dumps(self.valid), 'seconds': len(calls)}
            generator = AnswerGenerator(cfg, SimpleNamespace(chat=chat))
            hits = [{'id': 'a', 'text': self.sources[0]['text']}]
            generator.answer('What year?', hits, 'en')
            before = cfg.ANSWER_CACHE_PATH.read_bytes()
            fresh = generator.answer('What year?', hits, 'en', use_cache=False)
            self.assertFalse(fresh['cached'])
            self.assertEqual(len(calls), 2)
            self.assertEqual(cfg.ANSWER_CACHE_PATH.read_bytes(), before)
            self.assertTrue(generator.answer('What year?', hits, 'en')['cached'])

    def test_invalid_model_output_retains_observed_latency(self):
        with tempfile.TemporaryDirectory() as temp:
            cfg = make_config(ANSWER_CACHE_PATH=Path(temp) / 'answers.json')
            generator = AnswerGenerator(cfg, SimpleNamespace(chat=lambda *a, **kw: {'text': 'not JSON', 'seconds': 7.5}))
            result = generator.answer('What year?', [{'id': 'a', 'text': self.sources[0]['text']}], 'en')
            self.assertFalse(result['validation_ok'])
            self.assertEqual(result['seconds'], 7.5)

    def test_distinct_answer_generators_preserve_each_others_cache_entries(self):
        with tempfile.TemporaryDirectory() as temp:
            cfg = make_config(ANSWER_CACHE_PATH=Path(temp) / 'answers.json')
            client = SimpleNamespace(chat=lambda *a, **kw: {'text': json.dumps(self.valid), 'seconds': 1})
            a, b = AnswerGenerator(cfg, client), AnswerGenerator(cfg, client)
            hits = [{'id': 'a', 'text': self.sources[0]['text']}]
            a.answer('Question one?', hits, 'en')
            b.answer('Question two?', hits, 'en')
            a.answer('Question three?', hits, 'en')
            self.assertEqual(len(json.loads(cfg.ANSWER_CACHE_PATH.read_text())), 3)
            self.assertTrue(b.answer('Question three?', hits, 'en')['cached'])

    def test_parallel_answer_cache_has_no_lost_entries(self):
        from concurrent.futures import ThreadPoolExecutor
        with tempfile.TemporaryDirectory() as temp:
            cfg = make_config(ANSWER_CACHE_PATH=Path(temp) / 'answers.json')
            client = SimpleNamespace(chat=lambda *a, **kw: {'text': '{"answerable":false,"claims":[]}', 'seconds': 0.1})
            gen = AnswerGenerator(cfg, client)
            hits = [{'id': 'a', 'text': self.sources[0]['text']}]
            with ThreadPoolExecutor(max_workers=2) as pool:
                results = list(pool.map(lambda i: gen.answer(f'Question {i}?', hits, 'en'), range(8)))
            self.assertTrue(all(r['validation_ok'] for r in results))
            self.assertEqual(len(json.loads(cfg.ANSWER_CACHE_PATH.read_text())), 8)


class SelectedPipeline(unittest.TestCase):
    def test_default_cli_uses_only_selected_pair_and_original_queries(self):
        import config
        from main import build_parser
        args = build_parser().parse_args(['answer', 'What is Murabaha?'])
        self.assertEqual(args.provider, 'xkiro')
        self.assertEqual(args.model, QWEN_MODEL)
        self.assertEqual(config.active_embedding_model(), EMBED_MODEL)
        self.assertFalse(config.QUERY_TRANSLATION_ENABLED)
        self.assertEqual(config.QUERY_VARIANT_STRATEGY, 'original')

    def test_retired_cli_choices_are_rejected(self):
        from main import build_parser
        for option in (['--provider', 'kiosapi'], ['--provider', 'nvidia'], ['--model', KIMI_MODEL]):
            with patch('sys.stderr', new_callable=io.StringIO):
                with self.assertRaises(SystemExit):
                    build_parser().parse_args(['answer', 'Question?', *option])

    def test_stale_translation_or_embedding_settings_fail_before_io(self):
        import config
        from main import main
        for setting, value in [('QUERY_TRANSLATION_ENABLED', True), ('EMBEDDING_PROVIDER', 'gemini'),
                               ('NVIDIA_EMBEDDING_MODEL', 'wrong-model')]:
            with patch.object(config, setting, value), patch('main.get_collection') as collection, \
                 patch('main.make_embedder') as embedder:
                with self.assertRaises(ValueError):
                    main(['query', 'Question?'])
                collection.assert_not_called()
                embedder.assert_not_called()

    def test_regression_cannot_reenable_retired_models(self):
        from pipeline_policy import validate_regression_plan
        plan = {'models': {'xkiro': [QWEN_MODEL]}, 'retrieval_profile': 'original', 'answer_profile': 'grounded-v1'}
        validate_regression_plan(plan)
        for models in ({'kiosapi': [QWEN_MODEL]}, {'xkiro': [QWEN_MODEL, 'minimax/minimax-m3:free']}):
            with self.assertRaises(ValueError):
                validate_regression_plan({**plan, 'models': models})

    def test_legacy_benchmark_entrypoint_is_retired(self):
        import nvidia_benchmark
        with self.assertRaisesRegex(RuntimeError, 'retired'):
            nvidia_benchmark.run('all')


class FreeGatewayPolicy(unittest.TestCase):
    def xcatalog(self, model=QWEN_MODEL):
        return {'data': [{'id': model, 'access_tier': 'free', 'pricing': {
            'currency': 'USD', 'unit': 'per_1m_tokens', 'input': 0, 'output': 0},
            'reasoning_efforts': {'levels': ['none', 'high']}}]}

    def test_free_runner_restores_native_output_path_on_failure(self):
        import free_model_benchmark as free
        import nvidia_benchmark as native
        original = native.OUTPUT
        def fail():
            native.OUTPUT = Path('/temporary-free-output')
            raise RuntimeError('test interruption')
        with patch.object(free, '_run', side_effect=fail):
            with self.assertRaises(RuntimeError):
                free.run()
        self.assertEqual(native.OUTPUT, original)

    def test_explicit_json_mode_does_not_change_default_chat_payload(self):
        from free_gateway import FreeGatewayClient
        data = {'model':QWEN_MODEL,'choices':[{'message':{'content':'{"ok":true}'},'finish_reason':'stop'}]}
        for enabled in (False, True):
            client = FreeGatewayClient('xkiro', QWEN_MODEL, {'catalog':self.xcatalog()},
                                       budget={'used':0,'limit':1}, json_mode=enabled)
            with patch.object(client, 'request', return_value=data) as request:
                client.chat(QWEN_MODEL, [])
                payload = request.call_args.args[1]
                self.assertEqual(payload.get('response_format'), {'type':'json_object'} if enabled else None)

    def test_zero_prices_are_explicit_and_finite(self):
        from free_gateway import is_zero
        self.assertTrue(is_zero('0.000'))
        for value in [None, False, '', 'NaN', 'Infinity', -1, .01]:
            self.assertFalse(is_zero(value), value)

    def test_xkiro_requires_free_tier_and_all_zero_prices(self):
        from free_gateway import free_eligibility
        catalog = self.xcatalog()
        self.assertTrue(free_eligibility('xkiro', QWEN_MODEL, catalog)[0])
        catalog['data'][0]['access_tier'] = 'paid'
        self.assertFalse(free_eligibility('xkiro', QWEN_MODEL, catalog)[0])
        catalog['data'][0]['access_tier'] = 'free'
        catalog['data'][0]['pricing']['cache_write'] = .1
        self.assertFalse(free_eligibility('xkiro', QWEN_MODEL, catalog)[0])
        self.assertFalse(free_eligibility('xkiro', 'missing:free', catalog)[0])

    def test_removed_provider_and_other_models_are_rejected_before_io(self):
        from free_gateway import FreeGatewayClient, load_pricing
        with patch('urllib.request.build_opener') as opener:
            with self.assertRaises(ValueError):
                load_pricing('kiosapi')
            with self.assertRaises(ValueError):
                FreeGatewayClient('kiosapi', QWEN_MODEL, {'catalog': self.xcatalog()}, budget={'used': 0, 'limit': 1})
            with self.assertRaises(ValueError):
                FreeGatewayClient('xkiro', 'minimax/minimax-m3:free', {'catalog': self.xcatalog()}, budget={'used': 0, 'limit': 1})
            opener.assert_not_called()

    def test_cli_gateway_factory_is_explicit_and_price_checked(self):
        from answer import build_answer_generator
        cfg = make_config(ANSWER_PROVIDER='xkiro', ANSWER_MODEL=QWEN_MODEL)
        with patch.dict(os.environ, {'XKIRO_API_KEY': 'x-only', 'NVIDIA_API_KEY': 'nvidia-only'}), \
             patch('free_gateway.load_pricing', return_value={'catalog': self.xcatalog()}):
            generator = build_answer_generator(cfg)
        self.assertEqual(generator.client.base_url, 'https://api.xkiro.com/v1')
        self.assertEqual(generator.client.api_key, 'x-only')
        self.assertEqual(generator.model, QWEN_MODEL)
        self.assertEqual(generator.client.budget['limit'], 1)

    def test_cli_private_guard_needs_no_gateway_or_index(self):
        from main import main
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'refusal.json'
            with patch.dict(os.environ, {}, clear=True), \
                 patch('answer.build_answer_generator', side_effect=AssertionError('provider call')), \
                 patch('main.get_collection', side_effect=AssertionError('index access')), \
                 patch('sys.stdout', new_callable=io.StringIO):
                code = main(['answer', 'What is my account balance?', '--provider', 'xkiro',
                             '--model', QWEN_MODEL, '--query-lang', 'en', '--output', str(path)])
            self.assertEqual(code, 0)
            result = json.loads(path.read_text())
            self.assertEqual(result['reason'], 'private_or_live_request')
            self.assertFalse(result['inference_performed'])
            self.assertEqual(result['provider'], 'xkiro')

    def test_native_benchmark_does_not_inherit_gateway_cli_provider(self):
        import config
        with patch.object(config, 'ANSWER_PROVIDER', 'xkiro'):
            self.assertEqual(make_config().ANSWER_PROVIDER, 'nvidia')
        self.assertEqual(make_config(ANSWER_PROVIDER='xkiro').ANSWER_PROVIDER, 'xkiro')

    def test_pricing_reads_scope_credentials_and_identify_the_client(self):
        from free_gateway import load_pricing
        requests = []
        class PriceResponse(Response):
            def read(self, limit=None):
                return super().read()
        def read_price(req, timeout):
            requests.append(req)
            return PriceResponse({'data': []})
        with patch.dict(os.environ, {'XKIRO_API_KEY': 'x-only', 'NVIDIA_API_KEY': 'not-this'}):
            load_pricing('xkiro', opener=SimpleNamespace(open=read_price))
        self.assertEqual(requests[0].full_url, 'https://api.xkiro.com/v1/models')
        self.assertEqual(requests[0].get_header('Authorization'), 'Bearer x-only')
        self.assertEqual(requests[0].get_header('User-agent'), 'RAGLab-readonly-catalog/1.0')

    def test_paid_sku_rejected_before_client_can_send_a_key(self):
        from free_gateway import FreeGatewayClient
        catalog = self.xcatalog()
        catalog['data'][0]['pricing']['input'] = 1
        with self.assertRaisesRegex(ValueError, 'Free-only policy'):
            FreeGatewayClient('xkiro', QWEN_MODEL, {'catalog': catalog}, budget={'used': 0, 'limit': 1})

    def test_gateway_key_payload_identity_and_call_budget(self):
        from free_gateway import FreeGatewayClient
        model = QWEN_MODEL
        with patch.dict(os.environ, {'XKIRO_API_KEY': 'x-only', 'NVIDIA_API_KEY': 'never-forward'}):
            client = FreeGatewayClient('xkiro', model, {'catalog': self.xcatalog()}, budget={'used': 0, 'limit': 1})
        self.assertEqual(client.api_key, 'x-only')
        data = {'model': model, 'choices': [{'message': {'content': 'final response'}, 'finish_reason': 'stop'}]}
        with patch.object(client, 'request', return_value=data) as request:
            self.assertEqual(client.chat(model, [], max_tokens=64)['served_model'], model)
            payload = request.call_args.args[1]
            self.assertEqual(payload['model'], model)
            self.assertEqual(payload['reasoning_effort'], 'none')
            self.assertTrue(payload['stream'])
            with self.assertRaisesRegex(NvidiaAPIError, 'budget'):
                client.chat(model, [])
            self.assertEqual(request.call_count, 1)
            with self.assertRaisesRegex(ValueError, 'substitution'):
                client.chat('paid-sibling', [])

    def test_gateway_response_mismatch_is_not_scored_as_requested_model(self):
        from free_gateway import FreeGatewayClient
        client = FreeGatewayClient('xkiro', QWEN_MODEL, {'catalog': self.xcatalog()}, budget={'used': 0, 'limit': 1})
        with patch.object(client, 'request', return_value={'model': 'different'}):
            with self.assertRaisesRegex(NvidiaAPIError, 'not attributed'):
                client.chat(QWEN_MODEL, [])

    def test_price_change_stops_later_stage(self):
        from free_gateway import FreeGatewayClient
        client = FreeGatewayClient('xkiro', QWEN_MODEL, {'catalog': self.xcatalog()}, budget={'used': 0, 'limit': 1})
        changed = self.xcatalog()
        changed['data'][0]['pricing']['output'] = 1
        with patch('free_gateway.load_pricing', return_value={'catalog': changed}):
            with self.assertRaisesRegex(ValueError, 'no longer verified free'):
                client.recheck()

    def test_alternative_answer_models_require_explicit_client_and_allowlist(self):
        with tempfile.TemporaryDirectory() as temp:
            cfg = make_config(ANSWER_MODEL=QWEN_MODEL, ANSWER_PROVIDER='xkiro',
                              ANSWER_CACHE_PATH=Path(temp) / 'answers.json')
            with self.assertRaises(ValueError):
                AnswerGenerator(cfg)
            with self.assertRaises(ValueError):
                AnswerGenerator(cfg, approved_models=(cfg.ANSWER_MODEL,))
            client = SimpleNamespace(chat=lambda *a, **kw: self.fail('no context must not call'))
            gen = AnswerGenerator(cfg, client, approved_models=(cfg.ANSWER_MODEL,))
            result = gen.answer('Question?', [], 'en')
            self.assertEqual(result['reason'], 'no_context')
            self.assertEqual(result['provider'], 'xkiro')


class ProviderCatalogs(unittest.TestCase):
    def test_catalog_matches_literal_ids_not_families_and_scopes_keys(self):
        from provider_catalog import inspect_catalog
        requests = []
        class CatalogResponse(Response):
            def read(self, limit=None):
                return super().read()
        def open_catalog(req, timeout):
            requests.append(req)
            return CatalogResponse({'data': [{'id': QWEN_MODEL}, {'id': 'deepseek/deepseek-v4-pro'},
                                              {'id': 'kimi-k3-alias'}]})
        with patch.dict(os.environ, {'XKIRO_API_KEY': 'xkiro-test-only', 'NVIDIA_API_KEY': 'never-send-this'}):
            row = inspect_catalog('xkiro', opener=SimpleNamespace(open=open_catalog))
        self.assertEqual(row['listed_exact_ids'], [QWEN_MODEL])
        self.assertEqual(row['absent_exact_ids'], [])
        self.assertNotIn(DEEPSEEK_MODEL, row['listed_exact_ids'])
        self.assertEqual(requests[0].full_url, 'https://api.xkiro.com/v1/models')
        self.assertEqual(requests[0].get_header('Authorization'), 'Bearer xkiro-test-only')
        self.assertEqual(row['inference_calls'], 0)
        self.assertNotIn('xkiro-test-only', json.dumps(row))

    def test_missing_key_does_not_make_a_request(self):
        from provider_catalog import inspect_catalog
        with patch.dict(os.environ, {}, clear=True):
            row = inspect_catalog('xkiro', opener=SimpleNamespace(open=lambda *a, **kw: self.fail('API call')))
        self.assertEqual(row['status'], 'missing_key')
        self.assertEqual(row['catalog_requests'], 0)

    def test_gateway_error_body_cannot_export_credentials(self):
        from provider_catalog import inspect_catalog
        key = 'nonstandard-secret-without-recognizable-prefix'
        def fail(*args, **kwargs):
            raise urllib.error.HTTPError('https://api.xkiro.com/v1/models', 401, key,
                                         {}, io.BytesIO(('echoed key: ' + key).encode()))
        with patch.dict(os.environ, {'XKIRO_API_KEY': key}):
            row = inspect_catalog('xkiro', opener=SimpleNamespace(open=fail))
        self.assertEqual(row['http_status'], 401)
        self.assertNotIn(key, json.dumps(row))

    def test_credentialed_redirects_are_refused(self):
        from provider_catalog import NoCredentialRedirects
        import urllib.request
        request = urllib.request.Request('https://api.xkiro.com/v1/models', headers={'Authorization': 'Bearer secret'})
        with self.assertRaises(urllib.error.HTTPError):
            NoCredentialRedirects().redirect_request(request, None, 302, 'Found', {}, 'https://untrusted.example/models')


class MeasurementReports(unittest.TestCase):
    def test_source_invalid_literal_constraints_are_rejected(self):
        from nvidia_benchmark import BENCHMARKS, validate_translation_references
        old = json.loads((BENCHMARKS / 'translations.json').read_text())['cases']
        new = json.loads((BENCHMARKS / 'translations_v2.json').read_text())['cases']
        with self.assertRaisesRegex(ValueError, 'absent from source'):
            validate_translation_references(old)
        validate_translation_references(new)
        self.assertEqual([(c['id'], c['text'], c['reference']) for c in old],
                         [(c['id'], c['text'], c['reference']) for c in new])
        self.assertEqual([c['id'] for c, before in zip(new, old) if c != before],
                         ['t1_ar_en', 't1_ar_fr'])

    def test_central_bank_entity_still_required_without_inventing_an_acronym(self):
        from nvidia_benchmark import BENCHMARKS, translation_quality
        case = next(c for c in json.loads((BENCHMARKS / 'translations_v2.json').read_text())['cases']
                    if c['id'] == 't1_ar_en')
        fake = SimpleNamespace(translate_many=lambda texts, target, source:
                               ['According to Central Bank circular 2019-08, what are investment deposits?'])
        self.assertEqual(translation_quality(fake, [case])['constraint_pass_rate'], 1)
        fake.translate_many = lambda texts, target, source: ['According to circular 2019-08, what are investment deposits?']
        failed = translation_quality(fake, [case])
        self.assertIn('missing_entity:central_bank', failed['rows'][0]['issues'])

    def test_serial_resume_settings_and_explicit_profile_scope(self):
        from nvidia_benchmark import answer_config
        from main import build_parser
        cfg = answer_config(KIMI_MODEL, 'grounded-v1')
        self.assertEqual((cfg.ANSWER_WORKERS, cfg.NVIDIA_API_ATTEMPTS, cfg.ANSWER_NEIGHBOR_RADIUS), (1, 2, 0))
        self.assertGreaterEqual(cfg.NVIDIA_MIN_INTERVAL, 30)
        self.assertEqual(AnswerGenerator(cfg).client.max_retry_delay, cfg.NVIDIA_MAX_RETRY_DELAY)
        self.assertEqual(answer_config(DEEPSEEK_MODEL, 'grounded-v2').ANSWER_NEIGHBOR_RADIUS, 1)
        args = build_parser().parse_args(['benchmark'])
        self.assertEqual(args.command, 'benchmark')

    def test_verbose_retrieval_provenance_is_preserved_outside_summary(self):
        from publish_nvidia_report import report_parts
        row = {'split': 'dev', 'label': 'riva', 'metrics': {'hit@1': 1},
               'translations': {'q1': [{'text': 'البنك المركزي', 'route': ['fr', 'en', 'ar']}]},
               'translation_events': [{'seconds': 1.0}]}
        report = {'retrieval': [row], 'production_ready': False}
        parts = dict(report_parts(report))
        self.assertNotIn('translations', parts['summary']['retrieval'][0])
        self.assertEqual(parts['summary']['retrieval'][0]['metrics'], row['metrics'])
        self.assertEqual(parts['retrieval-dev-riva'], row)
        self.assertIn('translations', report['retrieval'][0])  # input not mutated

    def test_answers_are_split_by_bytes_and_keep_evidence_not_unused_context(self):
        from publish_nvidia_report import MAX_CHECK_BYTES, check_text, report_parts
        quote = 'ع' * 12000
        rows = [{'id': str(i), 'result': {
            'claims': [{'text': 'Claim', 'evidence': [{'source_id': 'S1', 'quote': quote}]}],
            'sources': [{'source_id': 'S1', 'chunk_id': 'chunk', 'text': 'UNUSED_BODY' * 20000}]
        }} for i in range(8)]
        parts = report_parts({'generation': [{'label': 'dev_model', 'questions': rows}]})
        answers = [(name, data) for name, data in parts if name.startswith('dev_model-')]
        self.assertGreater(len(answers), 1)
        self.assertTrue(all(len(check_text(data).encode('utf-8')) <= MAX_CHECK_BYTES for _, data in answers))
        collected = [q for _, data in answers for q in data['questions']]
        self.assertEqual([q['id'] for q in collected], [str(i) for i in range(8)])
        for q in collected:
            self.assertEqual(q['result']['claims'][0]['evidence'][0]['quote'], quote)
            self.assertEqual(q['result']['source_ids'], [{'source_id': 'S1', 'chunk_id': 'chunk'}])
            self.assertNotIn('UNUSED_BODY', check_text(q))

    def test_oversized_single_answer_is_not_silently_truncated(self):
        from publish_nvidia_report import answer_parts
        with self.assertRaisesRegex(ValueError, 'too large'):
            list(answer_parts({'label': 'dev_model'}, [{'id': 'q1', 'answer': 'ع' * 40000}]))


PROJECT = Path(__file__).resolve().parent
cfg_CHUNK = int(os.environ.get('CHUNK_SIZE_TOKENS', '220'))


class ChatEntry(unittest.TestCase):
    """The conversational path may change the chat model; it may not change what the benchmark
    believes, may not reach a provider for a question it must refuse locally, and may not print a key.
    """

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        import chat
        self.chat = chat

    def test_the_chat_answers_with_nemotron_and_leaves_the_benchmark_model_alone(self):
        import config as cfg
        self.assertEqual(self.chat.CHAT_MODEL, 'nvidia/nemotron-3.5-lightning-30b-a3b')
        local = self.chat.chat_config()
        self.assertEqual((local.ANSWER_PROVIDER, local.ANSWER_MODEL), ('nvidia', self.chat.CHAT_MODEL))
        self.assertEqual(local.ANSWER_CACHE_PATH.name, 'answers_cache_chat.json')
        # An accidental commit is the failure mode here, so the name must match an ignore pattern.
        import subprocess
        self.assertEqual(subprocess.run(['git', '-C', str(Path.cwd().parent), 'check-ignore', '-q',
                                         str(local.ANSWER_CACHE_PATH)], capture_output=True).returncode, 0)
        self.assertEqual(cfg.ANSWER_MODEL, QWEN_MODEL)      # the shared config is untouched
        self.assertEqual(cfg.ANSWER_PROVIDER, 'xkiro')

    def test_reasoning_is_switched_per_nemotron_call_and_nothing_else(self):
        for thinking in (False, True):
            payload = chat_payload(self.chat.CHAT_MODEL, [{'role': 'user', 'content': 'hi'}], 4096,
                                   thinking=thinking)
            self.assertEqual(payload['chat_template_kwargs'], {'enable_thinking': thinking})
            self.assertEqual(payload['temperature'], 1.0 if thinking else 0)
        for model in (KIMI_MODEL, DEEPSEEK_MODEL):          # each keeps its own documented switch
            self.assertNotIn('enable_thinking', json.dumps(chat_payload(model, [], 4096, thinking=True)))

    def test_the_generator_wrapper_forwards_the_flag_the_shared_signature_cannot_carry(self):
        seen = []

        class Fake:
            base_url = 'https://example.invalid/v1'
            api_key = 'k'

            def chat(self, model, messages, *, max_tokens=2048, thinking=False):
                seen.append(thinking)
                return {'text': '{}'}

        self.chat.ReasoningSwitch(Fake(), True).chat('m', [], max_tokens=9)
        self.chat.ReasoningSwitch(Fake(), False).chat('m', [], max_tokens=9)
        self.assertEqual(seen, [True, False])
        self.assertEqual(self.chat.ReasoningSwitch(Fake(), False).base_url, 'https://example.invalid/v1')

    def test_a_private_or_live_question_needs_neither_retrieval_nor_a_provider_call(self):
        local = self.chat.chat_config()
        local.CHAT_DATA_DIRS, local.CHAT_ALLOW_INGEST = [], False

        class Generator:
            def __init__(self):
                self.asked = []

            def answer(self, question, hits, language=None, use_cache=True,
                        allowed_documents=None):
                self.asked.append(question)
                return {'answer': 'unreachable'}

        generator = Generator()
        with patch('retrieval.retrieve', side_effect=AssertionError('must not retrieve')):
            result = self.chat.ask(local, None, None, generator, "what is my balance today?")
        self.assertEqual(result['status'], 'refused')
        self.assertEqual(result['reason'], 'private_or_live_request')
        self.assertEqual(generator.asked, [])
        self.assertEqual(result['model'], self.chat.CHAT_MODEL)

    def test_chat_commands_are_commands_and_only_questions_are_questions(self):
        import contextlib
        settings = {'top_k': 5, 'show_context': False, 'log': None, 'model': 'm', 'thinking': False,
                    'chunking': 220, 'overlap': 40}
        buffer = io.StringIO()
        lines = [':k 7', '  ', 'quel est le délai', ':show', ':nope', ':quit', 'never asked']
        with contextlib.redirect_stdout(buffer):
            asked = list(self.chat.read_questions(lines, settings))
        self.assertEqual(asked, ['quel est le délai'])
        self.assertEqual(settings['top_k'], 7)
        self.assertTrue(settings['show_context'])
        self.assertIn('unknown command', buffer.getvalue())

    def test_the_default_corpus_covers_the_shipped_documents_not_only_the_samples(self):
        import chat
        dirs = [path.name for path in chat.data_dirs()]
        self.assertIn('docs', dirs)                 # the four real documents
        files = sum(len(list(path.glob('*'))) for path in chat.data_dirs())
        self.assertGreaterEqual(files, 4)
        self.assertEqual([path.name for path in chat.data_dirs([str(Path(chat.__file__).parent)])],
                         ['raglab'])                # --data-dir replaces the list outright

    def test_readiness_report_masks_the_key_and_names_the_fix_for_an_empty_index(self):
        import contextlib
        class Collection:
            def __init__(self, count):
                self._count = count

            def count(self):
                return self._count

            def get(self, include=None, limit=None):
                return {'metadatas': [{'chunk_fp': 'fp-current'}] if self._count else []}

        with patch.dict(os.environ, {'NVIDIA_API_KEY': 'nvapi_SECRETVALUE99'}), \
             patch('store.chunk_fp', return_value='fp-current'), \
             patch('store.collection_languages', return_value=['ar', 'fr']):
            with patch('store.get_collection', return_value=Collection(256)):
                buffer = io.StringIO()
                with contextlib.redirect_stdout(buffer):
                    ready = self.chat.check()
            with patch('store.get_collection', return_value=Collection(0)):
                buffer = io.StringIO()
                with contextlib.redirect_stdout(buffer):
                    empty = self.chat.check()
            with patch('store.get_collection', side_effect=ValueError('sqlite file is locked')):
                buffer = io.StringIO()
                with contextlib.redirect_stdout(buffer):
                    broken = self.chat.check()
        text = buffer.getvalue()
        self.assertEqual((ready, empty, broken), (0, 1, 1))   # a chat with no index is not 'ready'
        self.assertIn('nvapi_SE', text)
        self.assertNotIn('SECRETVALUE', text)                  # reported presence never echoes the key
        self.assertIn('nemotron-3.5-lightning', buffer.getvalue())
        self.assertIn('sqlite file is locked', text)           # reported, not raised out of --check

    def test_the_empty_index_line_points_at_the_ingest_command(self):
        import contextlib
        class Collection:
            def count(self):
                return 0
        buffer = io.StringIO()
        with patch.dict(os.environ, {'NVIDIA_API_KEY': 'nvapi-x'}), \
             patch('store.get_collection', return_value=Collection()), \
             contextlib.redirect_stdout(buffer):
            self.chat.check()
        self.assertIn('--ingest', buffer.getvalue())

    def test_the_chat_indexes_at_the_pinned_chunking_not_the_app_default(self):
        import chat
        size, overlap, source = chat.plan_chunking(PROJECT / 'benchmarks' / 'hard_harness_plan.json')
        self.assertEqual((size, overlap), (640, 40))          # the measured pin, not 220
        self.assertIn('hard_harness_plan.json', source)
        missing = chat.plan_chunking(PROJECT / 'benchmarks' / 'nope.json')
        self.assertEqual(missing[0], cfg_CHUNK)               # documented fallback
        self.assertIn('default', missing[2])
        local = chat.chat_config()
        self.assertEqual((local.CHUNK_SIZE_TOKENS, local.CHUNK_OVERLAP_TOKENS), (640, 40))
        # A separate collection, so the app's index and the chat's index cannot argue over one store.
        self.assertEqual(local.CHROMA_COLLECTION_NAME, 'raglab_chat')

    def test_the_context_ceiling_sizes_from_k_so_a_hit_is_not_quietly_dropped(self):
        import chat
        self.assertEqual(chat.context_budget(5, 640, 40), 5 * 680)
        self.assertEqual(chat.context_budget(1, 220, 40), 3000)   # never below the app floor
        local = chat.chat_config(top_k=5)
        self.assertGreaterEqual(local.ANSWER_CONTEXT_TOKENS, 5 * (640 + 40))

    def test_the_chats_config_object_is_a_config_object_not_just_its_constants(self):
        """The crash that made this test: the chat's settings are a copy of config, and a copy of only
        the UPPERCASE names satisfies every read except the ones that *call* something on cfg — so
        store.py's chunk tagging raised AttributeError mid-ingest, while its embedding-space guard
        (gated on hasattr) had already been skipped silently. Anything passed as `cfg` must carry the
        module's callables too, for every module the chat hands it to.
        """
        import chat
        local = chat.chat_config()
        self.assertEqual(local.active_embedding_model(), EMBED_MODEL)
        for module_name in ('store', 'retrieval', 'chunker', 'answer'):
            module = __import__(module_name)
            source = inspect.getsource(module)
            for attr in sorted(set(re.findall(r'\bcfg\.([A-Za-z_][A-Za-z0-9_]*)', source))):
                self.assertTrue(hasattr(local, attr),
                                f'{module_name}.py reads cfg.{attr}, which the chat config lacks')

    def test_a_missing_callable_is_refused_instead_of_skipping_the_space_check(self):
        import chat
        from types import SimpleNamespace as NS
        broken = NS(**{key: value for key, value in vars(chat.cfg).items()
                       if key.isupper() and not callable(value)})
        self.assertFalse(hasattr(broken, 'active_embedding_model'))
        # store.ensure_fresh_chunks gates its embedding-space check on exactly that hasattr, so a
        # bare-constants namespace must never reach it.
        with self.assertRaises(ValueError) as raised:
            chat.checked_config(broken)
        self.assertIn('active_embedding_model', str(raised.exception))

    def test_a_stale_index_stops_the_chat_with_the_rebuild_command(self):
        class Collection:
            def count(self):
                return 127

            def get(self, include=None, limit=None):
                return {'metadatas': [{'chunk_fp': 'built-with-220'}]}

        local = self.chat.chat_config(top_k=5)
        local.CHAT_DATA_DIRS, local.CHAT_ALLOW_INGEST = [], True
        with patch('store.get_collection', return_value=Collection()), \
             patch('store.ensure_fresh_chunks',
                   side_effect=RuntimeError('[store] collection is STALE')):
            with self.assertRaises(ValueError) as raised:
                self.chat.open_collection(local, SimpleNamespace(batch_size=16))
        self.assertIn('--reset --ingest', str(raised.exception))

    def test_retrieval_is_checked_against_the_chats_settings_not_the_modules(self):
        # retrieve() enforces the chunking fingerprint from the cfg it is handed: passing the module
        # config here let a 220-token index pass a 640-token chat's own settings silently.
        seen = {}

        def fake_retrieve(cfg_arg, embedder, collection, text, **kwargs):
            seen['cfg'] = cfg_arg
            seen.update(kwargs)
            return [], [{'label': 'en(original)', 'lang': 'en', 'text': text}]

        class Generator:
            def answer(self, question, hits, language=None, use_cache=True,
                        allowed_documents=None):
                return {'status': 'refused', 'reason': 'no_context', 'answer': 'no', 'sources': [],
                        'model': language and 'm'}

        local = self.chat.chat_config(top_k=9)
        with patch('retrieval.retrieve', side_effect=fake_retrieve), \
             patch('retrieval.expand_neighbors', side_effect=lambda c, h, radius=0: h):
            self.chat.ask(local, None, None, Generator(), 'what is the minimum capital?',
                          mode='rrf', language='fr', lang_filter='ar')
        self.assertIs(seen['cfg'], local)
        self.assertEqual(seen['top_k'], 9)
        self.assertEqual((seen['mode'], seen['lang_filter'], seen['language']), ('rrf', 'ar', 'fr'))

    def test_a_refusal_shows_what_was_read_and_which_failure_it_was(self):
        result = {'status': 'refused', 'reason': 'insufficient_evidence', 'answer': 'cannot answer',
                  'question': 'can i buy a pc', 'retrieved': 5, 'context_tokens': 3400,
                  'question_language_mismatch': True,
                  'sources': [{'source_id': 'S1', 'document': 'Guide.docx', 'chunk_id': 'a',
                               'heading': 'المرابحة', 'text': 'تمويل شراء سيارة جديدة ' + 'x' * 400}]}
        text = self.chat.format_turn(result)
        self.assertIn('abstained, which is the contract', text)
        self.assertIn('Guide.docx', text)                        # the excerpt the model was handed
        self.assertIn('cross-lingual', text)                      # the language mismatch, named
        self.assertIn('-k 12', text)
        self.assertNotIn('x' * 250, text)                          # previewed, not dumped
        other = self.chat.format_turn({**result, 'reason': 'no_context', 'retrieved': 0, 'sources': []})
        self.assertIn('index is empty', other)
        guard = self.chat.format_turn({**result, 'reason': 'private_or_live_request', 'sources': []})
        self.assertIn('before any model call', guard)

    def test_an_arrow_key_cannot_become_part_of_a_question(self):
        """Without readline, input() hands over the raw CSI bytes of a cursor key and they end up
        embedded, matched against Arabic legal prose and quoted back. So the question that gets answered
        is not the question that was typed - unless control bytes are stripped at the edge."""
        self.assertEqual(self.chat.sanitize('\x1b[C what is murabaha\x1b[?2004h'), 'what is murabaha')
        self.assertEqual(self.chat.sanitize('\x1b]0;window title\x07next'), 'next')
        self.assertEqual(self.chat.sanitize('\x1b[C\x1b[D'), '')
        # Stripping control bytes must not touch the text a user actually means:
        self.assertEqual(self.chat.sanitize('  ما هي المرابحة؟  '), 'ما هي المرابحة؟')
        self.assertEqual(self.chat.sanitize('keep (parens), commas? and 12.5%'),
                         'keep (parens), commas? and 12.5%')

    def test_a_pasted_prompt_is_a_question_and_an_escaped_command_is_a_command(self):
        import contextlib
        settings = {'top_k': 5, 'show_context': False, 'log': None, 'model': 'm', 'thinking': False,
                    'chunking': 640, 'overlap': 40, 'mode': 'vector', 'language': None}
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            asked = list(self.chat.read_questions(['> how do I finance a PC?', '\x1b[C:k 9', ':k'],
                                                 settings))
        self.assertEqual(asked, ['how do I finance a PC?'])
        self.assertEqual(settings['top_k'], 9)          # the escape-prefixed command still ran
        self.assertIn('usage: :k 5', buffer.getvalue())

    def test_an_excerpt_without_a_heading_can_still_be_found_in_the_document(self):
        result = {'status': 'refused', 'reason': 'insufficient_evidence', 'answer': 'cannot answer',
                  'question': 'q', 'retrieved': 1, 'context_tokens': 3400,
                  'sources': [{'source_id': 'S1', 'document': 'Loi_2016-48.pdf', 'heading': '',
                               'chunk_id': 'Loi_2016-48.pdf::chunk_067', 'text': 'نص القانون' * 20}]}
        text = self.chat.format_turn(result)
        self.assertIn('Loi_2016-48.pdf::chunk_067', text)   # an empty heading is not an anonymous chunk

    def test_the_repl_understands_language_and_mode_commands(self):
        import contextlib
        settings = {'top_k': 5, 'show_context': False, 'log': None, 'model': 'm', 'thinking': False,
                    'chunking': 640, 'overlap': 40, 'mode': 'vector', 'language': None}
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            asked = list(self.chat.read_questions([':lang ar', ':mode', ':mode hybrid', ':mode rrf',
                                                   ':lang xx', 'ما هو'], settings))
        self.assertEqual(asked, ['ما هو'])
        self.assertEqual((settings['language'], settings['mode']), ('ar', 'rrf'))
        printed = buffer.getvalue()
        self.assertIn('unmeasured', printed)          # an unmeasured arm is allowed but labelled
        self.assertIn('usage: :mode', printed)
        self.assertIn(':mode takes vector, rrf or blend', printed)
        self.assertIn(':lang takes ar, fr, en or auto', printed)
        self.assertNotIn('xx\n> ', printed)           # a bad value is never asked as a question

    def test_check_reports_and_ingest_builds_and_together_do_both(self):
        # The reported loop: --check printed "run ./raglab/chat.sh --ingest --check", and that command
        # reported and exited before building anything. Check-with-ingest now builds, then reports.
        calls = []

        class FakeCollection:
            def __init__(self, count):
                self._count = count

            def count(self):
                return self._count

        class Embedder:
            provider_name, model, batch_size, api_calls, cache_hits = 'nvidia', 'emb', 16, 3, 0

            def embed_texts(self, texts):
                return [[0.0] * 4 for _ in texts]

        with patch.dict(os.environ, {'NVIDIA_API_KEY': 'nvapi-x'}), \
             patch('embedder.build_embedder', return_value=Embedder()), \
             patch('chat.open_collection', side_effect=lambda *a, **k: calls.append(k) or FakeCollection(127)), \
             patch('chat.build_generator', side_effect=AssertionError('must not build a client')), \
             patch('chat.check', return_value=0) as reported, \
             patch('store.collection_languages', return_value=['ar', 'fr', 'en']):
            self.assertEqual(self.chat.main(['--check', '--ingest']), 0)
        self.assertEqual(len(calls), 1)                    # it built
        # The report is printed once, after the build: the state a user acts on has to be the state the
        # index is actually in, and open_collection already printed the cost before embedding.
        self.assertEqual(reported.call_count, 1)

        calls.clear()
        with patch('chat.check', return_value=1) as reported:
            self.assertEqual(self.chat.main(['--check']), 1)
        self.assertEqual(calls, [])
        self.assertEqual(reported.call_count, 1)           # report only: no index, no embedding

    def test_the_not_ready_line_offers_a_command_that_really_builds(self):
        import contextlib

        class Collection:
            def count(self):
                return 0

            def get(self, include=None, limit=None):
                return {'metadatas': []}

        buffer = io.StringIO()
        with patch.dict(os.environ, {'NVIDIA_API_KEY': 'nvapi-x'}), \
             patch('store.get_collection', return_value=Collection()), \
             contextlib.redirect_stdout(buffer):
            self.chat.check()
        text = buffer.getvalue()
        self.assertIn('./raglab/chat.sh --ingest', text)
        self.assertNotIn('--ingest --check\n', text)        # the command that only re-reported itself

    def test_one_offline_turn_satisfies_the_citation_contract_on_this_model(self):
        from answer import AnswerGenerator
        local = self.chat.chat_config(cache_path=self.path / 'answers.json')
        quote = 'Le capital minimal est fixé à vingt millions de dinars.'
        reply = json.dumps({'answerable': True, 'claims': [
            {'text': 'Le minimum est de vingt millions de dinars.',
             'evidence': [{'source_id': 'S1', 'quote': quote}]}]})
        body = {'model': local.ANSWER_MODEL, 'usage': {},
                'choices': [{'message': {'content': reply}, 'finish_reason': 'stop'}]}
        sent = []

        class Opener:
            def open(self, request, timeout=None):
                sent.append(json.loads(request.data.decode()))
                return Response(body)

        client = self.chat.ReasoningSwitch(
            NvidiaClient(api_key='nvapi-x', min_interval=0, opener=Opener()), False)
        generator = AnswerGenerator(local, client=client, approved_models=(local.ANSWER_MODEL,))
        hits = [{'id': 'c1', 'text': quote, 'metadata': {'document': 'loi-2016-48.pdf'}}]
        result = generator.answer('Quel est le capital minimal ?', hits, 'fr')
        self.assertEqual(result['status'], 'answered', result.get('raw_preview') or result.get('error'))
        self.assertEqual(sent[0]['model'], self.chat.CHAT_MODEL)
        self.assertEqual(sent[0]['chat_template_kwargs'], {'enable_thinking': False})
        self.assertIn('[S1]', result['answer'])
        printed = self.chat.format_turn({**result, 'sources': [{'source_id': 'S1', 'document':
                                                                'loi-2016-48.pdf', 'chunk_id': 'c1',
                                                                'text': quote}]})
        self.assertIn(quote, printed)                # the verbatim quote is shown under the claim
        self.assertEqual(json.loads((self.path / 'answers.json').read_text()) and 1, 1)  # cached for re-reads


class ManualChunkMaps(unittest.TestCase):
    """Chunk boundaries as a reviewed artifact, and the guards that make that safe to run on.

    These are the argument that the feature is more than a formatter: a map has to reproduce the document
    exactly (no gap, no silent rewording), stay tied to the text it was written against, and change the
    corpus identity - because retrieval, the citation contract and any published score all depend on
    which segmentation produced them.
    """

    TEXT = ('# نظام التمويل\n\nالفصل 1 : تعريف المرابحة وهي بيع السلعة بثمن الشراء زائدا الربح.\n\n'
            'الفصل 2 : يحدد البنك نسبة الربح according to the tenor chosen by the client.\n\n'
            '## الإجراءات\n\nالفصل 3 : يستوجب الملف جواز سفر وأذن بالأجرة.\n\n'
            'الفصل 4 : يسلم العقد بعد أخذ موافقة اللجنة.\n')
    DOC = {'text': TEXT, 'source': 'docs/manual_test.md', 'language': 'ar'}

    @staticmethod
    def _map(*pairs):
        """A map from (start, end) pairs. No 'tokens' key on purpose: that field is drafted metadata, and
        claiming a count the text does not hold is exactly what the drift check exists to catch."""
        return [{'start': a, 'end': b} for a, b in pairs]

    def cfg(self, directory, **extra):
        values = {'CHUNKING_MODE': 'manual', 'CHUNK_MAP_DIR': Path(directory),
                  'CHUNK_MAX_TOKENS': 900, 'CHUNK_SIZE_TOKENS': 640, 'CHUNK_OVERLAP_TOKENS': 40,
                  'SPLIT_ON_HEADINGS_FIRST': True, 'CHUNK_OVERLAP_SENTENCE_AWARE': True}
        values.update(extra)
        return SimpleNamespace(**values)

    def test_map_round_trip_reproduces_the_document(self):
        entries = sc.propose(dict(self.DOC), target_tokens=80, max_tokens=200, min_tokens=20)
        self.assertGreaterEqual(len(entries), 2)
        # Structure only, on purpose: which token counter is configured decides sizes, not whether the
        # map is a partition of the document, and a test must not fail for the second reason because of
        # the first (that is exactly the trap these maps would fall into between a laptop and CI).
        # The message carries the boundaries and per-block counts so a machine whose counter differs
        # (estimator vs cl100k_base) reports the shape it built, not just the rule name.
        diag = [(e['start'], e['end'], sc.count_tokens(self.TEXT[e['start']:e['end']]))
                for e in entries]
        self.assertEqual([], sc.validate(self.TEXT, entries, max_tokens=10 ** 6, min_tokens=1),
                         f'entries={diag} tokenizer={sc.tokenizer_identity()}')
        joined = ''.join(self.TEXT[e['start']:e['end']] for e in entries)
        self.assertEqual(' '.join(self.TEXT.split()), ' '.join(joined.split()),
                         'the chunks must be the document, not a summary of it')
        chunks = sc.chunks_from_map(dict(self.DOC), entries, self.cfg('.'))
        for chunk in chunks:
            self.assertIn(' '.join(chunk.text.split()), ' '.join(self.TEXT.split()))

    def test_validate_finds_a_gap_and_a_reworded_chunk(self):
        """Boundaries written by hand, so the assertion is about the rule and not about the proposer's
        arithmetic: which pieces a token counter would cut is a different question from whether a map may
        leave a hole."""
        a = self.TEXT.index('الفصل 2')
        b = self.TEXT.index('## الإجراءات')
        whole = len(self.TEXT)
        pieces = self._map((0, a), (a, b), (b, whole))
        # hard=False: this fixture's last piece holds a section heading and both of its articles, which
        # is the other rule's business. Coverage is what is under test here.
        self.assertEqual([], sc.validate(self.TEXT, pieces, max_tokens=10 ** 6, min_tokens=1, hard=False))
        holed = [dict(e) for e in pieces]
        holed[0]['end'] = holed[0]['end'] - 20                  # characters in no chunk at all
        self.assertTrue(any('gap' in problem or 'no chunk' in problem
                            for problem in sc.validate(self.TEXT, holed, max_tokens=10 ** 6,
                                                        min_tokens=1, hard=False)),
                        sc.validate(self.TEXT, holed, max_tokens=10 ** 6, min_tokens=1))
        clipped = [dict(e) for e in pieces]
        clipped[0]['start'] = 20                                # the document's opening line dropped
        self.assertTrue(any('belong to no chunk' in problem
                            for problem in sc.validate(self.TEXT, clipped, max_tokens=10 ** 6,
                                                       min_tokens=1, hard=False)))
        truncated = [dict(e) for e in pieces]
        truncated[-1]['end'] = whole - 20                        # and its closing one
        self.assertTrue(any('after it' in problem
                           for problem in sc.validate(self.TEXT, truncated, max_tokens=10 ** 6,
                                                      min_tokens=1, hard=False)))
        invented = [dict(e) for e in pieces]
        invented[1]['end'] = invented[1]['end'] + 5              # overlapping, i.e. not a partition
        self.assertTrue(sc.validate(self.TEXT, invented, max_tokens=10 ** 6, min_tokens=1,
                                            hard=False))

    def test_a_map_refuses_a_changed_document(self):
        with tempfile.TemporaryDirectory() as tmp:
            entries = sc.propose(dict(self.DOC), target_tokens=80, max_tokens=200, min_tokens=20)
            path = sc.map_path(self.DOC['source'], tmp)
            sc.write_map(path, dict(self.DOC), entries)
            edited = dict(self.DOC, text=self.TEXT.replace('جواز سفر', 'بطاقة تعريف'))
            with self.assertRaises(ValueError) as caught:
                sc.load_map(path, edited)
            self.assertIn('offsets', str(caught.exception))   # says why, not just 'mismatch'

    def test_maps_are_part_of_the_corpus_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            plain = chunker.chunk_fingerprint(640, 40, True, True)
            with_maps = chunker.chunk_fingerprint(640, 40, True, True, maps='abc')
            self.assertNotEqual(plain, with_maps)
            entries = sc.propose(dict(self.DOC), target_tokens=80, max_tokens=200, min_tokens=20)
            sc.write_map(sc.map_path(self.DOC['source'], tmp), dict(self.DOC), entries)
            cfg = self.cfg(tmp)
            first = chunk_fp(cfg)
            (Path(tmp) / sc.map_path(self.DOC['source'], tmp).name).write_text(
                json.loads(sc.map_path(self.DOC['source'], tmp).read_text()).__class__ and
                json.dumps({**json.loads(sc.map_path(self.DOC['source'], tmp).read_text()),
                            'chunks': [{**json.loads(sc.map_path(self.DOC['source'], tmp).read_text())
                                       ['chunks'][0], 'end': 10}]}), encoding='utf-8')
            self.assertNotEqual(first, chunk_fp(cfg),
                                'moving one boundary must look like the new corpus version it is')

    def test_manual_mode_needs_a_map_for_every_document(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError) as caught:
                sc.chunk_documents([dict(self.DOC)], self.cfg(tmp))
            self.assertIn('manual_test.md', str(caught.exception))
            loose, _problems = sc.chunk_documents([dict(self.DOC)], self.cfg(tmp), strict=False)
            self.assertEqual([], loose)             # non-strict reports the gap instead of inventing one

    def test_soft_mode_packs_short_articles_and_hard_mode_does_not(self):
        """One map, two verdicts. Boundaries are given rather than proposed, so no token counter decides
        whether the test runs, and every piece but the deliberate merge is already article-aligned, so
        nothing except the mode is at stake."""
        tail = len(self.TEXT)
        marks = [m.start() for m in sc.ARTICLE.finditer(self.TEXT)]
        split = self._map((0, marks[0]), *[(x, y) for x, y in zip(marks, marks[1:])],
                          (marks[-1], tail))
        both = self._map((0, marks[2]), *[(x, y) for x, y in zip(marks[2:], marks[3:])],
                         (marks[-1], tail))          # the first chunk holds الفصل 1 and الفصل 2 together
        self.assertEqual([], sc.validate(self.TEXT, split, max_tokens=10 ** 6, min_tokens=1, hard=True))
        hard = sc.validate(self.TEXT, both, max_tokens=10 ** 6, min_tokens=1, hard=True)
        self.assertTrue(any('numbered articles' in problem for problem in hard), hard)
        self.assertEqual([], sc.validate(self.TEXT, both, max_tokens=10 ** 6, min_tokens=1, hard=False))
        entries = sc.propose(dict(self.DOC), target_tokens=200, max_tokens=10 ** 6, min_tokens=1)
        soft_entries = sc.propose(dict(self.DOC), target_tokens=200, max_tokens=10 ** 6, min_tokens=1,
                                  hard_subjects=False)
        self.assertLessEqual(len(soft_entries), len(entries),
                             'soft mode may only ever pack further, never cut deeper')

    def test_chat_runs_on_maps_in_its_own_collection(self):
        with tempfile.TemporaryDirectory() as tmp:
            local = chat.chat_config(chat.CHAT_MODEL, chunking='manual', map_dir=tmp,
                                     cache_path=str(Path(tmp) / 'answers.json'))
            self.assertEqual('manual', local.CHUNKING_MODE)
            self.assertEqual('raglab_chat_manual', local.CHROMA_COLLECTION_NAME)
            self.assertIn('manual maps', local.CHUNK_SOURCE)
            size = chat.chat_config(chat.CHAT_MODEL, cache_path=str(Path(tmp) / 'a2.json'))
            self.assertEqual('raglab_chat', size.CHROMA_COLLECTION_NAME,
                             'the two segmentations must not share an index')



class GazetteAndDominanceRepair(unittest.TestCase):
    """The 2026-09-28 corpus-defect treatments (owner decision, diff-reviewed).

    Stage-1 fixes grounded in the Loi audit findings and in the owner's
    review of the first repair diff (whole-line flipping was rejected as
    insufficient — the extraction scrambles each PHYSICAL rendered line as
    zones: pure-Arabic runs reversed, digit runs kept in logical order):
    (1) gazette running headers are stripped as a whole line-initial SPAN;
    (2) visual-order documents (document-dominance gate) get each rendered
        line zone-reconstructed: Arabic runs un-reversed, digit islands kept,
        letters fused to numbers split BEFORE zoning, marker prefixes and
        sentence-terminal punctuation preserved, the law-49 tail excluded.
    The full before/after diff for (2) is owner-reviewed in
    raglab/audits/Loi_2016-48_repair_diff.md before adoption.
    """

    def test_gazette_span_stripped_whole_from_fused_line(self):
        import restructure as rst
        for fused, remainder in (
            ("صفحة2518 للجم الرسمي الرائد التونسية هورية–– 15 جويلية 2016 عدد58 "
             "الفصل14 كل القانون هذا", "الفصل14"),
            ("عدد58 التونسية للجمهورية الرسمي الرائد–– 15 جويلية 2016 صفحة2519 "
             "لفائدة بالعمليات", "لفائدة"),
        ):
            report = rst.RestructureReport(name="t")
            out = rst._strip_gazette_header(fused, report)
            self.assertEqual(1, report.gazette_headers_removed)
            self.assertTrue(out.startswith(remainder), out[:60])
            self.assertNotIn("الرائد", out)
            self.assertNotIn("هورية", out)
        report = rst.RestructureReport(name="t")
        out = rst._strip_gazette_header(
            "صفحة2516 للجم الرسمي الرائد التونسية هورية–– 15 جويلية 2016 عدد58",
            report)
        self.assertEqual("", out)
        self.assertEqual(1, report.gazette_headers_removed)

    def test_gazette_citations_in_body_text_are_kept(self):
        import restructure as rst
        body = ("الرائد الرسمي للجمهورية التونسية وينفذ كقانون من قوانين الدولة "
                "ويقرأ في الجلسة العامة")
        report = rst.RestructureReport(name="t")
        self.assertEqual(body, rst._strip_gazette_header(body, report))
        citation = ("عدد 48 لسنة 2016 المتعلق بالبنوك والمؤسسات المالية ينشر "
                    "بالعديد الرسمي")
        self.assertEqual(citation, rst._strip_gazette_header(citation, report))
        self.assertEqual(0, report.gazette_headers_removed)

    def test_zone_reconstruction_unreverses_arabic_keeps_digits(self):
        import restructure as rst
        # art. 194 of Loi 2016-48, verbatim stored form (zones model):
        line = ("الفصول46و 47و 51و 52و 57 و58 اجل في القانون هذا من")
        out = rst._reconstruct_line_visual_order(line)
        self.assertEqual(
            "الفصول 46 و 47 و 51 و 52 و 57 و 58 من هذا القانون في اجل", out)
        # a word fused with its number splits BEFORE zoning so each part
        # lands in its own zone (art. 2 area):
        out = rst._reconstruct_line_visual_order(
            "عدد بالقانون الصادرة64 لسنة 2009 في المؤرخ 12 أوت")
        self.assertEqual(
            "الصادرة بالقانون عدد 64 لسنة 2009 المؤرخ في 12 أوت", out)

    def test_zone_reconstruction_preserves_marker_and_period(self):
        import restructure as rst
        out = rst._reconstruct_line_visual_order(
            "الفصل194 ـ تمارس التي المالية والمؤسسات البنوك على")
        self.assertTrue(out.startswith("الفصل194"), out)
        self.assertIn("على البنوك والمؤسسات المالية التي تمارس", out)
        # sentence-terminal punctuation is re-emitted at the raw line's end
        # and therefore lands at the TRUE end of the reconstructed line
        out = rst._reconstruct_line_visual_order("حكومي بأمر.")
        self.assertEqual("بأمر حكومي .", out)

    def test_dominance_gate_reconstructs_visual_order_document(self):
        import restructure as rst
        tails = ["سنويا", "شهريا", "دوريا", "فصليا", "دائما", "كليا", "جزئيا",
                 "تدريجيا", "نهائيا", "مؤقتا", "فوريا", "لاحقا", "حاليا",
                 "مستقبلا", "مباشرة", "استثنائيا", "طوارئ", "انتقاليا",
                 "تجريبيا", "استراتيجيا", "موسميا", "متكررا", "فرديا", "جماعيا"]
        # 24 reversed rendered lines + digit enumeration lines: a document the
        # pre-scan must classify as visual-order
        lines = [" ".join(reversed(
            f"يتم التمويل على اساس المرابحة بصفة متجددة في الغرض {tail}".split()))
            for tail in tails]
        lines.append("الفصول46و 47و 51و 52و 57 و58 اجل في القانون هذا من")
        text = "\n".join(lines)
        markdown, report = rst.normalize_structure(
            {"text": text, "name": "visual"}, repair_rtl=True)
        self.assertTrue(report.rtl_doc_dominant)
        self.assertGreaterEqual(report.rtl_lines_reconstructed, 20)
        self.assertIn("الفصول 46 و 47 و 51 و 52 و 57 و 58 من هذا القانون في اجل",
                      markdown)
        self.assertIn("يتم التمويل على اساس المرابحة", markdown)

    def test_logical_document_left_untouched(self):
        import restructure as rst
        tails = ["سنويا", "شهريا", "دوريا", "فصليا", "دائما", "كليا", "جزئيا",
                 "تدريجيا", "نهائيا", "مؤقتا", "فوريا", "لاحقا", "حاليا",
                 "مستقبلا", "مباشرة", "استثنائيا", "طوارئ", "انتقاليا",
                 "تجريبيا", "استراتيجيا", "موسميا", "متكررا", "فرديا", "جماعيا"]
        lines = [f"يتم التمويل على اساس المرابحة بصفة متجددة في الغرض {tail}"
                 for tail in tails]
        text = "\n".join(lines)
        markdown, report = rst.normalize_structure(
            {"text": text, "name": "logical"}, repair_rtl=True)
        self.assertFalse(report.rtl_doc_dominant)
        self.assertEqual(0, report.rtl_lines_reconstructed)
        self.assertIn("يتم التمويل على اساس المرابحة بصفة متجددة", markdown)

    def test_reconstruct_stops_at_signature_tail(self):
        import restructure as rst
        # everything after law 48's signature line stays in stored form
        text = ("شأنها من بأعمال القيام يمكنه لا الحالات كل وفي\n"
                "السبسي قايد الباجي محمد\n"
                "ايتعلق اتفاق على بالموافقة لقرض بتاريخ المبرم30 مارس")
        report = rst.RestructureReport(name="t")
        lines, _ = rst._prepare_lines(text, report, reconstruct=True)
        joined = "\n".join(lines)
        self.assertIn("وفي كل الحالات لا يمكنه القيام بأعمال من شأنها", joined)
        self.assertIn("محمد الباجي قايد السبسي", joined)
        self.assertIn("ايتعلق اتفاق على بالموافقة لقرض بتاريخ المبرم30 مارس", joined)


class AdoptedCodexRepair(unittest.TestCase):
    """The 2026-09-30 owner adoption of the corrected codices (all four docs).

    With RTL repair enabled, `restructure.py` replaces a document's stored
    text with its language-model-repaired codex in `audits/*_corrected.md`
    (Loi: 215 entries / 29 batches; Circulaire: 35; Guide: 23; Madkhal: 21 —
    each machine-checked by `audits/llm_repair_check.py`). This class pins
    the adoption contract: header stripped, loader-canonical form (PDFs keep
    rendered lines, DOCX reflow), block boundaries restored around
    marker-initial lines, each real document adopting its codex with the
    documented fixes visible in the markdown, and RESTRUCTURE_RTL_REPAIR=0
    still giving the raw stored arm.
    """

    def test_unknown_document_has_no_codex(self):
        import restructure as rst
        self.assertIsNone(rst._adopted_codex_text("Atlas_fiche.pdf"))
        self.assertIsNone(rst._adopted_codex_text(""))

    def test_codex_text_is_header_stripped_and_boundary_restored(self):
        import restructure as rst
        text = rst._adopted_codex_text("Loi_2016-48.pdf")
        self.assertIsNotNone(text)
        lines = text.split("\n")
        # header gone: the law text starts at the law's own title line
        self.assertTrue(lines[0].startswith("قانون عدد 48 لسنة 2016"),
                        lines[0][:60])
        self.assertNotIn("---", lines)
        # first line is its own block (title pick needs it as a line)
        self.assertEqual(lines[1], "")
        # every marker-initial line starts a block; 231 true structural
        # headings live in this codex (198 فصل + 33 عنوان/باب/قسم)
        marker_lines = 0
        for i, l in enumerate(lines):
            s = l.strip()
            if not s:
                continue
            if rst._MARKER_RE.match(s):
                marker_lines += 1
                self.assertTrue(
                    i == 0 or not lines[i - 1].strip(),
                    f"marker line not block-initial at {i}: {s[:40]}")
        self.assertEqual(marker_lines, 231)

    def test_real_documents_adopt_their_codices(self):
        import restructure as rst
        from loader import load_document
        docs_dir = Path(__file__).resolve().parent.parent / "docs"
        # (name, canaries that prove the documented fixes landed, n_h2, n_h3)
        expected = {
            "Loi_2016-48.pdf": (
                ["يتم الطعن بالاستئناف في الحكم الصادر"], 33, 198),
            "Circulaire_BCT_2019-08.pdf": (
                ["عشرة ايام عمل", "عدد 89 لسنة 1994",
                 "تعتزم تسويقها"], 4, 20),
            "Guide_Interne_Operations_Bancaires_Islamiques.docx": (
                ["4.2- توظيف الودائع على اساس الوكالة بالاستثمار",
                 "2.1- عملية التمويل بصيغة المرابحة"], 9, 15),
            "Madkhal_Sayrafa_Islamiya.docx": (
                ["(59 000)", "5- اهم منتجات الصيرفة الاسلامية"], 5, 12),
        }
        for name, (canaries, n_h2, n_h3) in expected.items():
            pdf = docs_dir / name
            if not pdf.is_file():
                self.skipTest(f"docs/{name} not present")
            doc = load_document(pdf, origin="docs/")
            md, report = rst.normalize_structure(doc, repair_rtl=True)
            self.assertTrue(report.codex_adopted, name)
            self.assertFalse(report.rtl_doc_dominant, name)  # logical order
            self.assertEqual(len(re.findall(r"^## ", md, re.M)), n_h2, name)
            self.assertEqual(len(re.findall(r"^### ", md, re.M)), n_h3, name)
            for probe in canaries:
                self.assertIn(probe, md, f"{name}: {probe}")

    def test_repair_disabled_keeps_the_stored_arm(self):
        import restructure as rst
        doc = {"name": "Loi_2016-48.pdf", "text": "نص مخزون خام"}
        md, report = rst.normalize_structure(doc, repair_rtl=False)
        self.assertFalse(report.codex_adopted)
        self.assertIn("نص مخزون خام", md)


class Phase2Machinery(unittest.TestCase):
    """Step 2.4 machinery re-introductions (owner: «طيب واصل المرحلة الثانية»,
    2026-10-01) — each sub-step individually reviewed, per the 2026-09-28
    decision that the Phase-2 machinery comes back one declared step at a time.

    2.4-أ: the five target-state categories join VALID_CATEGORIES (the four
    core categories unchanged); ambiguous cases must DECLARE their material
    ambiguity (ambiguity_note) — an undeclared ambiguity cannot be graded.
    """

    def _write_set(self, tmpdir, cases):
        import json as _json
        p = Path(tmpdir) / "q.json"
        p.write_text(_json.dumps({"cases": cases}, ensure_ascii=False), encoding="utf-8")
        return p

    def test_five_target_categories_are_valid(self):
        import evaluate as ev
        for cat in ("colloquial", "synonyms", "implicit", "compound", "ambiguous"):
            self.assertIn(cat, ev.VALID_CATEGORIES)
        for cat in ("verbatim", "paraphrase", "cross-lingual", "out-of-scope"):
            self.assertIn(cat, ev.VALID_CATEGORIES)
        self.assertNotIn("target-state", ev.VALID_CATEGORIES)

    def test_ambiguous_case_requires_declared_ambiguity(self):
        import evaluate as ev
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            ok = self._write_set(tmp, [dict(
                id="a1", question="كم تكلفة البطاقة؟", language="ar",
                category="ambiguous", expected_document="Loi_2016-48.pdf",
                expected_lang="ar", expected_substring="البطاقات",
                ambiguity_note="الغموض المؤثر: أي بطاقة؛ الإجابة الجيدة تطلب التحديد")])
            with patch("builtins.print"):  # silence the loader's info line
                ev.load_question_set(ok)   # no exception, no warning needed
            bare = self._write_set(tmp, [dict(
                id="a2", question="كم تكلفة البطاقة؟", language="ar",
                category="ambiguous", expected_document="Loi_2016-48.pdf",
                expected_lang="ar", expected_substring="البطاقات")])
            with patch("builtins.print") as pr:
                ev.load_question_set(bare)
            self.assertIn("ambiguity_note", "\n".join(str(a) for a in pr.call_args_list))

    def test_current_50_case_set_still_loads_clean(self):
        import evaluate as ev
        with patch("builtins.print"):
            cases = ev.load_question_set(Path(__file__).parent / "questions_50.json")
        self.assertEqual(len(cases), 50)

    def _hit(self, rank, text, doc="Loi_2016-48.pdf"):
        return {"rank": rank, "id": f"c{rank}", "text": text,
                "metadata": {"document": doc}}

    def test_requirement_completion_rank_semantics(self):
        import evaluate as ev
        case = {"expected_document": "Loi_2016-48.pdf",
                "expected_substrings": ["هامش ربح محدد مسبقا", "على اقساط معلومة"]}
        hits = [self._hit(1, "نص لا يحمل شيئا من المطالب"),
                self._hit(2, "الصيغة تتضمن هامش ربح محدد مسبقا فقط"),
                self._hit(3, "بعض النص"),
                self._hit(4, "ويتم تسديده على اقساط معلومة")]
        # ALL requirements are only covered once rank 4 is reached
        self.assertEqual(ev.requirement_completion_rank(case, hits), 4)
        cov = ev.requirement_coverage(case, [h["text"] for h in hits])
        self.assertEqual((cov["found"], cov["total"]), (2, 2))
        self.assertEqual(cov["missing"], [])
        # Uncovered set -> None (hit@k false at every k)
        self.assertIsNone(ev.requirement_completion_rank(
            case, hits[:3]))
        # Document scoping: same texts but wrong document -> never completes
        wrong_doc = [self._hit(1, "هامش ربح محدد مسبقا", doc="other.pdf"),
                     self._hit(2, "على اقساط معلومة", doc="other.pdf")]
        self.assertIsNone(ev.requirement_completion_rank(case, wrong_doc))
        # Cases without expected_substrings are untouched
        self.assertIsNone(ev.requirement_completion_rank(
            {"expected_substring": "x"}, hits))
        self.assertIsNone(ev.requirement_coverage(
            {"expected_substring": "x"}, [t["text"] for t in hits]))

    def test_expected_substrings_validation(self):
        import evaluate as ev
        import json as _json
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "q.json"
            good = {"cases": [dict(
                id="i1", question="ما الوثائق والشروط؟", language="ar",
                category="implicit", expected_document="Loi_2016-48.pdf",
                expected_lang="ar",
                expected_substrings=["عقد مضاربة", "عقد وكالة"])]}
            p.write_text(_json.dumps(good, ensure_ascii=False), encoding="utf-8")
            with patch("builtins.print"):
                ev.load_question_set(p)      # clean: no warnings expected
            bad = {"cases": [dict(
                id="i2", question="سؤال", language="ar", category="implicit",
                expected_document="Loi_2016-48.pdf", expected_lang="ar",
                expected_substrings=[])]}
            p.write_text(_json.dumps(bad, ensure_ascii=False), encoding="utf-8")
            with patch("builtins.print") as pr:
                ev.load_question_set(p)
            self.assertIn("expected_substrings", "\n".join(str(a) for a in pr.call_args_list))

    def test_answer_context_sources_simulates_budget(self):
        import evaluate as ev
        hits = [{"rank": i, "id": f"c{i}", "text": "نص " * 50,
                 "metadata": {"document": "d"}}
                for i in range(1, 30)]
        cfg = SimpleNamespace(ANSWER_CONTEXT_TOKENS=300)
        sources = ev.answer_context_sources(hits, cfg)
        self.assertLess(len(sources), len(hits))   # the budget drops sources

    def test_compute_metrics_requirements_and_context_blocks(self):
        import evaluate as ev

        def row(**extra):
            base = {"id": "q1", "question": "س", "category": "verbatim",
                    "language": "ar", "evaluable": True,
                    "hit_at_1": True, "hit_at_3": True, "hit_at_5": True,
                    "is_out_of_scope": False,
                    "hits": [{"id": "c1", "score": 0.5, "text": "نص"}],
                    "correct_rank": 1, "correct_score": 0.5, "correct_id": "c1"}
            base.update(extra)
            return base

        # legacy rows (no multi/context fields): both blocks stay None
        out = ev.compute_metrics([row()])
        self.assertIsNone(out["requirements"])
        self.assertIsNone(out["context"])
        # multi rows populate the requirements block; context stays None
        multi = row(category="implicit", correct_rank=2,
                    requirements_found=2, requirements_total=2,
                    requirements_missing=[],
                    requirements_all_in_context=True)
        out2 = ev.compute_metrics([multi])
        self.assertEqual(out2["requirements"]["n"], 1)
        self.assertEqual(out2["requirements"]["all_requirements_top_k"], 1.0)
        self.assertEqual(out2["requirements"]["all_requirements_in_context"], 1.0)
        self.assertIsNone(out2["context"])
        # a single-requirement row with evidence_in_context populates context
        out3 = ev.compute_metrics([row(evidence_in_context=True)])
        self.assertEqual(out3["context"]["n"], 1)
        self.assertEqual(out3["context"]["evidence_in_context"], 1.0)

    def test_print_report_handles_sets_without_out_of_scope(self):
        """Regression (live run 36838058003): a question set with ZERO
        out-of-scope cases (the Phase-3 target set) crashed print_report on
        None max_top1_score. The report must print an explicit absence."""
        import io
        import evaluate as ev

        def row(**extra):
            base = {"id": "t1", "question": "س", "category": "colloquial",
                    "language": "ar", "evaluable": True,
                    "hit_at_1": True, "hit_at_3": True, "hit_at_5": True,
                    "is_out_of_scope": False,
                    "hits": [{"id": "c1", "score": 0.5, "text": "نص"}],
                    "correct_rank": 1, "correct_score": 0.5, "correct_id": "c1"}
            base.update(extra)
            return base

        rows = [row(), row(id="t2", hit_at_1=False, hit_at_3=True,
                           correct_rank=2, correct_score=0.4)]
        run = {
            "generated_at": "2026-10-01T00:00:00+00:00",
            "config": {"provider": "fake", "embedding_model": "fake",
                       "chunk_size_tokens": 640, "chunk_overlap_tokens": 40,
                       "split_on_headings_first": True, "retrieval_top_k": 20,
                       "hybrid": False, "retrieval_mode": "vector"},
            "questions": rows,
            "metrics": ev.compute_metrics(rows),
        }
        self.assertEqual(run["metrics"]["out_of_scope"]["n"], 0)
        buf = io.StringIO()
        with patch("sys.stdout", buf):
            ev.print_report(run)          # must not raise
        self.assertIn("no out-of-scope questions", buf.getvalue())


class ModelIndependenceAB(unittest.TestCase):
    """answer_ab.run_comparison (step 2.4-ج): fixed retrieval, swapped answer
    models — one retrieval per question, identical hits to every arm, the
    citation gate judging each arm."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        self.cfg = make_config(CHROMA_DIR=self.path / 'chroma',
                               ANSWER_CACHE_PATH=self.path / 'answers.json')

    def _collection(self):
        def chunk(index, source, language, text):
            return SimpleNamespace(index=index, source=source, language=language,
                                   text=text, heading='', origin='test/',
                                   section_type='content', token_count=20)
        chunks = [chunk(0, 'ar.md', 'ar', 'تاسس بنك البركة في 1983.'),
                  chunk(0, 'fr.md', 'fr', 'La banque Baraka a été fondée en 1983.')]
        vectors = [[1.0] + [0.0] * 2047, [0.0, 1.0] + [0.0] * 2046]
        collection = get_collection(self.cfg, reset=True)
        store_chunks(collection, list(zip(chunks, vectors)), self.cfg)
        return collection, vectors

    def test_jaccard_edge_cases(self):
        from answer_ab import _jaccard
        self.assertIsNone(_jaccard([], []))
        self.assertEqual(_jaccard(['a'], ['a']), 1.0)
        self.assertEqual(_jaccard(['a'], ['b']), 0.0)
        self.assertAlmostEqual(_jaccard(['a', 'b'], ['b', 'c']), 1 / 3)

    def test_same_hits_all_arms_and_divergence_detection(self):
        from answer_ab import run_comparison
        collection, vectors = self._collection()
        fake_embedder = SimpleNamespace(embed_query=lambda text: vectors[0])

        class FakeGen:
            def __init__(self, status, reason='insufficient_evidence'):
                self.status, self.reason = status, reason
                self.seen = []

            def answer(self, question, hits, language, use_cache=True):
                self.seen.append([h['id'] for h in hits])
                if self.status == 'answered':
                    return {'status': 'answered', 'reason': None, 'model': 'fake',
                            'claims': [{'text': 'x'}],
                            'sources': [{'chunk_id': hits[0]['id']}],
                            'validation_ok': True}
                return {'status': self.status, 'reason': self.reason, 'model': 'fake',
                        'claims': [], 'sources': [], 'validation_ok': True}

        a, b = FakeGen('answered'), FakeGen('refused')
        cases = [{'id': 'q1', 'question': 'بنك البركة', 'language': 'ar',
                  'category': 'verbatim'}]
        report = run_comparison(self.cfg, fake_embedder, collection, cases,
                                [('ref', a), ('alt', b)], top_k=2, mode='vector')
        # both arms received the IDENTICAL hit ids (model independence)
        self.assertEqual(a.seen, b.seen)
        self.assertTrue(a.seen and a.seen[0])
        # divergence detected: statuses differ -> substance-divergent question
        pair = report['pairwise']['ref vs alt']
        self.assertEqual(pair['status_agreement'], 0.0)
        self.assertIn('q1', pair['substance_divergent_questions'])
        self.assertEqual(report['per_model']['ref']['answered'], 1)
        self.assertEqual(report['per_model']['alt']['refused'], 1)

    def test_a_failing_arm_is_data_not_a_dead_run(self):
        from answer_ab import anno_line, run_comparison
        collection, vectors = self._collection()
        fake_embedder = SimpleNamespace(embed_query=lambda text: vectors[0])

        class NativeErrorGen:
            # answer.py returns status="error" (provider_error) as a dict
            def answer(self, question, hits, language, use_cache=True):
                return {'status': 'error', 'reason': 'provider_error',
                        'model': 'nat', 'claims': [], 'sources': [],
                        'validation_ok': False, 'provider_ok': False}

        class RaisingGen:
            def answer(self, question, hits, language, use_cache=True):
                raise RuntimeError('boom: capacity 502')

        cases = [{'id': 'q1', 'question': 'س', 'language': 'ar',
                  'category': 'verbatim'}]
        report = run_comparison(self.cfg, fake_embedder, collection, cases,
                                [('nat', NativeErrorGen()), ('raise', RaisingGen())],
                                top_k=2, mode='vector')
        self.assertEqual(report['per_model']['nat']['errors'], 1)
        self.assertEqual(report['per_model']['raise']['errors'], 1)
        self.assertEqual(report['per_model']['nat']['answered'], 0)
        row = report['per_question'][0]['arms']['raise']
        self.assertEqual(row['status'], 'error')
        self.assertIn('RuntimeError', row['reason'])
        anno = anno_line(report)
        self.assertIn('nat: answered=0 refused=0 errors=1 gate=0', anno)
        self.assertIn('raise: answered=0 refused=0 errors=1 gate=0', anno)

    def test_anno_line_carries_the_gate_numbers(self):
        from answer_ab import anno_line, run_comparison
        collection, vectors = self._collection()
        fake_embedder = SimpleNamespace(embed_query=lambda text: vectors[0])

        class RefGen:
            def answer(self, question, hits, language, use_cache=True):
                return {'status': 'answered', 'reason': None, 'model': 'ref',
                        'claims': [{'text': 'a'}],
                        'sources': [{'chunk_id': h['id']} for h in hits[:1]],
                        'validation_ok': True}

        class AltGen:
            def answer(self, question, hits, language, use_cache=True):
                return {'status': 'refused', 'reason': 'no_evidence', 'model': 'alt',
                        'claims': [], 'sources': [],
                        'validation_ok': True}

        cases = [{'id': 'q1', 'question': 'س', 'language': 'ar',
                  'category': 'verbatim'}]
        report = run_comparison(self.cfg, fake_embedder, collection, cases,
                                [('ref', RefGen()), ('alt', AltGen())],
                                top_k=2, mode='vector')
        anno = anno_line(report)
        self.assertTrue(anno.startswith('ANNO| '))
        self.assertIn('n=1', anno)
        self.assertIn('retrieval=vector/top2 fixed (model-independent)', anno)
        self.assertIn('ref: answered=1 refused=0 errors=0 gate=0', anno)
        self.assertIn('alt: answered=0 refused=1 errors=0 gate=0', anno)
        self.assertIn('ref vs alt: agree=0.000', anno)
        self.assertIn('divergent(1)=[q1]', anno)

    def test_agreement_when_arms_match(self):
        from answer_ab import run_comparison
        collection, vectors = self._collection()
        fake_embedder = SimpleNamespace(embed_query=lambda text: vectors[0])

        class AgreeingGen:
            def answer(self, question, hits, language, use_cache=True):
                return {'status': 'answered', 'reason': None, 'model': 'fake',
                        'claims': [{'text': 'a'}, {'text': 'b'}],
                        'sources': [{'chunk_id': h['id']} for h in hits[:1]],
                        'validation_ok': True}

        cases = [{'id': 'q1', 'question': 'بنك البركة', 'language': 'ar',
                  'category': 'verbatim'},
                 {'id': 'q2', 'question': 'البركة 1983', 'language': 'ar',
                  'category': 'verbatim'}]
        report = run_comparison(self.cfg, fake_embedder, collection, cases,
                                [('ref', AgreeingGen()), ('alt', AgreeingGen())],
                                top_k=2, mode='vector')
        pair = report['pairwise']['ref vs alt']
        self.assertEqual(pair['status_agreement'], 1.0)
        self.assertEqual(pair['mean_source_jaccard'], 1.0)
        self.assertEqual(pair['substance_divergent_questions'], [])


class LexiconExpansion(unittest.TestCase):
    """Phase-3 intervention 2: the governed institutional lexicon — a
    deterministic, non-generative query expansion before embedding and BM25,
    strictly separate from the retired translation path."""

    def test_identity_when_nothing_matches(self):
        import lexicon
        self.assertEqual(lexicon.expand_query('ما هي شروط فتح الحساب؟'),
                         ['ما هي شروط فتح الحساب؟'])
        self.assertEqual(lexicon.expansion_pairs('ordinary words here'), [])

    def test_english_equivalent_and_abbreviation(self):
        import lexicon
        q = 'What is murabaha at the BCT?'
        out = lexicon.expand_query(q)
        self.assertEqual(out[0], q)
        self.assertIn('What is المرابحة at the BCT?', out)
        self.assertIn('What is murabaha at the البنك المركزي التونسي?', out)
        self.assertIn('What is المرابحة at the البنك المركزي التونسي?', out)
        labels = [label for label, _ in lexicon.expansion_pairs(q)]
        self.assertIn('lexicon:murabaha', labels)
        self.assertIn('lexicon:BCT', labels)
        self.assertIn('lexicon:combined', labels)

    def test_case_insensitive_and_no_substring_false_positives(self):
        import lexicon
        self.assertIn('المرابحة definition', lexicon.expand_query('MURABAHA definition'))
        self.assertEqual(lexicon.expand_query('murabahax contracts'),
                         ['murabahax contracts'])
        self.assertEqual(lexicon.expand_query('تسليفات متعددة'),
                         ['تسليفات متعددة'])

    def test_french_equivalents(self):
        import lexicon
        q = "Qu'est-ce que la mourabaha et les dépôts ?"
        out = lexicon.expand_query(q)
        self.assertIn("Qu'est-ce que la المرابحة et les dépôts ?", out)
        self.assertIn("Qu'est-ce que la mourabaha et les الودائع ?", out)
        self.assertIn("Qu'est-ce que la المرابحة et les الودائع ?", out)

    def test_arabic_colloquial_definite_and_indefinite(self):
        import lexicon
        self.assertIn('كيف نحصل على القرض؟',
                      lexicon.expand_query('كيف نحصل على تسليف؟'))
        self.assertIn('شروط القرض السكني',
                      lexicon.expand_query('شروط التسليف السكني'))

    def test_every_seed_target_exists_in_the_adopted_codex(self):
        """Governance: the lexicon maps INTO the corpus vocabulary only."""
        import lexicon
        import restructure
        from evaluate import normalize_for_match
        from loader import load_all
        docs = load_all(DOCS_DIRS)
        codices = []
        for d in docs:
            t = restructure._adopted_codex_text(d['name'])
            codices.append(t if t is not None else d['text'])
        self.assertEqual(len(codices), 4)
        for entry in lexicon.SEED_ENTRIES:
            self.assertTrue(
                any(normalize_for_match(entry['to']) in normalize_for_match(c)
                    for c in codices),
                f"seed target not in the adopted codex: {entry['from']} -> {entry['to']}")

    def test_retrieval_wiring_off_by_default_and_on_when_enabled(self):
        import lexicon
        from retrieval import retrieve
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        cfg = make_config(CHROMA_DIR=Path(self.temp.name) / 'chroma',
                          ANSWER_CACHE_PATH=Path(self.temp.name) / 'answers.json')
        self.assertFalse(cfg.LEXICON_ENABLED)   # OFF by default

        def chunk(index, source, language, text):
            return SimpleNamespace(index=index, source=source, language=language,
                                   text=text, heading='', origin='test/',
                                   section_type='content', token_count=20)
        chunks = [chunk(0, 'ar.md', 'ar', 'المرابحة صيغة تمويل بصيغة البيع'),
                  chunk(0, 'en.md', 'en', 'murabaha is a financing form')]
        vectors = [[1.0] + [0.0] * 2047, [0.0, 1.0] + [0.0] * 2046]
        collection = get_collection(cfg, reset=True)
        store_chunks(collection, list(zip(chunks, vectors)), cfg)

        def embed(text):
            # the governed Arabic anchor routes the expanded variant to the
            # Arabic chunk; the original English text to the English chunk
            return vectors[0] if 'المرابحة' in text else vectors[1]
        fake_embedder = SimpleNamespace(embed_query=embed)

        q = 'What is murabaha?'
        hits, variants = retrieve(cfg, fake_embedder, collection, q,
                                  language='en', mode='vector', top_k=2,
                                  variant_strategy='original')
        self.assertEqual([v['label'] for v in variants], ['en(original)'])
        # only the original variant contributed: every hit carries its label
        self.assertTrue(all(h['from_variant'] == 'en(original)' for h in hits))

        cfg.LEXICON_ENABLED = True
        hits2, variants2 = retrieve(cfg, fake_embedder, collection, q,
                                    language='en', mode='vector', top_k=2,
                                    variant_strategy='original')
        labels = [v['label'] for v in variants2]
        self.assertEqual(labels[0], 'en(original)')
        self.assertIn('lexicon:murabaha', labels)
        by_id = {h['id']: h for h in hits2}
        self.assertIn('ar.md::chunk_0000', by_id)
        self.assertIn('en.md::chunk_0000', by_id)
        # the Arabic chunk was reached BY the governed variant
        self.assertEqual(by_id['ar.md::chunk_0000']['from_variant'],
                         'lexicon:murabaha')
        self.assertIn('lexicon:murabaha',
                      by_id['ar.md::chunk_0000']['variant_ranks'])


class RerankerDeterministic(unittest.TestCase):
    """Phase-3 intervention 3: the deterministic feature-based reranker —
    coverage/heading/phrase signals over a rank prior, ties keeping the
    retriever's order, no model calls, no network."""

    def test_query_terms_stop_filtered_and_stable(self):
        import rerank
        terms = rerank.query_terms('ما هي شروط المرابحة؟ What is the rule?')
        self.assertNotIn('ما', terms)
        self.assertNotIn('هي', terms)
        self.assertNotIn('the', terms)
        self.assertIn('شروط', terms)
        self.assertIn('المرابحة', terms)
        self.assertNotIn('what', terms)   # function word, filtered
        self.assertEqual(terms, rerank.query_terms('ما هي شروط المرابحة؟ What is the rule?'))

    def test_rerank_lifts_the_term_covering_chunk(self):
        import rerank
        q = 'شروط عملية المرابحة المصرفية'
        hits = [
            {"id": "a::1", "text": "عملية الإجارة تأجير ثم تملك", "heading": "الإجارة"},
            {"id": "a::2", "text": "شروط عملية المرابحة المصرفية محددة في هذا الفصل",
             "heading": "شروط المرابحة"},
            {"id": "a::3", "text": "نص آخر لا صلة له", "heading": ""},
        ]
        out = rerank.rerank_hits(q, hits)
        self.assertEqual(out[0]["id"], "a::2")
        self.assertTrue(out[0]["rerank"]["features"]["phrase_bonus"] > 0)
        self.assertEqual(out[0]["rerank"]["features"]["term_coverage"], 1.0)

    def test_ties_keep_the_retriever_order(self):
        import rerank
        hits = [{"id": f"a::{i}", "text": "نص واحد متطابق", "heading": "عنوان"}
                for i in range(1, 4)]
        out = rerank.rerank_hits('استعلام لا مصطلحات مشتركة', hits)
        self.assertEqual([h["id"] for h in out], ["a::1", "a::2", "a::3"])

    def test_deterministic_and_annotated(self):
        import rerank
        q = 'قواعد التمويل بالمرابحة'
        hits = [{"id": "a::1", "text": "قواعد عامة", "heading": "عام"},
                {"id": "a::2", "text": "التمويل بالمرابحة قواعده واضحة", "heading": "المرابحة"}]
        first = rerank.rerank_hits(q, hits)
        second = rerank.rerank_hits(q, hits)
        self.assertEqual(first, second)
        for h in first:
            self.assertIn("rerank", h)
            self.assertIn("score", h["rerank"])
            self.assertIn("base_position", h["rerank"]["features"])
        # inputs are not mutated
        self.assertNotIn("rerank", hits[0])

    def test_short_penalty_and_missing_fields(self):
        import rerank
        out = rerank.rerank_hits('المرابحة', [
            {"id": "a::1", "text": "قصير جدا", "heading": ""},
            {"id": "a::2", "text": "المرابحة صيغة تمويل معروفة بشروطها المفصلة هنا", "heading": ""},
        ])
        self.assertEqual(out[0]["id"], "a::2")

    def test_retrieval_wiring_off_by_default_and_on(self):
        from retrieval import retrieve
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        cfg = make_config(CHROMA_DIR=Path(self.temp.name) / 'chroma',
                          ANSWER_CACHE_PATH=Path(self.temp.name) / 'answers.json')
        # ON by default since the owner decision of 2026-10-01
        self.assertTrue(cfg.RERANK_ENABLED)
        cfg.RERANK_ENABLED = False   # explicit off for the before-leg

        def chunk(index, source, language, text, heading):
            return SimpleNamespace(index=index, source=source, language=language,
                                   text=text, heading=heading, origin='test/',
                                   section_type='content', token_count=20)
        chunks = [chunk(0, 'ar.md', 'ar', 'شروط المرابحة كما وردت في النص', 'المرابحة'),
                  chunk(1, 'ar.md', 'ar', 'نص عام عن الصيرفة', 'عام')]
        # vector space: chunk 1 (general) is the retriever's top-1
        vectors = [[0.0, 1.0] + [0.0] * 2046, [1.0, 0.0] + [0.0] * 2046]
        collection = get_collection(cfg, reset=True)
        store_chunks(collection, list(zip(chunks, vectors)), cfg)
        # the retriever's top-1 is the GENERAL chunk (cosine 1.0 with it)
        fake_embedder = SimpleNamespace(embed_query=lambda text: vectors[1])

        q = 'شروط المرابحة'
        hits, _ = retrieve(cfg, fake_embedder, collection, q,
                           language='ar', mode='vector', top_k=2,
                           variant_strategy='original')
        self.assertEqual(hits[0]['id'], 'ar.md::chunk_0001')  # retriever order

        cfg.RERANK_ENABLED = True
        hits2, _ = retrieve(cfg, fake_embedder, collection, q,
                            language='ar', mode='vector', top_k=2,
                            variant_strategy='original')
        self.assertEqual(hits2[0]['id'], 'ar.md::chunk_0000')  # lifted by terms
        self.assertIn('rerank', hits2[0])
        # the deployed default (no explicit flag) reranks too
        cfg2 = make_config(CHROMA_DIR=Path(self.temp.name) / 'chroma2',
                           ANSWER_CACHE_PATH=Path(self.temp.name) / 'a2.json')
        collection2 = get_collection(cfg2, reset=True)
        store_chunks(collection2, list(zip(chunks, vectors)), cfg2)
        hits3, _ = retrieve(cfg2, fake_embedder, collection2, q,
                            language='ar', mode='vector', top_k=2,
                            variant_strategy='original')
        self.assertEqual(hits3[0]['id'], 'ar.md::chunk_0000')
        self.assertIn('rerank', hits3[0])


class GovernanceRegistry(unittest.TestCase):
    """Phase-4 item 2: the governance-axes registry — declared axes riding on
    chunk metadata at ingest; law fully registered, the other three corpus
    documents explicitly DEFERRED (minimal axes are a later reviewed
    extension); fingerprint checks keep registry and corpus in sync."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        self.cfg = make_config(NVIDIA_MIN_INTERVAL=0, NVIDIA_CHAT_STREAM=False,
                              QUERY_TRANSLATION_PROMPT='basic-v1',
                              NVIDIA_EMBEDDING_CACHE_PATH=self.path / 'embeddings.json',
                              EMBEDDING_CACHE_PATH=self.path / 'embeddings.json',
                              QUERY_TRANSLATION_CACHE_PATH=self.path / 'translations.json',
                              ANSWER_CACHE_PATH=self.path / 'answers.json',
                              CHROMA_DIR=self.path / 'chroma')

    @staticmethod
    def _corpus_docs():
        import harness50
        from loader import load_all
        return [d["name"] for d in load_all(harness50.DOCS_DIRS)]

    def test_registry_covers_corpus_exactly(self):
        import governance
        docs = self._corpus_docs()
        self.assertEqual(len(docs), 4)             # the corpus contract
        self.assertTrue(governance.registry_ok(docs), governance.registry_report(docs))
        report = governance.registry_report(docs)
        self.assertEqual(report["registered"], ["Loi_2016-48.pdf"])
        self.assertEqual(len(report["deferred"]), 3)

    def test_axes_for_law_full_others_empty(self):
        import governance
        law = governance.axes_for("Loi_2016-48.pdf")
        self.assertEqual(set(law), set(governance.AXES))
        self.assertEqual(law["gov_type"], "قانون")
        self.assertEqual(law["gov_status"], "نافذ")
        for deferred in governance.DEFERRED_DOCUMENTS:
            self.assertEqual(governance.axes_for(deferred), {})

    def test_fingerprint_checks_flag_phantom_and_undeclared(self):
        import governance
        # a registry entry for a document not in the corpus → phantom
        report = governance.registry_report(["Only_Doc.pdf"])
        self.assertEqual(report["phantom_entries"], ["Loi_2016-48.pdf"])
        self.assertEqual(report["unregistered_undeclared"], ["Only_Doc.pdf"])
        self.assertFalse(governance.registry_ok(["Only_Doc.pdf"]))
        # the real corpus passes
        self.assertTrue(governance.registry_ok(self._corpus_docs()))

    def test_axes_ride_on_chunk_metadata_at_ingest(self):
        import governance
        def chunk(index, source, text):
            return SimpleNamespace(index=index, source=source, language="ar", text=text,
                                   heading="", origin="test/", section_type="content",
                                   token_count=20)
        chunks = [chunk(0, "Loi_2016-48.pdf", "الفصل الاول ..."),
                  chunk(0, "Guide_Interne_Operations_Bancaires_Islamiques.docx", "دليل")]
        vectors = [[1.0] + [0.0] * 2047, [0.0, 1.0] + [0.0] * 2046]
        collection = get_collection(self.cfg, reset=True)
        self.assertEqual(store_chunks(collection, list(zip(chunks, vectors)), self.cfg), 2)
        got = collection.get(include=["metadatas"])["metadatas"]
        law_meta = next(m for m in got if m["document"] == "Loi_2016-48.pdf")
        guide_meta = next(m for m in got if m["document"].startswith("Guide"))
        for axis in governance.AXES:
            self.assertIn(axis, law_meta)          # all six axes on the law
            self.assertEqual(law_meta[axis], governance.axes_for("Loi_2016-48.pdf")[axis])
            self.assertNotIn(axis, guide_meta)     # deferred doc stays clean


class RelationsExtraction(unittest.TestCase):
    """Phase-4 item 3: the two minimal relations tables — grounding (declared,
    evidence-checked) and internal cross-references (deterministic from the
    item-1 units; only the literal «من هذا القانون» form becomes an edge, so
    other-law references like «من المجلة الجزائية» never do)."""

    @staticmethod
    def _units_and_internal():
        import restructure, units, relations
        codex = restructure._adopted_codex_text("Loi_2016-48.pdf")
        law_units = units.extract_law_units(codex)
        return law_units, relations.extract_internal_references(law_units)

    def test_validation_clean_and_counts(self):
        import restructure, units, relations
        codex = restructure._adopted_codex_text("Loi_2016-48.pdf")
        law_units = units.extract_law_units(codex)
        report = relations.validate_relations(law_units)
        self.assertEqual(report["violations"], [])
        self.assertEqual(report["grounding_edges"], 1)
        self.assertEqual(report["internal_edges"], 77)   # 68 singular + 9 from و-lists
        self.assertEqual(report["self_references"], 0)

    def test_grounding_edge_is_documented_and_verbatim(self):
        import restructure, relations
        edge = relations.GROUNDING_EDGES[0]
        self.assertEqual(edge["to_unit"], "loi-2016-48:art011")
        circ = restructure._adopted_codex_text(edge["from_document"])
        self.assertIn(edge["evidence"], circ)            # verbatim in the codex
        self.assertIn("MASTER_INDEX", edge["documented_by"])

    def test_internal_extraction_deterministic(self):
        _, first = self._units_and_internal()
        _, second = self._units_and_internal()
        self.assertEqual(first, second)
        self.assertTrue(all(e["relation"] == "إحالة داخلية" for e in first))

    def test_other_law_references_never_become_edges(self):
        import relations
        synthetic = [{"unit_id": "loi-2016-48:art999", "text":
                      "وخاضعة للعقوبات المنصوص عليها بالفصل254 من المجلة الجزائية ."}]
        self.assertEqual(relations.extract_internal_references(synthetic), [])

    def test_list_references_yield_one_edge_per_number(self):
        import relations
        synthetic = [{"unit_id": "loi-2016-48:art999", "text":
                      "تطبق احكام الفصول 49 و50 و51 من هذا القانون على الحالة."}]
        edges = relations.extract_internal_references(synthetic)
        self.assertEqual([e["to_unit"] for e in edges],
                         ["loi-2016-48:art049", "loi-2016-48:art050",
                          "loi-2016-48:art051"])
        self.assertTrue(all(e["evidence"] == "الفصول 49 و50 و51 من هذا القانون"
                            for e in edges))

    def test_emitted_file_matches_extraction(self):
        import json as _json
        from pathlib import Path as _Path
        law_units, internal = self._units_and_internal()
        data = _json.loads((_Path(__file__).resolve().parent
                            / "relations_loi_2016_48.json").read_text(encoding="utf-8"))
        self.assertEqual(data["internal"], internal)
        self.assertEqual(data["grounding"], __import__("relations").GROUNDING_EDGES)


class LegalNumbersExtraction(unittest.TestCase):
    """Phase-4 item 4: the structured legal-numbers path — deterministic
    extraction from the item-1 units via the governed NUM_WORDS vocabulary
    (percentages, monetary limits, financial penalties, deadlines), every
    record anchored to its unit with a verbatim raw span; plus the Circulaire
    corrections table (option ج) whose official values are anchored verbatim
    in the adopted Circulaire codex."""

    @staticmethod
    def _records():
        import restructure, units, legal_numbers
        codex = restructure._adopted_codex_text("Loi_2016-48.pdf")
        law_units = units.extract_law_units(codex)
        return law_units, legal_numbers.extract_legal_numbers(law_units)

    def test_parse_number_phrase_governed_vocabulary(self):
        import legal_numbers as ln
        self.assertEqual(ln.parse_number_phrase("خمسة وعشرين مليون"), 25_000_000)
        self.assertEqual(ln.parse_number_phrase("مائة الف"), 100_000)
        self.assertEqual(ln.parse_number_phrase("عشرة ملايين"), 10_000_000)
        self.assertEqual(ln.parse_number_phrase("مليوني"), 2_000_000)
        self.assertEqual(ln.parse_number_phrase("مائتي"), 200)
        self.assertEqual(ln.parse_number_phrase("خمسة عشرة"), 15)
        self.assertIsNone(ln.parse_number_phrase("راس مال"))     # never guesses
        self.assertIsNone(ln.parse_number_phrase("ثلاثة اعوام"))  # duration ≠ number

    def test_validation_clean_with_exact_counts(self):
        import restructure, units, legal_numbers as ln
        codex = restructure._adopted_codex_text("Loi_2016-48.pdf")
        law_units = units.extract_law_units(codex)
        report = ln.validate_numbers(law_units)
        self.assertEqual(report["violations"], [])
        self.assertEqual(report["records"], 53)
        self.assertEqual(report["by_kind"],
                         {"نسبة": 14, "أجل": 28, "حد مالي": 5, "عقوبة مالية": 6})

    def test_deterministic_and_verbatim_anchors(self):
        law_units, first = self._records()
        _, second = self._records()
        self.assertEqual(first, second)
        by_id = {u["unit_id"]: u for u in law_units}
        for r in first:
            self.assertIn(r["raw"], by_id[r["unit_id"]]["text"])   # verbatim anchor

    def test_known_values(self):
        _, records = self._records()
        by_key = {(r["unit_id"], r["raw"]): r for r in records}
        self.assertEqual(by_key[("loi-2016-48:art032", "خمسين مليون دينار")]["value"],
                         50_000_000)
        self.assertEqual(by_key[("loi-2016-48:art183", "مائة الف دينار")]["kind"],
                         "عقوبة مالية")
        self.assertEqual(by_key[("loi-2016-48:art183", "مائة الف دينار")]["value"],
                         100_000)
        self.assertEqual(by_key[("loi-2016-48:art181", "مائتي دينار")]["value"], 200)
        word_pct = [r for r in records if r["raw"].endswith("في المائة")]
        self.assertEqual(len(word_pct), 1)
        self.assertEqual(word_pct[0]["value"], 5)

    def test_corrections_table_anchored_in_adopted_codex(self):
        import legal_numbers as ln
        self.assertEqual(len(ln.CIRCULAIRE_CORRECTIONS), 15)
        self.assertEqual(ln.validate_corrections(), [])
        sections = {e["section"] for e in ln.CIRCULAIRE_CORRECTIONS}
        self.assertEqual(sections, {"الترويسة", "الإحالات التشريعية", "ترقيم الفصول 14–20"})

    def test_emitted_file_matches_extraction(self):
        import json as _json
        from pathlib import Path as _Path
        law_units, records = self._records()
        data = _json.loads((_Path(__file__).resolve().parent
                            / "legal_numbers_loi_2016_48.json").read_text(encoding="utf-8"))
        self.assertEqual(data["law_numbers"], records)


class UnitIndexMeasure(unittest.TestCase):
    """Phase-4 item 5: the units index — the 198 description lines on the
    EXISTING retrieval machinery in an isolated collection; the 12 law cases
    judged by the shared normalized-containment rule (expected substring
    inside the unit's verbatim article text)."""

    def test_law_case_filter(self):
        import json as _json
        import unit_index
        cases = _json.loads((Path(__file__).resolve().parent
                             / "questions_50.json").read_text(encoding="utf-8"))["cases"]
        loi = unit_index.law_cases(cases)
        self.assertEqual(len(loi), 12)
        self.assertTrue(all(c["expected_document"] == "Loi_2016-48.pdf" for c in loi))

    def test_unit_contains_evidence_rule(self):
        import unit_index
        art = "يعاقب بالسجن من ثلاثة اشهر وبخطية من مائة الف دينار"
        case = {"expected_substring": "بخطية من مائة الف دينار"}
        self.assertTrue(unit_index.unit_contains_evidence(art, case))
        case_wrong = {"expected_substring": "غير موجود إطلاقًا"}
        self.assertFalse(unit_index.unit_contains_evidence(art, case_wrong))
        case_empty = {"expected_substring": ""}
        self.assertFalse(unit_index.unit_contains_evidence(art, case_empty))

    def test_end_to_end_with_fake_embedder(self):
        """Machinery check (not quality): a fake embedder that ranks the
        article containing 'المرابحة' first for its question must produce
        hit@1 for that case and full hit metrics."""
        import tempfile
        import unit_index
        units_list = [
            {"unit_id": "loi-2016-48:art012", "heading": "الفصل12", "path": "p",
             "description": "يعتبر تمويلا بالمرابحة كل عملية شراء",
             "text": "يعتبر تمويلا بالمرابحة كل عملية يقوم فيها البنك بشراء منقولات",
             "type": "definition", "numeric": False},
            {"unit_id": "loi-2016-48:art013", "heading": "الفصل13", "path": "p",
             "description": "يعتبر تمويلا بالاجارة",
             "text": "يعتبر تمويلا بالاجارة كل عملية تملك منقولات او عقارات",
             "type": "definition", "numeric": False},
        ]
        cases = [{"id": "qX", "question": "ما المرابحة؟", "language": "ar",
                  "category": "verbatim", "expected_document": "Loi_2016-48.pdf",
                  "expected_lang": "ar",
                  "expected_substring": "شراء منقولات"}]
        embedder = SimpleNamespace(
            embed_texts=lambda ts: [[1.0, 0.0] if "المرابحة" in t else [0.0, 1.0]
                                    for t in ts],
            embed_query=lambda t: [1.0, 0.0])
        with tempfile.TemporaryDirectory() as tmp:
            result = unit_index.evaluate_units_index(
                units_list, cases, embedder, Path(tmp) / "chroma")
        self.assertEqual(result["metrics"]["n"], 1)
        self.assertEqual(result["questions"][0]["correct_unit_rank"], 1)
        self.assertTrue(result["questions"][0]["hit_at_1"])
        self.assertEqual(result["questions"][0]["top_unit"], "loi-2016-48:art012")

    def test_baseline_extraction_from_eval_json(self):
        import tempfile
        import unit_index
        run = {"questions": [
            {"id": "q10", "hit_at_1": True, "hit_at_3": True, "hit_at_5": True,
             "is_out_of_scope": False},
            {"id": "q11", "hit_at_1": False, "hit_at_3": True, "hit_at_5": True,
             "is_out_of_scope": False},
            {"id": "q99", "hit_at_1": True, "hit_at_3": True, "hit_at_5": True,
             "is_out_of_scope": False},
            {"id": "q28", "hit_at_1": False, "hit_at_3": False, "hit_at_5": False,
             "is_out_of_scope": True},
        ]}
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "eval.json"
            p.write_text(json.dumps(run), encoding="utf-8")
            base = unit_index.baseline_for_law_questions(p, ["q10", "q11", "q28"])
        self.assertEqual(base["n"], 2)          # the OOS case never counts
        self.assertEqual(base["hit@1"], 0.5)
        self.assertEqual(base["hit@5"], 1.0)


class IntentClassification(unittest.TestCase):
    """Phase-5 item 1: deterministic-first intent — declared INTENT_RULES (first
    match wins, every decision explained), the DECLARED default path for the
    unclassifiable, personal framing, explicit-document scope only, compound
    splitting, and influential-ambiguity detection (MASTER_INDEX §6 shared
    concepts) resolved by scope. The layer is INERT: nothing in the deployed
    retrieval path imports it."""

    def test_known_types_from_the_adopted_sets(self):
        import intent
        cases = {
            "ما العقوبتان الجزائيتان المقررتان في القانون؟": "عقوبي",
            "قارن بين صيغتي المضاربة من حيث حرية البنك في استثمار أموال الحريف.": "مقارن",
            "ما الصيغة التي يشارك فيها العميل بخبرته وإدارته فقط؟": "تعريفي",
            "ما الحد الأقصى لرأس مال البنك المقيم؟": "رقمي",
            "على أي أساس يمنح البنك تسليفا للحريف دون فوائد؟": "إجرائي",
        }
        for q, expected in cases.items():
            r = intent.classify(q)
            self.assertEqual(r["intent_type"], expected, q)
            self.assertTrue(r["fired_rules"], q)          # explained decisions

    def test_unclassifiable_gets_the_declared_default_path(self):
        import intent
        r = intent.classify("اشتريت سيارة أمس وذهبت إلى السوق")
        self.assertEqual(r["intent_type"], "غير مصنف")
        self.assertTrue(r["default_path"])
        self.assertFalse(r["fired_rules"])

    def test_personal_framing(self):
        import intent
        r = intent.classify("هل يمكنني فتح حساب مرابحة لدى البنك؟")
        self.assertTrue(r["personal"])
        self.assertFalse(intent.classify("ما هي المرابحة؟")["personal"])

    def test_scope_only_from_explicit_document_naming(self):
        import intent
        self.assertEqual(intent.classify("ماذا يقول الدليل عن المرابحة؟")["scope"],
                         "Guide_Interne_Operations_Bancaires_Islamiques.docx")
        self.assertEqual(intent.classify("ماذا يقول قانون عدد 48؟")["scope"],
                         "Loi_2016-48.pdf")
        self.assertIsNone(intent.classify("ما هي المرابحة؟")["scope"])  # never assumed

    def test_influential_ambiguity_and_scope_resolution(self):
        import intent
        r = intent.classify("ما هي المضاربة؟")
        self.assertEqual(r["ambiguous_concepts"], ["المضاربة"])
        r2 = intent.classify("ما هي المضاربة حسب المنشور؟")
        self.assertEqual(r2["scope"], "Circulaire_BCT_2019-08.pdf")
        self.assertEqual(r2["ambiguous_concepts"], [])    # scope resolves it
        self.assertIn(("scope-resolves-ambiguity",
                       "Circulaire_BCT_2019-08.pdf"), r2["fired_rules"])

    def test_compound_splitting(self):
        import intent
        q = ("في المضاربة المقيدة: من يتحمل مخاطر الاستثمار، وعلى ماذا يرتبط "
             "عائد الودائع الاستثمارية؟")
        r = intent.classify(q)
        self.assertIsNotNone(r["sub_questions"])
        self.assertEqual(len(r["sub_questions"]), 2)
        self.assertIsNone(intent.classify("ما هي المرابحة؟")["sub_questions"])

    def test_unclassified_ratio_zero_on_both_adopted_sets(self):
        """The measured contract (plan item 1.7's threshold): 0% unclassified."""
        import json as _json
        import intent
        for name in ("questions_50.json", "questions_targets.json"):
            cases = _json.loads((Path(__file__).resolve().parent / name)
                                .read_text(encoding="utf-8"))["cases"]
            report = intent.unclassified_ratio(cases)
            self.assertEqual(report["unclassified"], 0, name)
            self.assertEqual(report["n"], 50 if "50" in name else 10)

    def test_layer_is_inert_in_the_deployed_path(self):
        """Nothing in the deployed retrieval/answer path imports intent."""
        import subprocess
        out = subprocess.run(
            ["grep", "-l", "import intent", "retrieval.py", "store.py",
             "evaluate.py", "answer_ab.py", "service.py", "main.py"],
            capture_output=True, text=True, cwd=Path(__file__).resolve().parent)
        self.assertEqual(out.stdout.strip(), "")


class EvidencePlanDerivation(unittest.TestCase):
    """Phase-5 item 2: evidence plans derived from intent WITHOUT assuming the
    answer — declared DERIVATION_RULES (veto requirement by requirement),
    compound requests get one sub-plan per sub-question, and on both adopted
    sets every typed-evidence case is consistent with its derived plan (100%)."""

    def test_derivation_deterministic_and_explained(self):
        import evidence_plan as ep
        p1 = ep.derive_plan("ما هي المرابحة؟")
        p2 = ep.derive_plan("ما هي المرابحة؟")
        self.assertEqual(p1, p2)
        self.assertEqual(p1["intent_type"], "تعريفي")
        self.assertEqual([r["req"] for r in p1["requirements"]],
                         ["definition_or_purpose_unit"])
        # عقوبي derives TWO requirements: the penalty unit AND its governing rule
        pen = ep.derive_plan("ما العقوبة على ممارسة النشاط البنكي دون ترخيص؟")
        self.assertEqual([r["req"] for r in pen["requirements"]],
                         ["penalty_unit", "with_governing_rule"])
        # unclassified → the declared wide requirement (no type assumption)
        wide = ep.derive_plan("اشتريت سيارة أمس وذهبت إلى السوق")
        self.assertEqual([r["req"] for r in wide["requirements"]], ["wide_evidence"])

    def test_compound_gets_sub_plans(self):
        import evidence_plan as ep
        q = ("في المضاربة المقيدة: من يتحمل مخاطر الاستثمار، وعلى ماذا يرتبط "
             "عائد الودائع الاستثمارية؟")
        plan = ep.derive_plan(q)
        self.assertEqual(len(plan["sub_plans"]), 2)
        self.assertTrue(all(sp["requirements"] for sp in plan["sub_plans"]))

    def test_requirement_satisfaction_kinds(self):
        import evidence_plan as ep
        case = {"question": "ما هي المرابحة؟", "expected_substring": "x"}
        definition = {"type": "definition", "text": "تعتبر تمويلا بالمرابحة",
                      "unit_id": "u1", "numeric": False}
        self.assertTrue(ep.requirement_satisfied(
            {"req": "definition_or_purpose_unit"}, case, [definition], set()))
        # entity/purpose questions accept institutional provisions
        entity_case = {"question": "ما هو الصندوق الذي أحدثه القانون؟"}
        general = {"type": "general", "text": "تتكون لجنة الانقاذ من",
                   "unit_id": "u2", "numeric": False}
        self.assertTrue(ep.requirement_satisfied(
            {"req": "definition_or_purpose_unit"}, entity_case, [general], set()))
        penalty = {"type": "penalty", "text": "يعاقب طبقا للفصل24",
                   "unit_id": "u3", "numeric": True}
        self.assertTrue(ep.requirement_satisfied(
            {"req": "penalty_unit"}, case, [penalty], set()))
        self.assertTrue(ep.requirement_satisfied(
            {"req": "with_governing_rule"}, case, [penalty], set()))
        self.assertTrue(ep.requirement_satisfied(
            {"req": "numeric_evidence"}, case, [penalty], set()))
        self.assertIsNone(ep.requirement_satisfied(   # untyped → never guessed
            {"req": "penalty_unit"}, case, [], set()))

    def test_coverage_contract_on_both_adopted_sets(self):
        """The plan's measurable contract: 100% typed-evidence consistency."""
        import json as _json
        import evidence_plan as ep
        import restructure, units, legal_numbers
        codex = restructure._adopted_codex_text("Loi_2016-48.pdf")
        law_units = units.extract_law_units(codex)
        numeric_ids = {r["unit_id"] for r in
                       legal_numbers.extract_legal_numbers(law_units)}
        for name, n_typed in (("questions_50.json", 16), ("questions_targets.json", 2)):
            cases = _json.loads((Path(__file__).resolve().parent / name)
                                .read_text(encoding="utf-8"))["cases"]
            m = ep.measure_coverage(cases, law_units, numeric_ids)
            self.assertEqual(m["derived_plans"], m["n"], name)
            self.assertEqual(m["typed_evidence_cases"], n_typed, name)
            self.assertEqual(m["consistency_ratio"], 1.0, name)

    def test_layer_is_inert_in_the_deployed_path(self):
        import subprocess
        out = subprocess.run(
            ["grep", "-l", "import evidence_plan", "retrieval.py", "store.py",
             "evaluate.py", "answer_ab.py", "service.py", "main.py", "intent.py"],
            capture_output=True, text=True, cwd=Path(__file__).resolve().parent)
        self.assertEqual(out.stdout.strip(), "")


class CitationGateExpansion(unittest.TestCase):
    """Phase-6 item 1: stable unit ids on sources + the deterministic
    expansion of the citation gate — a cited source must be actually
    retrieved (by construction), ALLOWED (its document is in the deployed
    corpus set, passed by the caller — never assumed) and IN FORCE
    (gov_status == نافذ where the axis is registered; deferred documents
    carry no status claim and pass unflagged)."""

    @staticmethod
    def _hit(doc="Loi_2016-48.pdf", heading="الفصل52", gov=None, hid="h1"):
        meta = {"document": doc, "source": doc, "language": "ar",
                "heading": heading}
        if gov:
            meta["gov_status"] = gov
        return {"id": hid, "text": "نص الفصل 52 ينص على أحكام معينة",
                "metadata": meta}

    def test_resolve_unit_id_stable(self):
        import answer
        self.assertEqual(answer.resolve_unit_id("Loi_2016-48.pdf", "الفصل52"),
                         "loi-2016-48:art052")
        # untyped documents / unknown headings -> None (never invented)
        self.assertIsNone(answer.resolve_unit_id("Guide_Interne_x.docx", "2.1-"))
        self.assertIsNone(answer.resolve_unit_id("Loi_2016-48.pdf", "لا-فصل"))
        self.assertIsNone(answer.resolve_unit_id(None, "الفصل52"))

    def test_build_sources_carries_unit_id_and_status(self):
        import answer
        law = self._hit(gov="نافذ")
        other = self._hit(doc="Guide_Interne_Operations_Bancaires_Islamiques.docx",
                          heading="2.1- عملية التمويل بصيغة المرابحة", hid="h2")
        sources = answer.build_sources([law, other], 3000)
        by_doc = {s["document"]: s for s in sources}
        self.assertEqual(by_doc["Loi_2016-48.pdf"]["unit_id"],
                         "loi-2016-48:art052")
        self.assertEqual(by_doc["Loi_2016-48.pdf"]["gov_status"], "نافذ")
        self.assertNotIn("unit_id", by_doc[
            "Guide_Interne_Operations_Bancaires_Islamiques.docx"])
        self.assertNotIn("gov_status", by_doc[
            "Guide_Interne_Operations_Bancaires_Islamiques.docx"])

    def _validated(self, output, sources, allowed=None):
        import answer
        return answer.validate_answer(output, sources,
                                      allowed_documents=allowed)

    def test_gate_rejects_document_outside_the_corpus(self):
        import answer
        hit = self._hit(doc="Phantom_Doc.pdf", heading="")
        sources = answer.build_sources([hit], 3000)
        out = {"answerable": True, "claims": [
            {"text": "claim", "evidence": [
                {"source_id": "S1", "quote": sources[0]["text"][:40]}]}]}
        # allowed corpus set does not contain Phantom_Doc.pdf -> refused
        with self.assertRaises(ValueError):
            self._validated(out, sources, allowed={"Loi_2016-48.pdf"})
        # with the right allowlist it passes (quote membership still enforced)

    def test_gate_rejects_document_not_in_force(self):
        import answer
        hit = self._hit(gov="ملغى")     # registered axis, not نافذ
        sources = answer.build_sources([hit], 3000)
        out = {"answerable": True, "claims": [
            {"text": "claim", "evidence": [
                {"source_id": "S1", "quote": sources[0]["text"][:40]}]}]}
        with self.assertRaises(ValueError):
            self._validated(out, sources, allowed={"Loi_2016-48.pdf"})
        # نافذ passes; no axis (deferred doc) passes unflagged
        ok = answer.build_sources([self._hit(gov="نافذ")], 3000)
        self._validated(out, {**out, "claims": [{"text": "claim", "evidence": [
            {"source_id": "S1", "quote": ok[0]["text"][:40]}]}]} and [
            {"source_id": "S1", "chunk_id": "h1", "document": "Loi_2016-48.pdf",
             "heading": "الفصل52", "text": ok[0]["text"], "gov_status": "نافذ"}],
            allowed={"Loi_2016-48.pdf"})
        deferred = answer.build_sources([
            self._hit(doc="Madkhal_Sayrafa_Islamiya.docx", heading="ب-")], 3000)
        self._validated(
            {"answerable": True, "claims": [{"text": "claim", "evidence": [
                {"source_id": "S1", "quote": deferred[0]["text"][:40]}]}]},
            deferred, allowed={"Madkhal_Sayrafa_Islamiya.docx"})

    def test_policy_declared(self):
        import answer
        self.assertEqual(answer.CITATION_GATE_POLICY["required_status"], "نافذ")
        self.assertIn("registered", answer.CITATION_GATE_POLICY["status_check"])


class AuditTrail(unittest.TestCase):
    """Phase-6 item 3: bounded, PII-safe request audit trail (JSONL)."""

    def test_record_recent_roundtrip_and_defaults(self):
        import audit
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "audit_log.jsonl"
            row = audit.record(path, {"endpoint": "/answer",
                                      "question": "سؤال اختبار",
                                      "status": "answered"})
            self.assertEqual(len(row["trace_id"]), 12)
            self.assertIn("T", row["ts"])
            rows = audit.recent(path, 10)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["question"], "سؤال اختبار")

    def test_retention_trims_oldest(self):
        import audit
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "audit_log.jsonl"
            saved = audit.RETENTION_MAX_ENTRIES
            try:
                audit.RETENTION_MAX_ENTRIES = 3
                for i in range(5):
                    audit.record(path, {"endpoint": "/answer",
                                        "question": f"q{i}", "status": "ok"})
                rows = audit.recent(path, 10)
                self.assertEqual(len(rows), 3)
                self.assertEqual([r["question"] for r in rows],
                                 ["q4", "q3", "q2"])   # newest first
            finally:
                audit.RETENTION_MAX_ENTRIES = saved

    def test_torn_line_tolerated_and_schema_bounded(self):
        import audit
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "audit_log.jsonl"
            path.write_text('{"trace_id": "abc", "endpoint": "/answer"\n'
                            '{"trace_id": "def", "endpoint": "/answer"}\n',
                            encoding="utf-8")
            rows = audit.recent(path, 10)
            self.assertEqual(len(rows), 1)      # the torn line is skipped
            self.assertTrue(set(rows[0]) <= set(audit.ENTRY_FIELDS))


class DecompositionCheck(unittest.TestCase):
    """Phase-5 item 6 (owner directive 2026-10-01): the intermediate
    decomposition layer — EVERY question becomes micro-questions, each with
    its intent, its ONE requirement (from the item-2 plan), and the Phase-4
    answerable surfaces (live counts); influential ambiguity generates
    advisory clarifications. Read-only layer; INTENT_RULES and
    DERIVATION_RULES consumed as-is."""

    Q50 = "ما هي عملية التمويل بالمرابحة على معنى منشور البنك المركزي التونسي؟"
    Q_PENALTY = "ما العقوبة الجزائية لممارسة العمليات البنكية خلافا لأحكام القانون عدد 48 لسنة 2016؟"
    Q_COMPOUND = ("ما قيمة غرامة التأخير اليومية عن عدم تقديم الوثائق المطلوبة، "
                  "ومن يحدد مبلغها النهائية؟")
    Q_COMPARE = ("قارن بين صيغتي المضاربة من حيث حرية البنك في استثمار أموال "
                 "الحريف وعائد الحريف")

    def test_subject_extraction_declared_rules(self):
        import decompose as D
        self.assertEqual(D.extract_subject(self.Q50), "عملية التمويل بالمرابحة")
        # scope clause (خلافا لأحكام القانون...) stripped
        subj = D.extract_subject(self.Q_PENALTY)
        self.assertTrue(subj.startswith("العقوبة الجزائية"))
        self.assertNotIn("القانون", subj)
        # never empty — fallback keeps the text minus the question mark
        self.assertTrue(D.extract_subject("كيف؟").strip())

    def test_simple_question_one_micro_with_live_surfaces(self):
        import decompose as D
        d = D.decompose(self.Q50)
        self.assertEqual(d["intent_type"], "تعريفي")
        self.assertEqual(len(d["micro_questions"]), 1)
        m = d["micro_questions"][0]
        self.assertEqual(m["requirement"], "definition_or_purpose_unit")
        self.assertIn("عملية التمويل بالمرابحة", m["text"])   # faithful subject
        # the answerable surface carries the LIVE catalog count (28 definitions)
        self.assertTrue(any("definition (28)" in s for s in m["answerable_from"]))

    def test_penalty_question_two_micros_bijection(self):
        import decompose as D
        d = D.decompose(self.Q_PENALTY)
        reqs = [m["requirement"] for m in d["micro_questions"]]
        self.assertEqual(reqs, ["penalty_unit", "with_governing_rule"])
        self.assertTrue(all(m["answerable_from"] for m in d["micro_questions"]))
        self.assertTrue(any("penalty (7)" in s for m in d["micro_questions"]
                            for s in m["answerable_from"]))

    def test_compound_decomposition_per_sub_question(self):
        import decompose as D
        d = D.decompose(self.Q_COMPOUND)
        self.assertGreaterEqual(len(d["micro_questions"]), 4)
        subs = {m["sub_question_index"] for m in d["micro_questions"]}
        self.assertIn(1, subs)
        self.assertIn(2, subs)
        # the numeric sub-question asks for the structured-numbers surface
        numeric = [m for m in d["micro_questions"]
                   if m["requirement"] == "numeric_evidence"]
        self.assertTrue(numeric)
        self.assertTrue(any("53" in s for s in numeric[0]["answerable_from"]))

    def test_comparison_decomposes_by_aspect(self):
        import decompose as D
        d = D.decompose(self.Q_COMPARE)
        aspect_micros = [m for m in d["micro_questions"] if m.get("aspect")]
        self.assertEqual(len(aspect_micros), 2)
        self.assertEqual({m["aspect"] for m in aspect_micros},
                         {"حرية البنك في استثمار أموال الحريف", "عائد الحريف"})
        self.assertTrue(all(m["requirement"] == "both_sides_evidence"
                            for m in aspect_micros))
        self.assertTrue(all("صيغتي المضاربة" in m["text"] for m in aspect_micros))

    def test_ambiguity_generates_advisory_clarification(self):
        import decompose as D
        d = D.decompose(self.Q_COMPARE)
        self.assertIn("المضاربة", d["ambiguous_concepts"])
        self.assertTrue(d["clarifications"])
        self.assertIn("المضاربة", d["clarifications"][0])
        # advisory: the decomposition still proceeds
        self.assertTrue(d["micro_questions"])

    def test_oos_question_still_decomposes(self):
        # understanding does not need evidence: an OOS question decomposes,
        # its requirements carry the answerable surfaces, and the EVIDENCE
        # layer (sufficiency) is what reports absence
        import decompose as D
        d = D.decompose("ما هو سعر سهم شركة تسلا اليوم في بورصة نيويورك؟")
        self.assertTrue(d["micro_questions"])
        self.assertTrue(all(m["answerable_from"] for m in d["micro_questions"]))

    def test_determinism_same_input_same_output(self):
        import decompose as D
        a = D.decompose(self.Q_COMPOUND)
        b = D.decompose(self.Q_COMPOUND)
        self.assertEqual(a, b)

    def test_measure_contract_on_adopted_sets(self):
        import decompose as D
        for name in ("questions_50.json", "questions_targets.json"):
            cases = json.loads((Path(__file__).parent / name)
                               .read_text(encoding="utf-8"))["cases"]
            m = D.measure(cases)
            self.assertEqual(m["decomposed"], m["n"], name)
            self.assertEqual(m["bijection_violations"], 0, name)
            self.assertEqual(m["empty_surface_micros"], 0, name)
            self.assertEqual(m["ambiguous_without_clarification"], 0, name)

    def test_inert_layer_no_deployed_execution(self):
        """decompose executes in the deployed path ONLY inside the
        ANSWER_SUFFICIENCY_COMMITMENT gate (Phase-6 item 2 — the partial
        regeneration and the referral are built from its micro-questions);
        never on the default path."""
        import subprocess
        for f in ["main.py", "retrieval.py", "store.py", "answer.py",
                  "answer_ab.py", "chat.py"]:
            out = subprocess.run(
                ["grep", "-nE", "import decompose|decompose\.", f],
                capture_output=True, text=True)
            self.assertEqual(out.returncode, 1,
                             f"{f} executes decompose in the deployed path")
        # service.py: every reference is inside the gated callback (the
        # commitment branches — indented deeper than any top-level code)
        out = subprocess.run(
            ["grep", "-nE", "import decompose|decompose\.", "service.py"],
            capture_output=True, text=True)
        self.assertEqual(out.returncode, 0, "service.py lost its gated use")
        refs = [l for l in out.stdout.splitlines() if l.strip()]
        self.assertGreaterEqual(len(refs), 2)
        for line in refs:
            code = line.split(":", 1)[1]
            self.assertGreaterEqual(len(code) - len(code.lstrip()), 12,
                                    "decompose reference outside the gate: " + line)


class SufficiencyCheck(unittest.TestCase):
    """Phase-5 item 4: deterministic sufficiency & conflict — coverage of the
    item-2 evidence plan against the RETRIEVED pool (requirement by
    requirement), declared adjustable precedence, the four explicit states
    (كافٍ/غير كافٍ/متعارض/غير محسوم), bounded guided rounds, and refusal tied
    to evidence absence (never classification failure). Read-only layer."""

    @staticmethod
    def _hit(text, doc="Guide_Interne_Operations_Bancaires_Islamiques.docx",
             heading="", hid="h1"):
        return {"id": hid, "text": text, "keyword_score": 1.0, "rank": 1,
                "metadata": {"document": doc, "source": doc, "language": "ar",
                             "heading": heading}}

    def test_anchor_bar_relative_and_junk_excluded(self):
        import sufficiency as S
        # long question: the bar caps at AR_ANCHOR_MIN
        q = ("ما الصيغة التي يشارك فيها العميل بخبرته وإدارته فقط دون أن "
             "يدفع رأس مال")
        self.assertEqual(S._anchor_bar(q), S.AR_ANCHOR_MIN)
        # two-term question: ceil(0.6*2)=2
        self.assertEqual(S._anchor_bar("من يتحمل الخسائر؟"), 2)
        # «يها» residue (فيها minus ف) is junk — never a question term
        self.assertNotIn("يها", S.question_terms("كيف ينظر إلى النقد فيها؟"))

    def test_cross_anchor_df_checked_digits(self):
        import sufficiency as S
        df = {"48": 240, "2016": 240, "2019": 13, "مرابحة": 16}
        q = "According to BCT circular 80/2019, who gives the order to purchase?"
        self.assertTrue(S._is_cross_script(q))
        # header digits (df=240) carry no signal — no anchor
        header = "الفصل 1 من القانون عدد 48 لسنة 2016"
        self.assertFalse(S._anchors(q, header, df))
        # specific rare terms anchor (>= CROSS_ANCHOR_MIN distinctive)
        good = "يتولى البنك الشراء بمقتضى المنشور عدد 80 لسنة 2019"
        self.assertTrue(S._anchors(q, good, df))

    def test_mixed_script_question_uses_cross_rules(self):
        import sufficiency as S
        q = "Under the BCT circular, the investment-deposit mudaraba (المضاربة)"
        self.assertTrue(S._is_cross_script(q))   # 1 Arabic token <= MIXED_AR_MAX

    def test_sufficient_and_insufficient_states(self):
        import sufficiency as S
        # covered definition question -> كافٍ
        q = "ما هي عملية التمويل بالمرابحة على معنى منشور البنك المركزي التونسي؟"
        hit = self._hit("تعتبر عملية التمويل بالمرابحة على معنى منشور البنك "
                        "المركزي التونسي عملية تمويل حيث يشتري البنك المرابحة",
                        doc="Circulaire_BCT_2019-08.pdf")
        res = S.check(q, [hit])
        self.assertEqual(res["state"], "كافٍ")
        self.assertTrue(all(r["covered"] for r in res["requirements"]))
        # nothing anchors -> غير كافٍ, refusal names the missing requirement
        oos = "ما هو سعر سهم شركة تسلا اليوم في بورصة نيويورك؟"
        res2 = S.check(oos, [hit])
        self.assertEqual(res2["state"], "غير كافٍ")
        self.assertTrue(res2["reason"].startswith("evidence absent"))
        self.assertIn(res2["missing"][0], res2["reason"])
        # refusal is tied to evidence absence, never to the intent label
        self.assertNotIn(res2["intent_type"], res2["reason"])

    def test_conflict_resolved_by_precedence_and_adjustable(self):
        import sufficiency as S
        import legal_numbers
        row = legal_numbers.CIRCULAIRE_CORRECTIONS[3]   # القانون 48-2016 topic
        q = "ما القانون عدد 48 لسنة 2016 والمنشور حول الاعتمادات الاسلامية؟"
        circ = self._hit("القانون عدد 19 لسنة6142 المؤرخ في 44 جويلية 6142 "
                         "الاعتمادات الاسلامية",
                         doc="Circulaire_BCT_2019-08.pdf", hid="c1")
        loi = self._hit("القانون عدد 48 لسنة 2016 المؤرخ في 11 جويلية 2016 "
                        "الاعتمادات الاسلامية",
                        doc="Loi_2016-48.pdf", hid="l1")
        self.assertTrue(S._anchors(q, circ["text"], None))
        self.assertTrue(S._anchors(q, loi["text"], None))
        conflicts = S.detect_conflicts([circ, loi])
        self.assertEqual(len(conflicts), 1)
        self.assertEqual(conflicts[0]["resolution"], S._CONFLICT_RESOLVED)
        # declared order: the law (rank 1) wins — official evidence
        self.assertEqual(conflicts[0]["winning_text"], "official_evidence")
        # the table is ADJUSTABLE: flip the two levels and the winner flips
        saved = dict(S.PRECEDENCE)
        try:
            S.PRECEDENCE["Circulaire_BCT_2019-08.pdf"] = 1
            S.PRECEDENCE["Loi_2016-48.pdf"] = 2
            flipped = S.detect_conflicts([circ, loi])
            self.assertEqual(flipped[0]["winning_text"], "raw_form")
        finally:
            S.PRECEDENCE.clear()
            S.PRECEDENCE.update(saved)
        # an unranked document -> unresolvable -> غير محسوم at check level
        unknown = self._hit(circ["text"], doc="Unknown_Doc.pdf", hid="u1")
        res = S.check(q, [loi, unknown])
        self.assertEqual(res["state"], "غير محسوم")

    def test_guided_rounds_bounded_and_rescuing(self):
        import sufficiency as S
        q = "ما هي شروط التمويل بالمشاركة حسب الدليل الداخلي للبنك؟"
        weak = self._hit("البنك يقدم خدمات مالية للحرفاء داخل الفرع",
                         hid="w1")
        # gold evidence carries the rare term the first pool lacks
        gold = self._hit("التمويل بالمشاركة تعني حسب الدليل الداخلي للبنك "
                         "اتفاق الطرفين على حصص المشاركة ونسبتها",
                         hid="g1")
        calls = {"n": 0}

        def search_fn(query, k):
            calls["n"] += 1
            return [dict(gold)] if "مشاركة" in query else []

        res = S.check(q, [weak], search_fn=search_fn)
        self.assertEqual(res["state"], "كافٍ")
        self.assertEqual(len(res["guided_rounds"]), 1)
        self.assertLessEqual(calls["n"], S.MAX_GUIDED_ROUNDS)

        # junk rounds only: bounded by the declared cap, state stays insufficient
        def junk_fn(query, k):
            return [{"id": f"j{calls['n']}", "text": "نص آخر مختلف تمامًا",
                     "metadata": {"document": "Guide_Interne_Operations_"
                                              "Bancaires_Islamiques.docx"}}]

        res2 = S.check(q, [weak], search_fn=junk_fn)
        self.assertEqual(res2["state"], "غير كافٍ")
        self.assertEqual(len(res2["guided_rounds"]), S.MAX_GUIDED_ROUNDS)

    def test_multi_requirement_differentiation(self):
        import sufficiency as S
        # compound question: sub-question 2 (numeric) missing from the pool
        q = "في المضاربة المقيدة: من يتحمل مخاطر الاستثمار، وكم تبلغ نسبة عائد الحريف؟"
        base = self._hit("المضاربة المقيدة يتحمل فيها البنك مخاطر الاستثمار "
                         "في حدود معينة",
                         doc="Circulaire_BCT_2019-08.pdf", hid="a")
        res = S.check(q, [base])
        subs = {r["sub_question"] for r in res["requirements"]}
        self.assertIn(1, subs)
        self.assertIn(2, subs)
        self.assertEqual(res["state"], "غير كافٍ")
        # the refusal names WHICH requirement is missing (differentiation)
        self.assertTrue(res["missing"])

    def test_precedence_table_declared(self):
        import sufficiency as S
        self.assertEqual(list(S.PRECEDENCE.values()), [1, 2, 3, 4])
        self.assertEqual(S.PRECEDENCE["Loi_2016-48.pdf"], 1)

    def test_inert_layer_no_deployed_imports(self):
        """Read-only layer: sufficiency executes in the deployed path ONLY
        inside the SUFFICIENCY_FIELDS_ENABLED gate (item 5) — never on the
        default path. service.py may reference it nowhere else; every other
        deployed module must not reference it at all."""
        import subprocess
        for f in ["main.py", "retrieval.py", "store.py", "answer.py",
                  "answer_ab.py", "chat.py"]:
            out = subprocess.run(
                ["grep", "-nE", "import sufficiency|sufficiency\.", f],
                capture_output=True, text=True)
            self.assertEqual(out.returncode, 1,
                             f"{f} executes sufficiency outside the gate")
        # service.py: exactly ONE reference, and it is the gated import
        # (indented inside `if want_sufficiency and ...`)
        out = subprocess.run(["grep", "-n", "import sufficiency", "service.py"],
                             capture_output=True, text=True)
        self.assertEqual(out.returncode, 0, "service.py lost its gated import")
        lines = [l for l in out.stdout.splitlines() if "import sufficiency" in l]
        self.assertEqual(len(lines), 1)
        code = lines[0].split(":", 1)[1]      # drop grep's line-number prefix
        self.assertTrue(code.startswith("            import sufficiency"),
                        "the sufficiency import is not inside the gate branch")


class RelationalExpansion(unittest.TestCase):
    """Phase-5 item 3: retrieval expansion along the Phase-4 relations under
    the DECLARED intent policy — internal edges only for إجرائي (from) and
    استثناء (both directions), the grounding edge only when the request names
    BOTH the Circulaire and the law; depth <= 2, node cap; extras occupy the
    TAIL of the top-k window (head vector hits keep their ranks); env-gated
    default OFF."""

    @staticmethod
    def _cfg():
        return SimpleNamespace(RELATIONAL_EXPANSION_ENABLED=True)

    @staticmethod
    def _law_hit(heading="الفصل52", rank=1, score=5.0):
        return {"id": f"Loi_2016-48.pdf::chunk_{rank:04d}", "rank": rank,
                "text": "chunk text", "score": score,
                "metadata": {"document": "Loi_2016-48.pdf",
                             "source": "Loi_2016-48.pdf", "language": "ar",
                             "heading": heading}}

    def test_procedural_query_expands_with_documented_edges(self):
        import relational_expansion as rex
        hits = [self._law_hit(rank=i, score=5.0 - i * 0.1) for i in range(1, 6)]
        out = rex.expand(self._cfg(), "كيف تُمنح التراخيص للبنوك؟", hits, top_k=5)
        extras = [h for h in out if h["metadata"].get("via_relation")]
        self.assertTrue(extras)
        self.assertLessEqual(len(extras), rex.NODE_CAP)
        # art 52 references الفصول 49-51 — those units ride along, documented
        self.assertIn("loi-2016-48:art049", [h["id"] for h in extras])
        self.assertTrue(all(h["metadata"]["via_relation"].startswith("internal")
                            for h in extras))

    def test_tail_slots_only_head_preserved(self):
        import relational_expansion as rex
        hits = [self._law_hit(rank=i, score=5.0 - i * 0.1) for i in range(1, 6)]
        out = rex.expand(self._cfg(), "كيف تُمنح التراخيص للبنوك؟", hits, top_k=5)
        extras = [h for h in out if h["metadata"].get("via_relation")]
        head = out[:len(out) - len(extras)]
        self.assertEqual([h["id"] for h in head],
                         [h["id"] for h in hits[:len(head)]])   # head untouched
        self.assertEqual([h["rank"] for h in out], list(range(1, len(out) + 1)))
        self.assertLessEqual(len(out), 5)                        # top_k respected

    def test_expansion_rescues_missing_evidence(self):
        """The mechanism proof: a case whose expected evidence lives in a
        REFERENCED article's verbatim text is served by the expansion."""
        import relational_expansion as rex
        import evaluate
        hits = [self._law_hit(rank=i, score=5.0 - i * 0.1) for i in range(1, 6)]
        out = rex.expand(self._cfg(), "كيف تُمنح التراخيص للبنوك؟", hits, top_k=5)
        law_units = rex._law_context()[0]
        art49 = next(u for u in law_units if u["unit_id"].endswith("art049"))
        case = {"expected_document": "Loi_2016-48.pdf", "expected_lang": "ar",
                "expected_substring": art49["text"][100:220]}
        ch = evaluate.find_correct_hit(case, out)
        self.assertIsNotNone(ch)
        self.assertLessEqual(ch["rank"], 5)

    def test_policy_is_narrow(self):
        import relational_expansion as rex
        # definitional / numeric / comparison requests: no expansion at all
        for q in ("ما هي المرابحة؟", "ما الحد الأقصى لرأس المال؟",
                  "قارن بين صيغتي المضاربة"):
            out = rex.expand(self._cfg(), q, [self._law_hit()], top_k=5)
            self.assertEqual([h for h in out if h.get("metadata", {}).get("via_relation")],
                             [], q)
        # grounding fires only when BOTH document families are named AND
        # Circulaire evidence was retrieved (the edge is Circulaire→art011)
        circ_hit = {"id": "Circulaire_BCT_2019-08.pdf::chunk_0002", "rank": 1,
                    "text": "منشور", "score": 4.0,
                    "metadata": {"document": "Circulaire_BCT_2019-08.pdf",
                                 "source": "Circulaire_BCT_2019-08.pdf",
                                 "language": "ar", "heading": "الفصل 2"}}
        both = rex.expand(self._cfg(),
                          "ماذا يقول المنشور وقانون عدد 48 عن المضاربة؟",
                          [circ_hit, self._law_hit(rank=2, score=3.0)], top_k=5)
        self.assertIn("loi-2016-48:art011",
                      [h["id"] for h in both])
        # the law alone (no Circulaire hit) does not trigger the grounding
        law_only = rex.expand(self._cfg(), "ماذا يقول قانون عدد 48 عن المضاربة؟",
                              [self._law_hit()], top_k=5)
        self.assertNotIn("loi-2016-48:art011",
                         [h["id"] for h in law_only
                          if h.get("metadata", {}).get("via_relation")])

    def test_default_off_and_inert_when_disabled(self):
        import config
        self.assertFalse(config.RELATIONAL_EXPANSION_ENABLED)   # default OFF
        # the retrieval wiring is guarded: disabled flag → retrieve untouched
        src = (Path(__file__).resolve().parent / "retrieval.py").read_text(encoding="utf-8")
        self.assertIn('getattr(cfg, "RELATIONAL_EXPANSION_ENABLED", False)', src)


    def test_extras_carry_mode_score_key_regression(self):
        """Live-run regression (run 36868987998): extras without the active
        mode's ranking key left score=None in the evaluate record and the
        separation block crashed on max(None, float). Extras must carry the
        mode key at 0.0 so downstream score math never sees None."""
        import relational_expansion as rex
        hits = [self._law_hit(rank=i, score=5.0 - i * 0.1) for i in range(1, 6)]
        for key in ("similarity", "rrf_score", "blend_score"):
            # live head hits carry the mode key (from query_vector / fusion)
            live = [dict(h, **{key: h["score"]}) for h in hits]
            out = rex.expand(self._cfg(), "كيف تُمنح التراخيص للبنوك؟",
                             live, top_k=5, score_key=key)
            extras = [h for h in out if h["metadata"].get("via_relation")]
            self.assertTrue(extras)
            # every hit has a non-None score under the active mode's key —
            # the evaluate separation max() is safe on the full window
            scores = [h.get(key) for h in out]
            self.assertTrue(all(s is not None for s in scores))
            self.assertTrue(all(h[key] == 0.0 for h in extras))  # tail rank
            self.assertEqual(max(scores), max(5.0 - i * 0.1 for i in range(1, 6)))


class UnitsExtraction(unittest.TestCase):
    """Phase-4 item 1: Loi 2016-48 knowledge units — article-level split from
    the adopted codex through the SHARED marker machinery (restructure.
    _extract_markers), verbatim bodies (paragraph breaks preserved), stable
    article-based ids, governed functional types."""

    @staticmethod
    def _units():
        import restructure, units
        codex = restructure._adopted_codex_text("Loi_2016-48.pdf")
        return units.extract_law_units(codex), codex

    def test_full_coverage_and_verbatim(self):
        import units
        uls, codex = self._units()
        cov = units.coverage_report(uls, codex)
        self.assertEqual(cov["n_units"], 198)
        self.assertEqual(cov["first_article"], 1)
        self.assertEqual(cov["last_article"], 198)
        self.assertEqual(cov["missing_numbers"], [])
        self.assertEqual(cov["duplicate_ids"], 0)
        self.assertEqual(cov["empty_bodies"], 0)
        self.assertEqual(cov["empty_descriptions"], 0)
        self.assertTrue(cov["all_text_verbatim_in_codex"])

    def test_stable_ids_and_determinism(self):
        uls, _ = self._units()
        again, _ = self._units()
        self.assertEqual(uls, again)
        self.assertEqual([u["unit_id"] for u in uls],
                         [f"loi-2016-48:art{i:03d}" for i in range(1, 199)])

    def test_ordinal_first_article_and_heading_paths(self):
        uls, _ = self._units()
        first = uls[0]
        self.assertEqual(first["heading"], "الفصل الاول")
        self.assertIn("الفصل الاول", first["path"])
        self.assertIn(">", first["path"])          # العنوان > الباب > الفصل
        art183 = next(u for u in uls if u["unit_id"].endswith("art183"))
        self.assertEqual(art183["path"],
                         "العنوان التاسع > الباب الثاني > الفصل183")

    def test_governed_types_on_known_articles(self):
        uls, _ = self._units()
        by_id = {u["unit_id"]: u for u in uls}
        self.assertEqual(by_id["loi-2016-48:art183"]["type"], "penalty")
        self.assertTrue(by_id["loi-2016-48:art183"]["numeric"])
        self.assertEqual(by_id["loi-2016-48:art010"]["type"], "definition")
        # the type table's declared order is the governance: first match wins
        import units
        self.assertEqual(units._classify("يعاقب بالسجن...")[0], "penalty")
        self.assertEqual(units._classify("تعتبر خدمات دفع على معنى")[0], "definition")
        self.assertEqual(units._classify("نص عام لا قواعد له")[0], "general")

    def test_description_is_the_body_head(self):
        import re
        uls, _ = self._units()
        for u in uls[::37]:               # deterministic sample
            flat = re.sub(r"\s+", " ", u["text"]).strip()
            self.assertTrue(flat.startswith(u["description"][:40]))
            self.assertLessEqual(len(u["description"]), 181)

    def test_emitted_data_file_matches_extraction(self):
        import json as _json
        from pathlib import Path as _Path
        uls, _ = self._units()
        data = _json.loads((_Path(__file__).resolve().parent
                            / "units_loi_2016_48.json").read_text(encoding="utf-8"))
        self.assertEqual(data["units"], uls)


class TargetSetIntegrity(unittest.TestCase):
    """Phase-3 authoring round: the target-category set (questions_targets.json)
    stays schema-valid, balanced (2 per target category) and evidence-anchored
    to the four adopted documents. PRESENTED-FOR-ADOPTION state is part of the
    file's _comment; adoption flips it in this test's expectations."""

    def test_target_set_loads_balanced_and_anchored(self):
        import evaluate
        cases = evaluate.load_question_set(
            Path(__file__).resolve().parent / "questions_targets.json")
        self.assertEqual(len(cases), 10)
        cats = {}
        for c in cases:
            cats[c["category"]] = cats.get(c["category"], 0) + 1
            self.assertIn(c["expected_document"], (
                "Circulaire_BCT_2019-08.pdf",
                "Guide_Interne_Operations_Bancaires_Islamiques.docx",
                "Loi_2016-48.pdf", "Madkhal_Sayrafa_Islamiya.docx"))
            self.assertEqual(c["language"], "ar")
            if c["category"] == "ambiguous":
                self.assertTrue((c.get("ambiguity_note") or "").strip())
            if c["category"] in ("implicit", "compound"):
                self.assertEqual(len(c["expected_substrings"]), 2)
        self.assertEqual(cats, {"colloquial": 2, "synonyms": 2, "implicit": 2,
                                "compound": 2, "ambiguous": 2})
        # ids are unique and namespaced away from the adopted 50-set
        self.assertEqual(len({c["id"] for c in cases}), 10)
        self.assertTrue(all(c["id"].startswith("t") for c in cases))


class RealReportModes(unittest.TestCase):
    """Phase 2 step 3.2: real_report.py builds the retrieval-mode comparison
    (vector vs rrf vs blend on the restructure arm) and carries it into the
    ANNO annotation line the workflow reads from the sandbox."""

    @staticmethod
    def _run(mode, ranks, generated_at="2026-10-01T00:00:00+00:00"):
        """Minimal evaluate.save_run-shaped JSON: ranks maps qid -> correct_rank
        (None = miss). Two answerable questions (ar + fr) and one OOS."""
        questions = []
        for qid, lang, rank in (("q1", "ar", ranks["q1"]),
                                ("q2", "fr", ranks["q2"])):
            hit = None if rank is None else {"id": f"{qid}::chunk_0001",
                                             "score": 0.5, "text": "نص",
                                             "document": "D", "heading": "H"}
            questions.append({"id": qid, "question": "س", "language": lang,
                              "category": "verbatim", "is_out_of_scope": False,
                              "correct_rank": rank,
                              "correct_any_lang_rank": None,
                              "hits": [hit] if hit else []})
        questions.append({"id": "q3", "question": "خارج", "language": "ar",
                          "category": "out-of-scope", "is_out_of_scope": True,
                          "correct_rank": None, "correct_any_lang_rank": None,
                          "hits": [{"id": "x", "score": 0.2, "text": "t",
                                    "document": "D", "heading": "H"}]})
        n_ans = 2
        hit1 = sum(1 for r in ranks.values() if r == 1) / n_ans
        hit3 = sum(1 for r in ranks.values() if r is not None and r <= 3) / n_ans
        hit5 = sum(1 for r in ranks.values() if r is not None and r <= 5) / n_ans
        fr_rank = ranks["q2"]
        return {
            "generated_at": generated_at,
            "config": {"provider": "nvidia", "embedding_model": "m",
                       "retrieval_mode": mode, "retrieval_top_k": 20,
                       "chunk_size_tokens": 220, "chunk_overlap_tokens": 40,
                       "hybrid_blend_lambda": 0.7 if mode == "blend" else None},
            "metrics": {
                "overall": {"n": n_ans, "hit@1": hit1, "hit@3": hit3,
                            "hit@5": hit5},
                "by_category": {"verbatim": {"n": n_ans, "hit@1": hit1,
                                             "hit@3": hit3, "hit@5": hit5}},
                "by_language": {
                    "ar": {"n": 1, "hit@1": 1.0 if ranks["q1"] == 1 else 0.0,
                           "hit@3": 1.0 if ranks["q1"] else 0.0,
                           "hit@5": 1.0 if ranks["q1"] else 0.0},
                    "fr": {"n": 1, "hit@1": 1.0 if fr_rank == 1 else 0.0,
                           "hit@3": 1.0 if fr_rank else 0.0,
                           "hit@5": 1.0 if fr_rank else 0.0}},
                "separation": {"n_correct_retrieved": n_ans,
                               "mean_correct_score": 0.5,
                               "n_with_best_incorrect": 1,
                               "mean_best_incorrect_score": 0.4,
                               "gap_mean_correct_minus_best_incorrect": 0.1},
                "out_of_scope": {"n": 1, "max_top1_score": 0.2,
                                 "mean_top1_score": 0.2},
            },
            "questions": questions,
        }

    def test_modes_section_and_anno(self):
        import real_report as rr
        size = self._run("vector", {"q1": 3, "q2": None})
        vector = self._run("vector", {"q1": 1, "q2": None})
        rrf = self._run("rrf", {"q1": 1, "q2": 2})
        blend = self._run("blend", {"q1": 2, "q2": 2})
        runs = {"size": size, "restructure": vector}
        modes = {"vector": vector, "rrf": rrf, "blend": blend}
        md = rr.build_md(runs, None, top_k=20, modes=modes)
        self.assertIn("Retrieval modes on the restructure arm", md)
        self.assertIn("rrf (vector + BM25, reciprocal rank fusion)", md)
        self.assertIn("blend (LAMBDA*cosine + BM25 score)", md)
        # misses lists per mode: q2 misses in vector, hits everywhere else
        self.assertIn("**misses vector (cosine only)** (1): q2 (fr/verbatim)", md)
        self.assertIn("**misses rrf (vector + BM25, reciprocal rank fusion)** (0): none", md)
        # flips between modes
        self.assertIn("**flips vector → rrf:** +1 (q2) −0 (none)", md)
        self.assertIn("Blend λ = 0.7", md)
        anno = rr.build_anno(runs, None, modes=modes)
        # q1: vector rank1, rrf rank1, blend rank2; q2: vector miss, rrf rank2,
        # blend rank2 -> overall 50/50/50, 50/100/100, 0/100/100
        self.assertIn("modes vector=50/50/50 rrf=50/100/100 blend=0/100/100",
                      anno)
        self.assertIn("modes_fr vector=0/0/0 rrf=0/100/100 blend=0/100/100",
                      anno)
        self.assertIn("mode_misses vector=[q2]", anno)
        self.assertIn("mode_misses rrf=[]", anno)
        # a rerank run joins the table as a fourth labeled row (Phase-3 int. 3)
        rerank_run = self._run("vector", {"q1": 1, "q2": 1})
        modes["rerank"] = rerank_run
        md3 = rr.build_md(runs, None, top_k=20, modes=modes)
        self.assertIn("vector + deterministic rerank (Phase-3 intervention 3)", md3)
        self.assertIn("| key | n | vector | rrf | blend | rerank |", md3)
        self.assertIn("**flips vector → rerank:**", md3)
        anno3 = rr.build_anno(runs, None, modes=modes)
        self.assertIn("rerank=100/100/100", anno3)
        self.assertIn("mode_misses rerank=[]", anno3)
        # without modes the section and the segments disappear
        md2 = rr.build_md(runs, None, top_k=20)
        self.assertNotIn("Retrieval modes on the restructure arm", md2)
        anno2 = rr.build_anno(runs, None)
        self.assertNotIn("modes_fr", anno2)


if __name__ == '__main__':
    unittest.main()
