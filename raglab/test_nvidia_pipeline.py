"""Offline contract, provenance, safety, and regression tests; no API access."""
import inspect
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

            def answer(self, question, hits, language=None, use_cache=True):
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
            def answer(self, question, hits, language=None, use_cache=True):
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


if __name__ == '__main__':
    unittest.main()
