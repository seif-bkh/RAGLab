"""Offline contract, provenance, safety, and regression tests; no API access."""
import io
import json
import os
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



if __name__ == '__main__':
    unittest.main()
