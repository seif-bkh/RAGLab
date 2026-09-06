"""Offline integrity/continuation contracts for the 3,000-question harness."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from artifacts import fingerprint, write_json
from hard_harness.common import CheckpointClient, HarnessPause, parse_object, write_jsonl, read_jsonl
from hard_harness.publishing import file_parts
from hard_harness.sources import units_from_text, image_message
from nvidia_api import NvidiaAPIError
from pipeline_policy import ANSWER_MODEL


class FakeClient:
    base_url = 'https://api.xkiro.com/v1'
    timeout, attempts, min_interval, max_retry_delay = 120, 2, 3, 60

    def __init__(self, result=None, error=None):
        self.calls = 0
        self.result = result or {'text': 'not valid JSON', 'seconds': 1, 'served_model': ANSWER_MODEL}
        self.error = error

    def chat(self, *args, **kwargs):
        self.calls += 1
        if self.error:
            raise self.error
        return self.result


class Checkpoints(unittest.TestCase):
    def setUp(self):
        from unittest.mock import patch
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        plan = Path(temporary.name)/'plan.json'
        write_json(plan, {'llm':{'provider':'xkiro','model':ANSWER_MODEL,'credential_secret':'XKIRO_API_KEY_JINKO'},
                          'google_fallback_authorized':True,'google_fallback_active':False})
        patched = patch('hard_harness.common.PLAN_PATH', plan)
        patched.start()
        self.addCleanup(patched.stop)

    def test_reference_provider_switch_does_not_switch_candidate_target(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as temp:
            plan=Path(temp)/'plan.json'
            write_json(plan,{'llm':{'provider':'google','model':'gemini-3.1-flash-lite',
                                  'credential_secret':'GEMINI_API_KEY','free_tier_project_confirmed':True},
                             'candidate_llm':{'provider':'xkiro','model':ANSWER_MODEL,'credential_secret':'XKIRO_API_KEY_JINKO'},
                             'google_fallback_authorized':True,'google_fallback_active':True})
            with patch('hard_harness.common.PLAN_PATH',plan):
                reference=CheckpointClient('question_author',call_limit=1,cache_root=temp,client=FakeClient())
                candidate=CheckpointClient('candidate',call_limit=1,cache_root=temp,client=FakeClient())
            self.assertEqual(reference.provider,'google')
            self.assertEqual(reference.model,'gemini-3.1-flash-lite')
            self.assertEqual(candidate.provider,'xkiro')
            self.assertEqual(candidate.model,ANSWER_MODEL)
            self.assertEqual(candidate.credential_alias,'XKIRO_API_KEY_JINKO')

    def test_json_mode_changes_request_identity_but_preserves_old_candidate_cache(self):
        with tempfile.TemporaryDirectory() as temp:
            fake = FakeClient(result={'text':'{"ok":true}'})
            fake.json_mode = False
            client = CheckpointClient('format-test', call_limit=3, cache_root=temp, client=fake)
            first = client.chat(ANSWER_MODEL, [])
            fake.json_mode = True
            second = client.chat(ANSWER_MODEL, [])
            self.assertNotEqual(first['_harness_request_hash'], second['_harness_request_hash'])
            self.assertEqual(fake.calls, 2)
            fake.json_mode = False
            self.assertTrue(client.chat(ANSWER_MODEL, [])['_harness_cached'])
            self.assertEqual(fake.calls, 2)

    def test_json_mode_is_reference_only_not_an_answerer_change(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as temp, \
             patch.dict(__import__('os').environ, {'XKIRO_API_KEY_JINKO':'test-only'}), \
             patch('hard_harness.common.load_pricing', return_value={}), \
             patch('hard_harness.common.FreeGatewayClient', return_value=FakeClient()) as factory:
            CheckpointClient('candidate', call_limit=2, cache_root=temp)
            self.assertFalse(factory.call_args.kwargs['json_mode'])
            CheckpointClient('question_author', call_limit=2, cache_root=temp)
            self.assertTrue(factory.call_args.kwargs['json_mode'])

    def test_completed_response_is_reused_even_if_later_json_validation_fails(self):
        with tempfile.TemporaryDirectory() as temp:
            fake = FakeClient()
            client = CheckpointClient('candidate-test', call_limit=2, cache_root=temp, client=fake)
            first = client.chat(ANSWER_MODEL, [{'role': 'user', 'content': 'question'}])
            with self.assertRaises(ValueError):
                parse_object(first['text'])
            repeated = client.chat(ANSWER_MODEL, [{'role': 'user', 'content': 'question'}])
            self.assertTrue(repeated['_harness_cached'])
            self.assertEqual(fake.calls, 1)
            other = CheckpointClient('candidate-test', call_limit=2, cache_root=temp, client=fake)
            self.assertTrue(other.chat(ANSWER_MODEL, [{'role': 'user', 'content': 'question'}])['_harness_cached'])
            self.assertEqual(fake.calls, 1)

    def test_roles_and_changed_inputs_do_not_share_responses(self):
        with tempfile.TemporaryDirectory() as temp:
            fake = FakeClient(result={'text': '{"ok":true}'})
            a = CheckpointClient('author-test', call_limit=3, cache_root=temp, client=fake)
            b = CheckpointClient('grader-test', call_limit=3, cache_root=temp, client=fake)
            a.chat(ANSWER_MODEL, [])
            b.chat(ANSWER_MODEL, [])
            a.chat(ANSWER_MODEL, [{'role': 'user', 'content': 'changed'}])
            self.assertEqual(fake.calls, 3)

    def test_terminal_output_error_is_preserved(self):
        with tempfile.TemporaryDirectory() as temp:
            fake = FakeClient(error=NvidiaAPIError('truncated output', 422))
            client = CheckpointClient('terminal-test', call_limit=2, cache_root=temp, client=fake)
            for _ in range(2):
                with self.assertRaises(NvidiaAPIError):
                    client.chat(ANSWER_MODEL, [])
            self.assertEqual(fake.calls, 1)

    def test_quota_pauses_without_trying_another_model(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as temp, patch('hard_harness.common.OUTPUT', Path(temp)/'output'):
            fake = FakeClient(error=NvidiaAPIError('quota exhausted', 429))
            client = CheckpointClient('quota-test', call_limit=2, cache_root=temp, client=fake)
            with self.assertRaises(NvidiaAPIError):
                client.chat(ANSWER_MODEL, [])
            with self.assertRaises(HarnessPause):
                client.check_pause()
            with self.assertRaises(NvidiaAPIError):
                client.chat(ANSWER_MODEL, [{'role': 'user', 'content': 'another question'}])
            self.assertEqual(fake.calls, 1)
            self.assertTrue((Path(temp)/'output/pause_quota-test.json').exists())

    def test_jsonl_is_round_trip_and_fingerprint_stable(self):
        with tempfile.TemporaryDirectory() as temp:
            rows = [{'id': 'ar_1', 'question': 'ما هي المرابحة؟'}, {'id': 'fr_1', 'question': 'Quel contrat ?'}]
            path = Path(temp)/'questions.jsonl'
            write_jsonl(path, rows)
            self.assertEqual(fingerprint(read_jsonl(path)), fingerprint(rows))

    def test_parse_requires_a_single_object(self):
        self.assertEqual(parse_object('```json\n{"ok":true}\n```'), {'ok': True})
        with self.assertRaises(ValueError):
            parse_object('[]')
        with self.assertRaises(ValueError):
            parse_object('commentary {"ok":true}')

    def test_source_units_flag_replacement_characters(self):
        text = ('هذه فقرة في دليل العمليات المصرفية وشروط العقود. ' * 8) + 'نص غير مقروء \ufffd'
        units = units_from_text('Guide.docx', text)
        self.assertEqual(len(units), 1)
        self.assertFalse(units[0]['eligible_for_reference'])
        self.assertIn('replacement_characters', units[0]['issues'])

    def test_reference_images_are_explicit_source_inputs(self):
        message = image_message('transcribe source', b'image-bytes')
        self.assertEqual(message['role'], 'user')
        self.assertEqual(message['content'][1]['type'], 'image_url')
        self.assertTrue(message['content'][1]['image_url']['url'].startswith('data:image/jpeg;base64,'))

    def test_large_checkpoint_file_is_split_without_losing_rows(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rows = [{'id': i, 'answer': 'ع' * 7000} for i in range(8)]
            path = root/'answers.json'
            write_json(path, rows)
            parts = list(file_parts(path, root))
            self.assertGreater(len(parts), 1)
            restored = [row for part in parts for row in part['data']]
            self.assertEqual(restored, rows)
            self.assertTrue(all(p['fingerprint'] == fingerprint(rows) for p in parts))


class GoogleFallbackPolicy(unittest.TestCase):
    def test_google_requires_explicit_free_project_confirmation(self):
        from hard_harness.google_client import GoogleHarnessClient
        with self.assertRaises(ValueError):
            GoogleHarnessClient('gemini-3.1-flash-lite','dummy',free_project_confirmed=False,budget={'used':0,'limit':1})
        with self.assertRaises(ValueError):
            GoogleHarnessClient('some-paid-model','dummy',free_project_confirmed=True,budget={'used':0,'limit':1})

    def test_google_capacity_retry_is_bounded_and_quota_does_not_retry(self):
        import io, urllib.error
        from unittest.mock import patch
        from hard_harness.google_client import GoogleHarnessClient
        class Response:
            def __enter__(self): return self
            def __exit__(self,*args): return False
            def read(self):
                return json.dumps({'candidates':[{'finishReason':'STOP','content':{'parts':[{'text':'{"ok":true}'}]}}],
                                   'modelVersion':'gemini-3.1-flash-lite'}).encode()
        seen=[]
        def capacity_then_success(*args,**kwargs):
            seen.append(1)
            if len(seen)<3:
                raise urllib.error.HTTPError('https://generativelanguage.googleapis.com',503,'busy',{},io.BytesIO(b'busy'))
            return Response()
        client=GoogleHarnessClient('gemini-3.1-flash-lite','test-key',free_project_confirmed=True,
                                  budget={'used':0,'limit':2},opener=SimpleNamespace(open=capacity_then_success))
        with patch('hard_harness.google_client.time.sleep') as sleep:
            self.assertEqual(client.chat(client.model,[])['text'],'{"ok":true}')
            self.assertEqual(client.calls,3)
            waits=[c.args[0] for c in sleep.call_args_list]
            self.assertIn(30,waits); self.assertIn(60,waits)
        def quota(*args,**kwargs):
            raise urllib.error.HTTPError('https://generativelanguage.googleapis.com',429,'quota',{},io.BytesIO(b'quota'))
        client.opener=SimpleNamespace(open=quota)
        with patch('hard_harness.google_client.time.sleep'):
            with self.assertRaises(NvidiaAPIError): client.chat(client.model,[])
        self.assertEqual(client.calls,4)  # one quota attempt, never a hidden project switch

    def test_google_translates_only_explicit_messages_and_local_images(self):
        from hard_harness.google_client import google_payload
        messages=[{'role':'system','content':'policy'},image_message('read source',b'image')]
        data=google_payload(messages,2000)
        self.assertEqual(data['systemInstruction']['parts'][0]['text'],'policy')
        self.assertEqual(data['generationConfig']['responseMimeType'],'application/json')
        self.assertIn('inlineData',data['contents'][0]['parts'][1])
        with self.assertRaises(ValueError):
            google_payload([{'role':'user','content':[{'type':'image_url','image_url':{'url':'https://untrusted.example/image'}}]}],100)


class FreeTierPacing(unittest.TestCase):
    """A 20-requests-per-minute free tier is a scheduling limit, not a stop signal.

    Pacing must absorb the advertised minute-window wait, yet a daily quota or an
    unlabelled 429 still has to halt the fleet for a user decision.
    """

    def setUp(self):
        from unittest.mock import patch
        import hard_harness.google_client as gc
        self.gc = gc
        self.saved = dict(gc._PACE_INTERVAL), dict(gc._LAST_REQUEST)
        for name in ('_PACE_INTERVAL', '_LAST_REQUEST', '_RATE_LIMIT_EVENTS'):
            patcher = patch.object(gc, name, {})
            patcher.start(); self.addCleanup(patcher.stop)

    def response(self, body='{"ok":true}', model='gemini-3.5-flash'):
        gc = self.gc
        class Response:
            def __enter__(self): return self
            def __exit__(self, *args): return False
            def read(self):
                return json.dumps({'candidates': [{'finishReason': 'STOP',
                                                   'content': {'parts': [{'text': body}]}}],
                                   'modelVersion': model}).encode()
        return Response()

    def http_error(self, code, body):
        import io, urllib.error
        return urllib.error.HTTPError('https://generativelanguage.googleapis.com', code, 'err', {},
                                      io.BytesIO(body.encode()))

    def client(self, opener, **pacing):
        return self.gc.GoogleHarnessClient('gemini-3.5-flash', 'test-key', free_project_confirmed=True,
                                           budget={'used': 0, 'limit': 50},
                                           opener=SimpleNamespace(open=opener), pacing=pacing or None)

    def test_advertised_minute_window_wait_is_paced_and_slows_the_shared_interval(self):
        from unittest.mock import patch
        seen = []
        def opener(*args, **kwargs):
            seen.append(1)
            if len(seen) == 1:
                raise self.http_error(429, 'Quota exceeded for metric: generate_content_free_tier_requests, '
                                           'limit: 20, model: gemini-3.5-flash. Please retry in 14.15s')
            return self.response()
        client = self.client(opener)
        with patch.object(self.gc.time, 'sleep') as sleep:
            result = client.chat(client.model, [])
        self.assertEqual(result['text'], '{"ok":true}')
        self.assertEqual(client.calls, 2)
        self.assertEqual(client.rate_limit_events, 1)
        self.assertEqual(result['rate_limit_waits'], 1)
        waits = [c.args[0] for c in sleep.call_args_list]
        # The provider asked for 14s; a paced wait must be at least that long, never
        # longer than the cap, and exactly one extra request was made.
        self.assertTrue(any(14.0 <= w <= 240.0 for w in waits), waits)
        self.assertGreaterEqual(min(waits), self.gc.DEFAULT_PACING['min_interval_seconds'])
        self.assertGreaterEqual(self.gc._PACE_INTERVAL[self.gc.BASE_URL], 20.0)

    def test_unlabelled_or_long_window_quota_raises_without_retries(self):
        from unittest.mock import patch
        for body in ('quota exhausted', 'Please retry in 3600s'):
            with self.subTest(body=body):
                calls = []
                def opener(*args, _calls=calls, _body=body, **kwargs):
                    _calls.append(1)
                    raise self.http_error(429, _body)
                client = self.client(opener)
                with patch.object(self.gc.time, 'sleep') as sleep:
                    with self.assertRaises(NvidiaAPIError) as caught:
                        client.chat(client.model, [])
                self.assertEqual(caught.exception.status_code, 429)
                self.assertEqual(len(calls), 1)   # no hidden retry loop on a real quota
                self.assertEqual(client.rate_limit_events, 0)
                # Only normal pacing may sleep here: never a quota wait loop.
                self.assertTrue(all(w <= client.min_interval for w in
                                    [c.args[0] for c in sleep.call_args_list]), sleep.call_args_list)

    def test_sustained_429s_stop_after_the_allowed_waits(self):
        from unittest.mock import patch
        calls = []
        def opener(*args, **kwargs):
            calls.append(1)
            raise self.http_error(429, 'Please retry in 30s')
        client = self.client(opener, quota_retry_attempts=2)
        with patch.object(self.gc.time, 'sleep') as sleep:
            with self.assertRaises(NvidiaAPIError):
                client.chat(client.model, [])
        self.assertEqual(len(calls), 3)           # original attempt + two paced waits
        self.assertEqual(client.calls, 3)
        self.assertEqual(client.rate_limit_events, 2)

    def test_one_shared_event_budget_stops_grinding_a_hard_limit(self):
        from unittest.mock import patch
        calls = []
        def opener(*args, **kwargs):
            calls.append(1)
            raise self.http_error(429, 'Please retry in 30s')
        # Four families would each retry; the shared budget must stop the shard
        # instead of spending the quota it is waiting for.
        client = self.client(opener, rate_limit_event_budget=3, quota_retry_attempts=2)
        for _ in range(3):
            with patch.object(self.gc.time, 'sleep'):
                with self.assertRaises(NvidiaAPIError):
                    client.chat(client.model, [])
        self.assertEqual(self.gc._rate_limit_events(self.gc.BASE_URL), 3)
        self.assertEqual(len(calls), 6)          # two paced waits, then one probe per call
        before = len(calls)
        with patch.object(self.gc.time, 'sleep') as sleep:
            with self.assertRaises(NvidiaAPIError):
                client.chat(client.model, [])
        self.assertEqual(len(calls), before + 1)   # a single probe, never a retry loop
        # No extra quota wait is taken — at most the ordinary pacing interval sleeps
        # — so the shard aborts instead of grinding away the quota it is waiting on.
        self.assertLessEqual(len(sleep.call_args_list), 1)
        self.assertEqual(self.gc._rate_limit_events(self.gc.BASE_URL), 3)
        self.assertEqual(client.rate_limit_events, 3)  # budget reached, not exceeded

    def test_pacing_keys_are_validated_and_never_silently_loosened(self):
        for pacing in ({'min_interval_seconds': 1}, {'max_retry_delay_seconds': 5},
                       {'quota_retry_attempts': 50}, {'rate_limit_floor_seconds': 1},
                       {'rate_limit_event_budget': 0}, {'requests_per_minute': 900}):
            with self.subTest(pacing=pacing):
                with self.assertRaises(ValueError):
                    self.client(lambda *a, **k: self.response(), **pacing)

    def test_plan_pacing_reaches_the_google_client(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as temp:
            plan = Path(temp)/'plan.json'
            write_json(plan, {'llm': {'provider': 'google', 'model': 'gemini-3.5-flash',
                                      'credential_secret': 'GEMINI_API_KEY',
                                      'free_tier_project_confirmed': True,
                                      'pacing': {'min_interval_seconds': 9.0}},
                              'candidate_llm': {'provider': 'xkiro', 'model': ANSWER_MODEL,
                                                'credential_secret': 'XKIRO_API_KEY_JINKO'},
                              'google_fallback_authorized': True, 'google_fallback_active': True})
            captured = {}
            class Recorder:
                def __init__(self, model, key, **kwargs): captured.update(kwargs)
                base_url, timeout, attempts = 'https://recorder.invalid', 120, 3
                min_interval, max_retry_delay = 9.0, 240.0
                calls, rate_limit_events = 0, 0
            with patch('hard_harness.common.PLAN_PATH', plan), \
                 patch.dict(__import__('os').environ, {'GEMINI_API_KEY': 'test-only'}), \
                 patch('hard_harness.google_client.GoogleHarnessClient', Recorder):
                CheckpointClient('question_author', call_limit=1, cache_root=temp)
            self.assertEqual(captured['pacing'], {'min_interval_seconds': 9.0})
            self.assertIs(captured['free_project_confirmed'], True)


class DeadlineCheckpoints(unittest.TestCase):
    def test_job_deadline_env_overrides_the_plan_and_is_validated(self):
        import os
        from unittest.mock import patch
        from hard_harness.common import deadline_reached, soft_deadline
        with patch.dict(os.environ, {'HARNESS_DEADLINE_MINUTES': '3'}):
            deadline = soft_deadline({'shard_deadline_minutes': 45})
            self.assertIsNotNone(deadline)
            self.assertFalse(deadline_reached(deadline))
            self.assertTrue(deadline_reached(deadline - 10_000))
        with patch.dict(os.environ, {}):
            self.assertIsNone(soft_deadline({}))
            self.assertIsNotNone(soft_deadline({'shard_deadline_minutes': 45}))
        for raw in ('abc', '0.5'):
            with self.subTest(raw=raw), patch.dict(os.environ, {'HARNESS_DEADLINE_MINUTES': raw}):
                with self.assertRaises(ValueError):
                    soft_deadline({})

    def test_a_lone_transport_blip_does_not_stop_the_shard_but_a_streak_does(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            plan = root/'plan.json'
            write_json(plan, {'llm': {'provider': 'xkiro', 'model': ANSWER_MODEL,
                                      'credential_secret': 'XKIRO_API_KEY_JINKO'},
                             'google_fallback_authorized': True, 'google_fallback_active': False,
                             'transport_failure_streak': 2})
            with patch('hard_harness.common.PLAN_PATH', plan), patch('hard_harness.common.OUTPUT', root/'out'):
                client = CheckpointClient('question_author', call_limit=9, cache_root=temp,
                                          client=FakeClient(error=NvidiaAPIError('connection reset', 0)))
                with self.assertRaises(NvidiaAPIError):
                    client.chat(ANSWER_MODEL, [{'role': 'user', 'content': 'x'}])
                self.assertIsNone(client.pause)          # one blip: the case retry may recover
                with self.assertRaises(NvidiaAPIError):
                    client.chat(ANSWER_MODEL, [{'role': 'user', 'content': 'y'}])
                self.assertIsNotNone(client.pause)       # the streak proves an outage
                self.assertTrue((root/'out'/'pause_question_author.json').exists())
                self.assertEqual(client.summary()['consecutive_transport_failures'], 2)
                fake = client.client
                with self.assertRaises(NvidiaAPIError):
                    client.chat(ANSWER_MODEL, [{'role': 'user', 'content': 'z'}])
                self.assertEqual(fake.calls, 2)           # a paused client makes no further calls

    def test_gateway_capacity_blips_are_paced_before_the_fleet_pauses(self):
        """A free gateway answers with transient 429/503 capacity blips.

        Those are waited out; a spent allowance then behaves exactly like the old
        quota path so the user is still prompted instead of a silent switch.
        """
        import os
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            plan = root/'plan.json'
            write_json(plan, {'llm': {'provider': 'xkiro', 'model': ANSWER_MODEL,
                                      'credential_secret': 'XKIRO_API_KEY_JINKO',
                                      'pacing': {'max_retry_delay_seconds': 30.0, 'quota_retry_attempts': 2}},
                              'google_fallback_authorized': True, 'google_fallback_active': False})
            error = NvidiaAPIError('temporarily at capacity', 429, retry_after=7.0)
            with patch('hard_harness.common.PLAN_PATH', plan), patch('hard_harness.common.OUTPUT', root/'out'):
                client = CheckpointClient('candidate', call_limit=9, cache_root=temp, client=FakeClient(error=error))
                with patch('hard_harness.common.time.sleep') as sleep:
                    with self.assertRaises(NvidiaAPIError):
                        client.chat(ANSWER_MODEL, [{'role': 'user', 'content': 'q'}])
            self.assertEqual(client.client.calls, 3)              # original + two paced waits
            self.assertEqual([c.args[0] for c in sleep.call_args_list], [7.0, 7.0])
            self.assertEqual(client.rate_limit_waits, 2)
            self.assertIsNotNone(client.pause)                     # allowance spent: ask the user

    def test_gateway_success_after_a_wait_resets_the_pace_counter(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            plan = root/'plan.json'
            write_json(plan, {'llm': {'provider': 'xkiro', 'model': ANSWER_MODEL,
                                      'credential_secret': 'XKIRO_API_KEY_JINKO',
                                      'pacing': {'max_retry_delay_seconds': 30.0, 'quota_retry_attempts': 1}},
                              'google_fallback_authorized': True, 'google_fallback_active': False})
            class BlipThenOk:
                base_url, timeout, attempts, min_interval, max_retry_delay = 'https://api.xkiro.com/v1', 120, 2, 3, 60
                def __init__(self): self.calls = 0
                def chat(self, model, messages, *, max_tokens=4096):
                    self.calls += 1
                    if self.calls % 2:
                        raise NvidiaAPIError('temporarily at capacity', 429, retry_after=5.0)
                    return {'text': '{"ok":true}', 'served_model': ANSWER_MODEL}
            with patch('hard_harness.common.PLAN_PATH', plan), patch('hard_harness.common.OUTPUT', root/'out'), \
                 patch('hard_harness.common.time.sleep'):
                client = CheckpointClient('candidate', call_limit=9, cache_root=temp, client=BlipThenOk())
                for expected in range(1, 4):        # three cases, each needing one retry
                    self.assertEqual(client.chat(ANSWER_MODEL, [{'role': 'user', 'content': f'q{expected}'}])['text'],
                                     '{"ok":true}')
            self.assertIsNone(client.pause)
            self.assertEqual(client.rate_retries, 0)
            self.assertEqual(client.rate_limit_waits, 3)

    def test_google_paces_internally_so_the_checkpoint_layer_adds_no_second_allowance(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            plan = root/'plan.json'
            write_json(plan, {'llm': {'provider': 'google', 'model': 'gemini-3.5-flash',
                                      'credential_secret': 'GEMINI_API_KEY', 'free_tier_project_confirmed': True,
                                      'pacing': {'quota_retry_attempts': 4, 'max_retry_delay_seconds': 240.0}},
                              'google_fallback_authorized': True, 'google_fallback_active': True})
            class Paced:
                paces_rate_limits = True
                base_url, timeout, attempts, min_interval, max_retry_delay = 'https://paced.invalid', 120, 3, 6, 240
                def __init__(self): self.calls = 0
                def chat(self, model, messages, *, max_tokens=4096):
                    self.calls += 1
                    raise NvidiaAPIError('Google HTTP 429: already paced by the adapter', 429, retry_after=14.0)
            with patch('hard_harness.common.PLAN_PATH', plan), patch('hard_harness.common.OUTPUT', root/'out'), \
                 patch('hard_harness.common.time.sleep') as sleep:
                client = CheckpointClient('question_author', call_limit=9, cache_root=temp, client=Paced())
                with self.assertRaises(NvidiaAPIError):
                    client.chat('gemini-3.5-flash', [{'role': 'user', 'content': 'q'}])
            self.assertEqual(client.client.calls, 1)     # no double retry loop
            sleep.assert_not_called()
            self.assertIsNotNone(client.pause)            # adapter gave up -> ask the user

    def test_quota_still_pauses_immediately_without_a_streak(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            plan = root/'plan.json'
            write_json(plan, {'llm': {'provider': 'xkiro', 'model': ANSWER_MODEL,
                                      'credential_secret': 'XKIRO_API_KEY_JINKO'},
                              'google_fallback_authorized': True, 'google_fallback_active': False,
                              'transport_failure_streak': 8})
            with patch('hard_harness.common.PLAN_PATH', plan), patch('hard_harness.common.OUTPUT', root/'out'):
                client = CheckpointClient('question_author', call_limit=9, cache_root=temp,
                                          client=FakeClient(error=NvidiaAPIError('quota exhausted', 429)))
                with self.assertRaises(NvidiaAPIError):
                    client.chat(ANSWER_MODEL, [])
            self.assertIsNotNone(client.pause)



class RoleProfiles(unittest.TestCase):
    def test_authoring_and_auditing_resolve_to_different_provider_profiles(self):
        """The user chose hybrid sourcing: Qwen drafts, another provider audits."""
        from unittest.mock import patch
        from hard_harness.common import role_profile
        with tempfile.TemporaryDirectory() as temp:
            plan = Path(temp)/'plan.json'
            write_json(plan, {'llm': {'provider': 'google', 'model': 'gemini-3.5-flash',
                                      'credential_secret': 'GEMINI_API_KEY',
                                      'free_tier_project_confirmed': True},
                              'author_llm': {'provider': 'xkiro', 'model': ANSWER_MODEL,
                                             'credential_secret': 'XKIRO_API_KEY_JINKO'},
                              'candidate_llm': {'provider': 'xkiro', 'model': ANSWER_MODEL,
                                                 'credential_secret': 'XKIRO_API_KEY_JINKO'},
                              'google_fallback_authorized': True, 'google_fallback_active': True})
            with patch('hard_harness.common.PLAN_PATH', plan):
                author = CheckpointClient('question_author', call_limit=1, cache_root=temp, client=FakeClient())
                auditor = CheckpointClient('reference_audit', call_limit=1, cache_root=temp, client=FakeClient())
                grader = CheckpointClient('semantic_grader', call_limit=1, cache_root=temp, client=FakeClient())
                candidate = CheckpointClient('candidate', call_limit=1, cache_root=temp, client=FakeClient())
            self.assertEqual((author.provider, author.model), ('xkiro', ANSWER_MODEL))
            self.assertEqual(author.credential_alias, 'XKIRO_API_KEY_JINKO')
            self.assertEqual((auditor.provider, auditor.model), ('google', 'gemini-3.5-flash'))
            self.assertEqual((grader.provider, grader.model), ('google', 'gemini-3.5-flash'))
            self.assertEqual((candidate.provider, candidate.model), ('xkiro', ANSWER_MODEL))
            # A role without its own profile inherits the shared reference profile.
            self.assertEqual(role_profile(json.loads(plan.read_text()), 'absence_audit')['provider'], 'google')
            # The author role's pacing block is used, and only it, so the audit gets
            # no accidental xKiro retry allowance.
            self.assertEqual(auditor.rate_allowance, 0)

    def test_an_unapproved_reference_profile_is_refused_before_any_call(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as temp:
            plan = Path(temp)/'plan.json'
            write_json(plan, {'llm': {'provider': 'openai', 'model': 'gpt-x', 'credential_secret': 'X'},
                             'google_fallback_authorized': True, 'google_fallback_active': True})
            with patch('hard_harness.common.PLAN_PATH', plan):
                with self.assertRaises(HarnessPause):
                    CheckpointClient('reference_audit', call_limit=1, cache_root=temp, client=FakeClient())


class RetrievalJudge(unittest.TestCase):
    """The LLM-free arm: labels are span containment, so nothing here is model-judged."""

    class Chunk:
        def __init__(self, text, source='doc-a.pdf', language='ar'):
            self.text, self.source, self.language = text, source, language

    def corpus(self, *texts):
        from hard_harness.retrieval_judge import Corpus
        return Corpus([self.Chunk(text) for text in texts])

    def family(self, quote, *, question='what must the bank obtain first?', languages=None):
        return {'id': 'hh0001', 'category': 'supported', 'evidence': [{'unit_id': 'u', 'quote': quote}],
                'languages': languages or {lang: {'question': question} for lang in ('ar', 'fr', 'en')}}

    def test_evidence_is_found_by_containment_not_by_a_similarity_vote(self):
        from hard_harness.retrieval_judge import Corpus, judge_families
        corpus = self.corpus('The bank must obtain written consent before amending the contract.',
                            'Fees are published in the tariff schedule each January.')
        family = self.family('bank must obtain written consent before amending')
        rows = judge_families([family], corpus,
                              rank_scores=lambda fid, lang, q, allowed=None: [1.0, 0.0])
        self.assertEqual(len(rows), 3)                     # one row per language
        self.assertEqual(rows[0]['gold_chunks'], [0])
        self.assertEqual(rows[0]['first_gold_rank'], 1)
        self.assertTrue(rows[0]['best_is_gold'])
        # A rank-1 answer is not credited to the chunk that merely shares a word.
        rows = judge_families([family], corpus, rank_scores=lambda fid, lang, q, allowed=None: [0.0, 1.0])
        # The evidence is still reported at its true rank rather than hidden because a
        # better-scoring chunk was wrong: rank 2, outside a top-1 answer window.
        self.assertEqual(rows[0]['first_gold_rank'], 2)
        self.assertEqual(rows[0]['best_is_gold'], False)

    def test_a_boundary_split_span_is_reported_as_partial_not_as_a_retrieval_miss(self):
        from hard_harness.retrieval_judge import judge_families
        corpus = self.corpus('alpha beta gamma delta', 'unrelated epsilon zeta eta theta')
        family = self.family('alpha beta gamma delta epsilon')     # 4 of 5 tokens in chunk 0
        rows = judge_families([family], corpus, rank_scores=lambda fid, lang, q, allowed=None: [1.0, 0.0])
        self.assertEqual(rows[0]['gold_chunks'], [])
        self.assertEqual(rows[0]['partial_chunks'], [0])
        self.assertEqual(rows[0]['first_partial_rank'], 1)
        summary = __import__('hard_harness.retrieval_judge', fromlist=['summarise']).summarise(rows, top_k=1)
        self.assertEqual(summary['overall']['recall@1'], 0.0)          # strict containment: not met
        self.assertEqual(summary['overall']['partial_only_rate'], 1.0)  # boundary problem
        self.assertEqual(summary['overall']['evidence_available_rate'], 1.0)
        self.assertEqual(summary['overall']['no_evidence_rate'], 0.0)

    def test_misses_stay_in_the_average(self):
        from hard_harness.retrieval_judge import judge_families, summarise
        corpus = self.corpus('the answer lives here in full words', 'first distractor text here',
                            'second distractor text here', 'third distractor text here')
        family = self.family('the answer lives here in full words')
        rows = judge_families([family], corpus, rank_scores=lambda fid, lang, q, allowed=None: [0.0, 3.0, 2.0, 1.0])
        self.assertEqual(rows[0]['first_gold_rank'], 4)
        summary = summarise(rows, top_k=3)
        self.assertEqual(summary['overall']['recall@3'], 0.0)
        self.assertEqual(summary['overall']['mrr'], 0.25)               # 1/4, not dropped
        self.assertEqual(summary['overall']['ndcg@3'], 0.0)            # a dropped miss would inflate this
        self.assertEqual(summary['overall']['median_rank'], 4)

    def test_the_semantic_only_slice_is_the_one_a_lexical_ranker_cannot_win(self):
        from hard_harness.retrieval_judge import judge_families, summarise
        corpus = self.corpus('البنك ملزم بالحصول على موافقة كتابية قبل التعديل', 'نص آخر لا علاقة له')
        family = self.family('البنك ملزم بالحصول على موافقة كتابية قبل التعديل',
                             languages={'ar': {'question': 'هل يلزم الحصول على موافقة كتابية قبل التعديل؟'},
                                        'fr': {'question': 'Quelle est la procédure requise ?'},
                                        'en': {'question': 'Which procedure is required?'}})
        rows = judge_families([family], corpus, rank_scores=lambda fid, lang, q, allowed=None: [1.0, 0.0])
        by_language = {row['language']: row for row in rows}
        self.assertFalse(by_language['ar']['semantic_only'])       # shares wording with the evidence
        self.assertTrue(by_language['en']['semantic_only'])        # no shared content word at all
        summary = summarise(rows, top_k=1)
        self.assertEqual(summary['overall']['semantic_only_queries'], 2)
        self.assertEqual(summary['by_language']['en']['semantic_only_queries'], 1)

    def test_absent_evidence_separates_and_the_threshold_is_bounded_on_the_answerable_side(self):
        from hard_harness.retrieval_judge import auc, threshold_at_fpr
        self.assertEqual(auc([5, 6, 7, 8], [1, 2, 3, 4]), 1.0)
        self.assertEqual(auc([1, 2, 3, 4], [5, 6, 7, 8]), 0.0)
        self.assertIsNone(auc([1], []))
        picked = threshold_at_fpr([10, 9, 8, 7, 6, 5, 4, 3, 2, 1], [0.5] * 10, fpr=0.0)
        self.assertEqual(picked['answerable_rejected'], 0.0)   # a hard bound is honoured
        self.assertEqual(picked['unanswerable_caught'], 1.0)    # every absent query falls below it
        self.assertLessEqual(picked['score'], 1)
        # A loose budget lets absent evidence leak in rather than rejecting answerable work.
        loose = threshold_at_fpr([10, 9, 8, 7, 6, 5, 4, 3, 2, 1], [8, 7, 6, 5, 4, 3, 2, 1, 0.5, 0.4], fpr=0.4)
        self.assertLessEqual(loose['answerable_rejected'], 0.4)

    def test_an_unreachable_embedding_arm_is_named_instead_of_replaced(self):
        from unittest.mock import patch
        import hard_harness.retrieval_judge as judge
        families = [self.family('alpha beta gamma delta epsilon zeta')]
        corpus = self.corpus('alpha beta gamma delta epsilon zeta', 'other text entirely here now')
        with tempfile.TemporaryDirectory() as temp:
            with patch('embedder.build_embedder', side_effect=RuntimeError('NVIDIA_API_KEY is not configured')):
                with self.assertRaisesRegex(ValueError, 'No arm could be built'):
                    judge.evaluate(arms=('vector',), out=Path(temp) / 'only', families=families, corpus=corpus)
                manifest = judge.evaluate(arms=('lexical', 'vector'), out=Path(temp) / 'both',
                                          families=families, corpus=corpus)
            self.assertEqual(manifest['arms'], ['lexical'])
            self.assertEqual(manifest['arm_status']['vector']['status'], 'unavailable')
            self.assertIn('NVIDIA_API_KEY', manifest['arm_status']['vector']['error'])
            self.assertTrue((Path(temp) / 'both' / 'REPORT.md').exists())
            self.assertTrue((Path(temp) / 'both' / 'lexical_rows.jsonl').exists())

    def test_the_vector_arm_is_measured_through_the_same_labels(self):
        """The embedding arm must reuse the deterministic labels, not invent its own."""
        from types import SimpleNamespace
        from unittest.mock import patch
        import hard_harness.retrieval_judge as judge
        corpus = self.corpus('alpha beta gamma delta epsilon zeta', 'unrelated text with other words here')
        # The question shares the evidence word 'alpha', which is what the fake embedder
        # keys on; a question phrased differently would legitimately rank lower.
        family = self.family('alpha beta gamma delta epsilon zeta', question='alpha question?')

        class FakeEmbedder:
            api_calls = 2

            def embed_texts(self, texts, *, input_type=''):
                # One direction for the evidence, the orthogonal direction for everything else.
                return [[1.0, 0.0] if 'alpha' in text else [0.0, 1.0] for text in texts]

        with tempfile.TemporaryDirectory() as temp:
            with patch('embedder.build_embedder', return_value=FakeEmbedder()):
                manifest = judge.evaluate(arms=('vector',), out=Path(temp), families=[family], corpus=corpus,
                                           cfg=SimpleNamespace(NVIDIA_EMBEDDING_MODEL='fake-embed'))
        self.assertEqual(manifest['arms'], ['vector'])
        self.assertEqual(manifest['arm_status']['vector']['status'], 'ok')
        self.assertEqual(manifest['arm_status']['vector']['model'], 'fake-embed')
        report = manifest['report']['vector']
        self.assertEqual(report['arm']['dimensions'], 2)
        self.assertEqual(report['overall']['recall@1'], 1.0)
        self.assertEqual(report['overall']['top1_is_gold'], 1.0)
        self.assertEqual(report['overall']['queries'], 3)
        # An absent-evidence query set cannot be built when the corpus has one document,
        # so the abstention figures stay empty instead of being reported as 0.0.
        self.assertIsNone(report['abstention']['auc'])

    def test_a_broken_arm_is_named_and_the_other_arm_still_reports(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        import hard_harness.retrieval_judge as judge
        corpus = self.corpus('alpha beta gamma delta epsilon zeta', 'unrelated text with other words here')
        family = self.family('alpha beta gamma delta epsilon zeta', question='alpha question?')

        class BrokenEmbedder:
            """Corpus vectors are 2-d, query vectors are 3-d: the arm fails while measuring."""
            def embed_texts(self, texts, *, input_type=''):
                if input_type == 'search_query':
                    return [[1.0, 0.0, 0.0] for _ in texts]
                return [[1.0, 0.0] for _ in texts]

        with tempfile.TemporaryDirectory() as temp:
            with patch('embedder.build_embedder', return_value=BrokenEmbedder()):
                manifest = judge.evaluate(arms=('lexical', 'vector'), out=Path(temp), families=[family],
                                          corpus=corpus, cfg=SimpleNamespace(NVIDIA_EMBEDDING_MODEL='broken'))
        self.assertEqual(manifest['arms'], ['lexical'])
        self.assertEqual(manifest['arm_status']['vector']['status'], 'failed')
        self.assertIn('dimension', manifest['arm_status']['vector']['error'])
        self.assertEqual(manifest['report']['lexical']['overall']['queries'], 3)

    def test_a_label_that_is_absent_from_the_corpus_is_not_charged_to_retrieval(self):
        """Two extraction paths disagreeing is a labelling defect, not a recall failure."""
        from types import SimpleNamespace
        import hard_harness.retrieval_judge as judge
        corpus = self.corpus('alpha beta gamma delta epsilon zeta', 'unrelated text with other words here')
        present = self.family('alpha beta gamma delta epsilon zeta', question='alpha question?')
        present['id'] = 'hh0001'
        absent = self.family('kappa lambda mu nu xi omicron pi', question='alpha question?')   # not in the corpus
        absent['id'] = 'hh0002'
        with tempfile.TemporaryDirectory() as temp:
            manifest = judge.evaluate(arms=('lexical',), out=Path(temp), families=[present, absent],
                                      corpus=corpus, cfg=SimpleNamespace(NVIDIA_EMBEDDING_MODEL='none'))
            report = (Path(temp) / 'REPORT.md').read_text()
        integrity = manifest['label_integrity']
        self.assertEqual((integrity['labels_present_in_corpus'], integrity['labels_absent']), (1, 1))
        self.assertEqual(integrity['present_rate'], 0.5)
        overall = manifest['report']['lexical']['overall']
        self.assertEqual(overall['queries'], 6)
        # Overall half the queries "fail"; measured only over labels that exist in this
        # corpus, retrieval got every single one into the top-1. That gap is the defect.
        self.assertEqual(overall['answer_ready_rate'], 0.5)
        self.assertEqual(overall['answer_ready_rate_label_findable'], 1.0)
        self.assertIn('unanswerable from this index whatever', report)

    def test_reference_answers_are_never_read_into_a_retrieval_metric(self):
        from hard_harness.retrieval_judge import Corpus, judge_families
        family = self.family('bank must obtain written consent',
                             languages={'en': {'question': 'What must the bank obtain?',
                                               'reference_answer': 'GOLD-ONLY-TOKEN-918',
                                               'required_facts': ['GOLD-ONLY-TOKEN-918']},
                                        'ar': {'question': 'ماذا يجب؟', 'reference_answer': 'GOLD-ONLY-TOKEN-918'},
                                        'fr': {'question': 'Que faut-il ?', 'reference_answer': 'GOLD-ONLY-TOKEN-918'}})
        corpus = Corpus([self.Chunk('The bank must obtain written consent before amending the contract.')])
        rows = judge_families([family], corpus, rank_scores=lambda fid, lang, q, allowed=None: [1.0])
        self.assertNotIn('GOLD-ONLY-TOKEN-918', json.dumps(rows, ensure_ascii=False))
        self.assertTrue(all('reference_answer' not in row and 'required_facts' not in row for row in rows))


class DatasetIntegrity(unittest.TestCase):
    def test_focused_primary_repair_keeps_stable_evidence_aliases(self):
        from hard_harness.authoring import author_messages
        units={uid:{'id':uid,'document':'test.docx','page':None,'quality':'test','text':'Original source evidence statement with enough detail.'}
               for uid in ('a','b','c')}
        spec={'id':'hh0001','category':'supported','source_unit_ids':['a','b','c'],'primary_unit_id':'b'}
        messages=author_messages([spec],units,'hh0001: reference must address its assigned primary source unit')
        payload=json.loads(messages[1]['content'])
        self.assertEqual([s['id'] for s in payload['sources']],['U2'])
        self.assertEqual(payload['assignments'][0]['primary_unit_id'],'U2')
        self.assertEqual(payload['assignments'][0]['source_unit_ids'],['U2'])

    def test_author_spans_are_exact_original_slices(self):
        from hard_harness.authoring import evidence_spans
        text = ('نص مصدر يوضح الشروط والاستثناءات والأحكام. ' * 25) + ' نهاية'
        spans = evidence_spans(text, max_chars=160)
        self.assertGreater(len(spans), 1)
        for span in spans:
            self.assertEqual(span['text'], text[span['start']:span['end']])
        self.assertEqual(len({s['id'] for s in spans}), len(spans))

    def test_author_resolves_evidence_ids_without_copying_model_quotes(self):
        from hard_harness.authoring import validate_family, evidence_spans
        text = 'لا يجوز تغيير شروط العقد دون موافقة الهيئة المختصة. وهذا نص مصدر ثابت.'
        units = {'u':{'text':text}}
        spec = {'id':'hh0001','category':'supported','source_unit_ids':['u'],'primary_unit_id':'u','expected_behavior':'answer'}
        family = {'id':'hh0001','fact_summary':'approval needed','rationale':'source rule',
                  'evidence':[{'unit_id':'u','span_ids':['E1']}],
                  'languages':{lang:{'question':question,'reference_answer':'approval required',
                                    'required_facts':['approval required'],'forbidden_claims':[]}
                    for lang,question in [('ar','هل يجوز تغيير شروط العقد؟'),('fr','Peut-on modifier le contrat ?'),('en','Can the contract be modified?')]}}
        result = validate_family(family,spec,units)
        self.assertEqual(result['evidence'][0]['quote'],evidence_spans(text)[0]['text'])
        self.assertEqual(result['evidence'][0]['quote_resolved_by'],'source_span_id')
        family['evidence'][0]['span_ids']=['invented']
        with self.assertRaisesRegex(ValueError,'span IDs'):
            validate_family(family,spec,units)

    def family(self, number, category='supported'):
        languages = {lang: {'question': f'{lang} distinct question {number}?',
                            'reference_answer': f'expected fact {number}',
                            'required_facts': ['required meaning'], 'forbidden_claims': []}
                     for lang in ('ar','fr','en')}
        return {'id':f'hh{number:04d}','category':category,'expected_behavior':'answer' if category=='supported' else 'abstain',
                'fact_summary':f'fact {number}','rationale':'test-only fixture, not a real benchmark case',
                'languages':languages,'evidence':[{'unit_id':'unit','quote':'original source evidence text'}] if category=='supported' else []}

    def test_reference_recovery_exports_later_cached_ids_before_retrying_a_gap(self):
        from unittest.mock import patch
        import hard_harness.authoring as authoring
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); out=root/'out'; work=root/'work'; (out/'sources').mkdir(parents=True)
            plan=root/'plan.json'; write_json(plan,{'author_shards':1,'llm':{'model':ANSWER_MODEL}})
            source_hash='verified-sources'; write_json(out/'sources/manifest.json',{'gold_unit_manifest':source_hash})
            unit={'text':'هذا نص مصدر ثابت يبين اشتراط الموافقة على تعديل العقد.'}
            specs=[{'id':f'hh{i:04d}','category':'supported','expected_behavior':'answer',
                    'source_unit_ids':['u'],'primary_unit_id':'u'} for i in range(1,4)]
            for spec in specs[1:]:
                family={'id':spec['id'],'category':'supported','fact_summary':'approval','rationale':'source',
                    'authoring_version':authoring.AUTHOR_VERSION,
                    'evidence':[{'unit_id':'u','quote':unit['text']}],
                    'languages':{l:{'question':q,'reference_answer':'approval required','required_facts':['approval']}
                                 for l,q in [('ar','هل يشترط الحصول على الموافقة؟'),('fr','Une approbation est-elle nécessaire ?'),('en','Is approval required?')]}}
                key=fingerprint({'version':authoring.AUTHOR_VERSION,'spec':spec,'source':source_hash})
                write_json(work/'draft_families'/f'{key}.json',{'family':family,'audit':{'id':spec['id'],'approved':True,'issues':[]}})
            with patch.object(authoring,'OUTPUT',out),patch.object(authoring,'WORK',work),patch.object(authoring,'PLAN_PATH',plan), \
                 patch.object(authoring,'make_specs',return_value=(specs,{'u':unit})), \
                 patch.object(authoring,'CheckpointClient',side_effect=AssertionError('no provider on recovery')):
                report=authoring.author_shard(0,recover_only=True)
            self.assertEqual(report['families'],2)
            self.assertEqual(report['new_model_calls'],0)
            self.assertEqual([r['id'] for r in read_jsonl(out/'author_00/families.jsonl')],['hh0002','hh0003'])

    def test_a_committed_snapshot_recovers_families_without_artifact_access(self):
        """Actions caches are evicted and artifacts expire; the repo copy must not."""
        from unittest.mock import patch
        import hard_harness.authoring as authoring
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); out=root/'out'; work=root/'work'; snapshot=root/'accepted'
            (out/'sources').mkdir(parents=True); snapshot.mkdir()
            plan=root/'plan.json'; write_json(plan,{'author_shards':1,'llm':{'model':ANSWER_MODEL}})
            source_hash='verified-sources'; write_json(out/'sources/manifest.json',{'gold_unit_manifest':source_hash})
            unit={'text':'هذا نص مصدر ثابت يبين اشتراط الموافقة على تعديل العقد.'}
            specs=[{'id':f'hh{i:04d}','category':'supported','expected_behavior':'answer',
                    'source_unit_ids':['u'],'primary_unit_id':'u'} for i in range(1,4)]
            rows=[]
            for spec in specs[:2]:
                family={'id':spec['id'],'category':'supported','fact_summary':'approval','rationale':'source',
                        'authoring_version':authoring.AUTHOR_VERSION,
                        'evidence':[{'unit_id':'u','quote':unit['text']}],
                        'languages':{l:{'question':q,'reference_answer':'approval required','required_facts':['approval']}
                                      for l,q in [('ar','هل يشترط الحصول على الموافقة؟'),('fr','Une approbation est-elle nécessaire ?'),('en','Is approval required?')]}}
                rows.append({'family':family,'audit':{'id':spec['id'],'approved':True,'issues':[]}})
            write_jsonl(snapshot/'author_00.jsonl',rows)
            # A rejected row in the snapshot must never be trusted back in.
            bad=json.loads(json.dumps(rows[0])); bad['family']['id']='hh0003'
            bad['audit']={'id':'hh0003','approved':False,'issues':['evidence missing']}
            write_jsonl(snapshot/'author_01.jsonl',[bad])
            with patch.object(authoring,'OUTPUT',out),patch.object(authoring,'WORK',work), \
                 patch.object(authoring,'PLAN_PATH',plan), \
                 patch.object(authoring,'ACCEPTED_SNAPSHOT',snapshot), \
                 patch.object(authoring,'make_specs',return_value=(specs,{'u':unit})), \
                 patch.object(authoring,'CheckpointClient',side_effect=AssertionError('recovery must not call a model')):
                report=authoring.author_shard(0,recover_only=True)
            self.assertEqual(report['families'],2)
            self.assertEqual([r['id'] for r in read_jsonl(out/'author_00/families.jsonl')],['hh0001','hh0002'])
            # The recovered family is re-materialised into the local cache, so a
            # later run does not need the snapshot at all.
            self.assertEqual(len(list((work/'draft_families').glob('*.json'))),2)

    def test_drafting_continues_when_the_audit_provider_is_spent(self):
        """The scarce audit provider must never block the fast drafting provider."""
        from unittest.mock import patch
        import hard_harness.authoring as authoring
        unit_text='هذا نص مصدر ثابت يبين اشتراط الموافقة على تعديل العقد.'
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); out=root/'out'; work=root/'work'; (out/'sources').mkdir(parents=True)
            plan=root/'plan.json'
            write_json(plan,{'author_shards':1,'llm':{'model':'gemini-audit','provider':'google'},
                             'author_llm':{'model':'qwen-author','provider':'xkiro'},
                             'authoring':{'audit_mode':'drafts_only'}})
            source_hash='verified-sources'; write_json(out/'sources/manifest.json',{'gold_unit_manifest':source_hash})
            specs=[{'id':f'hh{i:04d}','category':'supported','expected_behavior':'answer',
                    'source_unit_ids':['u'],'primary_unit_id':'u'} for i in range(1,4)]
            units={'u':{'id':'u','text':unit_text,'document':'t.docx','page':None,'quality':'test'}}

            spec_id=[specs[0]['id']]
            with patch.object(authoring,'OUTPUT',out),patch.object(authoring,'WORK',work), \
                 patch.object(authoring,'PLAN_PATH',plan), \
                 patch.object(authoring,'make_specs',return_value=(specs,units)), \
                 patch.object(authoring,'CheckpointClient',side_effect=lambda role,**kw: (
                     self._draft_client(role,spec_id,unit_text,paused_audit=True))):
                first=authoring.author_shard(0)
            self.assertEqual(first['audit_mode'],'drafts_only')
            self.assertEqual(first['families'],0)                     # nothing accepted without an audit
            self.assertEqual(first['drafted_pending_audit'],3)
            self.assertEqual(first['unresolved_count'],0)   # drafted, not failed
            self.assertEqual(first['audit_stop_reason'],'provider_pause')
            self.assertEqual(len(list((work/'draft_pending').glob('*.json'))),3)
            self.assertEqual(len(list((work/'draft_families').glob('*.json'))),0)
            # The queue is also published with the shard checkpoint, because the work
            # cache is best-effort storage and eviction must not cost re-drafting.
            published = read_jsonl(out/'author_00/pending_drafts.jsonl')
            self.assertEqual(sorted(row['spec_id'] for row in published),
                             ['hh0001','hh0002','hh0003'])
            for folder in (work/'draft_pending', work/'draft_families'):
                for stale in folder.glob('*.json'):
                    stale.unlink()          # simulate an evicted Actions cache
            # A later run reuses the drafts (no second authoring call) and promotes
            # every family the audit still allows.
            calls={'author':0}
            def audit_ok(role,**kw):
                return self._draft_client(role,spec_id,unit_text,paused_audit=False,calls=calls)
            with patch.object(authoring,'OUTPUT',out),patch.object(authoring,'WORK',work), \
                 patch.object(authoring,'PLAN_PATH',plan), \
                 patch.object(authoring,'make_specs',return_value=(specs,units)), \
                 patch.object(authoring,'CheckpointClient',side_effect=audit_ok):
                second=authoring.author_shard(0)
            self.assertEqual(second['status'],'drafts_complete')
            self.assertEqual(second['audits_promoted'],3)
            self.assertEqual([r['id'] for r in read_jsonl(out/'author_00/families.jsonl')],
                             ['hh0001','hh0002','hh0003'])
            self.assertEqual(list((work/'draft_pending').glob('*.json')),[])
            self.assertEqual(calls['author'],0)   # pending drafts are never re-drafted

    def _draft_client(self, role, spec_id, unit_text, *, paused_audit, calls=None):
        class Client:
            def __init__(self):
                self.pause=HarnessPause('spent',429) if (paused_audit and role=='reference_audit') else None
                self.model='qwen-author' if role=='question_author' else 'gemini-audit'
            def object(self, messages, *, max_tokens=0):
                if role=='reference_audit':
                    if paused_audit:
                        raise self.pause
                    payload=json.loads(messages[1]['content'])
                    ids=[f['id'] for f in payload['families']]
                    return ({'reviews':[{'id':i,'approved':True,'issues':[]} for i in ids]}, {})
                if calls is not None: calls['author']=calls.get('author',0)+1
                payload=json.loads(messages[1]['content'])
                assigned=payload['assignments'][0]['id']
                spec_id[0]=assigned
                return ({'families':[{'id':assigned,'fact_summary':'approval','rationale':'source','uncertain':False,
                                      'evidence':[{'unit_id':'U1','quote':unit_text}],
                                      'languages':{l:{'question':q,'reference_answer':'approval required',
                                                      'required_facts':['approval'],'forbidden_claims':[]}
                                                   for l,q in [('ar','هل يشترط الموافقة على تعديل العقد؟'),
                                                               ('fr','Une approbation est-elle requise pour modifier le contrat ?'),
                                                               ('en','Is approval required to modify the contract?')]}}]}, {})
            def check_pause(self):
                if self.pause is not None: raise self.pause
            def summary(self):
                return {'role': role, 'provider': 'xkiro' if role == 'question_author' else 'google',
                        'model': self.model, 'credential_alias': 'X', 'http_requests': 0,
                        'logical_request_budget': {'used': 0, 'limit': 400}, 'cache_hits': 0,
                        'rate_limit_waits': 0, 'paced_provider_retries': 0,
                        'consecutive_transport_failures': 0,
                        'pause': str(self.pause) if self.pause else None}
        return Client()

    def test_shard_deadline_stops_before_the_next_model_call(self):
        from unittest.mock import patch
        import hard_harness.authoring as authoring
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); out=root/'out'; work=root/'work'; (out/'sources').mkdir(parents=True)
            plan=root/'plan.json'; write_json(plan,{'author_shards':1,'shard_deadline_minutes':45,
                                                     'llm':{'model':ANSWER_MODEL}})
            source_hash='verified-sources'; write_json(out/'sources/manifest.json',{'gold_unit_manifest':source_hash})
            unit={'text':'هذا نص مصدر ثابت يبين اشتراط الموافقة على تعديل العقد.'}
            specs=[{'id':f'hh{i:04d}','category':'supported','expected_behavior':'answer',
                    'source_unit_ids':['u'],'primary_unit_id':'u'} for i in range(1,4)]
            for spec in specs[1:]:
                family={'id':spec['id'],'category':'supported','fact_summary':'approval','rationale':'source',
                    'authoring_version':authoring.AUTHOR_VERSION,
                    'evidence':[{'unit_id':'u','quote':unit['text']}],
                    'languages':{l:{'question':q,'reference_answer':'approval required','required_facts':['approval']}
                                 for l,q in [('ar','هل يشترط الحصول على الموافقة؟'),('fr','Une approbation est-elle nécessaire ?'),('en','Is approval required?')]}}
                key=fingerprint({'version':authoring.AUTHOR_VERSION,'spec':spec,'source':source_hash})
                write_json(work/'draft_families'/f'{key}.json',{'family':family,'audit':{'id':spec['id'],'approved':True,'issues':[]}})
            with patch.object(authoring,'OUTPUT',out),patch.object(authoring,'WORK',work),patch.object(authoring,'PLAN_PATH',plan), \
                 patch.object(authoring,'make_specs',return_value=(specs,{'u':unit})), \
                 patch.object(authoring,'deadline_reached',return_value=True), \
                 patch.object(authoring,'CheckpointClient',side_effect=AssertionError('no provider use at the deadline')):
                report=authoring.author_shard(0)
            # A timed-out shard is resumable, never a fake completion and never a
            # fleet-wide 'paused' signal that would gate the other shards.
            self.assertEqual(report['status'],'partial_deadline')
            self.assertEqual(report['stop_reason'],'shard_deadline')
            self.assertEqual(report['families'],2)
            self.assertNotIn('error',report)
            self.assertEqual([r['id'] for r in read_jsonl(out/'author_00/families.jsonl')],['hh0002','hh0003'])

    def test_duplicate_detection_preserves_first_family_and_flags_later_ids(self):
        from hard_harness.repair import duplicate_ids
        first=self.family(1); second=self.family(2); third=self.family(3)
        second['languages']['fr']['question']=first['languages']['fr']['question']
        third['languages']['en']['question']=first['languages']['en']['question']
        self.assertEqual(duplicate_ids([third,second,first]), {'hh0002':['hh0001'],'hh0003':['hh0001']})

    def test_frozen_reference_repair_is_forbidden(self):
        from unittest.mock import patch
        import hard_harness.repair as repair
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp)
            write_json(out/'dataset/manifest.json', {'status':'frozen'})
            with patch.object(repair,'OUTPUT',out), patch.object(repair,'CheckpointClient') as client:
                with self.assertRaisesRegex(ValueError,'Frozen'):
                    repair.repair_before_freeze([],{}, {},purpose='duplicate')
                client.assert_not_called()

    def test_three_thousand_public_questions_are_separate_from_keys(self):
        from hard_harness.dataset import make_adversarial, validate_and_split
        base = [self.family(i) for i in range(1,651)]
        base += [self.family(i,'out_of_scope') for i in range(651,851)]
        base += [self.family(i,'insufficient_information') for i in range(851,901)]
        units = {'unit':{'id':'unit','document':'test.docx','page':None,'text':'original source evidence text','quality':'test'}}
        plan = self.plan()
        families = base + make_adversarial(base, plan)
        public, private = validate_and_split(families, units, plan)
        for lang in ('ar','fr','en'):
            self.assertEqual(len(public[lang]),1000)
            self.assertEqual(len(private[lang]),1000)
            self.assertEqual({r['id'] for r in public[lang]},{r['id'] for r in private[lang]})
            self.assertTrue(all('reference_answer' not in row and 'required_facts' not in row and 'category' not in row for row in public[lang]))
        self.assertEqual(sum('context_injections' in q for q in public['en']),50)
        families[1]['languages']['en']['question'] = families[0]['languages']['en']['question']
        with self.assertRaisesRegex(ValueError,'duplicate'):
            validate_and_split(families,units,plan)

    def test_completed_prediction_shard_replays_without_any_provider(self):
        from unittest.mock import patch
        import hard_harness.predict as predict
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); out=root/'out'; work=root/'work'; (out/'retrieval').mkdir(parents=True)
            policy={'top_k':5,'context_tokens':3000,'max_tokens':4096,'prompt':'grounded-v1','translation':False}
            plan=root/'plan.json'; write_json(plan,{'answer_policy':policy})
            metadata={'status':'retrieval_complete','public_files':{'test':{'fingerprint':'fixed'}}}
            write_json(out/'retrieval/manifest.json',metadata)
            rows=[{'id':f'{i}.en','family_id':str(i),'language':'en','question':f'question {i}',
                   'hits':[],'retrieval_skipped':None} for i in range(100)]
            write_jsonl(out/'retrieval/00.jsonl',rows)
            prior=[{'id':r['id'],'terminal':True,'case_hash':predict.case_identity(r,policy),
                    'provider':'xkiro','model':'original-model','result':{'status':'refused'}} for r in rows]
            cache=work/'prediction_shards'/fingerprint(metadata['public_files'])[:16]/'00.jsonl'
            write_jsonl(cache,prior)
            with patch.object(predict,'OUTPUT',out),patch.object(predict,'WORK',work),patch.object(predict,'PLAN_PATH',plan), \
                 patch.object(predict,'CheckpointClient',side_effect=AssertionError('provider must not be built')):
                report=predict.predict_shard(0)
            self.assertEqual(report['status'],'predictions_complete')
            self.assertEqual(report['new_model_calls'],0)
            self.assertEqual(read_jsonl(out/'predictions_00/predictions.jsonl'),prior)

    def test_invalid_outputs_are_terminal_but_quota_is_not_an_answer(self):
        from hard_harness.predict import terminal_result
        self.assertTrue(terminal_result({'provider_ok':True,'validation_ok':False,'status':'refused'}))
        self.assertTrue(terminal_result({'provider_ok':False,'http_status':422}))
        self.assertFalse(terminal_result({'provider_ok':False,'http_status':429}))

    def test_scoring_separates_refusal_failures_and_provider_errors(self):
        from hard_harness.grading import deterministic_grade
        prediction={'id':'hh1.en','family_id':'hh1','language':'en','provider':'xkiro','model':'test',
                    'result':{'provider_ok':True,'validation_ok':True,'status':'refused','reason':'private_or_live_request','answer':'No access'}}
        reference={'category':'out_of_scope','fact_group':'g','expected_behavior':'abstain','forbidden_claims':[]}
        self.assertEqual(deterministic_grade(prediction,reference)['grade'],'correct_abstention')
        reference['expected_behavior']='answer'
        self.assertEqual(deterministic_grade(prediction,reference)['grade'],'over_refusal')
        prediction['result'].update(provider_ok=False,http_status=429,error='quota')
        self.assertIsNone(deterministic_grade(prediction,reference)['correct'])

    def test_semantic_judge_cannot_mark_unfaithful_wrong_language_as_correct(self):
        from hard_harness.grading import validate_judgments
        row={'id':'x','grade':'correct','language_ok':False,'grounded':True,'missing_facts':[],'unsupported_claims':[]}
        with self.assertRaisesRegex(ValueError,'Inconsistent'):
            validate_judgments({'judgments':[row]},['x'])
        row.update(language_ok=True,grade='reference_issue')
        self.assertEqual(validate_judgments({'judgments':[row]},['x'])[0]['grade'],'reference_issue')

    def test_a_scaled_version_freezes_from_accepted_shards_end_to_end(self):
        """The freeze path must work at 6 questions per language and at 1,000."""
        from unittest.mock import patch
        import hard_harness.dataset as dataset
        counts = {'supported': 4, 'out_of_scope': 1, 'adversarial': 1, 'insufficient_information': 0}
        per_language = sum(counts.values())
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); out = root / 'results'
            plan_path = root / 'plan.json'
            write_json(plan_path, {'version': 'test-scaled-v1', 'author_shards': 1,
                                   'counts_per_language': counts,
                                   'questions_per_language': per_language,
                                   'full_target_questions_per_language': 1000,
                                   'answer_shard_size': 100, 'answer_shards': 1,
                                   'random_seed': 7, 'source_runtime_manifest': 'runtime-hash',
                                   'llm': {'model': ANSWER_MODEL}})
            units = [{'id': 'unit', 'document': 'test.docx', 'page': None,
                      'text': 'original source evidence text', 'quality': 'test',
                      'eligible_for_reference': True}]
            (out / 'sources').mkdir(parents=True)
            write_json(out / 'sources/gold_units.json', units)
            source_hash = fingerprint(units)
            write_json(out / 'sources/manifest.json',
                       {'status': 'ready_for_reference_authoring', 'gold_unit_manifest': source_hash})
            base = [self.family(i) for i in range(1, 5)]
            # Deliberately high ID: a category the version does not request must be
            # left out of the freeze even though it was accepted and audited.
            base += [self.family(5, 'out_of_scope'), self.family(900, 'insufficient_information')]
            for family in base:
                family['author_provenance'] = {'served_model': ANSWER_MODEL}
                family['audit_provenance'] = {'served_model': 'gemini-3.5-flash'}
            (out / 'author_00').mkdir(parents=True)
            write_jsonl(out / 'author_00/families.jsonl', base)
            write_jsonl(out / 'author_00/reference_audit.jsonl',
                        [{'id': f['id'], 'approved': True, 'issues': []} for f in base])
            write_json(out / 'author_00/manifest.json',
                       {'source_manifest': source_hash, 'families': len(base)})

            class AbsenceAuditor:
                def __init__(self, role, *, call_limit):
                    self.role, self.pause = role, None
                def object(self, messages, *, max_tokens=0):
                    asked = json.loads(messages[1]['content'])['families']
                    return ({'reviews': [{'id': row['id'], 'decision': 'abstention_justified',
                                          'source_unit_ids': [], 'reason': 'fixture'} for row in asked]}, {})
                def check_pause(self):
                    pass

            def freeze():
                with patch.object(dataset, 'OUTPUT', out), patch.object(dataset, 'PLAN_PATH', plan_path), \
                     patch.object(dataset, 'ROOT', root), \
                     patch.object(dataset, 'CheckpointClient', AbsenceAuditor):
                    return dataset.compile_dataset()

            manifest = freeze()
            self.assertEqual(manifest['status'], 'frozen')
            self.assertTrue(manifest['scaled_version'])
            self.assertEqual((manifest['questions_per_language'], manifest['question_records']),
                             (per_language, 3 * per_language))
            self.assertEqual(manifest['answer_shards'], 1)
            self.assertEqual(manifest['base_families_accepted'], 6)
            # The zero-count category contributes nothing and nothing was padded: the
            # surplus insufficient family is simply not selected.
            self.assertEqual(manifest['base_families_selected'], 5)
            self.assertEqual(manifest['accepted_per_shard'], {0: 6})
            self.assertFalse(manifest['reference_model_independent_of_candidate'])
            self.assertEqual(manifest['provider_mix']['authoring'], {f'xkiro:{ANSWER_MODEL}': 5})
            self.assertEqual(manifest['provider_mix']['audit'], {'google:gemini-3.5-flash': 5})
            ids = set()
            for lang in ('ar', 'fr', 'en'):
                public = read_jsonl(out / 'dataset/public' / f'questions.{lang}.jsonl')
                private = read_jsonl(out / 'dataset/references' / f'answer_key.{lang}.jsonl')
                self.assertEqual((len(public), len(private)), (per_language, per_language))
                self.assertEqual({r['id'] for r in public}, {r['id'] for r in private})
                self.assertTrue(all('reference_answer' not in row and 'required_facts' not in row
                                    for row in public))
                ids |= {r['id'] for r in public}
            self.assertEqual(len(ids), 3 * per_language)
            self.assertNotIn('hh0900.en', {row['id'] for row in
                                          read_jsonl(out / 'dataset/public/questions.en.jsonl')})
            # The single adversarial record continues after the selected base families.
            self.assertIn('hh0006.en', {row['id'] for row in
                                       read_jsonl(out / 'dataset/public/questions.en.jsonl')})
            # A second call replays the frozen dataset without touching a provider.
            with patch.object(dataset, 'CheckpointClient',
                              side_effect=AssertionError('a frozen dataset must not re-audit')):
                with patch.object(dataset, 'OUTPUT', out), patch.object(dataset, 'PLAN_PATH', plan_path), \
                     patch.object(dataset, 'ROOT', root):
                    self.assertEqual(freeze()['dataset_id'], manifest['dataset_id'])
            # A version that cannot be filled is refused instead of padded.
            (out / 'dataset/manifest.json').unlink()
            rows = [row for row in read_jsonl(out / 'author_00/families.jsonl')
                    if row['id'] != 'hh0005']        # drop the only out-of-scope family
            write_jsonl(out / 'author_00/families.jsonl', rows)
            write_json(out / 'author_00/manifest.json',
                       {'source_manifest': source_hash, 'families': len(rows)})
            with patch.object(dataset, 'OUTPUT', out), patch.object(dataset, 'PLAN_PATH', plan_path), \
                 patch.object(dataset, 'ROOT', root), \
                 patch.object(dataset, 'CheckpointClient', AbsenceAuditor):
                with self.assertRaisesRegex(ValueError, 'still needed'):
                    dataset.compile_dataset()

    def test_prediction_public_reader_rejects_answer_key_fields(self):
        from hard_harness.predict import load_public_questions
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); manifest={'public_files':{},'questions_per_language':1000,'question_records':3000}
            for lang in ('ar','fr','en'):
                rows=[{'id':f'{i}.{lang}','family_id':str(i),'language':lang,'question':f'question {i}'} for i in range(1000)]
                if lang=='en': rows[0]['reference_answer']='leaked gold'
                name=f'questions.{lang}.jsonl';write_jsonl(root/name,rows)
                manifest['public_files'][name]={'fingerprint':fingerprint(rows),'count':1000}
            write_json(root/'manifest.json',manifest)
            with self.assertRaisesRegex(ValueError,'Non-public'):
                load_public_questions(root)

    def test_a_scaled_version_is_validated_against_its_own_counts(self):
        """Counts come from the plan, so a scaled pass cannot masquerade as the target."""
        from hard_harness.dataset import dataset_scale, select_accepted, validate_and_split
        units={'unit':{'id':'unit','document':'test.docx','page':None,'text':'original source evidence text','quality':'test'}}
        base=[self.family(i) for i in range(1,301)]
        base+=[self.family(i,'out_of_scope') for i in range(301,401)]
        base+=[self.family(i,'insufficient_information') for i in range(401,426)]
        plan=self.plan(scaled=True)
        scale=dataset_scale(plan)
        self.assertEqual((scale['per_language'],scale['answer_shards']),(475,15))
        self.assertEqual(scale['base_families'],425)
        chosen=select_accepted(base,scale)
        self.assertEqual(len(chosen),425)
        self.assertEqual({f['id'] for f in chosen},{f['id'] for f in base})
        # A version needs every category it claims; the lowest accepted IDs win.
        with self.assertRaisesRegex(ValueError,'still needed'):
            select_accepted(base[:-30],scale)
        families=chosen+[dict(f,category='adversarial') for f in []]
        from hard_harness.dataset import make_adversarial
        families=chosen+make_adversarial(chosen,plan)
        public,private=validate_and_split(families,units,plan)
        for lang in ('ar','fr','en'):
            self.assertEqual(len(public[lang]),475)
            self.assertEqual(len({r['id'] for r in public[lang]}),475)
        # adversarial IDs continue after the selected base, not after 900
        self.assertEqual({r['id'].rsplit('.',1)[0][:6] for r in private['en'] if r['category']=='adversarial'},
                         {f'hh{i:04d}' for i in range(426,476)})
        # A plan whose shard count disagrees with its own sizes is refused.
        with self.assertRaisesRegex(ValueError,'answer_shards must be'):
            dataset_scale(self.plan(scaled=True, answer_shards=30))

    def plan(self, *, scaled=False, **overrides):
        counts=({'supported':300,'out_of_scope':100,'adversarial':50,'insufficient_information':25} if scaled else
                {'supported':650,'out_of_scope':200,'adversarial':100,'insufficient_information':50})
        per_language=sum(counts.values())
        plan={'counts_per_language':counts,'questions_per_language':per_language,
              'answer_shard_size':100,'answer_shards':-(-3*per_language//100),'random_seed':20260905}
        plan.update(overrides)
        return plan


if __name__ == '__main__':
    unittest.main()
