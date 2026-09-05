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


if __name__ == '__main__':
    unittest.main()
