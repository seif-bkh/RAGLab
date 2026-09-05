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

    def test_google_translates_only_explicit_messages_and_local_images(self):
        from hard_harness.google_client import google_payload
        messages=[{'role':'system','content':'policy'},image_message('read source',b'image')]
        data=google_payload(messages,2000)
        self.assertEqual(data['systemInstruction']['parts'][0]['text'],'policy')
        self.assertEqual(data['generationConfig']['responseMimeType'],'application/json')
        self.assertIn('inlineData',data['contents'][0]['parts'][1])
        with self.assertRaises(ValueError):
            google_payload([{'role':'user','content':[{'type':'image_url','image_url':{'url':'https://untrusted.example/image'}}]}],100)


class DatasetIntegrity(unittest.TestCase):
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

    def test_three_thousand_public_questions_are_separate_from_keys(self):
        from hard_harness.dataset import make_adversarial, validate_and_split
        base = [self.family(i) for i in range(1,651)]
        base += [self.family(i,'out_of_scope') for i in range(651,851)]
        base += [self.family(i,'insufficient_information') for i in range(851,901)]
        families = base + make_adversarial(base)
        units = {'unit':{'id':'unit','document':'test.docx','page':None,'text':'original source evidence text','quality':'test'}}
        plan = {'counts_per_language':{'supported':650,'out_of_scope':200,'insufficient_information':50,'adversarial':100}}
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

    def test_prediction_public_reader_rejects_answer_key_fields(self):
        from hard_harness.predict import load_public_questions
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); manifest={'public_files':{}}
            for lang in ('ar','fr','en'):
                rows=[{'id':f'{i}.{lang}','family_id':str(i),'language':lang,'question':f'question {i}'} for i in range(1000)]
                if lang=='en': rows[0]['reference_answer']='leaked gold'
                name=f'questions.{lang}.jsonl';write_jsonl(root/name,rows)
                manifest['public_files'][name]={'fingerprint':fingerprint(rows),'count':1000}
            write_json(root/'manifest.json',manifest)
            with self.assertRaisesRegex(ValueError,'Non-public'):
                load_public_questions(root)


if __name__ == '__main__':
    unittest.main()
