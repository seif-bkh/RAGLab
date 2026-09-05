"""Checkpointing and provenance for the large, explicitly authorized harness."""
import json
import os
import re
import threading
from datetime import datetime, timezone
from pathlib import Path

from artifacts import fingerprint, write_json
from free_gateway import FreeGatewayClient, load_pricing
from nvidia_api import NvidiaAPIError, safe_error
from pipeline_policy import ANSWER_MODEL

ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = ROOT / 'benchmarks/hard_harness_plan.json'
WORK = ROOT.parent / '.cache/hard_harness'
OUTPUT = ROOT / 'results/hard_harness'
LANGUAGES = ('ar', 'fr', 'en')


def now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def write_jsonl(path, rows):
    # Same atomic replacement semantics as other artifacts, without a huge array.
    import tempfile
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def parse_object(text):
    value = text.strip()
    if value.startswith('```'):
        value = re.sub(r'^```(?:json)?\s*', '', value)
        value = re.sub(r'\s*```$', '', value)
    data = json.loads(value)
    if not isinstance(data, dict):
        raise ValueError('Expected a JSON object')
    return data


class HarnessPause(RuntimeError):
    def __init__(self, message, status_code=0):
        super().__init__(message)
        self.status_code = status_code


class CheckpointClient:
    """Cache provider responses, including ones later rejected by a validator.

    No successful-output-only sampling: a resumed candidate trial sees the same
    completed response. Quota/transport failures remain attempts, not answers.
    Google is not selected automatically: a provider switch requires an explicit
    versioned plan and is recorded separately from Qwen-only measurements.
    """
    def __init__(self, role, *, call_limit, cache_root=WORK / 'requests', client=None):
        self.role = role
        self.plan = read_json(PLAN_PATH)
        profile = self.plan['llm']
        if profile['provider'] == 'xkiro':
            if profile['model'] != ANSWER_MODEL:
                raise HarnessPause('The xKiro harness profile must use the selected Qwen SKU')
        elif profile['provider'] == 'google':
            if not self.plan.get('google_fallback_authorized') or not self.plan.get('google_fallback_active'):
                raise HarnessPause('Google is not active; ask the user before changing the harness plan')
            if profile.get('free_tier_project_confirmed') is not True:
                raise HarnessPause('Confirm the Google key is for a free-tier project before inference')
        else:
            raise HarnessPause('Unknown harness provider; no silent fallback')
        self.model = profile['model']
        self.provider = profile['provider']
        self.credential_alias = profile['credential_secret']
        self.cache_root = Path(cache_root)
        self._lock = threading.RLock()
        self.budget = {'used': 0, 'limit': call_limit}
        if client is not None:
            self.client = client
        else:
            key = os.environ.get(self.credential_alias, '').strip()
            if os.environ.get('HARNESS_CREDENTIAL_ALIAS') == self.credential_alias:
                key = os.environ.get('HARNESS_API_KEY', '').strip()
            if not key:
                raise HarnessPause(f'{self.credential_alias} is not configured for this harness; no older credential is used', 401)
            if self.provider == 'xkiro':
                self.client = FreeGatewayClient(self.provider, self.model,
                    load_pricing(self.provider, api_key=key), budget=self.budget, api_key=key)
            else:
                from hard_harness.google_client import GoogleHarnessClient
                self.client = GoogleHarnessClient(self.model, key,
                    free_project_confirmed=profile.get('free_tier_project_confirmed'), budget=self.budget)
        self.base_url = self.client.base_url
        self.timeout = self.client.timeout
        self.attempts = self.client.attempts
        self.min_interval = self.client.min_interval
        self.max_retry_delay = self.client.max_retry_delay
        self.cached_calls = 0
        self.pause = None
        self.events = []
        self.last_provenance = None

    @property
    def calls(self):
        return self.client.calls

    def chat(self, model, messages, *, max_tokens=4096):
        if model != self.model:
            raise ValueError('Harness model substitution is forbidden')
        # Deliberately exclude credential values/aliases from request identity.
        key = fingerprint({'schema': 'hard-harness-request-v1', 'role': self.role,
                           'provider': self.provider, 'endpoint': self.base_url,
                           'model': model, 'messages': messages, 'max_tokens': max_tokens})
        path = self.cache_root / self.role / f'{key}.json'
        with self._lock:
            if path.exists():
                record = read_json(path)
                if record['status'] == 'response':
                    self.cached_calls += 1
                    self.last_provenance = {k:record.get(k) for k in ('request_hash','role','provider','model','credential_alias','timestamp')}
                    self.last_provenance['cache_replayed'] = True
                    return {**record['response'], '_harness_cached': True, '_harness_request_hash': key}
                if record.get('terminal_output_error'):
                    self.cached_calls += 1
                    self.last_provenance = {k:record.get(k) for k in ('request_hash','role','provider','model','credential_alias','timestamp')}
                    self.last_provenance['cache_replayed'] = True
                    raise NvidiaAPIError(record['error'], record.get('http_status', 422))
            if self.pause is not None:
                raise NvidiaAPIError(str(self.pause), self.pause.status_code)
            prior = self.client.calls
            try:
                response = self.client.chat(model, messages, max_tokens=max_tokens)
            except Exception as exc:
                status = getattr(exc, 'status_code', 0)
                record = {'request_hash': key, 'role': self.role, 'provider': self.provider,
                          'model': model, 'credential_alias': self.credential_alias,
                          'timestamp': now(), 'status': 'error', 'http_status': status,
                          'error': safe_error(exc), 'terminal_output_error': status == 422,
                          'http_attempts': self.client.calls - prior}
                # Preserve every attempt; don't overwrite a previous provider failure.
                attempt = path.with_name(key + '.' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f') + '.error.json')
                write_json(attempt, record)
                self.last_provenance = {k:record.get(k) for k in ('request_hash','role','provider','model','credential_alias','timestamp')}
                self.last_provenance['cache_replayed'] = False
                if status == 422:
                    write_json(path, record)
                else:
                    self.pause = HarnessPause(safe_error(exc), status)
                    write_json(OUTPUT / f'pause_{self.role}.json', {**record,
                               'action': 'Checkpoint saved. Ask the user before changing key/provider; resume does not discard completed records.'})
                raise
            record = {'request_hash': key, 'role': self.role, 'provider': self.provider,
                      'model': model, 'credential_alias': self.credential_alias,
                      'timestamp': now(), 'status': 'response', 'response': response,
                      'http_attempts': self.client.calls - prior}
            write_json(path, record)
            self.last_provenance = {k:record.get(k) for k in ('request_hash','role','provider','model','credential_alias','timestamp')}
            self.last_provenance['cache_replayed'] = False
            self.events.append({k: record[k] for k in ('request_hash', 'role', 'provider', 'model', 'timestamp', 'http_attempts')})
            return {**response, '_harness_cached': False, '_harness_request_hash': key}

    def object(self, messages, *, max_tokens=8192):
        # Reference-authoring syntax repair is explicit and cached under a NEW
        # request hash. Candidate answers use chat(), never this repair path.
        budget_retry = False
        try:
            response = self.chat(self.model, messages, max_tokens=max_tokens)
        except NvidiaAPIError as exc:
            if exc.status_code != 422 or 'incomplete' not in str(exc).lower() or max_tokens > 10000:
                raise
            # Authoring/reference only: a separately identified longer-output
            # request may repair truncation. Candidate chat() is never retried
            # merely to obtain a better scoring answer.
            max_tokens *= 2
            budget_retry = True
            response = self.chat(self.model, messages, max_tokens=max_tokens)
        original_hash = response.get('_harness_request_hash')
        try:
            value = parse_object(response['text'])
        except (ValueError, TypeError) as exc:
            repair = [*messages, {'role': 'assistant', 'content': response['text']},
                      {'role': 'user', 'content': 'Return exactly ONE valid JSON object for the same task, no commentary or second object. '
                                                'Repair JSON formatting without inventing new facts. Parser error: ' + str(exc)[:300]}]
            response = self.chat(self.model, repair, max_tokens=max_tokens)
            value = parse_object(response['text'])
        provenance = {k: response.get(k) for k in
                      ('_harness_request_hash', '_harness_cached', 'usage', 'seconds', 'served_model')}
        provenance['source_call'] = self.last_provenance
        provenance['max_tokens'] = max_tokens
        provenance['budget_retry'] = budget_retry
        provenance['original_request_hash'] = original_hash
        provenance['format_repaired'] = provenance['_harness_request_hash'] != original_hash
        return value, provenance

    def check_pause(self):
        if self.pause is not None:
            raise self.pause

    def summary(self):
        return {'role': self.role, 'provider': self.provider, 'model': self.model,
                'credential_alias': self.credential_alias, 'http_requests': self.calls,
                'logical_request_budget': dict(self.budget), 'cache_hits': self.cached_calls,
                'pause': str(self.pause) if self.pause else None}
