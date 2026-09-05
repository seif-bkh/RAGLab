"""Explicit, zero-priced gateway experiments, separate from NVIDIA model claims."""
import json
import os
import time
import urllib.request
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation

from nvidia_api import NvidiaClient, NvidiaAPIError, final_content
from provider_catalog import PROVIDERS, NoCredentialRedirects

PRICING_URLS = {'xkiro': 'https://api.xkiro.com/v1/models',
                'kiosapi': 'https://kiosapi.com/api/pricing'}


def is_zero(value):
    if value is None or isinstance(value, bool):
        return False
    try:
        amount = Decimal(str(value))
        return amount.is_finite() and amount == 0
    except (InvalidOperation, ValueError):
        return False


def free_eligibility(provider, model, catalog):
    """Fail closed on missing prices, paid siblings, and mixed paid/free groups."""
    if provider == 'xkiro':
        entries = [m for m in catalog.get('data', []) if m.get('id') == model]
        if len(entries) != 1:
            return False, 'missing_or_ambiguous_catalog_entry', {}
        row = entries[0]
        prices = row.get('pricing') or {}
        free = (row.get('access_tier') == 'free' and prices.get('currency') == 'USD'
                and prices.get('unit') == 'per_1m_tokens'
                and all(is_zero(prices.get(k)) for k in ('input', 'output'))
                and all(is_zero(prices[k]) for k in ('cache_read', 'cache_write') if k in prices))
        return bool(free), 'live_zero_token_prices' if free else 'not_verified_zero_price', row
    if provider == 'kiosapi':
        entries = [m for m in catalog.get('data', []) if m.get('model_name') == model]
        free = (catalog.get('success') is True and bool(entries)
                and is_zero((catalog.get('group_ratio') or {}).get('Free'))
                and all(set(m.get('enable_groups', [])) == {'Free'}
                        and m.get('quota_type') == 0 and not m.get('billing_expr')
                        and 'openai' in m.get('supported_endpoint_types', []) for m in entries))
        # model_price=0 alone is NOT free for token-priced New API models.
        row = {'entries': entries, 'free_group_ratio': (catalog.get('group_ratio') or {}).get('Free'),
               'pricing_version': catalog.get('pricing_version')}
        return bool(free), 'exclusive_zero_multiplier_free_group' if free else 'free_group_not_exclusive_or_unverified', row
    raise ValueError('Unknown gateway provider')


def load_pricing(provider, opener=None):
    client = opener or urllib.request.build_opener(NoCredentialRedirects())
    headers = {'Accept': 'application/json', 'User-Agent': 'RAGLab-readonly-catalog/1.0'}
    key = os.environ.get(PROVIDERS[provider]['key_env'], '').strip()
    if key:
        headers['Authorization'] = 'Bearer ' + key
    request = urllib.request.Request(PRICING_URLS[provider], headers=headers)
    with client.open(request, timeout=30) as response:
        raw = response.read(2_000_001)
    if len(raw) > 2_000_000:
        raise ValueError('Pricing response exceeds the bounded read size')
    data = json.loads(raw)
    if not isinstance(data, dict) or not isinstance(data.get('data'), list):
        raise ValueError('Malformed pricing catalog')
    return {'checked_at': datetime.now(timezone.utc).isoformat(), 'url': PRICING_URLS[provider], 'catalog': data}


class FreeGatewayClient(NvidiaClient):
    """Reuse tested HTTPS/SSE transport, never its NVIDIA credential default.

Gateway-reported labels are NOT independent upstream identity attestation.
Only the selected, live-price-approved request SKU may be called; no fallback.
"""
    def __init__(self, provider, model, pricing, *, budget, opener=None):
        eligible, reason, metadata = free_eligibility(provider, model, pricing['catalog'])
        if not eligible:
            raise ValueError(f'Free-only policy rejected {provider}/{model}: {reason}')
        self.provider, self.approved_model = provider, model
        self.pricing = pricing
        self._checked_at = time.monotonic()
        self.model_metadata = metadata
        self.budget = budget
        settings = PROVIDERS[provider]
        super().__init__(base_url=settings['base_url'], api_key=os.environ.get(settings['key_env'], ''),
                         timeout=120, attempts=2, min_interval=3, max_retry_delay=60, stream=True,
                         opener=opener or urllib.request.build_opener(NoCredentialRedirects()))

    def recheck(self):
        pricing = load_pricing(self.provider)
        eligible, _, metadata = free_eligibility(self.provider, self.approved_model, pricing['catalog'])
        if not eligible:
            raise ValueError('Model is no longer verified free; inference stopped')
        self.pricing = pricing
        self.model_metadata = metadata
        self._checked_at = time.monotonic()

    def request(self, path, payload=None):
        try:
            return super().request(path, payload)
        except NvidiaAPIError as exc:
            raise NvidiaAPIError(str(exc).replace('NVIDIA', self.provider.upper()),
                                 exc.status_code, exc.retry_after) from None

    def chat(self, model, messages, *, max_tokens=4096):
        if model != self.approved_model:
            raise ValueError('Gateway model substitution is forbidden')
        if time.monotonic() - self._checked_at > 300:
            self.recheck()
        if self.budget['used'] >= self.budget['limit']:
            raise NvidiaAPIError('Experiment logical-call budget exhausted', 429)
        self.budget['used'] += 1
        payload = {'model': model, 'messages': messages, 'temperature': 0,
                   'max_tokens': max_tokens, 'stream': True}
        if self.provider == 'xkiro':
            levels = (self.model_metadata.get('reasoning_efforts') or {}).get('levels', [])
            effort = next((x for x in ('none', 'off', 'disabled', 'minimal', 'low') if x in levels), None)
            if effort:
                payload['reasoning_effort'] = effort
        data = self.request('chat/completions', payload)
        reported = data.get('model')
        if reported and reported != model:
            raise NvidiaAPIError(f'Gateway reported {reported!r} for requested {model!r}; not attributed to requested model', 422)
        try:
            choice = data['choices'][0]
            if choice.get('finish_reason') in {'length', 'content_filter'}:
                raise NvidiaAPIError('Incomplete gateway final output', 422)
            text = final_content(choice['message'].get('content'))
        except (KeyError, IndexError, TypeError):
            raise NvidiaAPIError('Malformed gateway completion', 422) from None
        return {'text': text, 'requested_model': model, 'served_model': reported,
                'identity_assurance': 'gateway_reported_not_independently_verified',
                'usage': data.get('usage') or {}, 'seconds': data.get('_request_seconds')}
