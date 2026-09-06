"""LLM-free retrieval judgement: does the index hand the evidence to the answerer?

This answers a narrower question than the graded answer harness, and it needs no
generative model at all. A question is *answer-ready* when one of the top-k chunks
contains the audited source span the question was built from. Containment of a span is
arithmetic, not taste: no judge can reward a fluent wrong answer or penalise a terse
correct one, and nothing here reads a reference answer or writes one.

What it cannot measure is printed in every report it produces: how an answer model phrased
a reply, whether that reply is faithful to the retrieved text, whether an abstention was
the right call for an answerable question, and whether an injection was resisted. Those
need a model or a human reader. Embeddings are used to *rank* and to separate absent
evidence; the referee is string containment, so the index is not graded by its own taste.
"""
from __future__ import annotations

import math
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from artifacts import fingerprint
from loader import normalize_arabic
from hard_harness.common import ROOT, OUTPUT, now, read_json, read_jsonl, write_json, write_jsonl
from hard_harness.reporting import clustered_interval

SNAPSHOT = ROOT / 'benchmarks' / 'hard_harness_accepted'
LANGUAGES = ('ar', 'fr', 'en')
# Function words only, so "no content overlap" really means the query shares nothing
# lexical with its evidence: that subset is where semantic retrieval has to earn its keep.
STOPWORDS = {
    'the', 'a', 'an', 'of', 'and', 'or', 'to', 'in', 'on', 'for', 'is', 'are', 'be', 'what', 'which',
    'who', 'how', 'many', 'does', 'do', 'did', 'it', 'its', 'this', 'that', 'with', 'by', 'at', 'as',
    'from', 'was', 'were', 'can', 'should', 'must', 'any', 'answer', 'question', 'mentioned',
    'le', 'la', 'les', 'des', 'du', 'de', 'et', 'ou', 'à', 'dans', 'pour', 'est', 'sont', 'quel',
    'quelle', 'quels', 'quelles', 'comment', 'combien', 'que', 'quoi', 'avec', 'par', 'sur', 'ce',
    'cette', 'ces', 'au', 'aux', 'un', 'une', 'quels', 'selon', 'figure', 'indiqué', 'mentionné',
    'هل', 'ما', 'من', 'في', 'على', 'إلى', 'عن', 'و', 'أو', 'أن', 'إن', 'هذا', 'هذه', 'ذلك', 'الذي',
    'التي', 'كم', 'كيف', 'مع', 'بين', 'عند', 'ب', 'ل', 'هل', 'بموجب', 'وفق', 'ما', 'هو', 'هي',
}
MIN_PARTIAL_COVERAGE = 0.8


def normalize(text):
    """The one comparison form: Arabic-normalised, case-folded, single-spaced."""
    return ' '.join(normalize_arabic(str(text or '')).split()).casefold()


def tokens(text):
    return [t for t in re.findall(r'[0-9a-z\u0600-\u06ff]+', normalize(text)) if len(t) > 1]


def content_tokens(text):
    return [t for t in tokens(text) if t not in STOPWORDS]


def gold_spans(family):
    """Exact source strings the question was built from — the whole label set."""
    return [normalize(event.get('quote')) for event in family.get('evidence', []) if event.get('quote')]


def accepted_families(snapshot=SNAPSHOT):
    """Audited families straight from the committed snapshot; no dataset freeze needed.

    Only `question` and `evidence` are ever read, so the reference answers stay in the
    file they live in and cannot leak into a retrieval metric.
    """
    families = []
    for path in sorted(Path(snapshot).glob('author_*.jsonl')):
        for record in read_jsonl(path):
            families.append(record['family'])
    return sorted(families, key=lambda family: family['id'])


@dataclass
class Corpus:
    """Chunk texts once, so a 469-family sweep is a scan rather than a re-tokenisation."""
    chunks: list
    texts: list = field(default_factory=list)
    token_sets: list = field(default_factory=list)
    content_sets: list = field(default_factory=list)
    by_document: dict = field(default_factory=dict)

    def __post_init__(self):
        self.chunks = list(self.chunks)
        self.texts = [normalize(chunk.text) for chunk in self.chunks]
        self.token_sets = [set(tokens(chunk.text)) for chunk in self.chunks]
        self.content_sets = [set(content_tokens(chunk.text)) for chunk in self.chunks]
        self.by_document = defaultdict(list)
        for position, chunk in enumerate(self.chunks):
            self.by_document[chunk.source].append(position)
        self.by_document = dict(self.by_document)

    def contains(self, spans):
        """Chunks holding a gold span in full, plus chunks holding almost all of it.

        A span split across a boundary is reported as partial, never as a plain miss:
        that is a chunking decision, and blaming retrieval for it would be wrong.
        """
        full, partial = set(), set()
        for span in spans:
            if not span:
                continue
            span_tokens = set(tokens(span))
            for position, text in enumerate(self.texts):
                if span in text:
                    full.add(position)
                elif len(span_tokens) >= 4 and len(span_tokens & self.token_sets[position]) / len(span_tokens) \
                        >= MIN_PARTIAL_COVERAGE:
                    partial.add(position)
        return full, partial - full


class LexicalIndex:
    """BM25 over the candidate corpus: a second, independent ranker, not a second opinion
    from the model under test."""

    name = 'lexical'

    def __init__(self, corpus: Corpus, *, k1=1.5, b=0.75):
        self.corpus = corpus
        self.postings = defaultdict(list)
        lengths = []
        for position, terms in enumerate(corpus.token_sets):
            counts = Counter(tokens(corpus.chunks[position].text))
            lengths.append(max(1, sum(counts.values())))
            for term, frequency in counts.items():
                self.postings[term].append((position, frequency))
        self.lengths = lengths or [1]
        self.average = sum(self.lengths) / len(self.lengths)
        self.total = len(corpus.chunks)
        self.k1, self.b = k1, b

    def scores(self, query, *, allowed=None):
        counts = Counter(tokens(query))
        scores = [0.0] * self.total
        for term, query_frequency in counts.items():
            postings = self.postings.get(term)
            if not postings:
                continue
            inverse = math.log(1 + (self.total - len(postings) + 0.5) / (len(postings) + 0.5))
            for position, frequency in postings:
                if allowed is not None and position not in allowed:
                    continue
                length = self.lengths[position]
                denominator = frequency + self.k1 * (1 - self.b + self.b * length / self.average)
                scores[position] += inverse * (frequency * (self.k1 + 1) / denominator) * math.log(1 + query_frequency)
        return scores


class VectorIndex:
    """Cosine over the corpus vectors produced by the embedding model under test."""

    name = 'vector'

    def __init__(self, corpus: Corpus, vectors, *, name=None):
        self.corpus = corpus
        self.vectors = [[float(value) for value in vector] for vector in vectors]
        if len(self.vectors) != len(corpus.chunks):
            raise ValueError('One vector per chunk is required')
        dimensions = {len(vector) for vector in self.vectors}
        if len(dimensions) > 1:
            raise ValueError('Chunk vectors must share a dimension')
        self.dimensions = dimensions.pop() if dimensions else 0
        self.norms = [math.sqrt(sum(value * value for value in vector)) or 1.0 for vector in self.vectors]
        self.name = name or self.name

    def scores(self, query, *, allowed=None):
        query = [float(value) for value in query]
        if len(query) != self.dimensions:
            raise ValueError(f'Query has {len(query)} dimensions, corpus has {self.dimensions}')
        norm = math.sqrt(sum(value * value for value in query)) or 1.0
        scores = []
        for position, vector in enumerate(self.vectors):
            if allowed is not None and position not in allowed:
                scores.append(-1.0)
                continue
            scores.append(sum(a * b for a, b in zip(query, vector)) / (norm * self.norms[position]))
        return scores


def order_by(scores, *, allowed=None):
    positions = range(len(scores)) if allowed is None else sorted(allowed)
    return sorted(positions, key=lambda position: (-scores[position], position))


def judge_families(families, corpus: Corpus, *, rank_scores, top_k=5, allowed=None):
    """One row per (family, language): rank order from `rank_scores`, labels from spans."""
    rows = []
    for family in families:
        spans = gold_spans(family)
        if not spans:
            continue                      # a negative carries no span; there is nothing to contain
        full, partial = corpus.contains(spans)
        gold_content = set()
        for position in full:
            gold_content |= corpus.content_sets[position]
        for language in LANGUAGES:
            version = (family.get('languages') or {}).get(language) or {}
            question = version.get('question')
            if not question:
                continue
            scores = list(rank_scores(family['id'], language, question))
            order = order_by(scores, allowed=allowed)
            top = order[:top_k]
            query_content = set(content_tokens(question))
            rows.append({
                'id': f"{family['id']}.{language}", 'family_id': family['id'], 'language': language,
                'category': family.get('category'), 'gold_chunks': sorted(full),
                'partial_chunks': sorted(partial),
                'gold_documents': sorted({corpus.chunks[position].source for position in full | partial}),
                'top': top, 'top_scores': [round(scores[position], 6) for position in top],
                'best_score': round(scores[order[0]], 6) if order else None,
                'best_is_gold': bool(order) and order[0] in full,
                'first_gold_rank': next((rank + 1 for rank, position in enumerate(order) if position in full), None),
                'first_partial_rank': next((rank + 1 for rank, position in enumerate(order) if position in partial),
                                           None),
                'available_rank': min([rank for rank in (next((r + 1 for r, i in enumerate(order) if i in full),
                                                              None),
                                                           next((r + 1 for r, i in enumerate(order) if i in partial),
                                                                 None)) if rank], default=None),
                'shared_content_tokens': len(query_content & gold_content) if gold_content else 0,
                'semantic_only': bool(gold_content) and not (query_content & gold_content),
            })
    return rows


def abstention_rows(families, corpus: Corpus, *, rank_scores_for, top_k=5):
    """The same questions with the supporting document removed from what is searchable.

    The unanswerable condition is manufactured by construction rather than written by a
    model: if the only document that supports the question cannot be retrieved, then no
    answer exists in the index, so a low top score is correct behaviour and a high one is
    a confident miss.
    """
    rows = []
    for family in families:
        spans = gold_spans(family)
        if not spans:
            continue
        full, partial = corpus.contains(spans)
        blocked = set()
        for position in full | partial:
            blocked |= set(corpus.by_document.get(corpus.chunks[position].source, []))
        allowed = set(range(len(corpus.chunks))) - blocked
        if not allowed:
            continue
        for language in LANGUAGES:
            version = (family.get('languages') or {}).get(language) or {}
            if not version.get('question'):
                continue
            scores = list(rank_scores_for(family['id'], language, version['question'], allowed))
            order = order_by(scores, allowed=allowed)
            rows.append({'id': f"{family['id']}.{language}", 'family_id': family['id'], 'language': language,
                         'searchable_chunks': len(allowed),
                         'absent_best_score': round(scores[order[0]], 6) if order else None,
                         'absent_best_chunk': order[0] if order else None,
                         'absent_still_contains_gold_span': bool(order) and order[0] in full})
    return rows


def auc(positive, negative):
    """Chance that an answerable query scores above an unanswerable one."""
    positive = [value for value in positive if value is not None]
    negative = [value for value in negative if value is not None]
    if not positive or not negative:
        return None
    wins = ties = 0
    for high in positive:
        for low in negative:
            wins += high > low
            ties += high == low
    return round((wins + 0.5 * ties) / (len(positive) * len(negative)), 4)


def threshold_at_fpr(positive, negative, *, fpr=0.05):
    """Abstain below this score, at a bounded rate of wrongly rejecting answerable questions.

    `answerable_rejected` is the cost a reader actually pays, so it is bounded and the
    catch rate on unanswerable questions is reported next to it, never blended into one
    friendly accuracy figure.
    """
    positive = [value for value in positive if value is not None]
    negative = [value for value in negative if value is not None]
    if not positive or not negative:
        return None
    for candidate in sorted({round(v, 6) for v in positive} | {round(v, 6) for v in negative}, reverse=True):
        rejected = sum(1 for value in positive if value < candidate) / len(positive)
        if rejected <= fpr:
            caught = sum(1 for value in negative if value < candidate) / len(negative)
            return {'score': candidate, 'answerable_rejected': round(rejected, 4),
                    'unanswerable_caught': round(caught, 4),
                    'note': f'highest threshold keeping at most {fpr:.0%} of answerable questions rejected'}
    return None


def _block(rows, *, top_k):
    if not rows:
        return {'queries': 0}
    ranks = [row['first_gold_rank'] for row in rows]
    found = [rank for rank in ranks if rank]

    def recall(limit):
        return round(sum(1 for rank in found if rank <= limit) / len(rows), 4)

    def ndcg(limit):
        # One relevant span set per query, so the ideal ranking puts it at rank 1 and the
        # ideal discount is 1.0; a miss contributes 0 rather than being skipped, because
        # dropping misses from the average would flatter exactly the hard cases.
        values = [(1 / math.log2(rank + 1)) if rank and rank <= limit else 0.0 for rank in ranks]
        return round(statistics.mean(values), 4)

    semantic = [row for row in rows if row['semantic_only']]
    return {
        'queries': len(rows), 'families': len({row['family_id'] for row in rows}),
        'recall@1': recall(1), 'recall@3': recall(3), f'recall@{top_k}': recall(top_k),
        'answer_ready_rate': recall(top_k),
        'top1_is_gold': round(sum(1 for row in rows if row['best_is_gold']) / len(rows), 4),
        'mrr': round(sum(1 / rank for rank in found) / len(rows), 4),
        f'ndcg@{top_k}': ndcg(top_k),
        'median_rank': statistics.median(found) if found else None,
        'partial_only_rate': round(sum(1 for row in rows if not row['gold_chunks'] and row['partial_chunks'])
                                   / len(rows), 4),
        'no_evidence_rate': round(sum(1 for row in rows if not row['gold_chunks'] and not row['partial_chunks'])
                                  / len(rows), 4),
        # An answer model that only sees top-k cannot cite what is not there: strict
        # containment is the ceiling it can meet, and containment-or-partial is the
        # generous reading. Report both instead of choosing the flattering one.
        'evidence_available_rate': round(sum(1 for row in rows if row['available_rank'] and
                                            row['available_rank'] <= top_k) / len(rows), 4),
        'semantic_only_queries': len(semantic),
        'semantic_only_recall': recall_of(semantic, top_k),
        'mean_top_score': round(statistics.mean(row['top_scores'][0] for row in rows if row['top_scores']), 4)
        if any(row['top_scores'] for row in rows) else None,
    }


def recall_of(rows, limit):
    if not rows:
        return None
    return round(sum(1 for row in rows if row['first_gold_rank'] and row['first_gold_rank'] <= limit) / len(rows), 4)


def summarise(rows, *, top_k=5, abstain=None):
    by_language, by_category = defaultdict(list), defaultdict(list)
    for row in rows:
        by_language[row['language']].append(row)
        by_category[row['category']].append(row)
    # Bootstrap over paired families: three languages of one family are not three
    # independent observations, so a per-question interval would overstate precision.
    groups = defaultdict(list)
    for row in rows:
        groups[row['family_id']].append(row)
    # Resample whole families: the three languages of one family travel together, so a
    # per-question interval would overstate how much evidence these numbers are.
    interval = clustered_interval([{'family_id': row['family_id'], 'correct': row['best_is_gold']} for row in rows]) \
        if len(groups) > 1 else None
    summary = {'overall': _block(rows, top_k=top_k),
               'by_language': {language: _block(group, top_k=top_k) for language, group in sorted(by_language.items())},
               'by_category': {category: _block(group, top_k=top_k) for category, group in sorted(by_category.items())},
               'top1_is_gold_ci95': interval, 'top_k': top_k}
    if abstain:
        summary['abstention'] = abstain
    return summary


def agreement(left, right):
    """Do two independent rankers find the same evidence? The anti-circularity check."""
    by_id = {row['id']: row for row in right}
    pairs = [(row, by_id[row['id']]) for row in left if row['id'] in by_id]
    if not pairs:
        return {'pairs': 0}
    def hits(rows):
        return sum(1 for row in rows for other in [row] if 1 in row['top'])
    return {
        'pairs': len(pairs),
        'same_top1_chunk': round(sum(1 for a, b in pairs if a['top'][:1] == b['top'][:1]) / len(pairs), 4),
        'both_find_evidence': round(sum(1 for a, b in pairs if a['best_is_gold'] and b['best_is_gold']) / len(pairs), 4),
        'only_left_finds': sum(1 for a, b in pairs if a['best_is_gold'] and not b['best_is_gold']),
        'only_right_finds': sum(1 for a, b in pairs if b['best_is_gold'] and not a['best_is_gold']),
        'neither_finds': sum(1 for a, b in pairs if not a['best_is_gold'] and not b['best_is_gold']),
    }


CAVEATS = [
    'This measures whether the evidence reaches the top of the index, not whether an answer model would use it '
    'correctly. Answer faithfulness, refusal appropriateness on answerable questions and injection resistance '
    'cannot be measured this way and are not claimed here.',
    'Questions come from model-authored families, so they are more templated than real user questions. The corpus '
    'text, the gold spans and the ranking maths are not model-judged.',
    'Labels are exact source-span containment. A chunk that answers without quoting the span verbatim is scored as '
    'a miss, which understates recall wherever paraphrase or table flattening changed the wording.',
    'The unanswerable condition is produced by deleting the supporting document from the searchable set. It tests '
    'abstain signal, not the model\'s willingness to refuse.',
    'Embeddings rank the corpus here; they do not judge correctness, so this is not a self-graded score. The '
    'lexical arm is reported alongside so a reader can see whether both rankers agree.',
    'The searchable corpus is only the four documents in docs/. That is a small index, so a high score is easy to '
    'reach and these rates describe this corpus, not a production retrieval service. Near-duplicate evidence across '
    'documents counts as a hit wherever it contains the span.',
]


def _tokenizer_identity():
    """Chunk boundaries decide what counts as a whole span, so say which tokenizer built them."""
    try:
        from chunker import tokenizer_identity
        return tokenizer_identity()
    except Exception as exc:      # noqa: BLE001 - identity is a label, not a gate
        return f'unavailable: {type(exc).__name__}'


def unit_coverage(corpus, unit_texts):
    """How much of a source unit fits inside a single chunk — measured with no queries at all.

    A low whole-unit rate means the *chunker* is splitting evidence that the references were
    built from, which no retrieval setting can fix. Reporting it separately keeps a boundary
    problem from being read as a retrieval failure.
    """
    counts = Counter()
    for text in unit_texts:
        span = normalize(text)
        if not span:
            continue
        span_tokens = set(tokens(span))
        if any(len(span) > 20 and span in text_norm for text_norm in corpus.texts):
            counts['whole_unit_in_one_chunk'] += 1
        elif span_tokens and any(len(span_tokens & chunk_tokens) / len(span_tokens) >= MIN_PARTIAL_COVERAGE
                                 for chunk_tokens in corpus.token_sets):
            counts['most_of_it'] += 1
        else:
            counts['split_or_absent'] += 1
    total = sum(counts.values())
    return {'units': total, **{key: counts.get(key, 0) for key in
                               ('whole_unit_in_one_chunk', 'most_of_it', 'split_or_absent')},
            'whole_unit_rate': round(counts.get('whole_unit_in_one_chunk', 0) / total, 4) if total else None,
            'note': 'A unit split across chunks is retrievable but never quotable in full from one chunk.'}


def gold_unit_texts():
    path = OUTPUT / 'sources' / 'gold_units.json'
    if not path.exists():
        return None
    return [unit.get('text', '') for unit in read_json(path) if unit.get('eligible_for_reference')]


def question_texts(families):
    """(family_id, language, question) for every query this corpus can produce."""
    keys = []
    for family in families:
        for language in LANGUAGES:
            version = (family.get('languages') or {}).get(language) or {}
            if version.get('question'):
                keys.append((family['id'], language, version['question']))
    return keys


def evaluate(*, top_k=5, arms=('lexical', 'vector'), out=None, families=None, corpus=None, cfg=None,
             fpr=0.05, chunk_tokens=None, chunk_overlap=None):
    """Run the requested arms over the accepted families and write a report.

    The embedding arm needs the embedding provider. If it cannot be reached the arm is
    recorded as unavailable with the reason, and the lexical arm is still reported — no
    other model is ever substituted, and no arm is silently dropped.
    """
    out = Path(out) if out else OUTPUT / 'retrieval_judge'
    out.mkdir(parents=True, exist_ok=True)
    if cfg is None:
        from hard_harness.predict import runtime_config
        cfg = runtime_config()
    if corpus is None:
        from chunker import chunk_all
        from loader import load_all
        # Chunking is a knob worth measuring against: whole-span evidence depends on it,
        # so a comparison run may override it while the manifest records what was used.
        if chunk_tokens:
            cfg.CHUNK_SIZE_TOKENS = int(chunk_tokens)
        if chunk_overlap is not None:
            cfg.CHUNK_OVERLAP_TOKENS = int(chunk_overlap)
        corpus = Corpus(chunk_all(load_all(ROOT.parent / 'docs'), cfg))
    families = accepted_families() if families is None else list(families)
    queries = question_texts(families)
    indices, arm_status = {}, {}
    if 'lexical' in arms:
        lexical = LexicalIndex(corpus)
        indices['lexical'] = lambda family_id, language, question, allowed=None: lexical.scores(
            question, allowed=allowed)
        arm_status['lexical'] = {'status': 'ok', 'ranker': 'BM25 over normalised tokens',
                                  'model_calls': 0}
    if 'vector' in arms:
        try:
            from embedder import build_embedder
            embedder = build_embedder(cfg)
            vectors = list(embedder.embed_texts([chunk.text for chunk in corpus.chunks],
                                                input_type='search_document'))
            query_vectors = list(embedder.embed_texts([question for _, _, question in queries],
                                                      input_type='search_query'))
            index = VectorIndex(corpus, vectors)
            lookup = {key: vector for key, vector in zip(queries, query_vectors)}
            indices['vector'] = lambda family_id, language, question, allowed=None: index.scores(
                lookup[(family_id, language)], allowed=allowed)
            arm_status['vector'] = {'status': 'ok', 'model': cfg.NVIDIA_EMBEDDING_MODEL,
                                    'dimensions': index.dimensions,
                                    'embedding_api_calls': getattr(embedder, 'api_calls', None),
                                    'answer_model_calls': 0}
        except Exception as exc:      # noqa: BLE001 - an unavailable arm must be named, not hidden
            from nvidia_api import safe_error
            arm_status['vector'] = {'status': 'unavailable', 'error': safe_error(exc)}
    if not indices:
        raise ValueError('No arm could be built; nothing was substituted')
    coverage = unit_coverage(corpus, gold_unit_texts()) if gold_unit_texts() else {
        'status': 'unavailable', 'note': 'results/hard_harness/sources/gold_units.json was not present, '
                                         'so the chunking diagnostic could not run; download the sources '
                                         'checkpoint to enable it'}
    report = {}
    for arm, score in indices.items():
        rows = judge_families(families, corpus, rank_scores=score, top_k=top_k)
        absent = abstention_rows(families, corpus, rank_scores_for=
                                 lambda family_id, language, question, allowed: score(
                                     family_id, language, question, allowed=allowed), top_k=top_k)
        present_scores = [row['top_scores'][0] for row in rows if row['top_scores']]
        absent_scores = [row['absent_best_score'] for row in absent]
        threshold = threshold_at_fpr(present_scores, absent_scores, fpr=fpr)
        abstain = {'auc': auc(present_scores, absent_scores), 'threshold': threshold,
                   'answerable_queries': len(present_scores), 'unanswerable_queries': len(absent_scores),
                   'note': 'positive = top score with the corpus intact; negative = top score with the supporting '
                           'document removed from the searchable set'}
        write_jsonl(out / f'{arm}_rows.jsonl', rows)
        write_jsonl(out / f'{arm}_absent.jsonl', absent)
        summary = summarise(rows, top_k=top_k, abstain=abstain)
        summary['arm'] = arm_status[arm]
        write_json(out / f'{arm}_summary.json', summary)
        report[arm] = summary
    if len(indices) > 1:
        pair = agreement(read_jsonl(out / 'lexical_rows.jsonl'), read_jsonl(out / 'vector_rows.jsonl'))
        write_json(out / 'agreement.json', pair)
        report['agreement'] = pair
    manifest = {'status': 'judged', 'created_at': now(), 'arms': sorted(indices), 'arm_status': arm_status,
                'top_k': top_k, 'fpr': fpr, 'questions': len(queries), 'families': len(families),
                'chunk_size_tokens': getattr(cfg, 'CHUNK_SIZE_TOKENS', None),
                'chunk_overlap_tokens': getattr(cfg, 'CHUNK_OVERLAP_TOKENS', None),
                'chunks': len(corpus.chunks), 'documents': sorted(corpus.by_document),
                'corpus_fingerprint': fingerprint([chunk.text for chunk in corpus.chunks]),
                'tokenizer': _tokenizer_identity(),
                'label_source': str(SNAPSHOT.relative_to(ROOT.parent)),
                'embedding_model': getattr(cfg, 'NVIDIA_EMBEDDING_MODEL', None),
                'judged_by': 'exact source-span containment; no model judged anything',
                'unit_coverage': coverage,
                'caveats': CAVEATS, 'report': report}
    write_json(out / 'manifest.json', manifest)
    (out / 'REPORT.md').write_text(markdown(manifest), encoding='utf-8')
    return manifest


def _interval(value):
    if not value:
        return 'n/a'
    return f"{value['low']:.3f}–{value['high']:.3f} over {value.get('groups')} paired families"


def _percent(value):
    return 'n/a' if value is None else f'{value:.1%}'


def _arm_lines(arm, summary):
    """One section per ranker. Numbers are read out of the summary in plain locals so the
    report never recomputes or rounds differently from the JSON it accompanies."""
    k = summary['top_k']
    overall = summary['overall']
    ready, first, mrr, ndcg = (overall.get(f'recall@{k}'), overall.get('recall@1'), overall.get('mrr'),
                              overall.get(f'ndcg@{k}'))
    abstain = summary.get('abstention') or {}
    threshold = abstain.get('threshold') or {}
    lines = ['', f'## {arm} ranker', '',
             f"- **answer-ready at top-{k}: {_percent(ready)}** — the retrieved chunk contains the source span the "
             f"question was built from. recall@1 {_percent(first)}, MRR {mrr}, nDCG@{k} {ndcg}",
             f"- top chunk is the evidence for {_percent(overall.get('top1_is_gold'))} of queries; median rank of "
             f"the evidence: {overall.get('median_rank')}",
             f"- ceiling for any answer model given top-{k}: {_percent(overall.get('answer_ready_rate'))} of queries "
             f"have the span in full inside a retrieved chunk, or {_percent(overall.get('evidence_available_rate'))} "
             f"counting a boundary-split span — an answer model cannot cite what was never handed to it",
             f"- 95% interval on the top-chunk-is-evidence rate, resampled by family: "
             f"{_interval(summary.get('top1_is_gold_ci95'))}",
             f"- queries sharing no content word with their evidence: {overall.get('semantic_only_queries')} "
             f"(answer-ready {_percent(overall.get('semantic_only_recall'))}) — the slice where an embedding has to "
             f"carry the retrieval on meaning alone",
             f"- chunking, not retrieval: spans split across a boundary with no whole-span chunk: "
             f"{_percent(overall.get('partial_only_rate'))}; no evidence at all in the top-{k}: "
             f"{_percent(overall.get('no_evidence_rate'))}"]
    if abstain:
        lines += [f"- abstain signal: AUC {abstain.get('auc')} separating an intact corpus from one where the "
                  f"supporting document is deleted; a threshold of {threshold.get('score', 'n/a')} would catch "
                  f"{_percent(threshold.get('unanswerable_caught'))} of those unanswerable queries while wrongly "
                  f"rejecting {_percent(threshold.get('answerable_rejected'))} of the answerable ones "
                  f"({abstain.get('spurious_top1_containing_gold_span', 0)} still returned a chunk containing the "
                  f"span, which is a duplicate-document artefact, not a win)"]
    block = summary.get('by_language') or {}
    if block:
        lines += ['', f"| language | answer-ready | recall@1 | MRR | semantic-only queries | semantic-only recall |",
                  '|---|---:|---:|---:|---:|---:|']
        for language, values in block.items():
            lines.append(f"| {language} | {_percent(values.get(f'recall@{k}'))} | "
                         f"{_percent(values.get('recall@1'))} | {values.get('mrr')} | "
                         f"{values.get('semantic_only_queries')} | {_percent(values.get('semantic_only_recall'))} |")
    return lines


def markdown(manifest):
    lines = ['# Retrieval quality measured without an answer model', '',
             f"{manifest['families']} audited families, {manifest['questions']} questions, {manifest['chunks']} "
             f"corpus chunks from {len(manifest['documents'])} documents, top_k={manifest['top_k']}. Generated "
             f"{manifest['created_at']}.", '',
             f"The label is exact source-span containment inside a retrieved chunk. No generative model wrote, "
             f"judged or repaired anything in this report, and no reference answer was read. Corpus chunks were "
             f"built with tokenizer `{manifest.get('tokenizer')}`; if that is not the pinned tiktoken build, the "
             f"boundary-sensitive rates (`partial_only_rate`) differ from the frozen harness corpus.", '',
             f"Arms: {', '.join(manifest['arms'])}. "]
    for arm in manifest['arms']:
        status = (manifest.get('arm_status') or {}).get(arm, {})
        if status.get('status') != 'ok':
            lines += ['', f'## {arm} ranker', '',
                      f"> Did not run: {status.get('error', 'unavailable')}. Nothing was substituted for it."]
            continue
        summary = (manifest.get('report') or {}).get(arm) or {}
        lines += _arm_lines(arm, summary)
    pair = (manifest.get('report') or {}).get('agreement')
    if pair:
        lines += ['', '## Do the two rankers agree?', '',
                  f"- same top chunk on {_percent(pair.get('same_top1_chunk'))} of queries; both find the evidence on "
                  f"{_percent(pair.get('both_find_evidence'))}; only the lexical arm on {pair.get('only_left_finds')} "
                  f"queries; only the embedding arm on {pair.get('only_right_finds')}; neither on "
                  f"{pair.get('neither_finds')}", '',
                  '> A lexical ranker that never saw the embedding model reaching the same conclusion is what keeps '
                  'this from being the index grading its own taste. Where they disagree, neither number above can '
                  'be trusted alone, and that is precisely where a model or a human reader is required.']
    coverage = manifest.get('unit_coverage') or {}
    if coverage.get('units'):
        lines += ['', '## Chunking, measured without any query at all', '',
                  f"- {coverage['whole_unit_rate']:.1%} of the {coverage['units']} audited source units fit inside "
                  f"one chunk ({coverage['most_of_it']} fit mostly, {coverage['split_or_absent']} are split or "
                  f"absent). This is a property of the chunker, not of ranking, and it bounds every recall number "
                  f"above: a span no chunk holds in full cannot be handed over whole."]
    lines += ['', '## What this report does not say', ''] + [f'- {item}' for item in manifest['caveats']]
    return '\n'.join(lines) + '\n'
