"""Draft, check and measure manual chunk maps - the workspace for one-idea-per-chunk segmentation.

    python chunk_maps.py draft   [--document X] [--target 380] [--max 760]   # write a starting map
    python chunk_maps.py check                                                # validate every map
    python chunk_maps.py measure                                              # compare maps with size chunking

`measure` answers the only question that matters before spending a provider call on a new corpus
version: when a chunk is what retrieval returns, does it contain the whole passage the answer needs?
It reads the accepted harness families' verbatim evidence quotes and asks, for each chunking, how many
of those quotes survive inside one chunk. That is a property of the segmentation and the documents
alone, so it costs nothing and cannot be argued about later.
"""
import argparse
import json
import os
import statistics
import sys
from pathlib import Path

import config as cfg
from chunker import chunk_all, count_tokens
from loader import load_all
import semantic_chunking as sc

PROJECT = Path(__file__).resolve().parent
# Also an environment setting, because a chunk map is a corpus version: two candidate segmentations have
# to be comparable side by side (chunk_maps vs chunk_maps_soft) without editing either, and
# store.chunk_fp hashes exactly this directory, so what is measured here is what would be ingested.
MAP_DIR = Path(os.getenv('CHUNK_MAP_DIR', str(PROJECT / 'benchmarks' / 'chunk_maps')))
ACCEPTED = PROJECT / 'benchmarks' / 'hard_harness_accepted'


def documents():
    return load_all([PROJECT.parent / 'docs', cfg.DATA_DIR])


family_questions = {}


def quote_index():
    """Every verbatim evidence quote the accepted harness families were built on, plus their questions."""
    quotes = []
    for path in sorted(ACCEPTED.glob('author_0*.jsonl')):
        for line in path.read_text(encoding='utf-8').splitlines():
            if not line.strip():
                continue
            family = json.loads(line).get('family') or {}
            languages = family.get('languages') or {}
            family_questions[family.get('id')] = {code: str((entry or {}).get('question') or '')
                                                 for code, entry in languages.items()}
            for evidence in family.get('evidence') or []:
                quote = str(evidence.get('quote') or '')
                if len(quote.strip()) >= 12:
                    quotes.append((family.get('id'), str(evidence.get('unit_id') or '').split(':')[0],
                                   ' '.join(quote.split())))
    return quotes


def integrity(chunks, quotes):
    """How much of each quote a single chunk carries, over every chunk of that document."""
    by_document = {}
    for chunk in chunks:
        by_document.setdefault(Path(str(chunk.source)).name, []).append(
            ' '.join(chunk.text.split()))
    whole = split = absent = 0
    share = []                                  # fraction of the quote covered by the best single chunk
    for family_id, source, quote in quotes:
        candidates = by_document.get(Path(str(source)).name)
        if not candidates:
            continue
        best = max((_coverage(candidate, quote) for candidate in candidates), default=0.0)
        if best >= 0.999:
            whole += 1
        elif best > 0.5:
            split += 1
        else:
            absent += 1
        share.append(best)
    counted = whole + split + absent
    return {'quotes': counted, 'whole_in_one_chunk': whole, 'split_across_chunks': split,
            'barely_found': absent, 'whole_rate': round(whole / counted, 4) if counted else None,
            'mean_best_coverage': round(sum(share) / len(share), 4) if share else None}


def _coverage(chunk_text, quote):
    """Largest run of the quote that this chunk contains, as a fraction of the quote.

    Substring testing is the honest version of 'would a grader find this quote here': the citation
    contract requires contiguous verbatim text, so a chunk that holds 60% of a quote is a chunk that
    fails the contract for that quote, not one that half-succeeds.
    """
    if quote in chunk_text:
        return 1.0
    if not quote:
        return 0.0
    low, high = 1, len(quote)
    best = 0
    while low <= high:
        mid = (low + high) // 2
        if any(quote[i:i + mid] in chunk_text for i in range(0, len(quote) - mid + 1, 7)):
            best = mid
            low = mid + 1
        else:
            high = mid - 1
    return best / len(quote)


def family_integrity(chunks, quotes):
    """Families where one chunk is enough for every piece of evidence — what retrieval must achieve."""
    by_document = {}
    for chunk in chunks:
        by_document.setdefault(Path(str(chunk.source)).name, []).append(' '.join(chunk.text.split()))
    per_family = {}
    for family_id, source, quote in quotes:
        if not family_id:
            continue
        candidates = by_document.get(Path(str(source)).name, [])
        if not candidates:
            continue                                  # that document is not in this chunk set
        per_family.setdefault(family_id, []).append(max((_coverage(c, quote) for c in candidates),
                                                         default=0.0))
    solved = sum(1 for coverages in per_family.values() if all(c >= 0.999 for c in coverages))
    return {'families': len(per_family), 'answerable_from_one_chunk': solved,
            'rate': round(solved / len(per_family), 4) if per_family else None}


def profile(chunks):
    sizes = [chunk.token_count or count_tokens(chunk.text) for chunk in chunks]
    return {'chunks': len(chunks), 'min': min(sizes) if sizes else 0,
            'median': int(statistics.median(sizes)) if sizes else 0,
            'max': max(sizes) if sizes else 0,
            'single_sentence': sum(1 for s in sizes if s < 40)}


def answerable_families(chunks, families_by_id, *, language='ar'):
    """The families this chunk set can support at all: some chunk holds a quote end to end.

    Recall measured over each row's own answerable set is not a comparison, because a row can score higher
    by having fewer hard questions in its denominator. So the ranking rows are all scored on one fixed
    set - the families the published 640-token pin can support - and a row that loses a question the pin
    could support is counted as a miss, which is the honest direction for the risk being measured."""
    by_document = {}
    for chunk in chunks:
        by_document.setdefault(Path(str(chunk.source)).name, []).append(' '.join(chunk.text.split()))
    wanted = {}
    for family_id, source, quote in families_by_id:
        wanted.setdefault(family_id, []).append((Path(str(source)).name, quote))
    supported = set()
    for family_id, entries in wanted.items():
        if not (family_questions.get(family_id) or {}).get(language):
            continue
        for name, quote in entries:
            if any(_coverage(text, quote) >= 0.999 for text in by_document.get(name, [])):
                supported.add(family_id)
                break
    return supported


def lexical_recall(chunks, families_by_id, *, k=5, language='ar', only=None):
    """Rank chunks with BM25 over each real question, and ask whether the supporting passage comes back.

    This is the axis semantic chunking is supposed to win: not 'is the quote inside one chunk' (size
    answers that) but 'does the chunk that supports the answer survive the top-k cut'. Gold labels are
    the harness's own verbatim evidence quotes, so no judgement is involved: a hit is a returned chunk
    that contains a supporting quote end to end. BM25 carries its own length normalisation, so chunk
    size is not automatically rewarded here.
    """
    from store import BM25Index
    by_document = {}
    for chunk in chunks:
        by_document.setdefault(Path(str(chunk.source)).name, []).append(chunk)
    flat = [chunk for name in by_document for chunk in by_document[name]]
    if not flat:
        return None
    index = BM25Index([' '.join(chunk.text.split()) for chunk in flat])
    positions = {id(chunk): position for position, chunk in enumerate(flat)}
    quotes_for = {}
    for family_id, source, quote in families_by_id:
        quotes_for.setdefault(family_id, []).append((Path(str(source)).name, quote))
    hits_at = {1: 0, k: 0}
    scored = 0
    for family_id, entries in quotes_for.items():
        question = (family_questions.get(family_id) or {}).get(language)
        if not question:
            continue
        if only is not None and family_id not in only:
            continue
        wanted = [(name, quote) for name, quote in entries]
        scores = index.score(question)
        order = sorted(range(len(flat)), key=lambda i: -scores[i])[:k]
        ranked = [flat[i] for i in order]
        def is_hit(chunk):
            text = ' '.join(chunk.text.split())
            return any(name == Path(str(chunk.source)).name and _coverage(text, quote) >= 0.999
                       for name, quote in wanted)
        flags = [is_hit(chunk) for chunk in ranked]
        scored += 1                                   # a question this row cannot retrieve is a miss here,
        # not an exclusion: dropping it would let a row look better by scoring fewer questions.
        if flags[0]:
            hits_at[1] += 1
        if any(flags[:k]):
            hits_at[k] += 1
    return {'answerable': scored, 'recall_at_1': round(hits_at[1] / scored, 4) if scored else None,
            'recall_at_k': round(hits_at[k] / scored, 4) if scored else None, 'k': k}


def main(argv=None):
    parser = argparse.ArgumentParser(prog='chunk_maps.py', description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest='command')
    draft = sub.add_parser('draft')
    draft.add_argument('--document', default=None, help='only this file name from the corpus')
    draft.add_argument('--target', type=int, default=320)
    draft.add_argument('--max', dest='max_tokens', type=int, default=560)
    draft.add_argument('--min', dest='min_tokens', type=int, default=60)
    draft.add_argument('--force', action='store_true', help='overwrite an existing map')
    draft.add_argument('--soft', action='store_true',
                       help='pack at article boundaries but do not force a cut there: one heading per '
                            'chunk, several short articles allowed (so a law stays readable)')
    check = sub.add_parser('check')
    check.add_argument('--max-tokens', type=int, default=900)
    check.add_argument('--soft', action='store_true', help='do not fault a chunk for holding 2+ articles')
    measure = sub.add_parser('measure')
    measure.add_argument('--soft', action='store_true',
                         help='score the maps in $CHUNK_MAP_DIR as drafted with --soft')
    args = parser.parse_args(argv)

    docs = documents()
    if args.command == 'draft':
        for doc in docs:
            name = Path(str(doc.get('source'))).name
            if args.document and args.document not in name:
                continue
            path = sc.map_path(doc.get('source'), MAP_DIR)
            if path.exists() and not args.force:
                print(f'[draft] {name}: a map already exists (use --force to re-draft)')
                continue
            entries = sc.propose(doc, target_tokens=args.target, max_tokens=args.max_tokens,
                                 min_tokens=args.min_tokens, hard_subjects=not args.soft)
            problems = sc.validate(doc.get('text'), entries, max_tokens=args.max_tokens,
                                   min_tokens=10, hard=not args.soft)
            payload = sc.write_map(path, doc, entries,
                                   note=f'drafted by chunk_maps.py (target {args.target}, max '
                                        f'{args.max_tokens} tokens); needs human review of each idea label')
            sizes = [entry['tokens'] for entry in entries] or [0]
            print(f'[draft] {name}: {payload["chunk_count"]} chunk(s), median {int(statistics.median(sizes))} '
                  f'tokens, {len(problems)} structural complaint(s) -> {path.relative_to(PROJECT.parent)}')
            for problem in problems[:5]:
                print(f'         {problem}')
        return 0
    if args.command == 'check':
        bad = 0
        for doc in docs:
            name = Path(str(doc.get('source'))).name
            path = sc.map_path(doc.get('source'), MAP_DIR)
            if not path.exists():
                print(f'[check] {name}: no map')
                bad += 1
                continue
            data = sc.load_map(path, doc)
            problems = sc.validate(doc.get('text'), data['chunks'], max_tokens=args.max_tokens,
                                   hard=not args.soft)
            sizes = [count_tokens(' '.join(str(doc['text'])[int(e['start']):int(e['end'])].split()))
                     for e in data['chunks']]
            print(f'[check] {name}: {len(data["chunks"])} chunk(s), median {int(statistics.median(sizes))}, '
                  f'{"OK" if not problems else str(len(problems)) + " problem(s)"}')
            for problem in problems[:6]:
                print(f'         {problem}')
                bad += 1
        return 1 if bad else 0

    # measure: the number that decides whether a new corpus version is worth paying for
    quotes = quote_index()
    usable = set()
    for doc in docs:
        path = sc.map_path(doc.get('source'), MAP_DIR)
        if not path.exists():
            continue
        try:
            data = sc.load_map(path, doc)
        except Exception:                                              # noqa: BLE001
            continue
        if not sc.validate(doc.get('text'), data['chunks'], max_tokens=900,
                           hard=not getattr(args, 'soft', False)):
            usable.add(Path(str(doc.get('source'))).name)
    mapped = usable
    if mapped and len(mapped) < len(docs):
        # Every row is restricted to documents all rows can cover: scoring a manual chunk set over 4
        # documents against size chunking over 6 measures the corpus, not the chunking.
        # Comparing a manual row over 2 documents against size rows over 6 would flatter one of them.
        docs = [doc for doc in docs if Path(str(doc.get('source'))).name in mapped]
        allowed = {Path(str(source)).name for _, source, _ in quotes} & mapped
        quotes = [row for row in quotes if Path(str(row[1])).name in allowed]
        print(f'[measure] head-to-head restricted to the {len(mapped)} document(s) that have maps')
    print(f'documents in this comparison: {len(docs)} ({", ".join(sorted(Path(str(d.get("source"))).name for d in docs))})')
    print(f'evidence quotes under test: {len(quotes)} from {ACCEPTED.relative_to(PROJECT.parent)}')
    settings = [('size 220/40', 220, 40), ('size 640/40', 640, 40), ('size 420/40', 420, 40),
                ('manual maps', None, None)]
    chunk_sets = {}
    for label, size, overlap in settings:
        if size is None:
            try:
                chunk_sets[label], problems = sc.chunk_documents(docs, _map_cfg(args), strict=False,
                                                                  hard=not args.soft)
            except Exception as exc:                                   # noqa: BLE001
                print(f'[measure] manual: {exc}')
                chunk_sets[label] = []
                continue
            if problems:
                print(f'[measure] manual: {len(problems)} document(s) excluded for an invalid map: '
                      + ', '.join(sorted(problems)))
        else:
            cfg.CHUNK_SIZE_TOKENS, cfg.CHUNK_OVERLAP_TOKENS = size, overlap
            chunk_sets[label] = chunk_all(docs, cfg)
    reference = chunk_sets.get('size 640/40') or []
    only = answerable_families(reference, quotes) if reference else None
    if only:
        print(f'questions scored on every row: {len(only)} (those the pinned 640/40 can support)')
    rows = []
    for label, size, overlap in settings:
        chunks = chunk_sets[label]
        whole = integrity(chunks, quotes)
        families = family_integrity(chunks, quotes)
        rows.append((label, profile(chunks), whole, families,
                     lexical_recall(chunks, quotes, k=5, language='ar', only=only)))
    print(f'\n{"chunking":14s} {"chunks":>6s} {"med":>5s} {"tiny":>5s} {"quote whole":>12s} '
          f'{"family in 1":>12s} {"lex R@1":>8s} {"lex R@5":>8s}')
    for label, prof, whole, fam, lex in rows:
        lex = lex or {}
        print(f'{label:14s} {prof["chunks"]:6d} {prof["median"]:5d} {prof["single_sentence"]:5d} '
              f'{_pct(whole["whole_rate"])} {_pct(fam["rate"])} {_pct(lex.get("recall_at_1"))} '
              f'{_pct(lex.get("recall_at_k"))}   (of {lex.get("answerable", 0)} answerable questions)')
    print('\n"quote whole" = share of verbatim evidence quotes a single chunk contains end to end; that '
          'is what the citation contract needs from one retrieval hit.\n"family in 1" = share of '
          'question families whose every evidence quote fits in one chunk: the ideal for k=1 recall.')
    return 0


def _pct(value):
    return f'{value * 100:10.1f}%' if value is not None else f'{"n/a":>11s}'


def _map_cfg(args):
    class MapCfg:
        CHUNK_MAP_DIR = MAP_DIR
        CHUNK_MAX_TOKENS = 900
    return MapCfg()


if __name__ == '__main__':
    sys.exit(main())
