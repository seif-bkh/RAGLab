"""Manual, structure-aware chunk maps: one semantically complete idea per chunk.

Why this exists. Size-based chunking asks 'how many tokens fit' and gets whatever text lands in that
window. A semantic chunk map asks 'where does this document change subject' and records the answer.
The measured reason to care is that retrieval and this repo's citation contract want different things:
retrieval wants a chunk big enough to hold the whole idea, and quote-membership wants a chunk small
enough that a verbatim excerpt is not drowning in neighbours. 640/40 was chosen because it beat 420
and 220 on whole-document recall; a map can beat 640 on both at once, but only if boundaries land
where the text's own boundaries are.

Why a committed file and not a model call at run time. If a model produced the segmentation during a
run, the corpus would change whenever the model, prompt or temperature changed - which is precisely the
failure the frozen plan and the chunk fingerprint exist to prevent. So the map is data: reviewed by a
human, versioned, hashed into the collection's identity, and refused outright if the document it was
written for has since changed by a single character. Proposing boundaries mechanically is fine and is
what `propose` does; the artifact that retrieval runs on is the reviewed file.

The invariant that makes this safe to try: every character of the document must belong to exactly one
chunk, and a chunk's text must be a verbatim slice of the document. Coverage is checked, not assumed -
`validate` returns the specific gaps, overlaps, off-by-one slices and heading-crossing boundaries, and
`load_map` refuses a fingerprint mismatch rather than silently slicing the wrong characters.
"""
import hashlib
import json
import re
from pathlib import Path

from chunker import Chunk, count_tokens

# Structure markers that mean "a new idea starts here" in this corpus: markdown headings (the docx
# exports), Tunisian legal article numbers, and French/Arabic circular section numbers.
HEADING_PATTERNS = (
    r'^#{1,6}\s+\S',                       # markdown heading (docx conversion)
    r'^الفصل\s*[١-٩0-9]+',                  # "Article N" in the law
    r'^(?:Article|Chapitre|Titre)\s+\d',    # French legal headings
    r'^\d+\s*[.\-)]\s+\S',                  # numbered clauses
    r'^[-*]\s+\S',                          # list item: a distinct condition or product row
    r'^\|.*\|\s*$',                         # one table row = one product or one condition
)
# A chunk that starts mid-thought usually shows up as one of these at its very beginning.
# A chunk that opens mid-sentence is the usual sign a thought was cut. Bullets, table rows and headings
# are legitimate openers in this corpus, so they are excluded rather than special-cased away later.
ORPHAN_TAIL = re.compile(r'^(?:[,;:.\-–—)\]]|[a-zà-ÿ]{1,3}$)')


def text_fingerprint(text):
    """Hash of the exact characters the map indexes. Offsets are only meaningful for this text."""
    return hashlib.sha256(str(text).encode('utf-8')).hexdigest()[:32]


def maps_fingerprint(directory):
    """One hash over every map in a directory, so a chunk fingerprint changes when any boundary does."""
    directory = Path(directory)
    digest = hashlib.sha256()
    for path in sorted(directory.glob('*.json')) if directory.is_dir() else []:
        digest.update(path.name.encode('utf-8'))
        digest.update(path.read_bytes())
    return digest.hexdigest()[:32] if digest.digest() != hashlib.sha256(b'').digest() else 'no-maps'


def map_path(document_source, directory):
    stem = Path(str(document_source)).name
    return Path(directory) / (stem + '.json')


def _blocks(text, *, max_tokens):
    """The document as atomic pieces: a heading, a table row, a list item, a paragraph.

    These are the units a boundary may fall between, which is what makes the result reviewable: a chunk
    is a run of blocks, so 'one idea' can be checked as 'no new top-level heading inside it' and 'no
    table row mixed with prose', instead of as a feeling about the text.
    """
    heading_re = re.compile('(?:' + '|'.join(HEADING_PATTERNS) + ')', re.M)
    cuts = {0, len(text)}
    for match in heading_re.finditer(text):
        cuts.add(match.start())
        cuts.add(match.end())
    for match in re.finditer(r'\n\s*\n', text):                     # paragraph breaks
        cuts.add(match.end())
    for match in re.finditer(r'\n\|', text):                        # each table row
        cuts.add(match.start() + 1)
    for match in re.finditer(r'\n\s*[-*]\s', text):                 # each list item
        cuts.add(match.start() + 1)
    for match in re.finditer(r'(?<![.!?:])\.\s+(?=[A-ZÀ-ÖØ-öø-ÿالف-ي])', text):
        cuts.add(match.end())                                        # sentence end, as a last resort
    ordered = sorted(position for position in cuts if 0 <= position <= len(text))
    blocks, current = [], ordered[0]
    for position in ordered[1:]:
        if position > current:
            blocks.append((current, position))
            current = position
    if current < len(text):
        blocks.append((current, len(text)))
    # A block longer than the ceiling is split at sentence ends, never inside a sentence.
    pieces = []
    for block_start, block_end in blocks:
        body = text[block_start:block_end]
        if count_tokens(body) <= max_tokens or '\n' not in body.strip():
            pieces.append((block_start, block_end))
            continue
        inner = [block_start] + [block_start + m.end() for m in
                                 re.finditer(r'(?<![.!?:])\.\s+', body) if m.end() < len(body) - 1] + [block_end]
        inner = sorted({i for i in inner if block_start <= i <= block_end})
        for a, b in zip(inner, inner[1:]):
            if b > a:
                pieces.append((a, b))
    merged = []
    for a, b in pieces:
        if not text[a:b].strip():
            if merged:
                merged[-1] = (merged[-1][0], b)
            elif pieces.index((a, b)) + 1 < len(pieces):
                continue
            continue
        merged.append((a, b))
    return [(a, b, _kind(text[a:b])) for a, b in merged]


def _kind(body):
    stripped = body.strip()
    if stripped.startswith('|'):
        return 'table-row'
    if stripped.startswith('#'):
        return 'heading'
    if ARTICLE.match(stripped):
        return 'article'
    if re.match(r'^[-*]\s', stripped):
        return 'list-item'
    return 'paragraph'


def _level(kind, body):
    """How strong a boundary this block opens. Higher number = smaller unit."""
    match = re.match(r'^(#{1,6})\s', body.strip())
    if match:
        return len(match.group(1))
    return {'heading': 1, 'article': 2, 'table-row': 4, 'list-item': 4, 'paragraph': 5}.get(kind, 5)


def propose(doc, *, target_tokens=300, max_tokens=520, min_tokens=60, hard_subjects=True):
    """Draft boundaries from the document's own structure. A starting point for review, not a verdict.

    A chunk is a run of blocks that stays under target_tokens, closes the moment a block opens a
    heading at or above the chunk's starting level, and never mixes table rows with prose. Sections
    shorter than min_tokens are folded into a neighbour rather than left as stubs, because a 20-token
    'chunk' is a retrieval hit with no context.
    """
    text = str(doc.get('text') or '')
    blocks = _blocks(text, max_tokens=max_tokens)
    chunks, run, run_tokens = [], [], 0
    for start, end, kind in blocks:
        body = text[start:end]
        size = count_tokens(body)
        level = _level(kind, body)
        if run:
            first = run[0]
            first_level = _level(first[2], text[first[0]:first[1]])
            clashes = (first[2] == 'table-row') != (kind == 'table-row')
            closes_new_subject = level <= first_level and (kind == 'heading'
                                                           or (hard_subjects and kind == 'article'))
            if clashes or closes_new_subject or run_tokens + size > target_tokens:
                chunks.append((first[0], run[-1][1]))
                run, run_tokens = [], 0
        run.append((start, end, kind))
        run_tokens += size
        while count_tokens(text[run[0][0]:run[-1][1]]) > max_tokens and len(run) > 1:
            chunks.append((run[0][0], run[-2][1]))
            run = run[-1:]
            run_tokens = count_tokens(text[run[0][0]:run[-1][1]])
    if run:
        chunks.append((run[0][0], run[-1][1]))
    entries = []
    for start, end in chunks:
        body = text[start:end]
        # Fold a stub into the chunk before it. A 15-token chunk is not 'one idea', it is a fragment that
        # retrieval can return as a hit with nothing in it - the failure mode 220-token chunks measured.
        if count_tokens(body) < min_tokens and entries:
            entries[-1] = _entry(entries[-1]['start'], end, text)
            continue
        entries.append(_entry(start, end, text))
    return enforce_one_subject(text, entries, min_tokens=min_tokens, hard=hard_subjects)


def _entry(start, end, text, idea=None):
    """One map row: offsets, a token count recomputed from the final slice, and a reviewable label."""
    body = text[start:end]
    heading = ''
    for line in body.splitlines():
        stripped = line.strip()
        if stripped.startswith('#') or re.match(HEADING_PATTERNS[1] + '|' + HEADING_PATTERNS[2], stripped):
            heading = stripped.lstrip('#').strip()
            break
    return {'start': int(start), 'end': int(end), 'tokens': count_tokens(body),
            'idea': idea or (heading[:110] if heading else ' '.join(body.split())[:80]),
            'heading': heading[:160]}


# Derived from the same list validate() counts, so the builder can never miss a boundary the checker then
# complains about: one definition, used by both. (It was written twice once, and the two copies disagreed
# about Arabic 'الفصل4' with no space - a chunk holding two articles that no amount of re-drafting split.)
#
# Only these patterns open a *subject*. The rest of HEADING_PATTERNS - numbered list items, table rows -
# are boundaries between blocks, not between ideas: a product table is one idea over eleven rows, and
# treating every row as a new subject would put the whole table back into pieces too small to answer from.
SUBJECT_PATTERNS = [pattern for pattern in HEADING_PATTERNS[:3]]
MARKER = re.compile('(?:' + '|'.join(SUBJECT_PATTERNS) + ')', re.M)
ARTICLE = re.compile('(?:' + '|'.join(SUBJECT_PATTERNS[1:]) + ')', re.M)


def one_subject(text, start, end, *, hard=True):
    """Would [start,end) still be one subject? Checked before every merge, so fixing a size never
    creates a chunk holding two articles - the trade this whole module is about."""
    if not hard:
        return True
    return len(ARTICLE.findall(text[start:end])) <= 1


def forced_stub(doc_text, entries, index, min_tokens, *, maximum=900):
    """True when a chunk under the floor cannot be enlarged without taking a neighbour's subject.

    Structure and size disagree on a document made of very short articles. Saying 'either way it is
    wrong' is not useful, so the disagreement is resolved in favour of structure and recorded here,
    where a reviewer sees it, instead of by quietly merging two ideas together.
    """
    entry = entries[index]
    start, end = int(entry['start']), int(entry['end'])

    def ok(a, b):
        return len(ARTICLE.findall(doc_text[a:b])) <= 1 and (b - a) and \
            count_tokens(doc_text[a:b]) <= maximum

    if index and ok(entries[index - 1]['start'], end):
        return False
    if index + 1 < len(entries) and ok(start, entries[index + 1]['end']):
        return False
    return True


def enforce_one_subject(text, entries, *, min_tokens=20, hard=True):
    """Split any chunk that still contains a second top-level marker, at that marker.

    The packer tries to close on structure, but a block boundary can be invisible to it: a PDF page that
    carries 'الفصل 3 …الفصل 4' on one line, a docx export that keeps an article heading inside a
    paragraph. Rather than shipping a chunk that holds two ideas and hoping retrieval forgives it, the
    rule is applied a second time to the assembled chunks. Fragments below the floor are rejoined to the
    piece before them, so enforcing one rule never creates another violation.
    """
    out = []
    for entry in entries:
        start, end = int(entry['start']), int(entry['end'])
        body = text[start:end]
        # In soft mode only a *heading* forces a cut; an article marker is where the packing would have
        # liked to break, which is enough to keep sentences whole without fragmenting a law into one
        # chunk per article - the failure mode that made the hard maps lose retrieval recall.
        finder = MARKER if hard else re.compile('(?m)^#{1,6}\s+\S', re.M)
        marks = [start + m.start() for m in finder.finditer(body) if m.start() > 0]
        pieces, cursor = [], start
        for mark in marks:
            pieces.append((cursor, mark))
            cursor = mark
        pieces.append((cursor, end))
        pieces = [piece for piece in pieces if piece[1] > piece[0]]
        # A bare heading is not a chunk, it is the label of what follows, so a heading-shaped stub is
        # pushed forward onto its section; any other stub is folded back into its neighbour. Anything
        # else that falls under the floor joins the piece before it rather than being shipped as a hit
        # with nothing in it.
        sized = [(a, b, count_tokens(text[a:b])) for a, b in pieces if one_subject(text, a, b, hard=hard)]
        merged = []
        for index, (start_piece, end_piece, size) in enumerate(sized):
            if size < min_tokens:
                # Two rules collide here: 'no stub' and 'one subject per chunk'. The subject rule wins, so
                # a stub is only folded away when the merge does not steal the neighbour's subject - a
                # short chunk forced by the document's own structure is a shape, not a mistake.
                heading_shaped = len([line for line in text[start_piece:end_piece].splitlines()
                                      if line.strip()]) <= 2
                follows = sized[index + 1] if index + 1 < len(sized) else None
                if heading_shaped and follows is not None \
                        and one_subject(text, start_piece, follows[1], hard=hard):
                    sized[index + 1] = (start_piece, follows[1], size + follows[2])
                    continue
                if merged and one_subject(text, merged[-1][0], end_piece, hard=hard):
                    merged[-1] = (merged[-1][0], end_piece, merged[-1][2] + size)
                    continue
            merged.append((start_piece, end_piece, size))
        for piece_start, piece_end, _size in merged:
            out.append(_entry(piece_start, piece_end, text))
    return out


def validate(doc_text, entries, *, max_tokens=900, min_tokens=20, hard=True):
    """Return every way this map fails to be a partition of the document into complete ideas.

    Empty list means: the chunks are verbatim slices, they cover the document with no gap or overlap,
    each is within the size band, and none swallows a later section heading (which is what 'exactly one
    idea' can be checked as mechanically).
    """
    text = str(doc_text or '')
    problems = []
    heading_re = re.compile('(?:' + '|'.join(p for p in HEADING_PATTERNS if not p.startswith(r'^\|')) + ')', re.M)
    ordered = sorted(range(len(entries)), key=lambda i: entries[i]['start'])
    if ordered:
        # Coverage means the whole document, including its opening and closing lines: a map that starts
        # at 40 silently drops the title, and nothing else in this file would notice.
        first = int(entries[ordered[0]]['start'])
        if first != 0:
            problems.append(f'the map starts at character {first}, so characters 0..{first} belong to '
                            'no chunk')
        last = int(entries[ordered[-1]]['end'])
        if last < len(text) and text[last:].strip():
            problems.append(f'the map ends at character {last}, so the document still has '                            f'{len(text) - last} characters of text after it')
    for position, index in enumerate(ordered):
        entry = entries[index]
        start, end = int(entry['start']), int(entry['end'])
        if start < 0 or end > len(text) or end <= start:
            problems.append(f'chunk {index}: offsets {start}..{end} are not inside 0..{len(text)}')
            continue
        body = text[start:end]
        if not body.strip():
            problems.append(f'chunk {index}: slices only whitespace')
        if 'tokens' in entry and abs(count_tokens(body) - int(entry['tokens'])) > 8:
            problems.append(f'chunk {index}: recorded {entry["tokens"]} tokens, text holds '
                            f'{count_tokens(body)} (the document changed since the map was drafted)')
        size = count_tokens(body)
        if size > max_tokens:
            problems.append(f'chunk {index}: {size} tokens exceeds the {max_tokens} ceiling')
        if size < min_tokens and not forced_stub(doc_text, entries, index, min_tokens, maximum=max_tokens):
            problems.append(f'chunk {index}: {size} tokens is below the {min_tokens} floor')
        stripped = body.strip()
        if ORPHAN_TAIL.match(stripped) and not re.match(r'^(?:[-*|]|#{1,6}\s)', stripped):
            problems.append(f'chunk {index}: opens on a fragment, so a thought was cut in half')
        # One idea means one *top-level* subject. Sub-headings under it are its structure, and a table
        # or a list is one idea spread over rows, so only a second sibling at the same level counts.
        tops = [m.group(0).strip() for m in re.finditer(r'(?m)^(#{1,6})\s+\S.*$', stripped)]
        if tops:
            shallow = min(len(head.split()[0]) for head in tops)
            same = [head for head in tops if len(head.split()[0]) == shallow]
            # '## 1. x', '### 1.1 y', '### 1.2 z' is one section with subsections. The parent number is
            # what makes it one idea; two children of the same parent are not two subjects.
            parents = {re.match(r'(?:#+\s*)?(\d+(?:\.\d+)?)', head.replace('#', '').strip())
                       for head in same}
            trimmed = sorted({(m.group(1) or '').rsplit('.', 1)[0] if m else ''
                              for m in (re.match(r'(?:#+\s*)?(\d+(?:\.\d+)?)',
                                                 head.replace('#', '').strip()) for head in same)})
            if len(same) > 1 and not (len(trimmed) == 1 and trimmed[0] != ''):
                problems.append(f'chunk {index}: carries {len(same)} headings at level {shallow}, so it '
                                'holds several ideas')
        articles = len(ARTICLE.findall(stripped))
        if hard and articles > 1:
            problems.append(f'chunk {index}: carries {articles} numbered articles in one chunk')
        if position:
            previous = entries[ordered[position - 1]]
            if int(previous['end']) > start:
                problems.append(f'chunk {index}: overlaps the previous chunk')
            elif int(previous['end']) < start:
                skipped = text[int(previous['end']):start]
                if skipped.strip():
                    problems.append(f'gap before chunk {index}: {len(skipped.strip())} character(s) of '
                                    'the document are in no chunk')
    if entries:
        covered = sum(int(e['end']) - int(e['start']) for e in entries)
        if covered != len(text):
            stray = sum(len(text[int(e['end']):next_start].strip())
                        for e, next_start in zip(entries, [entries[i + 1]['start'] for i in
                                                           range(len(entries) - 1)] + [len(text)]))
            if stray:
                problems.append(f'coverage: {stray} non-whitespace character(s) unassigned')
    return problems


def load_map(path, doc):
    """Read a reviewed map, or refuse. A map is offsets into one exact text, so a changed document is
    a broken map, not a small drift."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f'no chunk map at {path}')
    data = json.loads(path.read_text(encoding='utf-8'))
    recorded = data.get('source_text_fp')
    current = text_fingerprint(doc.get('text'))
    if recorded and recorded != current:
        raise ValueError(f'{path.name} was written for a different text of '
                         f'{Path(str(doc.get("source"))).name} (map {recorded}, document {current}); '
                         'the document changed, so these offsets point at the wrong characters. '
                         'Re-draft and review the map instead of reusing it.')
    return data


def chunks_from_map(doc, entries, cfg, *, index_offset=0):
    """Chunk objects from a validated map. Text is sliced, never re-derived, so quotes still match."""
    text = str(doc.get('text') or '')
    language = doc.get('language') or 'ar'
    source = str(doc.get('source') or doc.get('path') or '')
    chunks = []
    for position, entry in enumerate(entries):
        body = text[int(entry['start']):int(entry['end'])]
        chunks.append(Chunk(index=index_offset + position, text=body.strip(),
                            heading=entry.get('heading', ''), language=language, source=source,
                            token_count=count_tokens(body), origin=doc.get('origin', 'maps/'),
                            section_type=entry.get('section_type', 'content'),
                            notes=['manual-map', f'offsets {entry["start"]}-{entry["end"]}',
                                   f'idea: {entry.get("idea", "")[:70]}']))
    return chunks


def chunk_documents(docs, cfg, *, strict=True, hard=True):
    """Chunk every document from its map. Strict by design: mixing a mapped document with an unmapped
    one, chunked by size, would produce a corpus whose chunks mean two different things."""
    directory = Path(getattr(cfg, 'CHUNK_MAP_DIR', Path(__file__).resolve().parent / 'benchmarks' / 'chunk_maps'))
    maximum = int(getattr(cfg, 'CHUNK_MAX_TOKENS', 900))
    all_chunks, missing, problems = [], [], {}
    offset = 0
    for doc in docs:
        path = map_path(doc.get('source'), directory)
        if not path.exists():
            missing.append(Path(str(doc.get('source'))).name)
            continue
        data = load_map(path, doc)
        found = validate(doc.get('text'), data['chunks'], max_tokens=maximum, hard=hard)
        if found:
            problems[Path(str(doc.get('source'))).name] = found[:8]
            continue
        produced = chunks_from_map(doc, data['chunks'], cfg, index_offset=offset)
        offset += len(produced)
        all_chunks.extend(produced)
    if strict and (missing or problems):
        details = []
        if missing:
            details.append('no map for: ' + ', '.join(missing))
        for name, issues in problems.items():
            details.append(f'{name} map is invalid: ' + '; '.join(issues[:3]))
        raise ValueError('manual chunking refused - ' + ' | '.join(details) +
                         f' (maps in {directory})')
    return all_chunks, problems


def write_map(path, doc, entries, *, note=''):
    """Persist a reviewed map, with the fingerprint of the text its offsets index."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {'document': Path(str(doc.get('source'))).name,
               'source_text_fp': text_fingerprint(doc.get('text')),
               'chunk_count': len(entries), 'tokens_total': sum(int(e['tokens']) for e in entries),
               'note': note, 'chunks': entries}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    return payload
