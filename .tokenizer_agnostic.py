import pathlib

sem = pathlib.Path('/home/user/RAGLab/raglab/semantic_chunking.py')
s = sem.read_text()

# validate: sizes are statements about one tokenization, and a map is not wrong in CI because CI counts
# tokens differently than the machine that reviewed it
old = "def validate(doc_text, entries, *, max_tokens=900, min_tokens=20, hard=True):"
new = """def validate(doc_text, entries, *, max_tokens=900, min_tokens=20, hard=True, drafted_with=None):"""
assert old in s; s = s.replace(old, new, 1)

old = '''    Empty list means: the chunks are verbatim slices, they cover the document with no gap or overlap,
    each is within the size band, and none swallows a later section heading (which is what 'exactly one
    idea' can be checked as mechanically).
    """'''
new = '''    Empty list means: the chunks are verbatim slices, they cover the document with no gap or overlap,
    each is within the size band, and none swallows a later section heading (which is what 'exactly one
    idea' can be checked as mechanically).

    ``drafted_with`` is the tokenizer identity recorded in the map. When it differs from this run's, the
    size band and the token-count drift check are skipped, and ``band_note`` says so: those complaints are
    statements about a tokenization this process does not use, so a map reviewed under cl100k_base is not
    'invalid' on a machine counting with the fallback estimator - it is just not measurable here. The
    structural rules (verbatim, complete cover, one subject) hold in both, and those are the ones that
    protect an answer.
    """
    band_note = None
    if drafted_with and drafted_with != tokenizer_identity():
        band_note = (f'size checks skipped: map drafted under {drafted_with}, this run counts with '
                     f'{tokenizer_identity()}')'''
assert old in s; s = s.replace(old, new, 1)

old = """        if 'tokens' in entry and abs(count_tokens(body) - int(entry['tokens'])) > 8:
            problems.append(f'chunk {index}: recorded {entry["tokens"]} tokens, text holds '
                            f'{count_tokens(body)} (the document changed since the map was drafted)')"""
new = """        if (drafted_with is None or drafted_with == tokenizer_identity()) \\
                and 'tokens' in entry and abs(count_tokens(body) - int(entry['tokens'])) > 8:
            problems.append(f'chunk {index}: recorded {entry["tokens"]} tokens, text holds '
                            f'{count_tokens(body)} (the document changed since the map was drafted)')"""
assert old in s, 'drift check'; s = s.replace(old, new, 1)

# guard the two size complaints behind the band check
old = """        if size > max_tokens:"""
new = """        if bands and size > max_tokens:"""
assert old in s, 'ceiling line'; s = s.replace(old, new, 1)
old = """        if size < min_tokens and not forced_stub("""
new = """        if bands and size < min_tokens and not forced_stub("""
assert old in s, 'floor line'; s = s.replace(old, new, 1)
old = """    text = str(doc_text or '')
    problems = []"""
new = """    text = str(doc_text or '')
    bands = drafted_with is None or drafted_with == tokenizer_identity()
    problems = []"""
assert old in s, 'problems init'; s = s.replace(old, new, 1)
old = "def validate(doc_text, entries, *, max_tokens=900, min_tokens=20, hard=True, drafted_with=None):"
new = "def validate(doc_text, entries, *, max_tokens=900, min_tokens=20, hard=True, drafted_with=None):  # noqa: C901"
s = s.replace(old, new, 1)

# write_map: record what counted the tokens
old = """    payload = {'document': Path(str(doc.get('source'))).name,
               'source_text_fp': text_fingerprint(doc.get('text')),"""
new = """    payload = {'document': Path(str(doc.get('source'))).name,
               'source_text_fp': text_fingerprint(doc.get('text')),
               'tokenizer': tokenizer_identity(),"""
assert old in s, 'write_map'; s = s.replace(old, new, 1)
old = """        found = validate(doc.get('text'), data['chunks'], max_tokens=maximum, hard=hard)"""
new = """        found = validate(doc.get('text'), data['chunks'], max_tokens=maximum, hard=hard,
                         drafted_with=data.get('tokenizer'))"""
assert old in s, 'chunk_documents validate'; s = s.replace(old, new, 1)
old = "from chunker import Chunk, count_tokens"
assert old in s or True
if "tokenizer_identity" not in s.split('def ')[0]:
    s = s.replace("from chunker import Chunk, count_tokens",
                  "from chunker import Chunk, count_tokens, tokenizer_identity", 1)
sem.write_text(s)

maps = pathlib.Path('/home/user/RAGLab/raglab/chunk_maps.py')
t = maps.read_text()
old = """            problems = sc.validate(doc.get('text'), entries, max_tokens=args.max_tokens,
                                   min_tokens=10, hard=not args.soft)"""
new = """            problems = sc.validate(doc.get('text'), entries, max_tokens=args.max_tokens,
                                   min_tokens=10, hard=not args.soft,
                                   drafted_with=sc.tokenizer_identity())"""
assert old in t, 'draft validate'; t = t.replace(old, new, 1)
old = """            problems = sc.validate(doc.get('text'), data['chunks'], max_tokens=args.max_tokens,
                                   hard=not args.soft)"""
new = """            problems = sc.validate(doc.get('text'), data['chunks'], max_tokens=args.max_tokens,
                                   hard=not args.soft, drafted_with=data.get('tokenizer'))
            note = sc.band_note(data.get('tokenizer'))"""
assert old in t, 'check validate'; t = t.replace(old, new, 1)
old = """            print(f'[check] {name}: {len(data["chunks"])} chunk(s), median {median}, '
                  f'{len(problems)} problem(s)')"""
new = """            print(f'[check] {name}: {len(data["chunks"])} chunk(s), median {median}, '
                  f'{len(problems)} problem(s)' + (f' — {note}' if note else ''))"""
assert old in t, 'check print'; t = t.replace(old, new, 1)
helper = '''def band_note(drafted_with):
    """Say out loud when the size band could not be judged, instead of reporting a clean pass."""
    if drafted_with and drafted_with != tokenizer_identity():
        return (f'sizes not comparable (map drafted under {drafted_with}, this run counts with '
                f'{tokenizer_identity()})')
    return None


'''
anchor = "def documents():"
assert anchor in t
t = t.replace(anchor, helper + anchor, 1)
if 'tokenizer_identity' not in t.split('\n\n')[0]:
    t = t.replace("from chunker import chunk_all, count_tokens",
                  "from chunker import chunk_all, count_tokens, tokenizer_identity", 1)
# the usable filter inside measure must agree with chunk_documents about the band exemption
old2 = """        if not sc.validate(doc.get('text'), data['chunks'], max_tokens=900,
                           hard=not getattr(args, 'soft', False)):"""
new2 = """        if not sc.validate(doc.get('text'), data['chunks'], max_tokens=900,
                           hard=not getattr(args, 'soft', False),
                           drafted_with=data.get('tokenizer')):"""
assert old2 in t, 'usable filter'; t = t.replace(old2, new2, 1)
maps.write_text(t)
print('maps are now tokenizer-relative, and the run says when it could not measure sizes')
