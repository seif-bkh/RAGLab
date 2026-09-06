"""chat.py — ask questions about the corpus in a REPL, answered by an NVIDIA Nemotron chat model.

Why this exists next to `main.py answer`: the benchmark pins its answerer to xKiro Qwen and refuses
any other SKU, because published scores have to stay comparable to the frozen plan. This is the other
thing — a conversational path over the same index, the same verbatim-evidence validation and the same
capability refusals, driven by NVIDIA_API_KEY and nvidia/nemotron-3.5-lightning-30b-a3b (which has a
free endpoint on build.nvidia.com). It is a lab tool, not a benchmark result: nothing here writes to
`results/hard_harness/`, no harness number is attributed to this model, and the plan file is untouched.

Two properties are inherited rather than reinvented, and they are the reason to use this instead of a
plain chat client. Retrieval is the lab's own (Nemotron embeddings over chunks built by chunker.py), so
an answer is only as good as the excerpts that were actually found — `--show-context` prints them. And
the reply is not accepted as prose: it must come back as claims carrying contiguous verbatim quotes from
those excerpts, which are then checked to be really present. A model that improvises a fee or a number
gets `status=refused reason=invalid_output`, with its raw reply shown, instead of a fluent sentence
nobody can trace.

`python chat.py --check` first: it reports which keys loaded, the chunking and the collection count,
and makes no completion call.
"""
import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import config as cfg
from nvidia_api import NvidiaClient, safe_error

PROJECT_DIR = Path(__file__).resolve().parent
CHAT_MODEL = 'nvidia/nemotron-3.5-lightning-30b-a3b'
# The build endpoint documents this exact slug; sampling parameters come from its model card, and
# 'enable_thinking' is that model's documented switch rather than a universal flag.
NVIDIA_CHAT_BASE_URL = 'https://integrate.api.nvidia.com/v1'
HELP = """commands in the chat:
  :k 7        retrieve N excerpts per question (1-20)
  :show       also print the excerpts an answer was built from, not only the cited quotes
  :log PATH   append every turn as one JSON line (model, claims, quotes, timings)
  :context    print what this session is configured with
  :quit       leave (Ctrl-D works too)
Nemotron's reasoning mode is a start-up flag (--thinking), because it changes the sampling shape and
the reply ceiling rather than the prompt.
"""


def chat_config(model=CHAT_MODEL, *, thinking=False, cache_path=None):
    """The lab's settings with the chat's model and cache swapped in.

    A copy, deliberately: config.py is imported by `main.py answer` and by the harness, and it rejects
    stale model values on purpose, so a chat session must not be able to change what they believe.
    """
    local = SimpleNamespace(**{key: getattr(cfg, key) for key in dir(cfg) if key.isupper()})
    local.ANSWER_MODEL = model
    local.ANSWER_PROVIDER = 'nvidia'
    local.ANSWER_PROMPT_VERSION = 'grounded-v2' if thinking else 'grounded-v1'
    local.NVIDIA_CHAT_STREAM = False
    # Named to match the repo's existing ignore pattern (raglab/answers_cache*.json), so an
    # interactive session cannot leave a cache file for someone to commit by accident.
    local.ANSWER_CACHE_PATH = Path(cache_path or (PROJECT_DIR / 'answers_cache_chat.json'))
    local.CHAT_THINKING = thinking
    return local


def data_dirs(extra=None):
    """The documents the lab was built on, unless told otherwise.

    `main.py` defaults to raglab/data/ (two sample product sheets). The corpus this repo was really
    built on is ../docs, so the chat reads both by default: asking about a specific product and asking
    about the BCT circular should not need different commands. --data-dir replaces the list outright.
    """
    if extra:
        candidates = [Path(entry).expanduser().resolve() for entry in extra]
    else:
        candidates = [path for path in (PROJECT_DIR.parent / 'docs', cfg.DATA_DIR)
                      if path.is_dir() and any(path.glob('*'))]
        if not candidates:
            raise ValueError('no corpus to read: put documents in docs/ or raglab/data/, or pass '
                             '--data-dir PATH')
    missing = [str(path) for path in candidates if not path.is_dir()]
    if missing:
        raise ValueError(f'data directory not found: {", ".join(missing)}')
    return candidates


def pending_chunks(local):
    """How big the first-time index would be, so the cost is stated before it is paid."""
    from chunker import chunk_all
    from loader import load_all
    return len(chunk_all(load_all(local.CHAT_DATA_DIRS), local))


def open_collection(local, embedder, *, reset=False):
    """A live collection, built from the corpus only when the index is empty and --ingest allowed it."""
    from store import get_collection, store_chunks
    collection = get_collection(local, reset=reset)
    if collection.count():
        return collection
    if not local.CHAT_ALLOW_INGEST:
        raise ValueError(f'the index is empty. Re-run with --ingest to embed '
                         f'{pending_chunks(local)} chunk(s) from '
                         f'{" ,".join(str(path) for path in local.CHAT_DATA_DIRS)} once '
                         '(NVIDIA embedding calls, cached afterwards), or use `python main.py ingest`.')
    from chunker import chunk_all
    from loader import load_all
    chunks = chunk_all(load_all(local.CHAT_DATA_DIRS), local)
    if not chunks:
        raise ValueError('no chunks produced: is there a pdf/docx in the data directory?')
    print(f'[chat] embedding {len(chunks)} chunk(s) for a first-time index '
          f'(batch {embedder.batch_size})...', flush=True)
    vectors = embedder.embed_texts([chunk.text for chunk in chunks])
    store_chunks(collection, list(zip(chunks, vectors)), local)
    print(f'[chat] index built: {collection.count()} chunk(s) | embedding calls={embedder.api_calls} '
          f'cache hits={embedder.cache_hits}')
    return collection


def build_generator(local):
    """The lab's grounded generator, pointed at the NVIDIA client and the chat model.

    AnswerGenerator refuses any model outside its approved list, so the approved list is written out
    here: this session's model, nothing else, and no substitution when a call fails.
    """
    from answer import AnswerGenerator
    client = NvidiaClient(base_url=NVIDIA_CHAT_BASE_URL,
                          timeout=getattr(local, 'NVIDIA_API_TIMEOUT', 120),
                          attempts=getattr(local, 'NVIDIA_API_ATTEMPTS', 3),
                          min_interval=getattr(local, 'NVIDIA_MIN_INTERVAL', 1.6),
                          max_retry_delay=getattr(local, 'NVIDIA_MAX_RETRY_DELAY', 30))
    if not client.api_key:
        raise ValueError('NVIDIA_API_KEY is not set. Put it in raglab/.env (see .env.example), then '
                         're-run `python chat.py --check`.')
    client = ReasoningSwitch(client, getattr(local, 'CHAT_THINKING', False))
    return AnswerGenerator(local, client=client, approved_models=(local.ANSWER_MODEL,))


class ReasoningSwitch:
    """Forwards Nemotron's thinking flag through a generator that only passes max_tokens.

    AnswerGenerator calls `client.chat(model, messages, max_tokens=...)`, which leaves no room for the
    model's own switch, so the chat wraps the client instead of widening the shared generator's
    signature for one entry point.
    """

    def __init__(self, client, thinking):
        self._client = client
        self.thinking = bool(thinking)

    @property
    def base_url(self):
        return self._client.base_url

    @property
    def api_key(self):
        return self._client.api_key

    def chat(self, model, messages, *, max_tokens=2048):
        return self._client.chat(model, messages, max_tokens=max_tokens, thinking=self.thinking)


def ask(local, embedder, collection, generator, question, *, top_k=None, neighbor_radius=None,
        use_cache=True):
    """One turn: retrieve, optionally widen the excerpts, then answer with citations or abstain."""
    from answer import local_private_refusal
    from evaluate import prepare_query_text
    from retrieval import expand_neighbors, retrieve
    from translate import detect_language
    question = ' '.join(str(question).split())
    if not question:
        raise ValueError('empty question')
    language = detect_language(question)
    guarded = local_private_refusal(local, question, language)
    if guarded is not None:                       # refused locally: no retrieval and no model call
        return {**guarded, 'question': question, 'retrieved': 0}
    hits, variants = retrieve(cfg, embedder, collection, prepare_query_text(question),
                              language=language, translator=None,
                              top_k=top_k or local.ANSWER_TOP_K,
                              variant_strategy=local.QUERY_VARIANT_STRATEGY)
    hits = expand_neighbors(collection, hits, radius=local.ANSWER_NEIGHBOR_RADIUS
                            if neighbor_radius is None else neighbor_radius)
    result = generator.answer(question, hits, language, use_cache=use_cache)
    return {**result, 'question': question, 'retrieved': len(hits), 'query_variants': variants,
            'language': language}


def format_turn(result, *, show_context=False):
    """Printable form of one turn. A citation marker stays attached to the claim it supports."""
    lines = [result['answer']]
    cited = {e['source_id'] for claim in result.get('claims', []) for e in claim['evidence']}
    for source in result.get('sources', []):
        quotes = [e['quote'] for claim in result.get('claims', []) for e in claim['evidence']
                  if e['source_id'] == source['source_id']]
        if quotes or show_context:
            body = source['text'] if show_context else '\n'.join(f'    "{quote}"' for quote in quotes)
            marker = '' if source['source_id'] in cited else '  (retrieved, not cited)'
            lines.append(f"\n[{source['source_id']}] {source['document']} — {source['chunk_id']}"
                         f"{marker}\n{body}")
    rejected = result.get('raw_preview') or (result.get('error') if result.get('status') == 'error' else '')
    if rejected:
        lines.append(f"\n[model said, not accepted] {rejected}")
    return '\n'.join(lines)


def log_turn(path, result):
    if not path:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {key: value for key, value in result.items() if key not in {'hits', 'sources'}}
    record['sources'] = [{key: source[key] for key in ('source_id', 'document', 'chunk_id')}
                         for source in result.get('sources', [])]
    record['logged_at'] = datetime.now(timezone.utc).isoformat(timespec='seconds')
    with path.open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + '\n')


def read_questions(lines, settings):
    """Yield questions from an iterator of input lines, applying the ':' commands to `settings`.

    Split out from the REPL so a piped session (`echo "..." | ./chat.sh`) and the tests run the same
    command handling as an interactive one. Unknown commands are ignored rather than asked as
    questions, because a typo'd ':kk 3' becoming a retrieval query is worse than a warning.
    """
    for raw in lines:
        line = (raw or '').strip()
        if not line:
            continue
        if line in {':quit', ':q', ':exit'}:
            return
        if line in {':help', ':h'}:
            print(HELP)
            continue
        if line == ':show':
            settings['show_context'] = not settings['show_context']
            print(f'[chat] context display {"on" if settings["show_context"] else "off"}')
            continue
        if line.startswith(':k'):
            try:
                settings['top_k'] = max(1, min(20, int(line.split()[1])))
            except (IndexError, ValueError):
                print('[chat] usage: :k 5')
                continue
            print(f'[chat] retrieving {settings["top_k"]} excerpt(s) per question')
            continue
        if line.startswith(':log'):
            try:
                settings['log'] = Path(line.split(None, 1)[1].strip()).expanduser()
            except IndexError:
                print('[chat] usage: :log chat_log.jsonl')
                continue
            print(f'[chat] logging turns to {settings["log"]}')
            continue
        if line == ':context':
            print('[chat] ' + json.dumps({key: value for key, value in settings.items()
                                          if key != 'rebuild'}, ensure_ascii=False, default=str))
            print(f'[chat] model={settings["model"]} thinking={settings["thinking"]} '
                  f'chunks={settings["chunking"]}/{settings["overlap"]} tokens')
            continue
        if line.startswith(':'):
            print('[chat] unknown command; :help lists them')
            continue
        yield line


def check(model=CHAT_MODEL):
    """Say what this command would use, without a completion call. Loads .env the way phases do."""
    from hard_harness.preflight import load_project_env, masked
    env = load_project_env()
    local = chat_config(model)
    key = os.environ.get('NVIDIA_API_KEY', '').strip()
    print('# Chat readiness (no completion request is made here)')
    print(f"- NVIDIA_API_KEY: {'set (' + masked(key) + ')' if key else 'NOT SET'}; "
          f"raglab/.env {'read' if env['loaded'] is True else 'not read'} "
          f"({len(env['assignments'])} assignment(s); python-dotenv "
          f"{'installed' if env['dotenv_installed'] else 'MISSING'})")
    print(f"- chat model: {model} at {NVIDIA_CHAT_BASE_URL}/chat/completions — thinking off, so "
          f"temperature 0 and {local.ANSWER_MAX_TOKENS if hasattr(local, 'ANSWER_MAX_TOKENS') else 4096}"
          ' max_tokens; answer cache ' + Path(local.ANSWER_CACHE_PATH).name)
    print(f"- chunking: {local.CHUNK_SIZE_TOKENS} tokens with {local.CHUNK_OVERLAP_TOKENS} overlap — the "
          'application default, NOT the harness pin of 640/40, so retrieval quality is not comparable')
    status = 0 if key else 1
    try:
        dirs = data_dirs(None)
        print(f'- corpus: {", ".join(path.name for path in dirs)}')
        from store import get_collection
        count = get_collection(local, reset=False).count()
        print(f"- index: {count} chunk(s) in {Path(local.CHROMA_DIR).name}/{local.CHROMA_COLLECTION_NAME}")
        if not count:
            # An empty index is reported as not-ready on purpose: `chat.sh` opens the chat only when the
            # check passes, so the failure lands here with the fix named, not on the first question.
            print('- not ready: nothing to retrieve from. Build it once (NVIDIA embedding calls, cached '
                  'afterwards): ./raglab/chat.sh --ingest --check')
            status = 1
    except Exception as exc:                                             # noqa: BLE001
        print(f'- corpus/index: {safe_error(exc)}')
        status = 1
    return status


def main(argv=None):
    parser = argparse.ArgumentParser(prog='chat.py', description='Document-grounded chat over the RAGLab corpus.')
    parser.add_argument('question', nargs='*', help='ask once and exit; omit to enter the chat')
    parser.add_argument('--model', default=CHAT_MODEL,
                        help=f'chat model on the NVIDIA endpoint (default {CHAT_MODEL})')
    parser.add_argument('-k', '--top-k', type=int, default=cfg.ANSWER_TOP_K, dest='top_k',
                        help=f'excerpts per question (default {cfg.ANSWER_TOP_K})')
    parser.add_argument('--neighbor-radius', type=int, choices=[0, 1, 2], default=0, dest='radius',
                        help='widen each hit with adjacent chunks (more context, more tokens)')
    parser.add_argument('--thinking', action='store_true',
                        help="turn on Nemotron's reasoning mode; the reply ceiling rises to fit it")
    parser.add_argument('--ingest', action='store_true', help='embed the corpus first if the index is empty')
    parser.add_argument('--reset', action='store_true', help='drop and rebuild the collection')
    parser.add_argument('--data-dir', action='append', default=None, dest='data_dirs',
                        help='corpus directory (repeatable; default: the repo docs/)')
    parser.add_argument('--no-cache', action='store_true', dest='no_cache',
                        help='re-ask the model even for a question already answered')
    parser.add_argument('--show-context', action='store_true', dest='show_context',
                        help='print the retrieved excerpts, not only the quotes the answer cited')
    parser.add_argument('--log', default=None, metavar='PATH', help='append every turn as a JSON line')
    parser.add_argument('--check', action='store_true',
                        help='report configuration and index state, then exit (spends nothing)')
    parser.add_argument('--json', action='store_true', dest='as_json', help='print one JSON record')
    parser.add_argument('--max-tokens', type=int, default=None, dest='max_tokens',
                        help='reply ceiling (default 4096, or 12288 with --thinking)')
    parser.add_argument('--print-prompt', action='store_true', dest='print_prompt',
                        help='show the exact system and user messages, then answer')
    args = parser.parse_args(argv)
    if not args.model.startswith('nvidia/'):
        raise ValueError(f'{args.model!r} is not on the NVIDIA endpoint this command talks to; the '
                         'benchmark answer path uses xKiro Qwen and will not accept this model either')
    if args.check:
        return check(args.model)

    local = chat_config(args.model, thinking=args.thinking)
    local.CHAT_DATA_DIRS = data_dirs(args.data_dirs)
    local.CHAT_ALLOW_INGEST = args.ingest or args.reset
    local.ANSWER_NEIGHBOR_RADIUS = args.radius
    local.ANSWER_TOP_K = max(1, min(20, args.top_k))
    local.ANSWER_MAX_TOKENS = args.max_tokens or (12288 if args.thinking else 4096)
    if not os.environ.get('NVIDIA_API_KEY', '').strip():
        raise ValueError('NVIDIA_API_KEY is not set. Put it in raglab/.env (see .env.example), then '
                         '`python chat.py --check`.')
    if args.thinking:
        print(f'[chat] thinking on: reasoning tokens are generated and discarded here, so the ceiling is '
              f'{local.ANSWER_MAX_TOKENS} max_tokens')

    from embedder import build_embedder
    embedder = build_embedder(cfg)
    print(f'[chat] embeddings {embedder.provider_name}/{embedder.model} | chat {local.ANSWER_MODEL} | '
          f'{local.CHUNK_SIZE_TOKENS}-token chunks | index {Path(local.CHROMA_DIR).name}')
    collection = open_collection(local, embedder, reset=args.reset)
    generator = build_generator(local)
    if args.print_prompt:
        from answer import answer_messages
        demo = [{'source_id': 'S1', 'chunk_id': 'demo', 'document': 'demo.pdf', 'heading': '',
                 'text': '<a retrieved excerpt goes here>'}]
        question = ' '.join(args.question) or 'What is the minimum capital for a participatory bank?'
        print(json.dumps(answer_messages(question, 'en', demo, local.ANSWER_PROMPT_VERSION),
                         ensure_ascii=False, indent=2))

    settings = {'top_k': local.ANSWER_TOP_K, 'show_context': args.show_context,
                'log': Path(args.log).expanduser() if args.log else None, 'model': args.model,
                'thinking': args.thinking, 'chunking': local.CHUNK_SIZE_TOKENS,
                'overlap': local.CHUNK_OVERLAP_TOKENS}

    def one_turn(question):
        result = ask(local, embedder, collection, generator, question, top_k=settings['top_k'],
                     neighbor_radius=args.radius, use_cache=not args.no_cache)
        log_turn(settings['log'], result)
        return result

    if args.question:
        result = one_turn(' '.join(args.question))
        print(json.dumps(result, ensure_ascii=False, indent=2) if args.as_json
              else format_turn(result, show_context=settings['show_context']))
        print(f"\n[chat] status={result['status']} reason={result.get('reason')} model={result['model']} "
              f"cached={result.get('cached', False)} {result.get('seconds', 0)}s "
              f"retrieved={result.get('retrieved', 0)}")
        return 0 if result.get('validation_ok', True) else 2

    print('Ask about the documents. Ctrl-D or :quit to leave. :help for commands.')
    stream = sys.stdin if not sys.stdin.isatty() else _prompted_lines()
    for question in read_questions(stream, settings):
        try:
            result = one_turn(question)
        except (ValueError, RuntimeError) as exc:
            print(f'[chat] {safe_error(exc)}')
            continue
        print(format_turn(result, show_context=settings['show_context']))
        print(f"\n[chat] {result['status']}/{result.get('reason')} · {result['model']} · "
              f"{result.get('seconds', 0)}s · k={settings['top_k']}")
    return 0


def _prompted_lines():
    """input() as a generator, so interactive and piped sessions share one code path."""
    try:
        while True:
            yield input('\n> ')
    except EOFError:
        return


if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print('\n[chat] interrupted')
        sys.exit(130)
    except (ValueError, RuntimeError) as exc:
        print(f'[chat] ERROR: {safe_error(exc)}', file=sys.stderr)
        sys.exit(2)
