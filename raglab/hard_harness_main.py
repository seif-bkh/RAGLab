"""Large multilingual harness phase runner. Results are resumable, not fabricated."""
import argparse
import json
from pathlib import Path

from hard_harness.common import OUTPUT, WORK, PLAN_PATH, read_json, now
from artifacts import write_json
from nvidia_api import safe_error


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('sources')
    pub = sub.add_parser('publish'); pub.add_argument('--phase', required=True)
    get = sub.add_parser('collect'); get.add_argument('--repo', default='seif-bkh/RAGLab')
    get.add_argument('--sha', required=True); get.add_argument('--destination', default=str(OUTPUT))
    args = parser.parse_args(argv)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    if args.command == 'publish':
        from hard_harness.publishing import publish
        publish(args.phase); return 0
    if args.command == 'collect':
        from hard_harness.publishing import collect
        collect(args.repo, args.sha, args.destination); return 0
    try:
        if args.command == 'sources':
            from hard_harness.sources import prepare_sources
            report = prepare_sources()
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return 0 if report['status'] == 'ready_for_reference_authoring' else 2
    except Exception as exc:
        report = {'status': 'blocked', 'phase': args.command, 'timestamp': now(), 'error': safe_error(exc)}
        write_json(OUTPUT / args.command / 'manifest.json', report)
        print(json.dumps(report, ensure_ascii=False)); return 2
    return 2


if __name__ == '__main__':
    raise SystemExit(main())
