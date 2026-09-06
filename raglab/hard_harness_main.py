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
    for action in ('gate','record-pause'):
        command = sub.add_parser(action)
        command.add_argument('--phase', choices=['author','predict'], required=True)
        command.add_argument('--shard', type=int, required=True)
    author = sub.add_parser('author'); author.add_argument('--shard', type=int, required=True)
    sub.add_parser('compile-dataset')
    sub.add_parser('retrieve')
    predict = sub.add_parser('predict'); predict.add_argument('--shard', type=int, required=True)
    sub.add_parser('grade')
    sub.add_parser('report')
    sub.add_parser('snapshot')
    judge = sub.add_parser('judge')
    judge.add_argument('--arms', default='lexical,vector',
                       help="comma list of lexical,vector; no model judges anything in either arm")
    judge.add_argument('--top-k', type=int, default=5)
    judge.add_argument('--fpr', type=float, default=0.05)
    judge.add_argument('--out', default=None)
    pub = sub.add_parser('publish'); pub.add_argument('--phase', required=True)
    get = sub.add_parser('collect'); get.add_argument('--repo', default='seif-bkh/RAGLab')
    get.add_argument('--sha', required=True); get.add_argument('--destination', default=str(OUTPUT))
    args = parser.parse_args(argv)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    if args.command in {'gate','record-pause'}:
        from hard_harness.control import gate, record_pause
        (gate if args.command == 'gate' else record_pause)(args.phase,args.shard)
        return 0
    if args.command == 'publish':
        from hard_harness.publishing import publish
        publish(args.phase); return 0
    if args.command == 'snapshot':
        from hard_harness.authoring import accepted_snapshot
        from hard_harness.common import read_json as _read_json, PLAN_PATH as _plan
        shards = range(_read_json(_plan)['author_shards'])
        print(json.dumps(accepted_snapshot(shards), ensure_ascii=False, indent=2))
        return 0
    if args.command == 'report':
        from hard_harness.reporting import build_report
        report = build_report()
        print(json.dumps({k: report[k] for k in ('status', 'graded_questions', 'ungraded_questions',
                                                'by_language') if k in report},
                         ensure_ascii=False, indent=2))
        return 0
    if args.command == 'judge':
        from hard_harness.retrieval_judge import evaluate
        manifest = evaluate(arms=tuple(name.strip() for name in args.arms.split(',') if name.strip()),
                            top_k=args.top_k, fpr=args.fpr, out=args.out)
        report = manifest['report']
        print(json.dumps({
            'status': manifest['status'], 'arms': manifest['arms'],
            'arm_status': {arm: state.get('status') for arm, state in manifest['arm_status'].items()},
            'questions': manifest['questions'], 'families': manifest['families'],
            'headline': {arm: {metric: report[arm]['overall'].get(metric) for metric in
                               ('answer_ready_rate', 'evidence_available_rate', 'recall@1', 'recall@3',
                                'semantic_only_recall', 'partial_only_rate')}
                         for arm in manifest['arms']},
            'abstention_auc': {arm: report[arm].get('abstention', {}).get('auc') for arm in manifest['arms']},
            'agreement': report.get('agreement', {})}, ensure_ascii=False, indent=2))
        return 0
    if args.command == 'collect':
        from hard_harness.publishing import collect
        collect(args.repo, args.sha, args.destination); return 0
    try:
        if args.command == 'sources':
            from hard_harness.sources import prepare_sources
            report = prepare_sources()
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return 0 if report['status'] == 'ready_for_reference_authoring' else 2
        if args.command == 'author':
            from hard_harness.authoring import author_shard
            report = author_shard(args.shard)
            return 0 if report['status'] == 'drafts_complete' else 2
        if args.command == 'compile-dataset':
            from hard_harness.dataset import compile_dataset
            report = compile_dataset()
            return 0 if report['status'] == 'frozen' else 2
        if args.command == 'retrieve':
            from hard_harness.predict import prepare_retrieval
            report = prepare_retrieval()
            return 0 if report['status'] == 'retrieval_complete' else 2
        if args.command == 'predict':
            from hard_harness.predict import predict_shard
            report = predict_shard(args.shard)
            return 0 if report['status'] == 'predictions_complete' else 2
        if args.command == 'grade':
            from hard_harness.grading import grade_all
            report = grade_all()
            return 0 if report['status'] == 'complete' else 2
    except Exception as exc:
        phase = {'author': f'author_{getattr(args,"shard",0):02d}', 'compile-dataset': 'dataset',
                 'predict': f'predictions_{getattr(args,"shard",0):02d}', 'retrieve': 'retrieval', 'grade': 'grading'}.get(args.command,args.command)
        report = {'status': 'paused' if getattr(exc,'status_code',getattr(exc,'code',0)) in {401,402,403,429} else 'blocked',
                  'phase': args.command, 'timestamp': now(), 'error': safe_error(exc)}
        write_json(OUTPUT / phase / 'manifest.json', report)
        print(json.dumps(report, ensure_ascii=False)); return 2
    return 2


if __name__ == '__main__':
    raise SystemExit(main())
