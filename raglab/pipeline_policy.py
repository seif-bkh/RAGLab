"""The supported runtime model pair. Historical experiments do not override it."""

EMBEDDING_PROVIDER = 'nvidia'
EMBEDDING_MODEL = 'nvidia/nemotron-3-embed-1b'
ANSWER_PROVIDER = 'xkiro'
ANSWER_MODEL = 'qwen/qwen3.8-max:free'


def validate_retrieval_settings(cfg):
    if str(cfg.EMBEDDING_PROVIDER).strip().lower() != EMBEDDING_PROVIDER:
        raise ValueError('The supported pipeline uses NVIDIA Nemotron embeddings only; update EMBEDDING_PROVIDER in .env')
    if cfg.NVIDIA_EMBEDDING_MODEL != EMBEDDING_MODEL:
        raise ValueError(f'The supported embedding model is {EMBEDDING_MODEL}; no substitution is allowed')
    if cfg.NVIDIA_EMBEDDING_DIM not in (0, 2048):
        raise ValueError('Nemotron uses native 2048 dimensions (set NVIDIA_EMBEDDING_DIM=0 or 2048)')
    if getattr(cfg, 'QUERY_TRANSLATION_ENABLED', False):
        raise ValueError('Query translation is retired for this pipeline; set QUERY_TRANSLATION_ENABLED=0 in .env')


def validate_answer_selection(provider, model):
    if provider != ANSWER_PROVIDER or model != ANSWER_MODEL:
        raise ValueError(f'The supported answer path is {ANSWER_PROVIDER}/{ANSWER_MODEL}; no provider/model fallback is allowed')


def validate_regression_plan(plan):
    if plan.get('models') != {ANSWER_PROVIDER: [ANSWER_MODEL]}:
        raise ValueError('Regression plan must contain only the selected xKiro Qwen model')
    if plan.get('skip_providers'):
        raise ValueError('The selected pipeline cannot be skipped or replaced')
    if plan.get('retrieval_profile') != 'original' or plan.get('answer_profile') != 'grounded-v1':
        raise ValueError('The selected pipeline uses original-query retrieval and grounded-v1 answers')
