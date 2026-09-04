"""answer.py — PLACEHOLDER for LLM answer generation.

Deliberately NOT implemented: this lab stops at retrieval. The stub exists so
the pipeline has an explicit, clearly marked place to plug in a generator
later. It is not called by any command in main.py yet.
"""


def generate_answer(question: str, retrieved_chunks: list, model_name: str = "not-configured") -> str:
    """STUB — would build a grounded answer from retrieved chunks.

    Args:
        question: user question (already cleaned/normalized by the pipeline).
        retrieved_chunks: list of hit dicts from store.query_vector /
            rrf_merge, each with "text", "metadata", "similarity".
        model_name: name of the (future) generation model.

    Returns:
        A string. Currently always the placeholder below.

    Intended implementation (NOT included on purpose):
        1. keep the top-k chunks above a chosen similarity threshold,
        2. template a prompt in the question language (no translation),
        3. call a hosted chat-completion API with the official SDK,
        4. print the prompt and the completion for inspection,
        5. cite the chunk ids/sources it used.
    """
    n = len(retrieved_chunks)
    print(f"[answer] STUB: generate_answer called with {n} chunk(s), "
          f"model={model_name!r}; no LLM call is made in this lab.")
    return (
        "[STUB] no answer generation configured. "
        "This laboratory stops at retrieval; add a generator here if you want one."
    )
