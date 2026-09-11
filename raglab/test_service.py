"""test_service.py — offline tests for the RAGLab HTTP service (no network).

Same strategy as the app.py section of tests_offline.py: a NON-pinned profile
(local HF embeddings via a stubbed sentence-transformers + an injected fake
chat client) driven through the real service code path — TestClient ->
endpoints -> profiles runtime -> chunker/store/retrieval/answer. Providers are
never contacted; every artifact lands in a temp directory.

Run:  python -m unittest -v test_service
"""

import json
import os
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import service  # noqa: E402
from answer import AnswerGenerator  # noqa: E402
from profiles import build_lab_config  # noqa: E402

try:
    from fastapi.testclient import TestClient
except ImportError:  # pragma: no cover
    raise unittest.SkipTest("fastapi/httpx not installed; "
                            "pip install -r requirements-service.txt")

CORPUS = ("# Account\n\nThe Atlas current account has no management fee.\n\n"
          "## Fees\n\nThe Atlas card costs 10 dinars per year.\n")
QUESTION = "What does the Atlas card cost?"
PROFILE_MODEL = "nvidia/nemotron-3.5-lightning-30b-a3b"  # fake client, no call is made


class _FakeSentenceTransformer:
    """Deterministic 8-dim one-hot-ish embeddings, enough for retrieval."""

    def __init__(self, model_name, device=None):
        self.prompts = {"query": ""}

    @staticmethod
    def get_embedding_dimension():
        return 8

    def encode(self, texts, batch_size=None, normalize_embeddings=None,
               show_progress_bar=None, convert_to_numpy=None, **kwargs):
        rows = []
        for i in range(len(texts)):
            row = [0.0] * 8
            row[hash(texts[i]) % 8] = 1.0
            rows.append(row)
        return rows


class _FakeChatClient:
    """One claim citing a verbatim prefix of the first supplied source."""
    base_url = "https://fake.test/v1"
    api_key = "fake"

    def chat(self, model, messages, *, max_tokens=4096):
        payload = json.loads(messages[1]["content"])
        source = payload["sources"][0]
        quote = " ".join(source["text"].split())[:40]
        return {"text": json.dumps({"answerable": True, "claims": [
            {"text": "The Atlas card costs 10 dinars per year.",
             "evidence": [{"source_id": source["source_id"], "quote": quote}]}]}),
            "served_model": model, "usage": {}, "seconds": 0.0}


def _service_profile(tmp: Path) -> dict:
    from app import default_state  # the same profile shape the console uses
    state = default_state()
    state["embedding"] = {"provider": "huggingface", "model": "Qwen/Qwen3-Embedding-0.6B"}
    state["answer"] = {"provider": "nvidia", "model": PROFILE_MODEL}
    state["chunking"] = {"mode": "size", "size": 60, "overlap": 10}
    state["data_dirs"] = [str(tmp)]
    return state


class ServiceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        (cls.tmp / "note.md").write_text(CORPUS, encoding="utf-8")
        cls.profile = _service_profile(cls.tmp)
        overrides = {
            "CHROMA_DIR": cls.tmp / "chroma",
            "EMBEDDING_CACHE_PATH": cls.tmp / "emb.json",
            "NVIDIA_EMBEDDING_CACHE_PATH": cls.tmp / "emb.json",
            "ANSWER_CACHE_PATH": cls.tmp / "answers.json",
            "RESULTS_DIR": cls.tmp,
        }
        # The injected generator must share the service's lab config (same
        # cache paths), so build it from the same profile + overrides.
        local = build_lab_config(cls.profile)
        for key, value in overrides.items():
            setattr(local, key, value)
        generator = AnswerGenerator(local, client=_FakeChatClient(),
                                    approved_models=(PROFILE_MODEL,))
        sys.modules["sentence_transformers"] = types.ModuleType("sentence_transformers")
        sys.modules["sentence_transformers"].SentenceTransformer = _FakeSentenceTransformer
        cls.client = TestClient(service.create_app(
            cls.profile, generator=generator, allow_profile_switch=False,
            config_overrides=overrides))
        # Build the index once up front: unittest runs methods alphabetically,
        # so the answer/search tests must not depend on the ingest test running
        # first (that test rebuilds it through the background-job path).
        cls._wait_ingest(cls.client.post("/ingest").json())

    @staticmethod
    def _wait_ingest(_accepted):
        deadline = time.monotonic() + 60
        client = ServiceTest.client
        while time.monotonic() < deadline:
            status = client.get("/ingest/status").json()
            if status["state"] != "running":
                return status
            time.sleep(0.2)
        raise TimeoutError(status)

    @classmethod
    def tearDownClass(cls):
        sys.modules.pop("sentence_transformers", None)

    # -- informational ------------------------------------------------------

    def test_root_and_health(self):
        body = self.client.get("/health").json()
        self.assertEqual(self.client.get("/").status_code, 200)
        self.assertEqual(body["status"], "ok")
        self.assertEqual(body["profile"]["embedding"]["provider"], "huggingface")
        self.assertEqual(body["profile"]["answer"]["model"], PROFILE_MODEL)
        self.assertIn("collection", body["index"])
        # No key value may ever appear — only presence.
        self.assertNotIn("sk-", json.dumps(body))
        self.assertNotIn("nvapi-", json.dumps(body))

    def test_models_lists_registered_providers(self):
        body = self.client.get("/models").json()
        providers = {row["provider"] for row in body["answer"]}
        self.assertIn("kira", providers)
        self.assertIn("xkiro", providers)
        self.assertIn("glm-5.3-free",
                      {m for row in body["answer"] if row["provider"] == "kira"
                       for m in row["models"]})

    def test_profile_reports_collection_and_switching_state(self):
        body = self.client.get("/profile").json()
        self.assertEqual(body["profile"]["answer"]["provider"], "nvidia")
        self.assertTrue(body["collection"].startswith("raglab_app_huggingface_"))
        self.assertIn("disabled", body["switching"])

    def test_profile_switch_disabled_by_default(self):
        response = self.client.post("/profile",
                                    json={"answer": {"provider": "kira"}})
        self.assertEqual(response.status_code, 403)
        self.assertEqual(response.json()["detail"]["reason"], "profile_switching_disabled")

    # -- ingestion ------------------------------------------------------------

    def test_ingest_job_and_search(self):
        # setUpClass already built the index; this exercises the background
        # job path again with reset=true (cheap: 1 chunk, cached embeddings).
        accepted = self.client.post("/ingest?reset=true")
        self.assertEqual(accepted.status_code, 200)
        self.assertEqual(accepted.json()["state"], "accepted")
        status = self._wait_ingest(accepted.json())
        self.assertEqual(status["state"], "done", status)
        self.assertGreater(status["stored"], 0)
        self.assertEqual(self.client.get("/health").json()["index"]["count"],
                         status["stored"])

        found = self.client.post("/search", json={"question": QUESTION, "k": 3})
        self.assertEqual(found.status_code, 200, found.text)
        body = found.json()
        self.assertEqual(body["embedder"]["provider"], "huggingface")
        self.assertGreaterEqual(len(body["hits"]), 1)
        self.assertTrue(body["hits"][0]["text"])
        self.assertIn("rank", body["hits"][0])

    def test_answer_grounded_with_citations(self):
        response = self.client.post("/answer", json={"question": QUESTION})
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["status"], "answered", body)
        self.assertTrue(body["validation_ok"])
        self.assertEqual(body["model"], PROFILE_MODEL)
        self.assertTrue(body["claims"])
        self.assertTrue(body["claims"][0]["evidence"][0]["quote"])
        self.assertEqual(body["sources"][0]["source_id"], "S1")
        self.assertNotIn("text", body["sources"][0])  # excerpts opt-in only

    def test_answer_excerpts_are_opt_in(self):
        body = self.client.post("/answer",
                                json={"question": QUESTION,
                                      "include_excerpts": True}).json()
        self.assertIn("text", body["sources"][0])

    def test_answer_greeting_short_circuits_without_model(self):
        body = self.client.post("/answer", json={"question": "bonjour"}).json()
        self.assertEqual(body["status"], "greeting")
        self.assertFalse(body["inference_performed"])
        self.assertIn("Murabaha", body["answer"])  # suggests a real question

    # -- error contract ---------------------------------------------------------

    def test_empty_index_is_409_with_hint(self):
        fresh = Path(tempfile.mkdtemp())
        (fresh / "note.md").write_text(CORPUS, encoding="utf-8")
        overrides = {"CHROMA_DIR": fresh / "chroma",
                     "EMBEDDING_CACHE_PATH": fresh / "emb.json",
                     "NVIDIA_EMBEDDING_CACHE_PATH": fresh / "emb.json",
                     "ANSWER_CACHE_PATH": fresh / "answers.json",
                     "RESULTS_DIR": fresh}
        profile = _service_profile(fresh)
        with self.subTest(part="empty index"):
            client = TestClient(service.create_app(
                profile, generator=object(), config_overrides=overrides))
            response = client.post("/search", json={"question": QUESTION})
            self.assertEqual(response.status_code, 409)
            self.assertEqual(response.json()["detail"]["reason"], "empty_index")

    def test_validation_rejects_bad_requests(self):
        for payload in ({"question": ""}, {"question": "x" * 3000},
                        {"question": "hi", "k": 99}):
            response = self.client.post("/answer", json=payload)
            self.assertEqual(response.status_code, 422, payload)

    def test_env_profile_validation_fails_loudly(self):
        old = os.environ.get("RAGLAB_ANSWER_PROVIDER")
        os.environ["RAGLAB_ANSWER_PROVIDER"] = "not-a-provider"
        try:
            with self.assertRaises(SystemExit) as caught:
                service.profile_from_env()
            self.assertIn("not-a-provider", str(caught.exception))
        finally:
            if old is None:
                os.environ.pop("RAGLAB_ANSWER_PROVIDER", None)
            else:
                os.environ["RAGLAB_ANSWER_PROVIDER"] = old

    def test_profile_switch_flow_when_enabled(self):
        enabled = TestClient(service.create_app(
            self.profile, generator=object(), allow_profile_switch=True,
            config_overrides={"CHROMA_DIR": self.tmp / "chroma_switch"}))
        response = enabled.post("/profile",
                                json={"answer": {"provider": "kira",
                                                 "model": "glm-5.3-free"}})
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["profile"]["answer"]["provider"], "kira")
        self.assertTrue(body["collection"].startswith("raglab_app_huggingface_"))
        # An unknown provider is a 400 with the registered list, not a crash.
        bad = enabled.post("/profile", json={"answer": {"provider": "nope"}})
        self.assertEqual(bad.status_code, 400)
        self.assertIn("xkiro", bad.json()["detail"]["registered"])


if __name__ == "__main__":
    unittest.main()
