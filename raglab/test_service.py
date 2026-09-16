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
import profiles  # noqa: E402
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


class LocalFrontOverHttp(unittest.TestCase):
    """local_front.py's smoke suite against a REAL HTTP server.

    Boots uvicorn on an ephemeral port (thread) with the default profile, no
    keys and an empty index — exactly the state a fresh deployment is in —
    and requires the front's state-aware suite to pass end-to-end: the
    refusal paths (503 missing key, 403 switching, 422 validation) ARE the
    correct behavior being tested. local_front imports nothing from the lab,
    so this exercises the whole HTTP boundary.
    """

    @staticmethod
    def _boot(app):
        """Run a REAL uvicorn on an ephemeral port (in-thread); (api-url, stop)."""
        import threading
        import uvicorn
        server = uvicorn.Server(uvicorn.Config(
            app, host="127.0.0.1", port=0, log_level="warning"))
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline and not server.started:
            time.sleep(0.05)
        assert server.started, "uvicorn did not start"
        port = server.servers[0].sockets[0].getsockname()[1]

        def stop():
            server.should_exit = True
            thread.join(timeout=10)
        return f"http://127.0.0.1:{port}", stop

    def test_smoke_suite_over_real_http(self):
        import local_front
        try:
            import uvicorn  # noqa: F401
        except ImportError:  # pragma: no cover
            self.skipTest("uvicorn not installed")
        key_names = ("NVIDIA_API_KEY", "XKIRO_API_KEY", "GOOGLE_API_KEY",
                     "GEMINI_API_KEY", "KIRA_API_KEY", "JINA_API_KEY",
                     "OPENAI_API_KEY", "COHERE_API_KEY", "VOYAGE_API_KEY")
        saved = {name: os.environ.pop(name, None) for name in key_names}
        app = service.create_app(profiles.default_state(), generator=object(),
                                 allow_profile_switch=False)
        url, stop = self._boot(app)
        try:
            api = local_front.Api(url)
            passed, failed = local_front.run_suite(api, spend=False)
            self.assertEqual(failed, 0, f"{failed} smoke check(s) failed")
            self.assertGreaterEqual(passed, 18)
        finally:
            stop()
            for name, value in saved.items():
                if value is not None:
                    os.environ[name] = value

    def test_front_sends_the_service_token(self):
        import local_front
        try:
            import uvicorn  # noqa: F401
        except ImportError:  # pragma: no cover
            self.skipTest("uvicorn not installed")
        app = service.create_app(profiles.default_state(), generator=object(),
                                 allow_profile_switch=False,
                                 service_token="front-secret-1")
        url, stop = self._boot(app)
        try:
            locked = local_front.Api(url, token="")     # no header -> 401
            status, body = locked.get("/health")
            self.assertEqual(status, 401)
            self.assertEqual(body["detail"]["reason"], "unauthorized")
            api = local_front.Api(url, token="front-secret-1")
            status, body = api.get("/health")
            self.assertEqual(status, 200)
            self.assertEqual(body["status"], "ok")
        finally:
            stop()


class ConsoleEndpointsTest(unittest.TestCase):
    """The console-parity endpoints: keys, inspect, chunk search, sanity,
    evaluate, extended profile switching — stubbed providers, real app."""

    CORPUS = ("# Products\n\nOverview of the Atlas retail range.\n\n## Prices\n\n"
              + "".join(
                  f"The Atlas card costs 10 dinars per year, and this price "
                  f"sheet {i} confirms that annual maintenance is included "
                  f"for every Atlas customer without exception, number {i}.\n\n"
                  for i in range(30)))

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        (cls.tmp / "note.md").write_text(cls.CORPUS, encoding="utf-8")
        cls.profile = _service_profile(cls.tmp)
        cls.overrides = {
            "CHROMA_DIR": cls.tmp / "chroma",
            "EMBEDDING_CACHE_PATH": cls.tmp / "emb.json",
            "NVIDIA_EMBEDDING_CACHE_PATH": cls.tmp / "emb.json",
            "ANSWER_CACHE_PATH": cls.tmp / "answers.json",
            "RESULTS_DIR": cls.tmp,
        }
        local = build_lab_config(cls.profile)
        for key, value in cls.overrides.items():
            setattr(local, key, value)
        generator = AnswerGenerator(local, client=_FakeChatClient(),
                                    approved_models=(PROFILE_MODEL,))
        sys.modules["sentence_transformers"] = types.ModuleType("sentence_transformers")
        sys.modules["sentence_transformers"].SentenceTransformer = _FakeSentenceTransformer
        cls.client = TestClient(service.create_app(
            cls.profile, generator=generator, allow_profile_switch=True,
            config_overrides=cls.overrides))
        accepted = cls.client.post("/ingest")
        assert accepted.status_code == 200, accepted.text
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            status = cls.client.get("/ingest/status").json()
            if status["state"] != "running":
                break
            time.sleep(0.1)
        assert status["state"] == "done", status

    @classmethod
    def tearDownClass(cls):
        sys.modules.pop("sentence_transformers", None)

    def test_keys_endpoints(self):
        saved = os.environ.get("KIRA_API_KEY")
        os.environ.pop("KIRA_API_KEY", None)
        try:
            listed = self.client.get("/keys").json()["keys"]
            self.assertIn("KIRA_API_KEY", {row["env"] for row in listed})
            self.assertEqual(self.client.post("/keys", json={
                "key_env": "NOT_A_KEY", "value": "whatever12345"}).status_code, 400)
            self.assertEqual(self.client.post("/keys", json={
                "key_env": "KIRA_API_KEY", "value": "YOUR_KIRA_API_KEY"}).status_code, 400)
            response = self.client.post("/keys", json={
                "key_env": "KIRA_API_KEY", "value": "kira-endpoint-test-123"})
            self.assertEqual(response.status_code, 200, response.text)
            body = response.json()
            self.assertEqual(body["masked"], "kira-end…")
            self.assertFalse(body["persisted"])
            blob = json.dumps(self.client.get("/keys").json())
            self.assertNotIn("kira-endpoint-test-123", blob)   # value never echoed
            removed = self.client.delete("/keys/KIRA_API_KEY")
            self.assertEqual(removed.status_code, 200)
            self.assertEqual(removed.json()["status"], "missing")
        finally:
            if saved is not None:
                os.environ["KIRA_API_KEY"] = saved

    def test_inspect_reports_chunking_without_model_calls(self):
        response = self.client.get("/inspect", params={"limit": 2})
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertTrue(body["documents"])
        self.assertGreater(body["chunks"]["count"], 2)   # the repeated corpus splits
        self.assertEqual(len(body["sample"]), 2)

    def test_chunks_search_verdicts(self):
        sample = self.client.get("/inspect", params={"limit": 8}).json()["sample"]
        content = [row for row in sample if row["section"] != "front-matter"]
        self.assertGreaterEqual(len(content), 2)
        # Text inside ONE chunk: found in full.
        first = " ".join(content[0]["text"].split())
        body = self.client.post("/chunks/search",
                                json={"text": first[:40]}).json()
        self.assertTrue(body["full"], body)
        self.assertIn("first_full_text", body)
        # A phrase stitched from two DIFFERENT chunks crosses a boundary: no
        # single chunk holds it, and both fragments are still locatable —
        # exactly the "quote can never pass the citation gate" diagnostic.
        second = " ".join(content[1]["text"].split())
        body = self.client.post("/chunks/search", json={
            "text": first[-32:] + " " + second[:32]}).json()
        self.assertFalse(body["full"])
        self.assertTrue(body["head"])
        self.assertTrue(body["tail"])

    def test_embeddings_sanity_report(self):
        response = self.client.post("/embeddings/sanity")
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["provider"], "huggingface")
        self.assertEqual(body["dimension"], 8)
        self.assertEqual(len(body["cosines"]), 3)   # 3 phrases -> 3 pairs

    def test_evaluate_runs_a_question_set(self):
        cases = {"cases": [{
            "id": "t1", "question": "What does the Atlas card cost?",
            "language": "en", "category": "verbatim",
            "expected_substring": "10 dinars"}]}
        path = self.tmp / "mini_questions.json"
        path.write_text(json.dumps(cases), encoding="utf-8")
        response = self.client.post("/evaluate", json={"questions": str(path)})
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["metrics"]["overall"]["n"], 1)
        self.assertIn("hit@1", body["metrics"]["overall"])
        self.assertEqual(len(body["questions"]), 1)
        self.assertTrue(str(body["saved_to"]).startswith(str(self.tmp)))
        bad = self.client.post("/evaluate", json={"questions": "nope.json"})
        self.assertEqual(bad.status_code, 400)

    def test_profile_switch_accepts_chunking_and_rejects_bad_dirs(self):
        response = self.client.post("/profile", json={"chunking": {"mode": "size"}})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["profile"]["chunking"]["mode"], "size")
        bad = self.client.post("/profile", json={"data_dirs": ["/no/such/dir"]})
        self.assertEqual(bad.status_code, 400)
        self.assertEqual(bad.json()["detail"]["reason"], "bad_data_dirs")
        bad = self.client.post("/profile", json={"chunking": {"mode": "weird"}})
        self.assertEqual(bad.status_code, 400)
        # restore, so any test running after this one keeps the ingested profile
        self.client.post("/profile", json={"chunking": {"mode": "restructure"}})


    def test_profile_switch_honors_custom_model_ids(self):
        # A custom (unregistered) model ID must be applied verbatim — the bug
        # this guards against: consistent_model silently swapped it for the
        # provider's first registered model while still answering 200.
        response = self.client.post("/profile", json={
            "answer": {"provider": "kira", "model": "glm-custom-9"}})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["profile"]["answer"],
                         {"provider": "kira", "model": "glm-custom-9"})
        self.assertEqual(self.client.get("/profile").json()["profile"]["answer"],
                         {"provider": "kira", "model": "glm-custom-9"})
        # same for the embedding slot — and the collection name follows
        emb = self.client.post("/profile", json={
            "embedding": {"provider": "nvidia", "model": "custom-embed-x"}})
        self.assertEqual(emb.status_code, 200, emb.text)
        self.assertEqual(emb.json()["profile"]["embedding"]["model"], "custom-embed-x")
        self.assertIn("custom_embed_x", emb.json()["collection"])   # slugified
        # a model ID with spaces is rejected, not silently corrected
        bad = self.client.post("/profile", json={
            "answer": {"provider": "kira", "model": "two tokens"}})
        self.assertEqual(bad.status_code, 400)
        self.assertEqual(bad.json()["detail"]["reason"], "bad_model")
        # provider-only switch to the SAME provider keeps the custom model
        same = self.client.post("/profile", json={"answer": {"provider": "kira"}})
        self.assertEqual(same.status_code, 200)
        self.assertEqual(same.json()["profile"]["answer"]["model"], "glm-custom-9")
        # provider-only switch to ANOTHER provider preselects its first
        # registered model — it never carries the custom ID across providers
        other = self.client.post("/profile", json={"answer": {"provider": "nvidia"}})
        self.assertEqual(other.status_code, 200)
        self.assertEqual(other.json()["profile"]["answer"]["provider"], "nvidia")
        self.assertIn(other.json()["profile"]["answer"]["model"],
                      [m["id"] for m in profiles.ANSWER_PROVIDERS["nvidia"]["models"]])
        # restore the class profile for any test that runs after this one
        for payload in ({"answer": {"provider": "xkiro",
                                    "model": "qwen/qwen3.8-max:free"}},
                        {"embedding": {"provider": "nvidia",
                                       "model": "nvidia/nemotron-3-embed-1b"}}):
            back = self.client.post("/profile", json=payload)
            self.assertEqual(back.status_code, 200, back.text)


class _ScriptedChatClient:
    """Answers with a scripted claim; the evidence quote is a verbatim prefix
    of the source containing `marker` — enough to drive the citation gate
    (quote membership + the numeric check) and the output scrubber."""
    base_url = "https://fake.test/v1"
    api_key = "fake"
    marker = "10 dinars"
    claim = "The Atlas card costs 10 dinars per year."

    def chat(self, model, messages, *, max_tokens=4096):
        payload = json.loads(messages[1]["content"])
        source = next(s for s in payload["sources"]
                      if type(self).marker in s["text"])
        quote = " ".join(source["text"].split())[:200]
        return {"text": json.dumps({"answerable": True, "claims": [
            {"text": type(self).claim,
             "evidence": [{"source_id": source["source_id"], "quote": quote}]}]}),
            "served_model": model, "usage": {}, "seconds": 0.0}


class _ContactQuoteClient(_ScriptedChatClient):
    marker = "support@atlas.tn"
    claim = ("Atlas support is support@atlas.tn, phone +216 71 123 456, "
             "RIB 08 0000 0000 0000 0000 12, CIN 09123456.")


class _LyingNumberClient(_ScriptedChatClient):
    marker = "10 dinars"
    claim = "The Atlas card costs 99 dinars per year."


class ServiceAuthTest(unittest.TestCase):
    """X-Service-Token: when configured, every request must carry it; CORS
    preflights stay open so browsers can still negotiate."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        overrides = {"CHROMA_DIR": cls.tmp / "chroma"}
        cls.open_client = TestClient(service.create_app(
            profiles.default_state(), generator=object(), service_token="",
            config_overrides=overrides))
        cls.locked_client = TestClient(service.create_app(
            profiles.default_state(), generator=object(),
            service_token="test-service-token",
            config_overrides=overrides))

    def test_open_when_no_token_configured(self):
        self.assertEqual(self.open_client.get("/health").status_code, 200)

    def test_locked_without_and_with_wrong_token(self):
        for headers in ({}, {"X-Service-Token": "wrong"},
                        {"X-Service-Token": "test-service-token "}):
            response = self.locked_client.get("/health", headers=headers)
            self.assertEqual(response.status_code, 401, headers)
            self.assertEqual(response.json()["detail"]["reason"], "unauthorized")
        # mutating routes are protected by the same check
        denied = self.locked_client.post("/keys", json={
            "key_env": "KIRA_API_KEY", "value": "kira-x-123456789"})
        self.assertEqual(denied.status_code, 401)

    def test_locked_accepts_the_right_token(self):
        response = self.locked_client.get(
            "/health", headers={"X-Service-Token": "test-service-token"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")

    def test_preflight_passes_without_token(self):
        response = self.locked_client.options(
            "/answer", headers={"Origin": "https://front.example",
                                "Access-Control-Request-Method": "POST"})
        self.assertEqual(response.status_code, 200)   # CORS answers preflight


class OutputGuardsTest(unittest.TestCase):
    """The output-side guards: PII scrubbing after the citation gate, and the
    numeric half of the gate (a claim number absent from its evidence quote
    refuses as unsourced_number)."""

    CORPUS = ("# Atlas Bank\n\n"
              "Product sheet for testing.\n\n"
              "## Contact\n\n"
              "Atlas support: email support@atlas.tn, phone +216 71 123 456, "
              "RIB 08 0000 0000 0000 0000 12, CIN 09123456 for verification.\n\n"
              "## Fees\n\n"
              "The Atlas card costs 10 dinars per year.\n")

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        (cls.tmp / "note.md").write_text(cls.CORPUS, encoding="utf-8")
        profile = _service_profile(cls.tmp)
        overrides = {"CHROMA_DIR": cls.tmp / "chroma",
                     "EMBEDDING_CACHE_PATH": cls.tmp / "emb.json",
                     "NVIDIA_EMBEDDING_CACHE_PATH": cls.tmp / "emb.json",
                     "ANSWER_CACHE_PATH": cls.tmp / "answers.json",
                     "RESULTS_DIR": cls.tmp}
        sys.modules["sentence_transformers"] = types.ModuleType("sentence_transformers")
        sys.modules["sentence_transformers"].SentenceTransformer = _FakeSentenceTransformer

        def build(client_class):
            local = build_lab_config(profile)
            for key, value in overrides.items():
                setattr(local, key, value)
            generator = AnswerGenerator(local, client=client_class(),
                                        approved_models=(PROFILE_MODEL,))
            return TestClient(service.create_app(
                profile, generator=generator, allow_profile_switch=False,
                config_overrides=overrides))

        cls.contact_client = build(_ContactQuoteClient)
        cls.lying_client = build(_LyingNumberClient)
        accepted = cls.contact_client.post("/ingest")
        assert accepted.status_code == 200, accepted.text
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            status = cls.contact_client.get("/ingest/status").json()
            if status["state"] != "running":
                break
            time.sleep(0.1)
        assert status["state"] == "done", status

    @classmethod
    def tearDownClass(cls):
        sys.modules.pop("sentence_transformers", None)

    def test_answer_output_is_pii_scrubbed_after_the_gate(self):
        response = self.contact_client.post(
            "/answer", json={"question": "How do I contact Atlas support?",
                             "k": 5, "include_excerpts": True})
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        # the gate validated the RAW verbatim quotes; only the output is scrubbed
        self.assertEqual(body["status"], "answered", body)
        for label in ("[EMAIL]", "[PHONE]", "[RIB]", "[CIN]"):
            self.assertIn(label, body["answer"], label)
        claim = body["claims"][0]
        self.assertIn("[EMAIL]", claim["text"])
        self.assertIn("[RIB]", claim["evidence"][0]["quote"])
        blob = json.dumps(body)
        for raw in ("support@atlas.tn", "+216 71 123 456",
                    "08 0000 0000 0000 0000 12", "09123456"):
            self.assertNotIn(raw, blob, raw)
        # excerpts too (order-independent: the stub embedder's ranking varies
        # with the per-process hash seed)
        self.assertTrue(any("[EMAIL]" in row.get("text", "")
                            for row in body["sources"]), body["sources"])

    def test_search_hits_are_pii_scrubbed(self):
        response = self.contact_client.post(
            "/search", json={"question": "Atlas support contact email"})
        self.assertEqual(response.status_code, 200, response.text)
        blob = json.dumps(response.json())
        self.assertNotIn("support@atlas.tn", blob)
        self.assertIn("[EMAIL]", blob)

    def test_inspect_shows_raw_text(self):
        # diagnostics display the raw truth on purpose (admin-facing); the
        # scrub boundary is the user-facing /answer and /search outputs
        response = self.contact_client.get("/inspect", params={"limit": 10})
        self.assertEqual(response.status_code, 200)
        self.assertIn("support@atlas.tn", json.dumps(response.json()))

    def test_unsourced_number_refuses(self):
        response = self.lying_client.post(
            "/answer", json={"question": "What does the Atlas card cost?"})
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["status"], "refused")
        self.assertEqual(body["reason"], "unsourced_number")
        self.assertFalse(body["inference_performed"])
        self.assertIn("99", body.get("raw_preview") or "")
        self.assertIn("10", body.get("raw_preview") or "")     # the honest quote
        self.assertIn("99", body.get("error") or "")


class DocumentsApiTest(unittest.TestCase):
    """The gateway feed target: push/list/status/delete with content-based
    versioning, chunk purging, and corpus integration via data_dirs."""

    MAX_BYTES = 4096

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        (cls.tmp / "seed.md").write_text(
            "# Seed\n\nseed doc\n\n## Info\n\nAtlas was founded in 2016.\n",
            encoding="utf-8")
        profile = _service_profile(cls.tmp)
        overrides = {"CHROMA_DIR": cls.tmp / "chroma",
                     "EMBEDDING_CACHE_PATH": cls.tmp / "emb.json",
                     "NVIDIA_EMBEDDING_CACHE_PATH": cls.tmp / "emb.json",
                     "ANSWER_CACHE_PATH": cls.tmp / "answers.json",
                     "RESULTS_DIR": cls.tmp}
        local = build_lab_config(profile)
        for key, value in overrides.items():
            setattr(local, key, value)
        generator = AnswerGenerator(local, client=_FakeChatClient(),
                                    approved_models=(PROFILE_MODEL,))
        sys.modules["sentence_transformers"] = types.ModuleType("sentence_transformers")
        sys.modules["sentence_transformers"].SentenceTransformer = _FakeSentenceTransformer
        cls.client = TestClient(service.create_app(
            profile, generator=generator, allow_profile_switch=True,
            config_overrides=overrides,
            documents_dir=cls.tmp / "documents",
            max_document_bytes=cls.MAX_BYTES))
        cls._wait_ingest()

    @classmethod
    def _wait_ingest(cls):
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            status = cls.client.get("/ingest/status").json()
            if status["state"] != "running":
                return status
            time.sleep(0.1)
        raise TimeoutError(status)

    @classmethod
    def tearDownClass(cls):
        sys.modules.pop("sentence_transformers", None)

    GUIDED_MD = ("# Guide\n\nguide doc\n\n## Fees\n\n"
                 "The Atlas card costs 10 dinars per year.\n")

    def test_push_json_lifecycle_with_versions_and_statuses(self):
        # create -> 201, pending (nothing ingested since)
        r = self.client.post("/documents", json={
            "filename": "guide.md", "content": self.GUIDED_MD})
        self.assertEqual(r.status_code, 201, r.text)
        doc = r.json()["document"]
        self.assertEqual((doc["id"], doc["stored_as"], doc["version"],
                          doc["status"], doc["chunks_in_index"]),
                         ("guide", "pushed-guide.md", 1, "pending", 0))
        self.assertEqual(len(doc["sha256_16"]), 16)      # hash is truncated
        # identical bytes -> no-op, same version
        r = self.client.post("/documents", json={
            "filename": "guide.md", "content": self.GUIDED_MD})
        self.assertEqual(r.status_code, 200)
        self.assertEqual((r.json()["result"], r.json()["document"]["version"]),
                         ("unchanged", 1))
        # different bytes -> version 2
        r = self.client.post("/documents", json={
            "filename": "guide.md",
            "content": self.GUIDED_MD.replace("10 dinars", "12 dinars")})
        self.assertEqual(r.status_code, 200)
        self.assertEqual((r.json()["result"], r.json()["document"]["version"]),
                         ("replaced", 2))
        # ingest -> indexed (chunks present, ingest newer than the doc)
        self.client.post("/ingest")
        self._wait_ingest()
        row = self.client.get("/documents/guide").json()["document"]
        self.assertEqual(row["status"], "indexed")
        self.assertGreaterEqual(row["chunks_in_index"], 1)
        # replace AFTER the ingest -> stale (old chunks still serve); sleep
        # past the second boundary so updated_at is strictly newer than the
        # ingest's finished_at (both timestamps are second-precision)
        time.sleep(1.05)
        r = self.client.post("/documents", json={
            "filename": "guide.md",
            "content": self.GUIDED_MD.replace("10 dinars", "15 dinars")})
        row = r.json()["document"]
        self.assertEqual((row["version"], row["status"]), (3, "stale"))

    def test_push_with_index_true_runs_the_ingest(self):
        r = self.client.post("/documents", params={"id": "auto", "index": "true"},
                             json={"filename": "auto.md",
                                   "content": "# Auto\n\nauto doc\n\n## Body\n\n"
                                              "The branch opens at 8.\n"})
        self.assertEqual(r.status_code, 201, r.text)
        self.assertTrue(r.json()["index_started"])
        status = self._wait_ingest()
        self.assertEqual(status["state"], "done")
        row = self.client.get("/documents/auto").json()["document"]
        self.assertEqual(row["status"], "indexed")

    def test_push_multipart(self):
        r = self.client.post("/documents", files={
            "file": ("notes.txt", b"# Notes\n\nplain text note\n")})
        self.assertEqual(r.status_code, 201, r.text)
        self.assertEqual(r.json()["document"]["stored_as"], "pushed-notes.txt")
        r = self.client.post("/documents", params={"id": "with-id"},
                             files={"file": ("whatever.md", b"# W\n\nw\n\n## S\n\ns\n")})
        self.assertEqual(r.json()["document"]["id"], "with-id")

    def test_push_base64(self):
        import base64
        r = self.client.post("/documents", json={
            "id": "b64", "filename": "b64.md",
            "content": base64.b64encode("# B\n\nb\n\n## S\n\ns\n".encode()).decode(),
            "content_encoding": "base64"})
        self.assertEqual(r.status_code, 201, r.text)
        self.assertEqual(r.json()["document"]["bytes"], 16)

    def test_push_rejects_bad_requests(self):
        cases = [
            ({"id": "../evil", "filename": "x.md", "content": "hi there"}, "bad_document_id"),
            ({"id": "a b", "filename": "x.md", "content": "hi there"}, "bad_document_id"),
            ({"filename": "x.exe", "content": "hi"}, "bad_document_type"),
            ({"filename": "noext", "content": "hi"}, "bad_document_type"),
            ({"filename": "x.md", "content": "!!!", "content_encoding": "base64"},
             "invalid_document_content"),
            ({"filename": "x.md", "content": "x" * 5000}, "document_too_large"),
        ]
        for payload, reason in cases:
            r = self.client.post("/documents", json=payload)
            self.assertEqual(r.status_code, 400 if reason != "document_too_large" else 413,
                             (payload, r.text))
            self.assertEqual(r.json()["detail"]["reason"], reason, payload)
        r = self.client.post("/documents", data="not-json",
                             headers={"content-type": "application/json"})
        self.assertEqual(r.status_code, 400)
        self.assertEqual(r.json()["detail"]["reason"], "invalid_document_push")

    def test_unknown_document_404(self):
        r = self.client.get("/documents/nope")
        self.assertEqual(r.status_code, 404)
        self.assertEqual(r.json()["detail"]["reason"], "unknown_document")
        self.assertEqual(self.client.delete("/documents/nope").status_code, 404)

    def test_documents_join_the_corpus_of_every_profile(self):
        self.client.post("/documents", json={
            "id": "corpus", "filename": "corpus.md",
            "content": "# C\n\nc\n\n## Body\n\npart of every profile\n"})
        names = [d["name"] for d in
                 self.client.get("/inspect", params={"limit": 1}).json()["documents"]]
        self.assertIn("pushed-corpus.md", names)
        # switching data_dirs must not drop the documents dir
        r = self.client.post("/profile", json={"data_dirs": [str(self.tmp)]})
        self.assertEqual(r.status_code, 200, r.text)
        self.assertIn(str(self.tmp / "documents"),
                      r.json()["profile"]["data_dirs"])
        # restore for the other tests
        self.client.post("/profile", json={"chunking": {"mode": "size"}})

    def test_delete_purges_chunks_from_the_index(self):
        self.client.post("/documents", json={
            "id": "gone", "filename": "gone.md",
            "content": "# G\n\ng\n\n## Body\n\ndelete me entirely\n"})
        self.client.post("/ingest")
        self._wait_ingest()
        row = self.client.get("/documents/gone").json()["document"]
        self.assertGreaterEqual(row["chunks_in_index"], 1)
        r = self.client.delete("/documents/gone")
        self.assertEqual(r.status_code, 200, r.text)
        collection = r.json()["chunks_removed"]
        self.assertGreaterEqual(sum(collection.values()), 1, collection)
        self.assertEqual(self.client.get("/documents/gone").status_code, 404)
        # the chunks are really gone from the active collection
        fresh = self.client.get("/documents").json()
        self.assertNotIn("gone", [d["id"] for d in fresh["documents"]])


class ProviderFailureTest(unittest.TestCase):
    """Network-layer failures (the xKiro live free-price check, provider
    endpoints, DNS/proxy/timeout) must surface as the JSON error envelope —
    502 provider_unreachable — never a bare 'Internal Server Error'."""

    def test_runtime_wraps_generator_build_network_errors(self):
        import urllib.error
        saved = os.environ.get("XKIRO_API_KEY")
        os.environ["XKIRO_API_KEY"] = "xki-test-1234567890"
        original = profiles.build_generator

        def boom(local):
            raise urllib.error.URLError("connection refused")
        profiles.build_generator = boom
        try:
            runtime = service.Runtime(profiles.default_state())
            with self.assertRaises(service.ServiceError) as ctx:
                runtime.generator()
        finally:
            profiles.build_generator = original
            if saved is None:
                os.environ.pop("XKIRO_API_KEY", None)
            else:
                os.environ["XKIRO_API_KEY"] = saved
        self.assertEqual(ctx.exception.status_code, 502)
        self.assertEqual(ctx.exception.payload["reason"], "provider_unreachable")

    def test_runtime_wraps_embedder_build_network_errors(self):
        import urllib.error
        saved = os.environ.get("NVIDIA_API_KEY")
        os.environ["NVIDIA_API_KEY"] = "nvapi-test-1234567890"
        import embedder as embedder_mod
        original = embedder_mod.build_embedder

        def boom(local):
            raise urllib.error.URLError("name resolution failed")
        embedder_mod.build_embedder = boom
        try:
            runtime = service.Runtime(profiles.default_state())
            with self.assertRaises(service.ServiceError) as ctx:
                runtime.embedder()
        finally:
            embedder_mod.build_embedder = original
            if saved is None:
                os.environ.pop("NVIDIA_API_KEY", None)
            else:
                os.environ["NVIDIA_API_KEY"] = saved
        self.assertEqual(ctx.exception.status_code, 502)
        self.assertEqual(ctx.exception.payload["reason"], "provider_unreachable")

    def test_answer_midrequest_network_failure_is_json_not_500(self):
        import urllib.error

        class _NetworkDeadGenerator:
            def answer(self, *args, **kwargs):
                raise urllib.error.URLError("connection reset by peer")

        tmp = Path(tempfile.mkdtemp())
        (tmp / "note.md").write_text(CORPUS, encoding="utf-8")
        profile = _service_profile(tmp)
        overrides = {"CHROMA_DIR": tmp / "chroma",
                     "EMBEDDING_CACHE_PATH": tmp / "emb.json",
                     "NVIDIA_EMBEDDING_CACHE_PATH": tmp / "emb.json",
                     "ANSWER_CACHE_PATH": tmp / "answers.json",
                     "RESULTS_DIR": tmp}
        sys.modules["sentence_transformers"] = types.ModuleType("sentence_transformers")
        sys.modules["sentence_transformers"].SentenceTransformer = _FakeSentenceTransformer
        try:
            client = TestClient(service.create_app(
                profile, generator=_NetworkDeadGenerator(),
                allow_profile_switch=False, config_overrides=overrides))
            client.post("/ingest")
            deadline = time.monotonic() + 60
            while time.monotonic() < deadline:
                status = client.get("/ingest/status").json()
                if status["state"] != "running":
                    break
                time.sleep(0.1)
            self.assertEqual(status["state"], "done", status)
            response = client.post("/answer", json={"question": QUESTION})
            self.assertEqual(response.status_code, 502, response.text)
            self.assertEqual(response.json()["detail"]["reason"],
                             "provider_unreachable")
            self.assertIn("connection reset", response.text)
        finally:
            sys.modules.pop("sentence_transformers", None)


class IngestConcurrencyTest(unittest.TestCase):
    """One writer at a time: while the ingest job runs, index reads and
    collection mutations come back as 409 ingest_in_progress (not a bare 500
    from racing sqlite), greetings still work, and document rows report
    'indexing' without touching the store."""

    class _SlowEmbedder:
        def __init__(self, model_name, device=None):
            self.prompts = {"query": ""}

        @staticmethod
        def get_embedding_dimension():
            return 8

        def encode(self, texts, batch_size=None, normalize_embeddings=None,
                   show_progress_bar=None, convert_to_numpy=None, **kwargs):
            time.sleep(1.5)          # stretch the job across the assertions
            rows = []
            for text in texts:
                row = [0.0] * 8
                row[hash(text) % 8] = 1.0
                rows.append(row)
            return rows

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        (cls.tmp / "note.md").write_text(CORPUS, encoding="utf-8")
        profile = _service_profile(cls.tmp)
        overrides = {"CHROMA_DIR": cls.tmp / "chroma",
                     "EMBEDDING_CACHE_PATH": cls.tmp / "emb.json",
                     "NVIDIA_EMBEDDING_CACHE_PATH": cls.tmp / "emb.json",
                     "ANSWER_CACHE_PATH": cls.tmp / "answers.json",
                     "RESULTS_DIR": cls.tmp}
        local = build_lab_config(profile)
        for key, value in overrides.items():
            setattr(local, key, value)
        generator = AnswerGenerator(local, client=_FakeChatClient(),
                                    approved_models=(PROFILE_MODEL,))
        sys.modules["sentence_transformers"] = types.ModuleType("sentence_transformers")
        sys.modules["sentence_transformers"].SentenceTransformer = cls._SlowEmbedder
        cls.client = TestClient(service.create_app(
            profile, generator=generator, allow_profile_switch=True,
            config_overrides=overrides,
            documents_dir=cls.tmp / "documents"))
        # build the index once (slow embedder: this ingest takes seconds)
        cls.client.post("/ingest")
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            if cls.client.get("/ingest/status").json()["state"] != "running":
                break
            time.sleep(0.2)

    @classmethod
    def tearDownClass(cls):
        sys.modules.pop("sentence_transformers", None)

    def _run_ingest_and_probe(self):
        # a unique document guarantees at least one UNCACHED chunk, so the
        # re-ingest really spends time in the (sleeping) embedder while we probe
        import uuid
        unique = uuid.uuid4().hex[:8]
        pushed0 = self.client.post("/documents", json={
            "id": f"slow-{unique}", "filename": "slow.md",
            "content": f"# S\n\ns {unique}\n\n## Body\n\nslow embed target {unique}\n"})
        assert pushed0.status_code == 201, pushed0.text
        accepted = self.client.post("/ingest?reset=true")
        assert accepted.status_code == 200, accepted.text
        # the job is now embedding (the unique chunk sleeps in the fake) — probe
        greeting = self.client.post("/answer", json={"question": "bonjour"})
        self.assertEqual(greeting.status_code, 200, greeting.text)
        self.assertEqual(greeting.json()["status"], "greeting")
        for path, payload in (("/search", {"question": QUESTION}),
                              ("/answer", {"question": QUESTION})):
            response = self.client.post(path, json=payload)
            self.assertEqual(response.status_code, 409, response.text)
            self.assertEqual(response.json()["detail"]["reason"],
                             "ingest_in_progress", path)
        switched = self.client.post("/profile", json={"retrieval": {"top_k": 6}})
        self.assertEqual(switched.status_code, 409, switched.text)
        pushed = self.client.post("/documents", json={
            "id": "midjob", "filename": "mid.md",
            "content": "# M\n\nm\n\n## Body\n\npushed mid-job\n"})
        self.assertEqual(pushed.status_code, 201, pushed.text)   # store is fine
        rows = self.client.get("/documents").json()["documents"]
        self.assertEqual(next(r for r in rows if r["id"] == "midjob")["status"],
                         "indexing")                            # no store read
        deleted = self.client.delete("/documents/midjob")
        self.assertEqual(deleted.status_code, 409)              # purge would race
        # wait for the job to finish so other tests start clean
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            status = self.client.get("/ingest/status").json()
            if status["state"] != "running":
                return status
            time.sleep(0.2)
        raise TimeoutError(status)

    def test_reads_refused_during_ingest_then_work(self):
        status = self._run_ingest_and_probe()
        self.assertEqual(status["state"], "done", status)
        answered = self.client.post("/answer", json={"question": QUESTION})
        self.assertEqual(answered.status_code, 200, answered.text)
        self.assertIn(answered.json()["status"], {"answered", "refused"})


class InternalErrorNeverPlainTextTest(unittest.TestCase):
    """The safety net: an exception nothing catches still comes back as the
    JSON error envelope (500 internal_error), never FastAPI's plain-text
    'Internal Server Error'."""

    def test_unhandled_exception_returns_json_envelope(self):
        class _ExplodingGenerator:
            def answer(self, *args, **kwargs):
                raise KeyError("boom")     # not OSError/ValueError/NvidiaAPIError

        tmp = Path(tempfile.mkdtemp())
        (tmp / "note.md").write_text(CORPUS, encoding="utf-8")
        profile = _service_profile(tmp)
        overrides = {"CHROMA_DIR": tmp / "chroma",
                     "EMBEDDING_CACHE_PATH": tmp / "emb.json",
                     "NVIDIA_EMBEDDING_CACHE_PATH": tmp / "emb.json",
                     "ANSWER_CACHE_PATH": tmp / "answers.json",
                     "RESULTS_DIR": tmp}
        sys.modules["sentence_transformers"] = types.ModuleType("sentence_transformers")
        sys.modules["sentence_transformers"].SentenceTransformer = _FakeSentenceTransformer
        try:
            # raise_server_exceptions=False: Starlette's ServerErrorMiddleware
            # sends our JSON envelope and then RE-RAISES so the server logs it;
            # the production client sees the envelope, the TestClient would
            # surface the re-raise instead
            client = TestClient(service.create_app(
                profile, generator=_ExplodingGenerator(),
                allow_profile_switch=False, config_overrides=overrides),
                raise_server_exceptions=False)
            client.post("/ingest")
            deadline = time.monotonic() + 60
            while time.monotonic() < deadline:
                status = client.get("/ingest/status").json()
                if status["state"] != "running":
                    break
                time.sleep(0.1)
            self.assertEqual(status["state"], "done", status)
            response = client.post("/answer", json={"question": QUESTION})
            self.assertEqual(response.status_code, 500, response.text)
            body = response.json()
            self.assertEqual(body["detail"]["reason"], "internal_error")
            self.assertIn("boom", body["detail"]["error"])
            self.assertEqual(response.headers["content-type"],
                             "application/json")   # not text/plain plaintext
        finally:
            sys.modules.pop("sentence_transformers", None)


if __name__ == "__main__":
    unittest.main()
