"""Request inspection through HTTP; no native model or GPU is required."""
import json
import threading
import time
import unittest
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest import mock

from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer, EngineDied, MockEngine, Service, serve


class ReloadableEngine(MockEngine):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.running, self.unloaded = True, False

    def alive(self):
        return self.running

    def unload(self):
        self.running, self.unloaded = False, True

    def restart(self):
        self.running, self.unloaded = True, False


class MonitorOff(unittest.TestCase):
    """#332 is opt-in: by default no prompt or answer is kept, and the monitor's endpoints do not exist."""

    def test_off_by_default(self):
        tok = ByteTokenizer()
        svc = Service(MockEngine(tok, "Hello.", max_context=4096), tok,
                      ChatTemplate(Path(__file__).parent / "chat_template.jinja"))
        httpd = serve(svc, port=0)
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        try:
            req = urllib.request.Request(base + "/v1/chat/completions", headers={"Content-Type": "application/json"},
                                         data=json.dumps({"messages": [{"role": "user", "content": "secret"}]}).encode())
            with urllib.request.urlopen(req, timeout=10) as r:
                self.assertEqual(r.status, 200)
            self.assertEqual(len(svc.api_requests), 0)
            for path in ("/api/requests", "/api-monitor"):
                with self.assertRaises(urllib.error.HTTPError) as e:
                    urllib.request.urlopen(base + path, timeout=10)
                self.assertEqual(e.exception.code, 404, path)
                e.exception.close()
        finally:
            httpd.shutdown()
            httpd.server_close()


class Monitor(unittest.TestCase):
    def setUp(self):
        self.tok = ByteTokenizer()
        self.engine = ReloadableEngine(self.tok, "Thought.</think>\nHello <script>alert(1)</script>.", max_context=16384)
        self.svc = Service(self.engine, self.tok, ChatTemplate(Path(__file__).parent / "chat_template.jinja"))
        self.svc.api_monitor = True                       # opt-in ("api_monitor": true / --api-monitor)
        self.httpd = serve(self.svc, port=0)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def request(self, path, body=None, headers=None):
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode() if body is not None else None,
                                     headers={"Content-Type": "application/json", **(headers or {})})
        try:
            response = urllib.request.urlopen(req, timeout=10)
        except urllib.error.HTTPError as error:
            response = error
        with response:
            raw = response.read().decode()
            return response.status, json.loads(raw) if "application/json" in response.headers.get("Content-Type", "") else raw

    def chat(self, api="openai", **kwargs):
        # Anthropic thinks only when asked (#278): "thinking" there, reasoning_effort for OpenAI
        think = {"thinking": {"type": "enabled", "budget_tokens": 1024}} if api == "anthropic" else             {"reasoning_effort": "low"}
        return self.request("/v1/messages" if api == "anthropic" else "/v1/chat/completions",
                            {"messages": [{"role": "user", "content": "Say hello."}],
                             "max_tokens": 256, **think, **kwargs})

    def latest(self):
        summary = self.request("/api/requests")[1]["requests"][0]
        return self.request("/api/requests?id=" + summary["id"])[1]

    def test_output_and_reasoning_for_both_dialects_and_stream_modes(self):
        for api in ("openai", "anthropic"):
            for stream in (False, True):
                with self.subTest(api=api, stream=stream):
                    code, reply = self.chat(api, stream=stream)
                    self.assertEqual(code, 200)
                    record = self.latest()
                    self.assertEqual(record["state"], "completed")
                    self.assertEqual(record["http_status"], 200)
                    self.assertIn("Say hello.", record["input"])
                    self.assertIn("<script>alert(1)</script>", record["output"])
                    self.assertIn("Thought.", record["reasoning"])
                    self.assertGreater(record["usage"].get("completion_tokens", record["usage"].get("output_tokens", 0)), 0)
                    self.assertGreaterEqual(record["wallclock_s"], record["first_token_s"])
                    self.assertNotIn("_clock", record)
                    if not stream:
                        self.assertEqual(json.loads(record["response"]), reply)
                    else:
                        self.assertNotIn("response", record)

    def test_error_and_engine_death_are_not_successes(self):
        # a request the server refuses is an error record: the strict mode (fit_max_tokens off) refuses a max_tokens
        # that overshoots the context, the default shortens it (serve/test_server.py MaxTokens)
        self.svc.fit_max_tokens = False
        self.assertEqual(self.chat(max_tokens=20000)[0], 400)
        self.svc.fit_max_tokens = True
        self.assertEqual(self.latest()["state"], "error")
        for api in ("openai", "anthropic"):
            for stream in (False, True):
                with self.subTest(api=api, stream=stream), mock.patch.object(self.engine, "generate", side_effect=EngineDied("stopped")):
                    self.chat(api, stream=stream)
                    record = self.latest()
                    self.assertEqual(record["state"], "error")
                    self.assertIn("stopped", record["error"]["message"])
                    self.assertEqual(record["output"], "")
                    self.assertNotIn("timings", record)

    def test_history_is_authorized_and_does_not_capture_headers(self):
        self.svc.api_key = "secret"
        auth = {"Authorization": "Bearer secret"}
        self.assertEqual(self.request("/api/requests")[0], 401)
        self.assertEqual(self.request("/api/requests?id=missing")[0], 401)
        self.assertEqual(len(self.svc.api_requests), 0)
        self.assertEqual(self.request("/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}]}, auth)[0], 200)
        summary = self.request("/api/requests", headers=auth)[1]["requests"][0]
        self.assertNotIn("input", summary)
        self.assertNotIn("output", summary)
        self.assertEqual(self.request("/api/requests?id=" + summary["id"])[0], 401)
        detail = self.request("/api/requests?id=" + summary["id"], headers=auth)[1]
        self.assertNotIn("secret", json.dumps(detail))

    def test_capture_is_bounded_and_eviction_returns_404(self):
        huge = "x" * 262200
        first = self.svc.begin_request("/v1/chat/completions", {"messages": [{"content": huge}]})
        self.assertEqual(len(first["input"]), 262144)
        self.assertTrue(first["input_truncated"])
        for _ in range(100):
            self.svc.begin_request("/v1/chat/completions", {"messages": []})
        self.assertEqual(len(self.request("/api/requests")[1]["requests"]), 100)
        self.assertEqual(self.request("/api/requests?id=" + first["id"])[0], 404)
        # Exercise output capture without making a mock engine emit 262K tokens.
        handler = object.__new__(self.httpd.RequestHandlerClass)
        handler.record = self.svc.api_requests[-1]
        chunk = {"choices": [{"delta": {"content": huge, "reasoning_content": huge}}]}
        list(handler._capture((item for item in [chunk]), "openai"))
        for field in ("output", "reasoning"):
            self.assertEqual(len(handler.record[field]), 262144)
            self.assertTrue(handler.record[field + "_truncated"])

    def test_queue_wait_and_loading_are_attributed_separately(self):
        self.engine.unload()
        entered, release = threading.Event(), threading.Event()
        original = self.engine.restart
        def restart():
            entered.set()
            self.assertTrue(release.wait(5))
            original()
        with mock.patch.object(self.engine, "restart", side_effect=restart), ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(self.chat)
            self.assertTrue(entered.wait(5))
            second = pool.submit(self.chat)
            deadline = time.monotonic() + 5
            while len(self.svc.api_requests) < 2 and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertEqual(len(self.svc.api_requests), 2)
            history = self.request("/api/requests")[1]
            self.assertEqual({r["state"] for r in history["requests"]}, {"queued", "loading"})
            self.assertFalse(history["loaded"])
            self.assertTrue(history["auto_load"])
            time.sleep(0.03)
            release.set()
            self.assertEqual(first.result()[0], 200)
            self.assertEqual(second.result()[0], 200)
        records = list(self.svc.api_requests)
        self.assertGreater(records[0]["load_s"], 0.02)
        self.assertGreater(records[1]["queue_s"], 0.02)
        self.assertLess(records[1]["load_s"], records[0]["load_s"])
        self.assertTrue(all(r["state"] == "completed" for r in records))

    def test_page_and_relative_assets_are_served(self):
        code, html = self.request("/api-monitor")
        self.assertEqual(code, 200)
        self.assertIn('src="/web/monitor.js"', html)
        code, script = self.request("/web/monitor.js")
        self.assertEqual(code, 200)
        self.assertIn("textContent", script)
        self.assertNotIn("innerHTML", script)
        self.assertEqual(self.request("/web/../server.py")[0], 404)


class ConversationCacheCard(unittest.TestCase):
    """#596: /metrics' "conversation_cache": the parked conversations from the engine's log lines, the budget and
    slots from its INFO line, and how much of the prompts the cache gave back."""

    PARK = ("strata serve: conversation cache: parked 5000 tokens in 12.0 ms; parked=1 bytes=104857600 evictions=0 "
            "snapshot_bytes=104857600 reused_kv_bytes=0\n")
    RESTORE = "strata serve: conversation cache: restored 4000 tokens (exact) in 30.0 ms; parked=2 bytes=209715200\n"

    def test_the_log(self):
        from serve.server import ConvCacheLog
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            log = Path(d) / "strata.log"
            log.write_text("an earlier run\n" + self.PARK, encoding="utf-8")
            start = log.stat().st_size                 # this run starts here: the earlier run's park is not counted
            c = ConvCacheLog()
            self.assertEqual(c.poll(str(log), start)["parked"], 0)
            with open(log, "a", encoding="utf-8") as f:
                f.write("strata serve: something else\n" + self.PARK + self.RESTORE + "strata serve: conversation ca")
            st = c.poll(str(log), start)
            self.assertEqual((st["parked"], st["bytes"], st["parks"], st["restores"]), (2, 209715200, 1, 1))
            self.assertEqual((st["last_event"], st["last_tokens"]), ("restored", 4000))
            with open(log, "a", encoding="utf-8") as f:   # the cut line is read once it is whole
                f.write("che: dropped 1 superseded copy of this conversation; parked=1\n")
            self.assertEqual(c.poll(str(log), start)["parked"], 1)
            self.assertEqual(c.poll(str(log), log.stat().st_size)["parked"], 0)   # the engine started again
            self.assertEqual(c.poll(None, None)["parked"], 0)

    def test_metrics(self):
        tok = ByteTokenizer()
        engine = MockEngine(tok, "Thought.</think>\nHello.", max_context=4096)
        engine.info = {"conversation_cache_mib": 8192, "conversation_cache_slots": 4}
        svc = Service(engine, tok, ChatTemplate(Path(__file__).parent / "chat_template.jinja"))
        httpd = serve(svc, port=0)
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        try:
            with urllib.request.urlopen(base + "/metrics", timeout=10) as r:
                c = json.loads(r.read())["conversation_cache"]
            self.assertEqual((c["enabled"], c["budget_mib"], c["slots"], c["parked"], c["requests"]),
                             (True, 8192, 4, 0, 0))
            req = urllib.request.Request(base + "/v1/chat/completions", headers={"Content-Type": "application/json"},
                                         data=json.dumps({"messages": [{"role": "user", "content": "hi"}]}).encode())
            with urllib.request.urlopen(req, timeout=10) as r:
                r.read()
            with urllib.request.urlopen(base + "/metrics", timeout=10) as r:
                c = json.loads(r.read())["conversation_cache"]
            self.assertEqual(c["requests"], 1)
            self.assertGreater(c["last_prompt"], 0)
            engine.info = {}
            with urllib.request.urlopen(base + "/metrics", timeout=10) as r:
                self.assertFalse(json.loads(r.read())["conversation_cache"]["enabled"])   # off: the reuse lines only
        finally:
            httpd.shutdown()
            httpd.server_close()

    def test_the_card_is_on_the_page(self):
        web = Path(__file__).parent / "web"
        html, js = (web / "index.html").read_text(encoding="utf-8"), (web / "app.js").read_text(encoding="utf-8")
        for el in ("cc-card", "cc-slots-text", "cc-mem-text", "cc-facts", "cc-note"):
            self.assertIn(f'id="{el}"', html)
            self.assertIn(f'"{el}"', js)



class RequestAsChat(unittest.TestCase):
    """#332 "view as chat": with the API request monitor on, a request's messages (compact) and its reply are kept
    with its record, and the Monitor's request rows say which record that is."""

    def test_transcript(self):
        from serve.server import transcript_of, TRANSCRIPT_TEXT_MAX
        msgs = [{"role": "system", "content": "Be brief."},
                {"role": "user", "content": [{"type": "text", "text": "What is "}, {"type": "image", "source": {}},
                                             {"type": "text", "text": "this?"}]},
                {"role": "assistant", "content": "", "reasoning_content": "Look.",
                 "tool_calls": [{"function": {"name": "read", "arguments": {"path": "a.txt"}}}]},
                {"role": "tool", "content": "x" * (TRANSCRIPT_TEXT_MAX + 10)}]
        t = transcript_of(msgs)
        self.assertEqual(t["omitted"], 0)
        user, call, result = t["messages"][1], t["messages"][2], t["messages"][3]
        self.assertEqual((user["text"], user["images"]), ("What is this?", 1))
        self.assertEqual((call["reasoning"], call["tool_calls"]), ("Look.", [{"name": "read", "arguments": {"path": "a.txt"}}]))
        self.assertIn("10 more characters not kept", result["text"])
        # the next turn of the same conversation shares the messages that did not change
        later = transcript_of(msgs + [{"role": "assistant", "content": "Done."}], [t["messages"]])
        self.assertTrue(all(a is b for a, b in zip(later["messages"], t["messages"])))
        self.assertEqual(later["messages"][-1]["text"], "Done.")
        # past TRANSCRIPT_MAX characters the oldest messages are left out; the newest one always stays
        with mock.patch("serve.server.TRANSCRIPT_MAX", 30):
            t = transcript_of([{"role": "user", "content": "a" * 20}, {"role": "assistant", "content": "b" * 20},
                               {"role": "user", "content": "c" * 40}])
        self.assertEqual((t["omitted"], [m["text"][0] for m in t["messages"]]), (2, ["c"]))

    def test_request_rows_link_their_messages(self):
        tok = ByteTokenizer()
        engine = MockEngine(tok, "Thought.</think>\nHello.", max_context=4096)
        svc = Service(engine, tok, ChatTemplate(Path(__file__).parent / "chat_template.jinja"))
        httpd = serve(svc, port=0)
        base = f"http://127.0.0.1:{httpd.server_address[1]}"

        def chat(stream):
            req = urllib.request.Request(base + "/v1/chat/completions", headers={"Content-Type": "application/json"},
                                         data=json.dumps({"messages": [{"role": "system", "content": "Be brief."},
                                                                       {"role": "user", "content": "hi"}],
                                                          "stream": stream}).encode())
            with urllib.request.urlopen(req, timeout=10) as r:
                r.read()
            with urllib.request.urlopen(base + "/metrics", timeout=10) as r:
                return json.loads(r.read())
        try:
            m = chat(False)                              # the monitor is off: nothing is kept
            self.assertFalse(m["transcripts"])
            self.assertIsNone(m["requests"][0]["record_id"])
            svc.api_monitor = True
            for stream in (False, True):
                m = chat(stream)
                self.assertTrue(m["transcripts"])
                rid = m["requests"][0]["record_id"]
                with urllib.request.urlopen(f"{base}/api/requests?id={rid}", timeout=10) as r:
                    rec = json.loads(r.read())
                self.assertEqual([(x["role"], x["text"]) for x in rec["transcript"]["messages"]],
                                 [("system", "Be brief."), ("user", "hi")])
                self.assertEqual((rec["reply"]["reasoning"].strip(), rec["reply"]["content"].strip()),
                                 ("Thought.", "Hello."))
                self.assertEqual(rec["params"], {})
            with urllib.request.urlopen(base + "/api/requests", timeout=10) as r:   # the list stays light
                self.assertNotIn("transcript", json.loads(r.read())["requests"][0])
        finally:
            httpd.shutdown()
            httpd.server_close()

    def test_the_button_is_on_the_page(self):
        web = Path(__file__).parent / "web"
        html, js = (web / "index.html").read_text(encoding="utf-8"), (web / "app.js").read_text(encoding="utf-8")
        for el in ("rv", "rv-title", "rv-chat", "rv-side", "rv-close"):
            self.assertIn(f'id="{el}"', html)
        self.assertIn("viewButton(r)", js)
        self.assertIn("req-view", js)
        self.assertIn("api/requests?id=", js)
        self.assertIn('id="i-close"', (web / "sprite.svg").read_text(encoding="utf-8"))

    def test_request_params(self):
        from serve.server import request_params
        p = request_params({"messages": [], "temperature": 0.7, "stream": True, "max_tokens": 512,
                            "thinking": {"type": "enabled", "budget_tokens": 4000}, "output_config": {"effort": "high"},
                            "tools": [{"type": "function", "function": {"name": "read"}}, {"name": "bash"}],
                            "tool_choice": {"type": "function", "function": {"name": "read"}}})
        self.assertEqual(p, {"temperature": 0.7, "max_tokens": 512, "thinking": "enabled", "thinking_budget": 4000,
                             "effort": "high", "tool_choice": "read", "tools": 2, "tool_names": ["read", "bash"]})

if __name__ == "__main__":
    unittest.main()


class AutoSessionsCard(unittest.TestCase):
    """/metrics' "auto_sessions": the limits, the conversation the engine holds, the auto-*.bin files on disk (not
    the ones saved by hand) and what the switches did."""

    def test_metrics(self):
        import tempfile
        tok = ByteTokenizer()
        engine = MockEngine(tok, "Thought.</think>\nHello.", max_context=4096)
        svc = Service(engine, tok, ChatTemplate(Path(__file__).parent / "chat_template.jinja"))
        self.assertFalse(svc.auto_sessions_view()["enabled"])
        self.assertIsNone(svc.auto_sessions_view()["files"])
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "auto-aaaa.bin").write_bytes(b"x" * 100)
            (Path(d) / "by-hand.bin").write_bytes(b"x" * 1000)
            svc.slot_save_path, svc.auto_sessions = d, True
            svc.auto_conv = {"key": "aaaa", "tokens": 20000}
            svc._auto_session_note("restored", {"tokens": 20000, "ms": 140.0})
            httpd = serve(svc, port=0)
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{httpd.server_address[1]}/metrics", timeout=10) as r:
                    a = json.loads(r.read())["auto_sessions"]
            finally:
                httpd.shutdown()
                httpd.server_close()
        self.assertEqual((a["enabled"], a["files"], a["files_bytes"], a["keep"], a["min_tokens"]),
                         (True, 1, 100, 12, 16000))
        self.assertEqual((a["held_tokens"], a["held_on_disk"]), (20000, True))
        self.assertEqual((a["restores"], a["saves"], a["last_event"], a["last_tokens"]), (1, 0, "restored", 20000))

    def test_the_card_is_on_the_page(self):
        web = Path(__file__).parent / "web"
        html, js = (web / "index.html").read_text(encoding="utf-8"), (web / "app.js").read_text(encoding="utf-8")
        for el in ("as-card", "as-files-text", "as-disk-text", "as-facts", "as-note"):
            self.assertIn(f'id="{el}"', html)
        self.assertIn("renderAutoSessions(m.auto_sessions)", js)
        self.assertLess(html.index('id="cc-card"'), html.index('id="as-card"'))   # under the Conversation cache
