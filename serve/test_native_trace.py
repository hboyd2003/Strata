"""CPU-only protocol contract for opt-in native request correlation."""
from types import SimpleNamespace
import os
import queue
import threading
import unittest
from collections import deque
from unittest import mock

from serve import native_trace
from serve.server import Service, StrataEngine


class ActiveRequests:
    """CPU-only model of the IDs active in native batch slots."""

    def __init__(self):
        self._active: dict[str, int] = {}

    def begin(self, request_id: str, slot: int) -> None:
        native_trace.field(request_id)
        if request_id in self._active:
            raise ValueError("duplicate active correctness trace request ID")
        self._active[request_id] = slot

    def complete(self, request_id: str, slot: int, finish: str, produced: int) -> str:
        if self._active.get(request_id) != slot:
            raise ValueError("missing or mismatched correctness trace request ID")
        del self._active[request_id]
        # Deliberately excludes prompt, generated text, and token IDs.
        return (f"strata correctness: request={request_id} slot={slot} state=native_leg_complete "
                f"finish={finish} produced={produced}")


class NativeTraceProtocol(unittest.TestCase):
    def test_opt_in_field_is_opaque_and_validated(self):
        rid = "a" * 32
        with mock.patch.dict(os.environ, {native_trace.TRACE_ENV: "1"}):
            self.assertTrue(native_trace.enabled())
            self.assertEqual(native_trace.field(rid), f"trace_id={rid}")
            for invalid in (None, "", "x" * 32, "a" * 31, "A" * 32):
                with self.assertRaises(ValueError):
                    native_trace.field(invalid)

    def test_actual_engine_geni_command_contains_only_the_id_field(self):
        class Input:
            def __init__(self):
                self.lines = []

            def write(self, line):
                self.lines.append(line)

            def flush(self):
                pass

        rid = "c" * 32
        eng = object.__new__(StrataEngine)
        eng.alive = lambda: True
        eng.batch = 0
        eng.sampling_keys = lambda _: ""
        eng.progress = eng.progress_ms = eng.prefill_tok_s_mean = None
        eng.silence_s, eng.can_stop = 0, False
        eng.proc = SimpleNamespace(stdin=Input())
        eng.lines = queue.Queue()
        eng.lines.put("DONE 0")
        eng.last = {"finish": "stop"}
        eng._parse_done = lambda _: None
        with mock.patch.dict(os.environ, {native_trace.TRACE_ENV: "1"}), \
                mock.patch("serve.server.bias_api.validate_request"):
            self.assertEqual(list(eng.generate([7, 8], 3, {}, threading.Event(), embeddings="emb.bin",
                                               request_id=rid)), [])
        self.assertEqual(eng.proc.stdin.lines, [f"GENI 3 trace_id={rid} emb.bin 7,8\n"])

    def test_actual_batch_heads_cover_gen_to_bgen_image_transition(self):
        rid = "d" * 32
        with mock.patch.dict(os.environ, {native_trace.TRACE_ENV: "1"}):
            solo = native_trace.command("GENI", 12, " temperature=0.2", rid, embeddings="emb.bin")
            batch = native_trace.command("BGENI", 12, " temperature=0.2", rid, slot=3,
                                         embeddings="emb.bin")
        self.assertEqual(solo, f"GENI 12 temperature=0.2 trace_id={rid} emb.bin")
        self.assertEqual(batch, f"BGENI 3 12 temperature=0.2 trace_id={rid} emb.bin")

    def test_missing_id_fails_before_batch_control_acquire(self):
        eng = object.__new__(StrataEngine)
        eng.alive = lambda: True
        eng._take_control = mock.Mock()
        with mock.patch.dict(os.environ, {native_trace.TRACE_ENV: "1"}):
            gen = eng.generate_batched([1], 4, {}, threading.Event())
            with self.assertRaisesRegex(ValueError, "requires a 32-character"):
                next(gen)
        eng._take_control.assert_not_called()

    def test_request_id_is_lazy_and_monitor_ids_keep_their_length(self):
        svc = object.__new__(Service)
        svc.api_monitor = False
        svc.request_trace = threading.local()
        with mock.patch.dict(os.environ, {}, clear=True), mock.patch("serve.server.uuid.uuid4") as uuid4:
            self.assertIsNone(svc.begin_request("/v1/chat/completions", {}))
        uuid4.assert_not_called()
        self.assertIsNone(svc.request_trace.id)

        with mock.patch.dict(os.environ, {native_trace.TRACE_ENV: "1"}), \
                mock.patch("serve.server.uuid.uuid4") as uuid4:
            uuid4.return_value.hex = "f" * 32
            self.assertIsNone(svc.begin_request("/v1/chat/completions", {}))
        self.assertEqual(svc.request_trace.id, "f" * 32)

        svc.api_monitor = True
        svc.model = "model"
        svc.status_lock = threading.Lock()
        svc.api_requests = deque(maxlen=100)
        with mock.patch.dict(os.environ, {}, clear=True), mock.patch("serve.server.uuid.uuid4") as uuid4:
            uuid4.return_value.hex = "e" * 32
            record = svc.begin_request("/v1/chat/completions", {})
        self.assertEqual(record["id"], "e" * 12)
        self.assertEqual(svc.request_trace.id, "e" * 32)

    def test_engine_env_alone_turns_the_ids_on(self):
        # the switch only in the run config's "env" block: the engine traces, so the server must send the IDs
        with mock.patch.dict(os.environ, {}, clear=True):
            native_trace.follow_engine_env({native_trace.TRACE_ENV: "1"})
            try:
                self.assertTrue(native_trace.enabled())
                self.assertEqual(native_trace.command("GEN", 0, " prefill=1", "a" * 32),
                                 f"GEN 0 prefill=1 trace_id={'a' * 32}")
            finally:
                native_trace.follow_engine_env({})
            self.assertFalse(native_trace.enabled())

    def test_legacy_mode_adds_no_field(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertFalse(native_trace.enabled())
            self.assertEqual(native_trace.command("GEN", 3, " top_k=4", "a" * 32), "GEN 3 top_k=4")

    def test_concurrent_distinct_ids_complete_without_prompt_data(self):
        ledger = ActiveRequests()
        left, right = "1" * 32, "2" * 32
        ledger.begin(left, 0)
        ledger.begin(right, 1)
        msg = ledger.complete(right, 1, "stop", 4)
        self.assertIn(f"request={right}", msg)
        self.assertNotIn(left, msg)
        self.assertNotIn("prompt", msg)
        self.assertNotIn("secret prompt text", msg)
        self.assertIn("state=native_leg_complete", msg)
        self.assertNotIn("logical_complete", msg)
        self.assertIn("produced=4", msg)
        self.assertIn(left, ledger.complete(left, 0, "length", 8))

    def test_duplicate_active_and_missing_completion_are_rejected(self):
        ledger = ActiveRequests()
        rid = "f" * 32
        with self.assertRaises(ValueError):
            ledger.complete(rid, 0, "stop", 1)
        ledger.begin(rid, 0)
        with self.assertRaises(ValueError):
            ledger.begin(rid, 1)
        with self.assertRaises(ValueError):
            ledger.complete("e" * 32, 0, "stop", 1)
        with self.assertRaises(ValueError):
            ledger.complete(rid, 1, "stop", 1)
        ledger.complete(rid, 0, "stop", 1)
        with self.assertRaises(ValueError):
            ledger.complete(rid, 0, "stop", 1)


if __name__ == "__main__":
    unittest.main()
