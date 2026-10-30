"""Auto sessions: the conversation key and the prune (CPU only, no engine)."""
import os
import tempfile
import time
import unittest
from types import SimpleNamespace

from serve import frontend
from serve.server import Service, client_conversation_id, conversation_key


class ConversationKey(unittest.TestCase):
    def test_client_id_separates_chats_with_the_same_opening(self):
        msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "hello"}]
        self.assertEqual(conversation_key(msgs), conversation_key(msgs))
        a = conversation_key(msgs, client_conversation_id({"x-session-id": "a"}, {}))
        b = conversation_key(msgs, client_conversation_id({"x-session-id": "b"}, {}))
        self.assertNotEqual(a, b)
        self.assertNotEqual(a, conversation_key(msgs))
        self.assertEqual(conversation_key(msgs, client_conversation_id({}, {"prompt_cache_key": "q:1"})),
                         conversation_key(msgs, client_conversation_id({}, {"prompt_cache_key": "q:1"})))
        self.assertIsNone(client_conversation_id({}, {}))
        self.assertIsNone(conversation_key([{"role": "user", "content": " "}], "x"))

    def test_system_to_user_block_does_not_change_the_key(self):
        old = frontend.SYSTEM_TO_USER
        frontend.SYSTEM_TO_USER = frontend.SystemToUser(patterns=[r"^Date: "], learn=False, place="latest")
        try:
            sys_msg = {"role": "system", "content": "You help.\nDate: 2026-10-30"}
            turn1, _, _ = frontend.openai_to_messages(
                {"messages": [sys_msg, {"role": "user", "content": "first"}]})
            turn2, _, _ = frontend.openai_to_messages(
                {"messages": [dict(sys_msg, content="You help.\nDate: 2026-10-31"), {"role": "user", "content": "first"},
                              {"role": "assistant", "content": "ok"}, {"role": "user", "content": "next"}]})
            self.assertTrue(turn1[1]["content"].startswith("Date: "))     # the block moved into the first message
            self.assertEqual(conversation_key(turn1), conversation_key(turn2))
            self.assertEqual(conversation_key(turn1), conversation_key([{"role": "user", "content": "first"}]))
        finally:
            frontend.SYSTEM_TO_USER = old


class Prune(unittest.TestCase):
    def test_the_file_restored_next_is_spared(self):
        with tempfile.TemporaryDirectory() as d:
            now = time.time()
            for i, key in enumerate(("old", "mid", "new")):          # "old" is the oldest
                p = os.path.join(d, f"auto-{key}.bin")
                with open(p, "wb") as f:
                    f.write(b"x")
                os.utime(p, (now - 100 + i, now - 100 + i))
            svc = SimpleNamespace(auto_sessions_keep=1, auto_sessions_max_gb=0, slot_save_path=d,
                                  auto_stats={"pruned": 0}, _DELTA_LOG=Service._DELTA_LOG)
            svc._auto_session_files = lambda: Service._auto_session_files(svc)
            svc._remove_auto_session = lambda p: Service._remove_auto_session(svc, p)
            svc._remove_orphan_auto_logs = lambda: Service._remove_orphan_auto_logs(svc)
            Service._auto_session_prune(svc, spare="old")
            self.assertEqual(sorted(os.listdir(d)), ["auto-new.bin", "auto-old.bin"])


if __name__ == "__main__":
    unittest.main()
