"""Config "system_to_user": system-message lines that change from request to request (a client's date and time) are
learned, or named by pattern, and move to a user message, so new chats share the system turn and its prefix cache.
Not configured (the default), nothing moves."""
import json
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import serve.frontend as fe  # noqa: E402


def system(stamp="2026-10-18T10:44:15.670Z", local="2026-10-18 12:44:15 +02:00 (Sunday)", extra=""):
    return ("# `web_search` Runtime Context\n"
            f"Conversation Date & Time: {stamp}\n"
            "You are a precise assistant. Your knowledge cutoff is 2025-01.\n\n"
            "Current User: J\n"
            f"Current Date & Time: {local}\n"
            f"{extra}"
            "Instructions that never change.")


SYS_A = system()
SYS_B = system("2026-10-19T08:02:07.001Z", "2026-10-19 10:02:07 +02:00 (Monday)")
STABLE = ("# `web_search` Runtime Context\n"
          "You are a precise assistant. Your knowledge cutoff is 2025-01.\n\n"
          "Current User: J\n"
          "Instructions that never change.")


class SystemToUser(unittest.TestCase):
    def setUp(self):
        self.addCleanup(setattr, fe, "SYSTEM_TO_USER", fe.SYSTEM_TO_USER)
        self.log = []
        quiet = mock.patch.object(fe, "print", create=True, new=lambda *a, **k: self.log.append(" ".join(map(str, a))))
        quiet.start()
        self.addCleanup(quiet.stop)

    def use(self, **cfg):
        fe.SYSTEM_TO_USER = fe.SystemToUser(**cfg)
        return fe.SYSTEM_TO_USER

    def chat(self, sys_text, *turns):
        msgs = [{"role": "system", "content": sys_text}]
        for i, t in enumerate(turns or ("hi",)):
            msgs.append({"role": "user" if i % 2 == 0 else "assistant", "content": t})
        return fe.openai_to_messages({"messages": msgs})[0]

    def test_learned_from_two_chats_then_moved(self):
        self.use()
        self.assertEqual(self.chat(SYS_A)[0]["content"], SYS_A)          # nothing to compare with yet
        b = self.chat(SYS_B)
        self.assertEqual(b[0]["content"], STABLE)
        self.assertEqual(b[1]["content"], "Conversation Date & Time: 2026-10-19T08:02:07.001Z\n"
                                          "Current Date & Time: 2026-10-19 10:02:07 +02:00 (Monday)\n\nhi")
        self.assertEqual(sum("learned a changing line" in x for x in self.log), 2)
        c = self.chat(system("2026-10-20T00:00:00.000Z", "2026-10-20 02:00:00 +02:00 (Tuesday)"))
        self.assertEqual(c[0]["content"], STABLE)                          # every new chat shares the system turn

    def test_a_reworded_label_is_learned_too(self):
        self.use()
        self.chat(SYS_A.replace("Current Date & Time", "Datum/Uhrzeit"))
        out = self.chat(SYS_B.replace("Current Date & Time", "Datum/Uhrzeit"))
        self.assertNotIn("Datum/Uhrzeit", out[0]["content"])
        self.assertIn("Datum/Uhrzeit: 2026-10-19 10:02:07", out[1]["content"])

    def test_stable_and_edited_lines_stay(self):
        self.use()
        self.chat(SYS_A)
        out = self.chat(SYS_B.replace("Instructions that never change.", "Instructions that changed once."))
        self.assertIn("knowledge cutoff is 2025-01", out[0]["content"])      # never differed
        self.assertIn("Instructions that changed once.", out[0]["content"])  # other words: an edit, not a stamp

    def test_a_bare_number_line_is_not_learned(self):
        self.use()
        self.chat("Header text\n12:30\nrest")
        self.assertEqual(self.chat("Header text\n13:45\nrest")[0]["content"], "Header text\n13:45\nrest")

    def test_ignore_keeps_a_line_and_is_not_learned(self):
        stu = self.use(ignore=[r"^Conversation Date"])
        self.chat(SYS_A)
        out = self.chat(SYS_B)
        self.assertIn("Conversation Date & Time: 2026-10-19", out[0]["content"])
        self.assertNotIn("Current Date & Time", out[0]["content"])
        self.assertEqual(len(stu.learned), 1)

    def test_patterns_without_learning(self):
        stu = self.use(learn=False, patterns=[r"^Current Date & Time: "])
        out = self.chat(SYS_A)                                              # moves on the first request
        self.assertNotIn("Current Date & Time", out[0]["content"])
        self.assertIn("Conversation Date & Time", out[0]["content"])
        self.chat(SYS_B)
        self.assertEqual(stu.learned, set())
        self.assertEqual(stu.recent, {})

    def test_place_latest_and_first(self):
        self.use(patterns=[r"^Current Date & Time: "])
        latest = self.chat(SYS_A, "q1", "a1", "q2")
        self.assertEqual(latest[1]["content"], "q1")
        self.assertTrue(latest[3]["content"].startswith("Current Date & Time: ") and latest[3]["content"].endswith("q2"))
        self.use(patterns=[r"^Current Date & Time: "], place="first")
        first = self.chat(SYS_A, "q1", "a1", "q2")
        self.assertTrue(first[1]["content"].endswith("\n\nq1") and first[1]["content"].startswith("Current Date"))
        self.assertEqual(first[3]["content"], "q2")

    def test_latest_user_message_in_a_tool_loop(self):
        self.use(patterns=[r"^Current Date & Time: "])
        msgs = fe._system_to_user([
            {"role": "system", "content": SYS_A}, {"role": "user", "content": "lights?"},
            {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "x", "arguments": "{}"}}]},
            {"role": "tool", "content": "on"}])
        self.assertTrue(msgs[1]["content"].startswith("Current Date & Time: "))
        self.assertEqual(msgs[3]["content"], "on")

    def test_content_parts_get_a_leading_text_part(self):
        self.use(patterns=[r"^Current Date & Time: "])
        out = fe._system_to_user([{"role": "system", "content": SYS_A},
                                  {"role": "user", "content": [{"type": "image", "image": "x.png"}]}])
        self.assertEqual(out[1]["content"][0], {"type": "text", "text":
                                                "Current Date & Time: 2026-10-18 12:44:15 +02:00 (Sunday)\n\n"})
        self.assertEqual(out[1]["content"][1:], [{"type": "image", "image": "x.png"}])

    def test_learned_shapes_survive_a_restart(self):
        with tempfile.TemporaryDirectory() as d:
            state = str(Path(d) / "learned.json")
            self.use(state=state)
            self.chat(SYS_A)
            self.chat(SYS_B)
            self.assertEqual(len(json.loads(Path(state).read_text())["learned"]), 2)
            self.use(state=state)                                           # a new process
            self.assertEqual(self.chat(SYS_A)[0]["content"], STABLE)
            self.use(state=state, ignore=[r"Conversation"])                 # ignore also drops a stored shape
            self.assertIn("Conversation Date & Time", self.chat(SYS_A)[0]["content"])

    def test_bad_settings_are_refused(self):
        for bad in ({"learn": "yes"}, {"place": "middle"}, {"patterns": "x"}, {"ignore": ["("]}, {"state": ""}):
            with self.assertRaises(ValueError):
                fe.SystemToUser(**bad)

    def test_off_no_system_and_no_user_change_nothing(self):
        fe.SYSTEM_TO_USER = None
        self.assertEqual(self.chat(SYS_A)[0]["content"], SYS_A)
        self.use(patterns=[r"^Current Date & Time: "])
        only_system = [{"role": "system", "content": SYS_A}]
        self.assertIs(fe._system_to_user(only_system), only_system)
        no_system = [{"role": "user", "content": "hi"}]
        self.assertIs(fe._system_to_user(no_system), no_system)


if __name__ == "__main__":
    unittest.main()
