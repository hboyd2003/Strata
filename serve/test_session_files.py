"""Session files with their delta K/V logs (STRATA_SESSION_DELTA): removal and the auto-session orphan sweep."""
import os
import tempfile
import unittest
from types import SimpleNamespace

from serve import session_files
from serve.server import Service


def touch(d, name):
    with open(os.path.join(d, name), "wb") as f:
        f.write(b"x")


class SessionFiles(unittest.TestCase):
    def test_remove_takes_the_logs_and_only_them(self):
        with tempfile.TemporaryDirectory() as d:
            for n in ("s.bin", "s.bin.0123456789abcdef.kv", "s.bin.fedcba9876543210.kv", "s.bin.x.kv",
                      "s.bin2.0123456789abcdef.kv", "t.bin"):
                touch(d, n)
            session_files.remove(os.path.join(d, "s.bin"))
            self.assertEqual(sorted(os.listdir(d)), ["s.bin.x.kv", "s.bin2.0123456789abcdef.kv", "t.bin"])

    def test_a_log_without_its_head_goes_too(self):
        with tempfile.TemporaryDirectory() as d:
            touch(d, "h.session.0123456789abcdef.kv")
            session_files.remove(os.path.join(d, "h.session"))
            self.assertEqual(os.listdir(d), [])

    def test_auto_session_orphan_logs_are_swept(self):
        with tempfile.TemporaryDirectory() as d:
            for n in ("auto-a.bin", "auto-a.bin.0123456789abcdef.kv", "auto-b.bin.0123456789abcdef.kv",
                      "mine.bin.0123456789abcdef.kv"):
                touch(d, n)
            svc = SimpleNamespace(slot_save_path=d, _DELTA_LOG=Service._DELTA_LOG)
            self.assertEqual(Service._remove_orphan_auto_logs(svc), 1)
            self.assertEqual(sorted(os.listdir(d)),
                             ["auto-a.bin", "auto-a.bin.0123456789abcdef.kv", "mine.bin.0123456789abcdef.kv"])


if __name__ == "__main__":
    unittest.main()
