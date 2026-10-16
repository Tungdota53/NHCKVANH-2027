import subprocess
import sys
import unittest

import psutil

from yolo_tuner.runner import stop_process_tree


class ProcessCleanupTests(unittest.TestCase):
    def test_cleanup_kills_only_spawned_worker_and_descendant(self):
        child_code = "import time; time.sleep(60)"
        code = ("import subprocess,sys,time; "
                f"child=subprocess.Popen([sys.executable,'-c',{child_code!r}]); "
                "print(child.pid,flush=True); time.sleep(60)")
        process = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True)
        try:
            child_pid = int(process.stdout.readline())
            child = psutil.Process(child_pid)
            stop_process_tree(process)
            self.assertIsNotNone(process.poll())
            self.assertFalse(child.is_running() and child.status() != psutil.STATUS_ZOMBIE)
            self.assertTrue(psutil.Process().is_running())
        finally:
            stop_process_tree(process)
            process.stdout.close()


if __name__ == "__main__":
    unittest.main()
