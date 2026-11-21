import unittest

import torch

from runtime_memory import memory_snapshot, resolve_device


class RuntimeDeviceTests(unittest.TestCase):
    def test_cpu_device_is_explicit(self):
        self.assertEqual(resolve_device("cpu"), ("cpu", "CPU explicitly selected"))

    def test_auto_device_reports_why_cpu_was_selected(self):
        device, reason = resolve_device("auto")
        if torch.cuda.is_available():
            self.assertEqual(device, "cuda:0")
        else:
            self.assertEqual(device, "cpu")
            self.assertIn("CUDA unavailable", reason)

    def test_explicit_cuda_never_silently_falls_back(self):
        if not torch.cuda.is_available():
            with self.assertRaisesRegex(RuntimeError, "CUDA is unavailable"):
                resolve_device("cuda")

    def test_cpu_memory_snapshot_is_unavailable_not_zero(self):
        self.assertIsNone(memory_snapshot("cpu"))


if __name__ == "__main__":
    unittest.main()
