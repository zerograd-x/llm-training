"""CPU-only regression checks for the GPU suite's result/exit-code contract."""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("gpu_driver", Path(__file__).with_name("gpu_validation.py"))
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)


class DriverTests(unittest.TestCase):
    def run_fake(self, code):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            worker = root / "worker.py"
            worker.write_text(code)
            return driver.run_phase(worker, "lora_bf16", "baseline", root, timeout=5)

    def test_semantic_failure_with_zero_exit_is_failure(self):
        result = self.run_fake("import pathlib,sys\np=pathlib.Path(sys.argv[3])/sys.argv[1]/(sys.argv[2]+'.json')\np.write_text('{\"ok\": false}')\n")
        self.assertEqual(result["returncode"], 0)
        self.assertFalse(result["ok"])

    def test_missing_result_is_failure(self):
        self.assertFalse(self.run_fake("pass\n")["ok"])

    def test_nonzero_exit_cannot_pass(self):
        result = self.run_fake("import pathlib,sys\np=pathlib.Path(sys.argv[3])/sys.argv[1]/(sys.argv[2]+'.json')\np.write_text('{\"ok\": true}')\nsys.exit(2)\n")
        self.assertFalse(result["ok"])

    def test_success_requires_true_result_and_zero_exit(self):
        result = self.run_fake("import pathlib,sys\np=pathlib.Path(sys.argv[3])/sys.argv[1]/(sys.argv[2]+'.json')\np.write_text('{\"ok\": true}')\n")
        self.assertTrue(result["ok"])

    def test_verify_removes_distributed_environment(self):
        with patch.dict(os.environ, {"WORLD_SIZE": "2", "RANK": "1", "ACCELERATE_USE_DEEPSPEED": "true"}):
            env = driver.phase_environment("lora_zero3", "verify", Path("unused"))
        self.assertNotIn("WORLD_SIZE", env)
        self.assertNotIn("ACCELERATE_USE_DEEPSPEED", env)

    def test_offload_rejects_inherited_cuda_check_bypass(self):
        with patch.dict(os.environ, {"DS_SKIP_CUDA_CHECK": "1"}):
            with self.assertRaisesRegex(RuntimeError, "DS_SKIP_CUDA_CHECK"):
                driver.phase_environment("offload", "baseline", Path("unused"))


if __name__ == "__main__":
    unittest.main()
