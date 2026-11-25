"""Synthetic failure-path checks for the VS/GS subgroup probes."""
import contextlib
import importlib.util
import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "runtime" / "test" / "shaders" / "w64_graphics.py"
spec = importlib.util.spec_from_file_location("w64_graphics", SCRIPT)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class ProbeFailureTests(unittest.TestCase):
    def run_fixture(self, name, failed_width, diagnostic, probe_lane=63):
        """Run one fixture with `_draw_heap` stubbed. `probe_lane` is the largest lane id the wave-size probe writes:
        32..63 for a stage that really ran 64-wide, below 32 for a 32-wide one (MoltenVK/Metal reports 31)."""
        stage = name[:2]
        count = runner.VS_VERTICES if stage == "vs" else runner.VS_VERTICES // 3
        records = [runner.FIXTURES[name][1](i) for i in range(count)]

        def draw_heap(tools, spirv, fixture, size, tmp):
            if fixture == runner.PROBES[stage]:
                if size == failed_width:
                    return None, diagnostic
                return [(probe_lane, 0, 0, 0) for _ in range(count)], None
            return records, None

        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, clear=True), \
                patch.object(runner, "_draw_heap", side_effect=draw_heap), \
                patch.dict(runner._applied, {}, clear=True), patch.dict(runner._measured, {}, clear=True):
            spirv = Path(folder, "spirv")
            spirv.mkdir()
            (spirv / f"{name}.spv").touch()
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                status = runner.main([folder, name])
        return status, output.getvalue()

    def test_probe_dispatch_failure_is_not_a_pass(self):
        for name, width in (("vs_w64_a_skip", 64), ("gs_w64_a_ring", 32)):
            with self.subTest(fixture=name, width=width):
                status, output = self.run_fixture(name, width, "exit 1: synthetic dispatch failure")
                self.assertEqual(status, 1, output)
                self.assertIn(f"{width}-wide {name[:2]} subgroup probe", output)
                self.assertIn("synthetic dispatch failure", output)

    def test_missing_probe_heap_is_not_a_pass(self):
        status, output = self.run_fixture("vs_w64_a_skip", 64, "exit 0: no heap dump")
        self.assertEqual(status, 1, output)
        self.assertIn("no heap dump", output)

    def test_unsupported_probe_keeps_default_draw(self):
        status, output = self.run_fixture("vs_w64_a_skip", 64, "unsupported: no geometry stage")
        self.assertEqual(status, 0, output)
        self.assertIn("  ok", output)

    def test_successful_probe_without_width_change_keeps_default_draw(self):
        status, output = self.run_fixture("vs_w64_a_skip", None, "")
        self.assertEqual(status, 0, output)
        self.assertIn("  ok", output)

    def test_a_stage_measured_32_wide_skips_instead_of_failing(self):
        # MoltenVK/Metal has one subgroup size (32) and ignores RADV_PERFTEST, so the 64-wide words a vs_/gs_ fixture
        # derives cannot be produced there. That is a device the comparison cannot run on, not a fixture failure
        # (TEST-1968): the skip names the width the probe measured.
        for name, stage in (("vs_w64_a_skip", "vertex"), ("gs_w64_a_ring", "geometry")):
            with self.subTest(fixture=name):
                status, output = self.run_fixture(name, None, "", probe_lane=31)
                self.assertEqual(status, 0, output)
                self.assertIn(f"SKIP (the {stage} stage ran a 32-wide subgroup", output)
                self.assertIn("derived for 64-lane waves", output)
                self.assertNotIn("differ from the derived words", output)

    def test_a_stage_measured_64_wide_still_compares(self):
        for name in ("vs_w64_a_skip", "gs_w64_a_ring"):
            with self.subTest(fixture=name):
                status, output = self.run_fixture(name, None, "", probe_lane=63)
                self.assertEqual(status, 0, output)
                self.assertIn("  ok", output)
                self.assertNotIn("SKIP (the", output)


if __name__ == "__main__":
    unittest.main()
