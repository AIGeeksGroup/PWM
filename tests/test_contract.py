import ast
import importlib.util
import json
import math
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def module_at(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prepare = module_at("prepare", ROOT / "scripts/prepare_data.py")
runner = module_at("runner", ROOT / "scripts/run_scene.py")


def isolated_functions(path, names):

    tree = ast.parse(path.read_text())
    body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    if len(body) != len(names):
        raise AssertionError("Missing function")
    namespace = {"math": math}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


class ContractTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.scene_dir = self.root / "gaming1"
        self.scene_dir.mkdir()
        with h5py.File(self.scene_dir / "data.h5", "w") as f:
            for split, n, starts in (("train", 80, prepare.TRAIN_STARTS),
                                      ("test", 145, prepare.TEST_STARTS)):
                g = f.require_group(f"gaming1/{split}")
                vc = g.create_dataset("video_clip", (n,), dtype=h5py.vlen_dtype(np.dtype("uint8")))

                for i in range(n):
                    vc[i] = np.array([i, 1, 2, 3], dtype=np.uint8)
                g.create_dataset("poses", data=np.repeat(np.eye(4)[None], n, axis=0))
                g.create_dataset("chunk_start_frames", data=starts)
                g.create_dataset("chunk_end_frames_exclusive", data=np.array(starts) + 29)
                for field, values in (("chunk_captions", [f"{split} window {i} →" for i in range(len(starts))]),
                                      ("chunk_keys", ["W"] * len(starts)),
                                      ("chunk_mouse", ["·"] * len(starts))):
                    g.create_dataset(field, data=values, dtype=h5py.string_dtype())
        self.out = self.root / "prepared"

    def test_original_readers_get_all_five_captions(self):
        original_bytes = (self.scene_dir / "data.h5").read_bytes()
        prepare.prepare(self.scene_dir, self.out)
        for name in ("sample_5b_natsom.py", "sample_5b_sft.py", "sample_5b_som.py", "sample_5b_nft.py"):
            reader = isolated_functions(ROOT / "sample" / name, {"load_captions_from_h5"})
            with patch.dict(os.environ, {"NATIVE": "1", "CLEAN_CAP": "1"}):
                caps = reader["load_captions_from_h5"](str(self.out / "scene.h5"), "gaming1", "test")
            index = isolated_functions(ROOT / "sample/sample_5b_som.py", {"caption_grid_index"})
            selected = [caps[index["caption_grid_index"](k, 29, len(caps))] for k in range(5)]
            self.assertEqual(selected, [f"test window {i} →" for i in range(5)])
        self.assertEqual(original_bytes, (self.scene_dir / "data.h5").read_bytes())
        windows = json.loads((self.out / "trainwin_caps.json").read_text())
        self.assertEqual(list(windows), list(map(str, prepare.TRAIN_STARTS)))
        self.assertEqual(list(windows.values()), [f"train window {i} →" for i in range(6)])
        with h5py.File(self.out / "scene.h5") as f, h5py.File(self.scene_dir / "data.h5") as src:
            for split in ("train", "test"):
                a, b = f["gaming1"][split], src["gaming1"][split]
                np.testing.assert_array_equal(a["poses"][:], b["poses"][:])
                for i in range(len(a["video_clip"])):
                    np.testing.assert_array_equal(a["video_clip"][i], b["video_clip"][i])

    def test_rejects_overwrite_and_wrong_window(self):
        prepare.prepare(self.scene_dir, self.out)
        with self.assertRaises(FileExistsError):
            prepare.prepare(self.scene_dir, self.out)
        with h5py.File(self.scene_dir / "data.h5", "r+") as f:
            f["gaming1/test/chunk_start_frames"][1] = 28
        with self.assertRaises(ValueError):
            prepare.prepare(self.scene_dir, self.root / "bad")

    def test_commands_and_environment(self):
        for method, expected in (("rl", "fastvideo.sample.sample_5b_natsom"),
                                  ("sft", "fastvideo.sample.sample_5b_sft")):
            cmd = runner.train_command(method, self.out, "gaming1", self.root / "weights", self.root / method)
            self.assertIn(expected, cmd)
            self.assertEqual(cmd[cmd.index("--lora_rank") + 1], "8")
            self.assertEqual(cmd[cmd.index("--w_gt") + 1], "0.50")
            self.assertIn("--win_caps", cmd)
            if method == "sft":
                self.assertEqual(cmd[cmd.index("--sft_updates") + 1], "192")
            else:
                self.assertIn("--native_windows", cmd)
        for seed in (42, 123, 777):
            cmd = runner.infer_command(self.out, "gaming1", self.root, self.root, seed)
            self.assertEqual(cmd[cmd.index("--num_chunks") + 1], "5")
            self.assertNotIn("--lora_ckpt", cmd)
            self.assertNotIn("--eval_real_starts", cmd)
        with patch.dict(os.environ, {"KEEP_PX": "9", "RSTD_FLOOR": "1", "NATIVE": "1",
                                    "ALPHA_RAMP": "9", "HIST_NOISE": "1"}):
            env = runner.runtime_env(self.root)
            self.assertNotIn("NATIVE", env)
            self.assertNotIn("KEEP_PX", env)
            self.assertNotIn("RSTD_FLOOR", env)
            self.assertNotIn("ALPHA_RAMP", env)
            self.assertNotIn("HIST_NOISE", env)
            self.assertEqual(runner.runtime_env(self.root, True)["NATIVE"], "1")

    def test_nft_cli_contract(self):
        import argparse
        tree = ast.parse((ROOT / "sample/sample_5b_nft.py").read_text())
        entry = tree.body[-1]
        namespace = {"argparse": argparse}
        exec(compile(ast.Module(body=entry.body[:-1], type_ignores=[]), "nft_parser", "exec"), namespace)
        cmd = runner.train_command("nft", self.out, "gaming1", self.root / "weights", self.root / "nft")
        args = namespace["p"].parse_args(cmd[4:])
        self.assertEqual(cmd[3], "fastvideo.sample.sample_5b_nft")
        expected = dict(group_size=4, epochs=8, save_every=1, early_stop_patience=99,
                        num_chunks=8, num_euler_timesteps=4, ppo_epochs=4, lr=5e-5,
                        kl_coef=0.0, eps_clip=1e-4, beta=0.5, lora_rank=8, lora_alpha=16,
                        w_gt=0.5, w_lpips=0.3, w_cross=0.1, w_dyn=0.1, gt_offset=0,
                        nft=True, unshare_init=True, accum_windows=True, native_windows=True,
                        score_on_mean=True, nft_r_mode="global", nft_sigma_mode="sampler",
                        nft_anchor=0.0, nft_ema=0.0, nft_beta=0.5, nft_t_draws=2,
                        nft_zscale=2.0, win_starts="0,9,18,27,36,45", nft_per_window=False)
        for key, value in expected.items():
            self.assertEqual(getattr(args, key), value, key)
        setup = module_at("setup", ROOT / "scripts/setup_yume.py")
        self.assertIn("sample_5b_nft.py", setup.SAMPLES)
        with patch.dict(os.environ, {"PWM_SDE_FORM": "flowgrpo", "DIAG_DIV": "1"}):
            env = runner.runtime_env(self.root)
            self.assertNotIn("PWM_SDE_FORM", env)
            self.assertNotIn("DIAG_DIV", env)

    def test_soup_requires_and_averages_all_eight_epochs(self):
        import torch
        target = self.root / "soup"
        target.mkdir()
        for i in range(7):
            torch.save({"adapter": torch.tensor([float(i)])}, target / f"lora_ep{i}.pt")
        with self.assertRaises(RuntimeError):
            runner.make_soup(target)
        torch.save({"adapter": torch.tensor([7.0])}, target / "lora_ep7.pt")
        runner.make_soup(target)
        result = torch.load(target / "lora_soup_all.pt", weights_only=True)
        self.assertEqual(result["adapter"].item(), 3.5)

    def test_improvement_sign(self):
        improved = isolated_functions(ROOT / "evaluation/sgt_score.py", {"is_improved"})["is_improved"]
        self.assertTrue(improved(-0.00001))
        self.assertFalse(improved(0.0))
        self.assertFalse(improved(0.1))
        with self.assertRaises(ValueError):
            improved(float("nan"))


if __name__ == "__main__":
    unittest.main()
