import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (42, 123, 777)


def base_args(prepared, scene, weights):
    return ["--from_h5", "--h5_path", str(prepared / "scene.h5"), "--scene", scene,
            "--pwm_root", str(ROOT), "--ckpt_dir", str(weights),
            "--lora_rank", "8", "--lora_alpha", "16", "--gt_offset", "0",
            "--num_euler_timesteps", "4"]


def train_command(method, prepared, scene, weights, output):
    module = "sample_5b_natsom" if method == "rl" else "sample_5b_sft"
    cmd = [sys.executable, "-u", "-m", f"fastvideo.sample.{module}"]
    cmd += base_args(prepared, scene, weights)
    cmd += ["--mode", "train", "--split", "train", "--num_chunks", "8", "--group_size", "4",
            "--epochs", "8", "--save_every", "1", "--early_stop_patience", "99",
            "--early_stop_min_delta", "0.0", "--ppo_epochs", "4", "--kl_coef", "0.1",
            "--lr", "1e-4", "--beta", "0.5", "--w_gt", "0.50", "--w_lpips", "0.30",
            "--w_cross", "0.10", "--w_dyn", "0.1", "--lora_out", str(output),
            "--win_caps", str(prepared / "trainwin_caps.json")]
    if method == "rl":
        cmd += ["--score_on_mean", "--native_windows", "--win_starts", "0,9,18,27,36,45"]
    else:
        cmd += ["--sft", "--sft_updates", "192"]
    return cmd


def infer_command(prepared, scene, weights, output, seed, lora=None):
    cmd = [sys.executable, "-u", "-m", "fastvideo.sample.sample_5b_som"]
    cmd += base_args(prepared, scene, weights)
    cmd += ["--mode", "infer", "--split", "test", "--num_chunks", "5", "--seed", str(seed),
            "--w_gt", "0.40", "--w_lpips", "0.40", "--w_cross", "0.20", "--w_dyn", "0.0",
            "--infer_out", str(output), "--bench_json", str(output)]
    if lora:
        cmd += ["--lora_ckpt", str(lora)]
    return cmd


def runtime_env(yume, infer=False):
    env = os.environ.copy()

    for name in ("STATIC_MASK", "RSTD_FLOOR", "DETERM", "NATIVE", "KEEP_PX",
                 "STRICT_DET", "CUBLAS_WORKSPACE_CONFIG", "ALPHA_RAMP", "HIST_NOISE"):
        env.pop(name, None)
    env.update(PYTHONPATH=os.pathsep.join((str(yume), str(ROOT))),
               PYTHONNOUSERSITE="1", PYTHONUNBUFFERED="1", CLEAN_CAP="1",
               PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    if infer:
        env.update(DETERM="1", NATIVE="1", CUBLAS_WORKSPACE_CONFIG=":4096:8")
    return env


def run_logged(command, log, cwd, env):
    print(shlex.join(command), flush=True)
    with log.open("x") as f:
        result = subprocess.run(command, cwd=cwd, env=env, stdout=f, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"Command failed ({result.returncode}); see {log}")


def make_soup(directory):
    import torch
    files = [directory / f"lora_ep{i}.pt" for i in range(8)]
    if not all(p.is_file() for p in files):
        raise RuntimeError("Training incomplete: all eight checkpoints are required")
    states = [torch.load(p, map_location="cpu", weights_only=True) for p in files]
    if any(s.keys() != states[0].keys() for s in states[1:]):
        raise ValueError("Checkpoint keys differ")
    soup = {k: sum(s[k].float() for s in states) / 8 for k in states[0]}
    torch.save(soup, directory / "lora_soup_all.pt")


def validate_video(path):
    import cv2
    cap = cv2.VideoCapture(str(path))
    count = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame.shape[:2] != (480, 832):
            cap.release()
            raise ValueError(f"Wrong resolution: {path}")
        count += 1
    cap.release()
    if count != 145:
        raise ValueError(f"Expected 145 decoded frames: {path}, got {count}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("train", "infer", "evaluate"))
    p.add_argument("--method", choices=("rl", "sft"), default="rl")
    p.add_argument("--prepared", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--weights", type=Path, default=ROOT / "weights/Yume-5B-720P")
    p.add_argument("--yume", type=Path, default=ROOT / "third_party/YUME")
    p.add_argument("--lora", type=Path, help="Omit for Native-only inference")
    p.add_argument("--videos", type=Path, help="Inference directory to evaluate")
    p.add_argument("--dry-run", action="store_true", help="Print commands without loading models")
    a = p.parse_args()
    prepared, output, weights, yume = [x.resolve() for x in (a.prepared, a.out, a.weights, a.yume)]
    meta = json.loads((prepared / "scene.json").read_text())
    scene = meta["scene"]
    for name in ("scene.h5", "scene_natcap.h5", "trainwin_caps.json"):
        if not (prepared / name).is_file():
            raise FileNotFoundError(prepared / name)
    if a.action == "train":
        commands = [train_command(a.method, prepared, scene, weights, output)]
    elif a.action == "infer":
        lora = a.lora.resolve() if a.lora else None
        if lora and not lora.is_file() and not a.dry_run:
            raise FileNotFoundError(lora)
        commands = [infer_command(prepared, scene, weights, output / f"s{s}", s, lora) for s in SEEDS]
    else:
        if a.videos is None:
            p.error("evaluate requires --videos")
        commands = [[sys.executable, str(ROOT / "evaluation/sgt_score.py"), "--batch",
                     str(output / "jobs.json"), str(output / "results.json")]]
    if a.dry_run:
        for cmd in commands:
            print(shlex.join(cmd))
        return
    if a.action != "evaluate":
        if not (yume / ".pwm-overlay").is_file():
            raise FileNotFoundError("Run scripts/setup_yume.py before training or inference")
        if not weights.is_dir():
            raise FileNotFoundError(f"Download the Yume weights to {weights}")
    output.mkdir(parents=True, exist_ok=False)
    (output / "commands.json").write_text(json.dumps(commands, indent=2) + "\n")
    env = runtime_env(yume, infer=a.action != "train")
    if a.action == "train":
        run_logged(commands[0], output / "train.log", yume, env)
        if a.method == "sft" and "SFT_TRAIN_LOOP_DONE" not in (output / "train.log").read_text():
            raise RuntimeError("SFT completion marker missing")
        make_soup(output)
        print(f"Training and soup complete: {output}")
    elif a.action == "infer":
        for seed, cmd in zip(SEEDS, commands):
            out = output / f"s{seed}"
            out.mkdir()
            log = out / "infer.log"
            run_logged(cmd, log, yume, env)
            if "[caption-map] generation chunk indices=[0, 3, 6, 9, 12]" not in log.read_text():
                raise RuntimeError(f"Caption mapping not verified: {log}")
            validate_video(out / f"{scene}_test_baseline.mp4")
            if a.lora:
                validate_video(out / f"{scene}_test_lora.mp4")
        print(f"Three-seed inference complete: {output}")
    else:
        seeds = {}
        for s in SEEDS:
            folder = a.videos.resolve() / f"s{s}"
            videos = [folder / f"{scene}_test_lora.mp4", folder / f"{scene}_test_baseline.mp4"]
            for video in videos:
                validate_video(video)
            seeds[str(s)] = list(map(str, videos))
        jobs = {scene: {"h5": str(prepared / "scene.h5"), "scene": scene, "seeds": seeds}}
        (output / "jobs.json").write_text(json.dumps(jobs, indent=2) + "\n")
        env["HORIZON"] = "145"
        run_logged(commands[0], output / "evaluate.log", ROOT, env)
        print((output / "results.json").read_text())


if __name__ == "__main__":
    main()
