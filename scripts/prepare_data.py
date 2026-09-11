import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

import h5py
import numpy as np

TRAIN_STARTS = [0, 9, 18, 27, 36, 45]
TEST_STARTS = [0, 29, 58, 87, 116]


def validate_scene(path, scene):
    with h5py.File(path, "r") as f:
        root = f[scene]
        for split, n, starts in (("train", 80, TRAIN_STARTS), ("test", 145, TEST_STARTS)):
            g = root[split]
            if len(g["video_clip"]) != n or g["poses"].shape != (n, 4, 4):
                raise ValueError(f"{split}: expected {n} frames and poses ({n},4,4)")
            if g["chunk_start_frames"][:].tolist() != starts:
                raise ValueError(f"{split}: unexpected window starts")
            if g["chunk_end_frames_exclusive"][:].tolist() != [s + 29 for s in starts]:
                raise ValueError(f"{split}: expected 29-frame caption windows")
            for key in ("chunk_captions", "chunk_keys", "chunk_mouse"):
                if len(g[key]) != len(starts):
                    raise ValueError(f"{split}/{key}: window count mismatch")
            if any(not s.strip() for s in g["chunk_captions"].asstr()[:]):
                raise ValueError(f"{split}: empty caption")


def prepare(scene_dir, destination):
    scene_dir, destination = Path(scene_dir).resolve(), Path(destination).resolve()
    scene = scene_dir.name
    source = scene_dir / "data.h5"
    validate_scene(source, scene)
    if destination.exists():
        raise FileExistsError(f"Output already exists: {destination}; choose a new directory")
    destination.parent.mkdir(parents=True, exist_ok=True)
    string_dtype = h5py.string_dtype("utf-8")
    with tempfile.TemporaryDirectory(prefix=".pwm-prepare-", dir=destination.parent) as tmp:
        tmp = Path(tmp)
        with h5py.File(source, "r") as src, h5py.File(tmp / "scene.h5", "w") as dst, \
                h5py.File(tmp / "scene_natcap.h5", "w") as side:
            src.copy(scene, dst)
            for split, starts in (("train", TRAIN_STARTS), ("test", TEST_STARTS)):
                original, group = src[scene][split], dst[scene][split]
                n = len(group["video_clip"])
                caps = original["chunk_captions"].asstr()[:].tolist()
                grid = np.arange(n // 9, dtype=np.int64) * 9

                indices = [min(i // 3, 4) if split == "test" else min(i, 5)
                           for i in range(len(grid))]
                for field in ("chunk_captions", "chunk_keys", "chunk_mouse"):
                    values = original[field].asstr()[:].tolist()
                    del group[field]
                    group.create_dataset(field, data=[values[i] for i in indices], dtype=string_dtype)
                for field, values in (("chunk_start_frames", grid),
                                      ("chunk_end_frames_exclusive", grid + 9)):
                    del group[field]
                    group.create_dataset(field, data=values)
                side.require_group(f"{scene}/{split}").create_dataset(
                    "native_captions", data=caps, dtype=string_dtype)
                if split == "train":
                    (tmp / "trainwin_caps.json").write_text(
                        json.dumps(dict(zip(map(str, starts), caps)), ensure_ascii=False, indent=2) + "\n")
            dst.attrs["format"] = "PWM sample 9-frame caption index grid"
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        metadata = {"scene": scene, "source_sha256": digest, "train_frames": 80,
                    "test_frames": 145, "train_starts": TRAIN_STARTS,
                    "test_starts": TEST_STARTS, "caption_map": [0, 3, 6, 9, 12]}
        (tmp / "scene.json").write_text(json.dumps(metadata, indent=2) + "\n")

        os.rename(tmp, destination)
    print(f"Prepared {scene}: train80 / test145; original captions and frames preserved.")
    return destination


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scene-dir", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    a = p.parse_args()
    prepare(a.scene_dir, a.out)
