from __future__ import annotations

import cv2
import h5py
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModel, CLIPModel, CLIPProcessor

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


_dino_id = "facebook/dinov2-base"
_proc = AutoImageProcessor.from_pretrained(_dino_id)
_dino = AutoModel.from_pretrained(_dino_id).to(DEVICE).eval()


_clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE).eval()
_clip_p = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
CATS = {
    "person": ("a photo containing a person", "a photo of an empty scene with no people"),
    "vehicle": ("a photo containing a car or truck", "a photo of a street with no vehicles"),
    "animal": ("a photo containing an animal", "a photo of a scene with no animals"),
}
T_GEN, T_REF = 0.50, 0.30


def read_mp4(p, n):
    c = cv2.VideoCapture(p)
    out = []
    while len(out) < n:
        ok, im = c.read()
        if not ok:
            break
        out.append(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    c.release()
    return out


def h5_frames(path, scene, split, stride=1, max_n=None):
    with h5py.File(path, "r") as f:
        vc = f[scene][f"{split}/video_clip"]
        idx = list(range(0, len(vc), stride))
        if max_n is not None:
            idx = idx[:max_n]
        return [
            cv2.cvtColor(
                cv2.imdecode(np.frombuffer(vc[i], np.uint8), cv2.IMREAD_COLOR),
                cv2.COLOR_BGR2RGB,
            )
            for i in idx
        ]


@torch.no_grad()
def dino_cls(frames, bs=16):

    if not frames:
        return torch.zeros(0, 768, device=DEVICE)
    outs = []
    for i in range(0, len(frames), bs):
        batch = frames[i : i + bs]
        inp = _proc(images=batch, return_tensors="pt")
        inp = {k: v.to(DEVICE) for k, v in inp.items()}
        h = _dino(**inp).last_hidden_state[:, 0]
        outs.append(F.normalize(h, dim=-1))
    return torch.cat(outs, dim=0)


def appear_distance(gen_cls, gt_cls):
    n = min(gen_cls.shape[0], gt_cls.shape[0])
    if n == 0:
        return float("nan")
    sim = (gen_cls[:n] * gt_cls[:n]).sum(dim=-1)
    return float((1.0 - sim).mean().cpu())


def _norm(x):
    return x / x.norm(dim=-1, keepdim=True)


@torch.no_grad()
def clip_text_bank():
    out = {}
    dummy = np.zeros((64, 64, 3), np.uint8)
    for k, (pos, neg) in CATS.items():
        e = _clip(
            **_clip_p(
                text=[pos, neg],
                images=[dummy],
                return_tensors="pt",
                padding=True,
            ).to(DEVICE)
        ).text_embeds
        out[k] = _norm(e)
    return out


@torch.no_grad()
def clip_probs(frames, bank, bs=8):
    feats = []
    for i in range(0, len(frames), bs):
        e = _clip(
            **_clip_p(
                text=["x"],
                images=frames[i : i + bs],
                return_tensors="pt",
                padding=True,
            ).to(DEVICE)
        ).image_embeds
        feats.append(_norm(e))
    Fv = torch.cat(feats)
    return {
        k: ((Fv @ v.T) * 100).softmax(-1)[:, 0].cpu().numpy() for k, v in bank.items()
    }


def halluc_rate(frames, known, bank):
    if not frames:
        return float("nan"), 0
    pa = clip_probs(frames, bank)
    m = len(frames)
    any_hall = np.zeros(m, dtype=bool)
    for k in CATS:
        if known[k]:
            continue
        any_hall |= pa[k][:m] > T_GEN
    tot = int(any_hall.sum())
    return tot / m, tot
