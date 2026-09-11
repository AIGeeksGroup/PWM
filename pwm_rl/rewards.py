from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np


_FEAT_RES = 224
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


DEFAULT_WEIGHTS = {"gt": 0.45, "cross": 0.25, "dyn": 0.30, "geo": 0.0,
                   "action": 0.0, "temporal": 0.0, "quality": 0.0, "lpips": 0.0,
                   "motion": 0.0, "sharp": 0.0, "qual": 0.0, "precision": 0.0}


SHARP_WEIGHTS = {"sharp": 0.35, "gt": 0.25, "action": 0.20, "temporal": 0.10,
                 "cross": 0.10, "dyn": 0.0, "quality": 0.0, "lpips": 0.0,
                 "geo": 0.0, "motion": 0.0}


DOC_V1_WEIGHTS = {"cross": 0.30, "temporal": 0.30, "motion": 0.20, "quality": 0.20,
                  "gt": 0.0, "dyn": 0.0, "action": 0.0, "lpips": 0.0, "geo": 0.0}


MATURE_WEIGHTS = {"gt": 0.40, "cross": 0.15, "dyn": 0.10, "geo": 0.0,
                  "action": 0.20, "temporal": 0.10, "quality": 0.05}


LPIPS_WEIGHTS = {"lpips": 0.40, "gt": 0.20, "cross": 0.05, "dyn": 0.0, "geo": 0.0,
                 "action": 0.20, "temporal": 0.10, "quality": 0.05}


R_GT_CLS_W = 0.3


@dataclass
class RewardModels:
    dino: object
    raft: object
    depth: object | None
    device: str
    lpips: object | None = None
    clip: object | None = None          
    clip_tok: object | None = None       
    qual: object | None = None           
    scene_bank: object | None = None     


def load_reward_models(
    device: str = "cuda",
    dino_id: str = "facebook/dinov2-small",
    depth_id: str | None = None,  
    lpips_net: str | None = "alex",  
    clip_name: str | None = None,    
    qual_metric: str | None = None,  
) -> RewardModels:

    import torch
    from transformers import Dinov2Model
    from torchvision.models.optical_flow import Raft_Small_Weights, raft_small

    dino = Dinov2Model.from_pretrained(dino_id).to(device).eval()
    for p in dino.parameters():
        p.requires_grad_(False)

    raft = raft_small(weights=Raft_Small_Weights.DEFAULT, progress=False).to(device).eval()
    for p in raft.parameters():
        p.requires_grad_(False)

    depth = None
    if depth_id is not None:
        from transformers import AutoModelForDepthEstimation

        depth = AutoModelForDepthEstimation.from_pretrained(depth_id).to(device).eval()
        for p in depth.parameters():
            p.requires_grad_(False)

    lp = None
    if lpips_net is not None:
        try:
            import lpips as _lpips  

            lp = _lpips.LPIPS(net=lpips_net).to(device).eval()
            for p in lp.parameters():
                p.requires_grad_(False)
        except Exception as e:  
            print(f"[rewards] LPIPS load failed ({e}); R_lpips disabled")
            lp = None

    clip_m, clip_tok = None, None
    if clip_name is not None:
        try:
            import clip as _clip  

            cm, _ = _clip.load(clip_name, device=device)
            cm.eval()
            for p in cm.parameters():
                p.requires_grad_(False)
            clip_m, clip_tok = cm, _clip.tokenize
        except Exception as e:
            print(f"[rewards] CLIP load failed ({e}); R_clip disabled")

    qm = None
    if qual_metric is not None:
        try:
            import pyiqa

            qm = pyiqa.create_metric(qual_metric, device=device)
            qm.eval() if hasattr(qm, "eval") else None
        except Exception as e:
            print(f"[rewards] pyiqa '{qual_metric}' load failed ({e}); R_qual disabled")

    return RewardModels(dino=dino, raft=raft, depth=depth, device=device, lpips=lp,
                        clip=clip_m, clip_tok=clip_tok, qual=qm)


def _to_feat(frames):

    import torch
    import torch.nn.functional as F

    x = F.interpolate(frames, size=(_FEAT_RES, _FEAT_RES), mode="bilinear", align_corners=False)
    mean = torch.tensor(_IMAGENET_MEAN, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    std = torch.tensor(_IMAGENET_STD, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return (x - mean) / std


def _dino_feats(frames, dino):

    import torch

    with torch.no_grad():
        out = dino(pixel_values=_to_feat(frames).to(next(dino.parameters()).dtype))
    return out.last_hidden_state[:, 0]  


def _dino_cls_patch(frames, dino):

    import torch

    with torch.no_grad():
        out = dino(pixel_values=_to_feat(frames).to(next(dino.parameters()).dtype))
    h = out.last_hidden_state
    return h[:, 0], h[:, 1:]  


def _raft_flow(frame_a, frame_b, raft):

    import torch
    import torch.nn.functional as F

    a = F.interpolate(frame_a.float(), size=(_FEAT_RES, _FEAT_RES), mode="bilinear", align_corners=False)
    b = F.interpolate(frame_b.float(), size=(_FEAT_RES, _FEAT_RES), mode="bilinear", align_corners=False)
    a, b = a * 2 - 1, b * 2 - 1  
    with torch.no_grad():
        flow = raft(a, b)[-1]  
    mag = torch.sqrt(flow[:, 0] ** 2 + flow[:, 1] ** 2)  
    return mag.mean(dim=(1, 2))  


def _agg(per_frame):

    import torch

    t = per_frame if torch.is_tensor(per_frame) else torch.as_tensor(per_frame)
    return 0.7 * t.mean() + 0.3 * t.min()


def compute_r_gt(gen, gt, dino, patch_w=None):

    import torch.nn.functional as F

    cls_g, pat_g = _dino_cls_patch(gen, dino)
    cls_r, pat_r = _dino_cls_patch(gt, dino)
    cls_sim = F.cosine_similarity(cls_g, cls_r, dim=-1)                 
    ps = F.cosine_similarity(pat_g, pat_r, dim=-1)                      
    if patch_w is not None:


        _w = patch_w.to(ps.device, ps.dtype)
        patch_sim = (ps * _w).sum(dim=1) / (_w.sum(dim=1) + 1e-8)       
    else:
        patch_sim = ps.mean(dim=1)                                       
    sims = R_GT_CLS_W * cls_sim + (1.0 - R_GT_CLS_W) * patch_sim        
    return float(_agg(sims))


def build_scene_bank(frames, dino, max_patches=40000):

    import torch
    import torch.nn.functional as F

    _, pat = _dino_cls_patch(frames, dino)               
    bank = F.normalize(pat.reshape(-1, pat.shape[-1]).float(), dim=-1)  
    if bank.shape[0] > max_patches:
        idx = torch.randperm(bank.shape[0], device=bank.device)[:max_patches]
        bank = bank[idx]
    return bank.contiguous()


def compute_r_precision(gen, scene_bank, dino):

    import torch
    import torch.nn.functional as F

    _, pat = _dino_cls_patch(gen, dino)                  
    N, P, D = pat.shape
    g = F.normalize(pat.reshape(-1, D).float(), dim=-1)  
    sup = []
    for i in range(0, g.shape[0], 4096):
        sup.append((g[i:i + 4096] @ scene_bank.T.to(g.dtype)).max(dim=1).values)
    sup = torch.cat(sup).reshape(N, P).mean(dim=1)       
    return float(_agg(sup))


def compute_r_lpips(gen, gt, lpips_model, pix_mask=None):

    import torch
    import torch.nn.functional as F

    a = F.interpolate(gen, size=(_FEAT_RES, _FEAT_RES), mode="bilinear", align_corners=False)
    b = F.interpolate(gt, size=(_FEAT_RES, _FEAT_RES), mode="bilinear", align_corners=False)
    a, b = a * 2 - 1, b * 2 - 1  
    if pix_mask is not None:


        _m = pix_mask.to(a.device, a.dtype)
        if _m.dim() == 3:
            _m = _m.unsqueeze(1)
        a = a * _m
        b = b * _m
    with torch.no_grad():
        d = lpips_model(a.float(), b.float()).view(-1)  
    return float(_agg(1.0 - d))


_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def compute_r_clip(gen, caption, clip_model, clip_tok):

    import torch
    import torch.nn.functional as F

    x = F.interpolate(gen, size=(224, 224), mode="bilinear", align_corners=False)
    mean = torch.tensor(_CLIP_MEAN, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    std = torch.tensor(_CLIP_STD, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    x = (x - mean) / std
    toks = clip_tok([caption], truncate=True).to(x.device)
    with torch.no_grad():
        img_f = clip_model.encode_image(x.to(next(clip_model.parameters()).dtype))
        txt_f = clip_model.encode_text(toks).to(img_f.dtype)
    img_f = F.normalize(img_f, dim=-1)
    txt_f = F.normalize(txt_f, dim=-1)
    return float((img_f @ txt_f.T).squeeze(-1).mean())


def compute_r_qual(gen, qual_model, norm=100.0):

    import torch

    with torch.no_grad():
        s = qual_model(gen.float())            
    return float(s.float().mean()) / norm


def compute_r_cross(gen, prev_last, dino):

    import torch
    import torch.nn.functional as F

    pair = torch.stack([gen[0], prev_last.to(gen.device)], dim=0)  
    f = _dino_feats(pair, dino)
    return float(F.cosine_similarity(f[0:1], f[1:2], dim=-1).squeeze())


def compute_r_dyn(gen, gt, raft):

    m_gen = _raft_flow(gen[:-1], gen[1:], raft).mean()
    m_gt = _raft_flow(gt[:-1], gt[1:], raft).mean()
    rel_err = (abs(float(m_gen) - float(m_gt))) / (float(m_gt) + 1e-3)
    return float(np.clip(1.0 - rel_err, -1.0, 1.0))


def compute_r_motion(gen, raft, norm=10.0):

    mag = float(_raft_flow(gen[:-1], gen[1:], raft).mean())
    return float(np.clip(mag / norm, 0.0, 1.0))


def _mean_flow_vec(frames, raft):

    import torch
    import torch.nn.functional as F

    a = F.interpolate(frames[:-1].float(), size=(_FEAT_RES, _FEAT_RES), mode="bilinear", align_corners=False)
    b = F.interpolate(frames[1:].float(), size=(_FEAT_RES, _FEAT_RES), mode="bilinear", align_corners=False)
    a, b = a * 2 - 1, b * 2 - 1
    with torch.no_grad():
        flow = raft(a, b)[-1]  
    u = flow[:, 0].mean()
    v = flow[:, 1].mean()
    return float(u), float(v)


def _net_yaw_deg(poses):

    P = np.asarray(poses, dtype=np.float64)
    yaw = np.arctan2(P[:, 1, 0], P[:, 0, 0])
    d = np.diff(yaw)
    d = (d + np.pi) % (2 * np.pi) - np.pi
    return float(np.degrees(d.sum()))


def compute_r_action(gen, poses, raft, turn_deg_thresh=2.0, drift_norm=1.0):

    ug, _ = _mean_flow_vec(gen, raft)          
    net_yaw = _net_yaw_deg(poses)              
    if abs(net_yaw) < turn_deg_thresh:

        return float(np.clip(1.0 - abs(ug) / drift_norm, -1.0, 1.0))
    expected_sign = 1.0 if net_yaw > 0 else -1.0   


    mag_conf = min(abs(ug) / 3.0, 1.0)             
    return float(np.clip(np.sign(ug) * expected_sign * mag_conf, -1.0, 1.0))


def compute_r_temporal(gen):

    if gen.shape[0] < 3:
        return 1.0
    acc = float((gen[2:] - 2 * gen[1:-1] + gen[:-2]).abs().mean())  
    vel = float((gen[1:] - gen[:-1]).abs().mean())                  
    ratio = acc / (vel + 1e-4)                                      
    return float(np.clip(1.0 - ratio, -1.0, 1.0))


def compute_r_quality(gen):

    import torch

    bright = gen.mean(dim=(1, 2, 3))               
    var = gen.var(dim=(1, 2, 3))                   

    b_ok = ((bright > 0.05) & (bright < 0.95)).float()
    v_ok = (var > 0.002).float()
    score = (b_ok + v_ok) - 1.0                    
    return float(score.mean())


def compute_r_sharp(gen, blur_ks=3, norm=0.05):

    import torch
    import torch.nn.functional as F

    g = gen.mean(1, keepdim=True)                              
    g = F.avg_pool2d(g, blur_ks, 1, blur_ks // 2)              
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                      device=g.device, dtype=g.dtype).view(1, 1, 3, 3)
    ky = kx.transpose(2, 3)
    gx = F.conv2d(g, kx, padding=1)
    gy = F.conv2d(g, ky, padding=1)
    mag = torch.sqrt(gx * gx + gy * gy + 1e-8).mean()         
    return float(torch.clamp(mag / norm, 0.0, 1.0))


def _ssim(a, b, c1=0.01**2, c2=0.03**2):

    import torch.nn.functional as F

    mu_a, mu_b = F.avg_pool2d(a, 3, 1, 1), F.avg_pool2d(b, 3, 1, 1)
    sa = F.avg_pool2d(a * a, 3, 1, 1) - mu_a**2
    sb = F.avg_pool2d(b * b, 3, 1, 1) - mu_b**2
    sab = F.avg_pool2d(a * b, 3, 1, 1) - mu_a * mu_b
    ssim = ((2 * mu_a * mu_b + c1) * (2 * sab + c2)) / ((mu_a**2 + mu_b**2 + c1) * (sa + sb + c2))
    return ssim.clamp(-1, 1).mean()


def _depth(frame, depth_model):

    import torch
    import torch.nn.functional as F

    H, W = frame.shape[-2:]
    x = _to_feat(frame.unsqueeze(0)).to(next(depth_model.parameters()).dtype)
    with torch.no_grad():
        disp = depth_model(pixel_values=x).predicted_depth  
    disp = F.interpolate(disp.unsqueeze(1), size=(H, W), mode="bilinear", align_corners=False)[0, 0]
    disp = (disp - disp.min()) / (disp.max() - disp.min() + 1e-6)
    return 1.0 / (disp + 0.1)  


def compute_r_geo(gen, poses, K, depth_model, alpha=0.85, scales=(0.5, 0.75, 1.0, 1.5, 2.0, 3.0)):

    import torch

    dev = gen.device
    H, W = gen.shape[-2:]
    K = torch.as_tensor(np.asarray(K), dtype=torch.float32, device=dev)
    Kinv = torch.inverse(K)
    P = torch.as_tensor(np.asarray(poses), dtype=torch.float32, device=dev)  


    ys, xs = torch.meshgrid(torch.arange(H, device=dev), torch.arange(W, device=dev), indexing="ij")
    pix = torch.stack([xs.flatten(), ys.flatten(), torch.ones(H * W, device=dev)], 0).float()

    depths = [_depth(gen[t], depth_model) for t in range(gen.shape[0])]  

    def warp(src_img, depth_tgt, T_tgt2src, s):
        Z = (depth_tgt.flatten() * s).clamp(min=1e-3)
        cam_t = (Kinv @ pix) * Z  
        cam_t_h = torch.cat([cam_t, torch.ones(1, H * W, device=dev)], 0)
        cam_s = (T_tgt2src @ cam_t_h)[:3]  
        proj = K @ cam_s
        u = proj[0] / proj[2].clamp(min=1e-3)
        v = proj[1] / proj[2].clamp(min=1e-3)
        grid = torch.stack([2 * u / (W - 1) - 1, 2 * v / (H - 1) - 1], -1).view(1, H, W, 2)
        import torch.nn.functional as F

        return F.grid_sample(src_img.unsqueeze(0), grid, align_corners=True, padding_mode="border")[0]

    best = None
    for s in scales:
        errs = []
        for t in range(1, gen.shape[0] - 1):
            cand = []
            for tp in (t - 1, t + 1):
                T = torch.inverse(P[tp]) @ P[t]  
                I_warp = warp(gen[tp], depths[t], T, s)
                l1 = (gen[t] - I_warp).abs().mean()
                ssim = _ssim(gen[t].unsqueeze(0), I_warp.unsqueeze(0))
                cand.append(alpha * (1 - ssim) / 2 + (1 - alpha) * l1)
            errs.append(torch.stack(cand).min())  
        total = torch.stack(errs).mean()
        best = total if best is None else torch.minimum(best, total)
    return float(-best)


def compute_chunk_reward(
    gen,
    gt,
    prev_last,
    models: RewardModels,
    poses=None,
    K=None,
    weights: dict | None = None,
    captions=None,
    patch_w=None,
    static_patch=None,
    static_pix=None,
):

    w = {**DEFAULT_WEIGHTS, **(weights or {})}


    if static_patch is not None:
        patch_w = (static_patch if patch_w is None
                   else patch_w * static_patch.to(patch_w.device, patch_w.dtype))


    comp = {
        "cross": compute_r_cross(gen, prev_last, models.dino),
    }
    if w.get("gt", 0.0) != 0.0:
        comp["gt"] = compute_r_gt(gen, gt, models.dino, patch_w=patch_w)

    if w.get("precision", 0.0) != 0.0:
        assert models.scene_bank is not None, "R_precision enabled but scene_bank not built"
        comp["precision"] = compute_r_precision(gen, models.scene_bank, models.dino)
    if w.get("dyn", 0.0) != 0.0:
        comp["dyn"] = compute_r_dyn(gen, gt, models.raft)


    if w.get("motion", 0.0) != 0.0:
        comp["motion"] = compute_r_motion(gen, models.raft)


    if w.get("lpips", 0.0) != 0.0:
        assert models.lpips is not None, "R_lpips enabled but lpips model not loaded"
        comp["lpips"] = compute_r_lpips(gen, gt, models.lpips, pix_mask=static_pix)


    if w.get("action", 0.0) != 0.0:
        assert poses is not None, "R_action enabled but poses (the command) not provided"
        comp["action"] = compute_r_action(gen, poses, models.raft)
    if w.get("temporal", 0.0) != 0.0:
        comp["temporal"] = compute_r_temporal(gen)
    if w.get("quality", 0.0) != 0.0:
        comp["quality"] = compute_r_quality(gen)


    if w.get("sharp", 0.0) != 0.0:
        comp["sharp"] = compute_r_sharp(gen)

    if w.get("qual", 0.0) != 0.0:
        assert models.qual is not None, "R_qual enabled but pyiqa metric not loaded (qual_metric)"
        comp["qual"] = compute_r_qual(gen, models.qual)


    if w.get("clip", 0.0) != 0.0 and captions:
        assert models.clip is not None, "R_clip enabled but CLIP not loaded (clip_name)"
        for ci, cap in enumerate(captions):
            comp[f"clip{ci}"] = compute_r_clip(gen, cap, models.clip, models.clip_tok)
    if w.get("geo", 0.0) != 0.0:
        assert models.depth is not None and poses is not None and K is not None, \
            "R_geo enabled but depth model / poses / K not provided"
        comp["geo"] = compute_r_geo(gen, poses, K, models.depth)
    return {"components": comp, "reward": combine(comp, w)}


def combine(components: dict, weights: dict | None = None) -> float:
    w = {**DEFAULT_WEIGHTS, **(weights or {})}

    nclip = sum(1 for k in components if k.rstrip("0123456789") == "clip" and k != "clip")
    total = 0.0
    for k, v in components.items():
        base = k.rstrip("0123456789")
        if base == "clip" and k != "clip" and nclip > 0:
            total += (w.get("clip", 0.0) / nclip) * v
        else:
            total += w.get(k, 0.0) * v
    return float(total)


def load_chunk_gt(h5_path, scene, k, device="cuda", chunk_len=9, res_hw=(480, 832), offset=0, split=None):

    import cv2
    import h5py
    import torch

    H, W = res_hw
    with h5py.File(h5_path, "r") as f:
        g = f[scene] if split is None else f[scene][split]
        lo = k * chunk_len + offset
        hi = lo + chunk_len
        n = len(g["video_clip"])
        assert lo < n, f"chunk {k} starts at {lo} but scene has {n}"
        hi = min(hi, n)  
        frames = []
        for i in range(lo, hi):
            buf = np.frombuffer(g["video_clip"][i], dtype=np.uint8)
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)  
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if img.shape[:2] != (H, W):
                img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
            frames.append(img)
        gt = np.stack(frames).astype(np.float32) / 255.0  
        poses = np.asarray(g["poses"][lo:hi], dtype=np.float32)  
        K = np.asarray(g["intrinsics"], dtype=np.float32) if "intrinsics" in g else None  
    gt = torch.from_numpy(gt).permute(0, 3, 1, 2).contiguous().to(device)  
    return gt, poses, K
