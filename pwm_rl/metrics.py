from __future__ import annotations

from dataclasses import dataclass


METRIC_HIGHER_BETTER = {"lpips": False, "psnr": True, "ssim": True, "dino": True,
                        "fvd": False, "drift": False,
                        "action_follow": True, "smoothness": True}


_LPIPS_RES = 224


@dataclass
class MetricModels:
    lpips: object | None
    device: str
    inception: object | None = None   
    raft: object | None = None        


def load_metric_models(device: str = "cuda", lpips_net: str = "alex",
                       with_fid: bool = True, with_raft: bool = True) -> MetricModels:

    import torch

    lp = None
    try:
        import lpips as _lpips  
        lp = _lpips.LPIPS(net=lpips_net).to(device).eval()
        for p in lp.parameters():
            p.requires_grad_(False)
    except Exception as e:  
        print(f"[metrics] LPIPS unavailable ({type(e).__name__}: {e}); "
              f"falling back to PSNR/SSIM only. `pip install lpips` to enable.")

    inception = None
    if with_fid:
        try:
            from torchvision.models import Inception_V3_Weights, inception_v3
            m = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, aux_logits=True)
            m.fc = torch.nn.Identity()  
            inception = m.to(device).eval()
            for p in inception.parameters():
                p.requires_grad_(False)
        except Exception as e:  
            print(f"[metrics] InceptionV3 (FID) unavailable ({type(e).__name__}: {e}).")

    raft = None
    if with_raft:
        try:
            from torchvision.models.optical_flow import Raft_Small_Weights, raft_small
            raft = raft_small(weights=Raft_Small_Weights.DEFAULT, progress=False).to(device).eval()
            for p in raft.parameters():
                p.requires_grad_(False)
        except Exception as e:  
            print(f"[metrics] RAFT (action-follow) unavailable ({type(e).__name__}: {e}).")

    return MetricModels(lpips=lp, device=device, inception=inception, raft=raft)


def psnr(gen, gt, eps: float = 1e-8) -> float:

    import torch

    mse = ((gen - gt) ** 2).mean(dim=(1, 2, 3)).clamp(min=eps)  
    return float((-10.0 * torch.log10(mse)).mean())


def ssim(gen, gt, c1=0.01**2, c2=0.03**2) -> float:

    import torch.nn.functional as F

    win, pad = 7, 3
    mu_a = F.avg_pool2d(gen, win, 1, pad)
    mu_b = F.avg_pool2d(gt, win, 1, pad)
    sa = F.avg_pool2d(gen * gen, win, 1, pad) - mu_a**2
    sb = F.avg_pool2d(gt * gt, win, 1, pad) - mu_b**2
    sab = F.avg_pool2d(gen * gt, win, 1, pad) - mu_a * mu_b
    s = ((2 * mu_a * mu_b + c1) * (2 * sab + c2)) / ((mu_a**2 + mu_b**2 + c1) * (sa + sb + c2))
    return float(s.clamp(-1, 1).mean())


def lpips_score(gen, gt, lpips_model) -> float | None:

    if lpips_model is None:
        return None
    import torch
    import torch.nn.functional as F

    a = F.interpolate(gen, size=(_LPIPS_RES, _LPIPS_RES), mode="bilinear", align_corners=False)
    b = F.interpolate(gt, size=(_LPIPS_RES, _LPIPS_RES), mode="bilinear", align_corners=False)
    a, b = a * 2 - 1, b * 2 - 1  
    with torch.no_grad():
        d = lpips_model(a.float(), b.float())  
    return float(d.mean())


def inception_features(frames, inception):

    if inception is None:
        return None
    import torch
    import torch.nn.functional as F

    x = F.interpolate(frames, size=(299, 299), mode="bilinear", align_corners=False)
    mean = torch.tensor((0.485, 0.456, 0.406), device=x.device).view(1, 3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225), device=x.device).view(1, 3, 1, 1)
    x = (x - mean) / std
    with torch.no_grad():
        f = inception(x.float())  
    return f.detach().cpu().numpy()


def frechet_distance(feat_a, feat_b) -> float:

    import numpy as np
    from scipy import linalg

    mu_a, mu_b = feat_a.mean(0), feat_b.mean(0)
    cov_a = np.cov(feat_a, rowvar=False)
    cov_b = np.cov(feat_b, rowvar=False)
    diff = mu_a - mu_b
    covmean, _ = linalg.sqrtm(cov_a @ cov_b, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(cov_a + cov_b - 2 * covmean))


def action_follow(gen, gt, raft) -> float | None:

    if raft is None:
        return None
    import torch
    import torch.nn.functional as F

    def mvec(fr):
        a = F.interpolate(fr[:-1].float(), size=(224, 224), mode="bilinear", align_corners=False) * 2 - 1
        b = F.interpolate(fr[1:].float(), size=(224, 224), mode="bilinear", align_corners=False) * 2 - 1
        with torch.no_grad():
            fl = raft(a, b)[-1]
        return float(fl[:, 0].mean()), float(fl[:, 1].mean())

    ug, vg = mvec(gen)
    ut, vt = mvec(gt)
    ng, nt = (ug * ug + vg * vg) ** 0.5, (ut * ut + vt * vt) ** 0.5
    if nt < 0.5:
        return 1.0 if ng < 0.5 else float(max(-1.0, 1.0 - ng))
    if ng < 1e-3:
        return -1.0
    return float(max(-1.0, min(1.0, (ug * ut + vg * vt) / (ng * nt + 1e-6))))


def temporal_smoothness(gen) -> float:

    import torch

    if gen.shape[0] < 3:
        return 1.0
    acc = (gen[2:] - 2 * gen[1:-1] + gen[:-2]).abs().mean()
    return float(max(-1.0, min(1.0, 1.0 - float(acc) / 0.05)))


def compute_chunk_metrics(gen, gt, models: MetricModels) -> dict:

    return {
        "lpips": lpips_score(gen, gt, models.lpips),
        "psnr": psnr(gen, gt),
        "ssim": ssim(gen, gt),
        "action_follow": action_follow(gen, gt, models.raft),
        "smoothness": temporal_smoothness(gen),
    }


def start_end_contrast(frames, inception, frac: float = 0.15) -> float | None:

    if inception is None:
        return None
    import numpy as np

    T = frames.shape[0]
    k = max(1, int(T * frac))
    fe = inception_features(frames[:k], inception).mean(0)
    fl = inception_features(frames[-k:], inception).mean(0)
    cos = float(fe @ fl / (np.linalg.norm(fe) * np.linalg.norm(fl) + 1e-8))
    return 1.0 - cos
