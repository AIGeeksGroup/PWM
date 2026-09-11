import argparse
import gc
import json
import math
import os
import sys

import numpy as np
import torch


from fastvideo.sample.sample_5b import (  
    create_scaled_videos,
    get_sampling_sigmas,
    scale,
)

import wan23  
from wan23.configs import WAN_CONFIGS  

LATENT_FRAME_ZERO = 8   
KEEP_LATENT = 3         
KEEP_FRAME = 9          


def caption_grid_index(chunk_idx, keep_frame, n_captions):

    if chunk_idx < 0 or keep_frame <= 0 or n_captions <= 0:
        raise ValueError("chunk_idx>=0, keep_frame>0, and n_captions>0 required")
    return min((chunk_idx * keep_frame) // 9, n_captions - 1)


def sde_mean_std(latent_gen, v_pred_gen, sigma_now, sigma_next, beta):

    dt = sigma_next - sigma_now                      
    mean = latent_gen + dt * v_pred_gen
    std = beta * math.sqrt(abs(dt)) * max(sigma_now, 1e-4) + 1e-4
    return mean, std


def gaussian_logprob(x, mean, std):

    var = std * std
    lp = -0.5 * ((x - mean) ** 2 / var + math.log(2 * math.pi * var))
    return lp.mean()


def grpo_loss(new_logprob, old_logprob, advantage, eps_clip=0.2):
    ratio = torch.exp(new_logprob - old_logprob)
    unclipped = ratio * advantage
    clipped = torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip) * advantage
    return -torch.min(unclipped, clipped)


def build_timestep(sigma_now, mask2, seq_len, device):
    timestep = torch.tensor([sigma_now * 1000]).to(device)
    temp_ts = (mask2[0][0][:-LATENT_FRAME_ZERO, ::2, ::2]).flatten()
    temp_ts = torch.cat([temp_ts, temp_ts.new_ones(seq_len - temp_ts.size(0)) * timestep])
    return temp_ts.unsqueeze(0)


def prepare_first_chunk(wan_i2v, pixel_values_vid, caption0, max_area, device):
    pixel_values_vid = pixel_values_vid.squeeze().permute(1, 0, 2, 3).contiguous().to(device)
    model_input = pixel_values_vid
    model_input = torch.cat([model_input[:, 0].unsqueeze(1).repeat(1, 8, 1, 1), model_input[:, :33]], dim=1)
    frame = model_input.shape[1]
    model_input = torch.cat([
        wan_i2v.vae.encode([model_input.to(device)[:, :-32].to(device)])[0],
        wan_i2v.vae.encode([model_input.to(device)[:, -32:].to(device)])[0],
    ], dim=1)
    img = model_input[:, :-LATENT_FRAME_ZERO]
    with torch.no_grad():
        arg_c, arg_null, noise, mask2, img = wan_i2v.generate(
            caption0, frame_num=frame, max_area=max_area,
            latent_frame_zero=LATENT_FRAME_ZERO, img=img)
    return model_input, arg_c, noise, mask2, img


def prepare_next_chunk(wan_i2v, model_input, caption_k, max_area, device):
    with torch.no_grad():
        arg_c, arg_null, noise, mask2, img = wan_i2v.generate(
            caption_k, frame_num=(model_input.shape[1] - 1) * 4 + 1 + 32,
            max_area=max_area, latent_frame_zero=LATENT_FRAME_ZERO, img=model_input)
        model_input_1 = model_input
    model_input_1 = torch.cat(
        [model_input_1, torch.zeros(48, LATENT_FRAME_ZERO, model_input_1.shape[2],
                                    model_input_1.shape[3]).to(device)], dim=1)
    return arg_c, noise, mask2, img, model_input_1


def rollout_sde(transformer, arg_c, mask2, init_noise, hist_full, sigmas, beta, seed, device):

    g = torch.Generator(device=device).manual_seed(seed)
    seq_len = arg_c["seq_len"]
    latent = init_noise.clone()
    latent = torch.cat([hist_full[:, :-LATENT_FRAME_ZERO], latent[:, -LATENT_FRAME_ZERO:]], dim=1)
    records = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for i in range(len(sigmas)):
            ts = build_timestep(sigmas[i], mask2, seq_len, device)
            v_pred = transformer([latent.squeeze(0)], t=ts, **arg_c)[0]
            sigma_next = sigmas[i + 1] if i + 1 < len(sigmas) else 0.0
            latent_gen = latent[:, -LATENT_FRAME_ZERO:]
            mean, std = sde_mean_std(latent_gen, v_pred[:, -LATENT_FRAME_ZERO:],
                                     float(sigmas[i]), float(sigma_next), beta)
            sample_gen = mean + std * torch.randn(mean.shape, generator=g, device=device, dtype=mean.dtype)
            old_lp = gaussian_logprob(sample_gen.float(), mean.float(), std)
            records.append({
                "latent_before": latent.detach().cpu(),
                "sample_gen": sample_gen.detach().cpu(),
                "sigma_now": float(sigmas[i]),
                "sigma_next": float(sigma_next),
                "old_logprob": old_lp.detach().cpu(),
            })
            latent = torch.cat([hist_full[:, :-LATENT_FRAME_ZERO], sample_gen], dim=1)
            if i + 1 == len(sigmas):


                records[-1]["final_mean_full"] = torch.cat(
                    [hist_full[:, :-LATENT_FRAME_ZERO], mean], dim=1).detach().cpu()
    return latent, records


def load_start_frame_from_h5(h5_path, scene, split, H=480, W=832, total_frames=33, frame_idx=0):

    import cv2
    import h5py
    import torch.nn.functional as F

    with h5py.File(h5_path, "r") as f:
        g = f[scene] if split is None else f[scene][split]
        buf = np.frombuffer(g["video_clip"][frame_idx], dtype=np.uint8)
    img = cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0           
    t = F.interpolate(t.unsqueeze(0), size=(H, W), mode="bilinear", align_corners=False)[0]
    vid = torch.zeros(3, total_frames, H, W)
    vid[:, 0] = (t - 0.5) * 2                                            
    return vid.permute(1, 0, 2, 3).contiguous()                          


def load_captions_from_h5(h5_path, scene, split):

    import h5py

    import os
    import re
    with h5py.File(h5_path, "r") as f:
        g = f[scene] if split is None else f[scene][split]
        caps = [c.decode() if isinstance(c, (bytes, bytearray)) else str(c)
                for c in g["chunk_captions"][()]]


    if os.environ.get("NATIVE") == "1":
        side = h5_path[:-3] + "_natcap.h5"
        if os.path.exists(side):
            with h5py.File(side, "r") as f2:
                nat = [c.decode() if isinstance(c, (bytes, bytearray)) else str(c)
                       for c in f2[scene][split or "test"]["native_captions"][()]]
            caps = [nat[min(i // 3, len(nat) - 1)] for i in range(len(caps))]
            print(f"[native] sidecar captions loaded: {side} ({len(nat)} windows)",
                  flush=True)


    if os.environ.get("CLEAN_CAP") == "1":
        out = []
        for c in caps:
            c2 = re.sub(r'(autonomous vehicle in [^.]*\. ).*?(The vehicle moves through)', r'\1\2', c)
            c2 = re.sub(r'\s*at \d+(\.\d+)? meters per second', '', c2)
            out.append(c2)
        caps = out
        print("[clean_cap] e.g.:", caps[0][:220], flush=True)
    return caps


def emit9(vae, final_latent):

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        video = scale(vae, final_latent[:, -LATENT_FRAME_ZERO:])  
    gen = ((video[:, :KEEP_FRAME] + 1.0) / 2.0).clamp(0, 1)
    return gen.permute(1, 0, 2, 3).float()  


def emit9_grad(vae, final_latent):

    import torch.utils.checkpoint as _cp


    _n_lat = (KEEP_FRAME + 3) // 4
    lat = final_latent[:, -LATENT_FRAME_ZERO:][:, :_n_lat].to(torch.float32)
    def _dec(l):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return vae.decode([l])[0]
    video = _cp.checkpoint(_dec, lat, use_reentrant=False)  
    gen = ((video[:, :KEEP_FRAME] + 1.0) / 2.0).clamp(0, 1)
    return gen.permute(1, 0, 2, 3).float()  


def _dino_feats_grad(frames, dino):
    from pwm_rl.rewards import _to_feat
    out = dino(pixel_values=_to_feat(frames).to(next(dino.parameters()).dtype))
    h = out.last_hidden_state
    return h[:, 0], h[:, 1:]


def r_recon_diff(gen, gt, rm, w_gt, w_lpips):

    import torch.nn.functional as F
    from pwm_rl.rewards import R_GT_CLS_W

    R = gen.new_zeros(())
    if w_gt > 0:
        cg, pg = _dino_feats_grad(gen, rm.dino)
        with torch.no_grad():                 
            cr, pr = _dino_feats_grad(gt, rm.dino)
        cr = cr.detach(); pr = pr.detach()
        cls_sim = F.cosine_similarity(cg, cr, dim=-1)
        pat_sim = F.cosine_similarity(pg, pr, dim=-1).mean(dim=1)
        sims = R_GT_CLS_W * cls_sim + (1.0 - R_GT_CLS_W) * pat_sim
        R = R + w_gt * (0.7 * sims.mean() + 0.3 * sims.min())
    if w_lpips > 0 and rm.lpips is not None:
        a = gen * 2 - 1; b = gt * 2 - 1
        d = rm.lpips(a.to(torch.float32), b.to(torch.float32)).view(-1)
        r = 1.0 - d
        R = R + w_lpips * (0.7 * r.mean() + 0.3 * r.min())
    return R


def rollout_vader(transformer, arg_c, mask2, init_noise, hist_full, sigmas, k_grad, device, kl_coef=0.0):

    latent = init_noise.clone()
    latent = torch.cat([hist_full[:, :-LATENT_FRAME_ZERO], latent[:, -LATENT_FRAME_ZERO:]], dim=1)
    n = len(sigmas)
    kl_total = init_noise.new_zeros(())
    for i in range(n):
        grad_on = (i >= n - k_grad)
        ctx = torch.enable_grad() if grad_on else torch.no_grad()
        with ctx, torch.autocast("cuda", dtype=torch.bfloat16):
            ts = build_timestep(sigmas[i], mask2, arg_c["seq_len"], device)
            if grad_on:
                import torch.utils.checkpoint as _cp
                def _fwd(x):
                    return transformer([x], t=ts, **arg_c)[0]
                v_pred = _cp.checkpoint(_fwd, latent.squeeze(0), use_reentrant=False)
                if kl_coef > 0.0:


                    with torch.no_grad(), transformer.disable_adapter():
                        v_base = transformer([latent.squeeze(0).detach()], t=ts, **arg_c)[0]
                    kl_total = kl_total + ((v_pred[:, -LATENT_FRAME_ZERO:]
                                            - v_base[:, -LATENT_FRAME_ZERO:].detach()) ** 2).mean()
            else:
                v_pred = transformer([latent.squeeze(0)], t=ts, **arg_c)[0]
            s_now = float(sigmas[i]); s_next = float(sigmas[i + 1] if i + 1 < len(sigmas) else 0.0)
            gen = latent[:, -LATENT_FRAME_ZERO:] + (s_next - s_now) * v_pred[:, -LATENT_FRAME_ZERO:]
            latent = torch.cat([hist_full[:, :-LATENT_FRAME_ZERO], gen], dim=1)
        if not grad_on:
            latent = latent.detach()
    return latent, kl_total


def group_advantages(components_list, weights, min_std=1e-3, calibrate=False, rmax=1.0):

    keys = components_list[0].keys()
    adv = [0.0] * len(components_list)


    nclip = sum(1 for k in keys if k.rstrip("0123456789") == "clip" and k != "clip")
    for k in keys:
        base = k.rstrip("0123456789")
        if base == "clip" and k != "clip" and nclip > 0:
            w = weights.get("clip", 0.0) / nclip
        else:
            w = weights.get(k, 0.0)
        if w == 0.0:
            continue
        vals = np.array([c[k] for c in components_list], dtype=np.float64)
        ref = np.append(vals, rmax) if calibrate else vals   
        mu, s = ref.mean(), ref.std()
        if s < min_std:        
            continue
        z = (vals - mu) / (s + 1e-6)
        for i in range(len(adv)):
            adv[i] += w * z[i]
    return adv


def make_caption_views(caption, k):

    c = caption
    act_i = c.find("Person moves")
    if act_i < 0:
        act_i = c.find("Person ")
    num_i = c.find("Actual distance")
    scene = c[:act_i].strip() if act_i > 0 else c
    action = c[act_i:num_i].strip() if (act_i > 0 and num_i > act_i) else c
    nonum = c[:num_i].strip() if num_i > 0 else c
    views = [c, scene, action, nonum]
    out = []
    for v in views[:max(1, k)]:
        out.append(v if v else c)
    return out


def save_mp4(frames, path, fps=16):

    import cv2

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    vid = torch.cat([f.cpu() for f in frames], dim=0)  
    vid = (vid.clamp(0, 1).permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)  
    H, W = vid.shape[1:3]
    w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    for fr in vid:
        w.write(cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
    w.release()
    return path


def run_infer(args, transformer, vae, wan_i2v, pixel_values_vid, chunk_caps,
              reward_models, weights, device, load_chunk_gt, compute_chunk_reward):
    import contextlib

    if os.environ.get("DETERM") == "1":
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception as _e:
            print("[determ] use_deterministic_algorithms:", _e, flush=True)
        print("[determ] deterministic eval ON (CUBLAS_WORKSPACE_CONFIG=%s)"
              % os.environ.get("CUBLAS_WORKSPACE_CONFIG"), flush=True)

    global KEEP_FRAME, KEEP_LATENT
    if os.environ.get("NATIVE") == "1":
        KEEP_FRAME, KEEP_LATENT = 29, 8   
        print("[native] Yume native mode ON: emit29 / commit8 / GT stride 29", flush=True)
    _kp = os.environ.get("KEEP_PX")
    if _kp:  
        KEEP_FRAME = int(_kp); assert KEEP_FRAME % 4 == 1 and 9 <= KEEP_FRAME <= 29
        KEEP_LATENT = (KEEP_FRAME + 3) // 4
        print(f"[keep] custom commit: emit{KEEP_FRAME} / commit{KEEP_LATENT}", flush=True)
    _clen = KEEP_FRAME   
    if args.eval_real_starts and not args.from_h5:
        raise ValueError("--eval_real_starts requires --from_h5")
    if args.eval_real_starts and args.eval_legacy_caption_bug:
        raise ValueError("--eval_real_starts and --eval_legacy_caption_bug are mutually exclusive")

    def _correct_chunk_caption(k):
        return chunk_caps[caption_grid_index(k, KEEP_FRAME, len(chunk_caps))]

    def _chunk_caption(k):
        if args.eval_legacy_caption_bug:
            return chunk_caps[min(k, len(chunk_caps) - 1)]
        return _correct_chunk_caption(k)

    if args.eval_legacy_caption_bug:
        _cap_indices = [min(k, len(chunk_caps) - 1) for k in range(args.num_chunks)]
        print("[diagnostic] eval_legacy_caption_bug ON: reproducing the retired evaluator",
              flush=True)
    else:
        _cap_indices = [caption_grid_index(k, KEEP_FRAME, len(chunk_caps))
                        for k in range(args.num_chunks)]
    print(f"[caption-map] generation chunk indices={_cap_indices}", flush=True)
    if args.eval_real_starts:
        print("[diagnostic] eval_real_starts ON: every chunk starts from its real GT frame",
              flush=True)

    have_lora = bool(args.lora_ckpt)
    if have_lora:
        state = torch.load(args.lora_ckpt, map_location="cpu")
        res = transformer.load_state_dict(state, strict=False)
        print(f"[infer] loaded LoRA {args.lora_ckpt} ({len(state)} tensors, "
              f"{len(res.unexpected_keys)} unexpected)")
    else:
        print("[infer] no --lora_ckpt -> baseline only")
    transformer.eval()
    max_area = 480 * 832
    split_tag = args.split or "full"


    from pwm_rl.metrics import compute_chunk_metrics, load_metric_models
    metric_models = load_metric_models(device=device, lpips_net=args.lpips_net)
    bench_dir = args.bench_json or args.infer_out
    os.makedirs(bench_dir, exist_ok=True)

    _ramp_n = float(os.environ.get("ALPHA_RAMP", "0") or 0)
    _ramp_layers = []
    if _ramp_n > 0 and have_lora:
        from peft.tuners.lora import LoraLayer as _LL
        for _m in transformer.modules():
            if isinstance(_m, _LL):
                _ramp_layers.append((_m, dict(_m.scaling)))
        print(f"[alpha-ramp] ramp over {_ramp_n} chunks, {len(_ramp_layers)} lora layers",
              flush=True)

    def one_pass(tag, adapter_off):


        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        ctx = transformer.disable_adapter() if adapter_off else contextlib.nullcontext()
        frames, gt_frames, comps, mets = [], [], [], []
        with torch.no_grad(), ctx:
            model_input, arg_c, noise, mask2, img = prepare_first_chunk(
                wan_i2v, pixel_values_vid, _chunk_caption(0), max_area, device)
            gt0, _, _ = load_chunk_gt(args.h5_path, args.scene, 0, device=device,
                                      chunk_len=_clen, offset=args.gt_offset, split=args.split)
            prev_last = gt0[0]
            hist_full = img[0]


            _ca = float(getattr(args, "color_anchor", 0.0) or 0.0)
            if _ca > 0:
                def _lp(x):  
                    C, T, H, W = x.shape
                    y = x.reshape(C * T, 1, H, W)
                    y = torch.nn.functional.avg_pool2d(y, 5, stride=1, padding=2)
                    return y.reshape(C, T, H, W)
                _ref = model_input[:, :1].float()
                _ref_lo = _lp(_ref)
                _ref_mu = _ref_lo.mean(dim=(1, 2, 3), keepdim=True)
                _ref_sd = _ref_lo.std(dim=(1, 2, 3), keepdim=True).clamp_min(1e-6)
            for k in range(args.num_chunks):
                if args.eval_real_starts and k > 0:
                    start_vid = load_start_frame_from_h5(
                        args.h5_path, args.scene, args.split, frame_idx=k * _clen)
                    model_input, arg_c, noise, mask2, img = prepare_first_chunk(
                        wan_i2v, start_vid, _chunk_caption(k), max_area, device)
                    gtk, _, _ = load_chunk_gt(
                        args.h5_path, args.scene, k, device=device,
                        chunk_len=_clen, offset=args.gt_offset, split=args.split)
                    prev_last = gtk[0]
                    hist_full = img[0]
                sigmas = get_sampling_sigmas(args.num_euler_timesteps, 7.0)
                if _ramp_n > 0 and not adapter_off:
                    _f = min(1.0, (k + 1) / _ramp_n)
                    for _lm, _base in _ramp_layers:
                        for _ad, _v in _base.items():
                            _lm.scaling[_ad] = _v * _f


                init_noise = noise
                final_latent, _ = rollout_sde(transformer, arg_c, mask2, init_noise, hist_full,
                                              sigmas, beta=0.0, seed=args.seed + k, device=device)
                if os.environ.get("DEBUG_DET") == "1" and k < 2:
                    print(f"[DET k{k}] noise_sum={float(init_noise.double().sum()):.6f} "
                          f"hist_sum={float(hist_full.double().sum()):.6f} "
                          f"latent_sum={float(final_latent.double().sum()):.6f}", flush=True)
                gen = emit9(vae, final_latent)
                gt, poses, K = load_chunk_gt(args.h5_path, args.scene, k, device=device,
                                             chunk_len=_clen, offset=args.gt_offset, split=args.split)


                gen_s = gen[:gt.shape[0]] if gt.shape[0] < gen.shape[0] else gen

                out = compute_chunk_reward(gen_s, gt, prev_last, reward_models,
                                           poses=poses, K=K, weights=weights,
                                           captions=(make_caption_views(_correct_chunk_caption(k), args.clip_views)
                                                     if args.w_clip > 0 else None))
                comps.append(out["components"])

                met = compute_chunk_metrics(gen_s.to(device), gt.to(device), metric_models)
                mets.append(met)
                frames.append(gen)
                gt_frames.append(gt)
                cs = " ".join(f"{kk}={vv:.3f}" for kk, vv in out["components"].items())
                ms = " ".join(f"{kk}={vv:.3f}" for kk, vv in met.items() if vv is not None)
                print(f"[infer {tag} {split_tag} chunk {k:>2}] R={out['reward']:.3f} | {cs} || {ms}")
                prev_last = gen[KEEP_FRAME - 1]
                if args.eval_real_starts:
                    continue
                _commit = final_latent[:, -LATENT_FRAME_ZERO:][:, :KEEP_LATENT]
                if _ca > 0:
                    _cf = _commit.float()
                    _lo = _lp(_cf); _hi = _cf - _lo
                    _mu = _lo.mean(dim=(1, 2, 3), keepdim=True)
                    _sd = _lo.std(dim=(1, 2, 3), keepdim=True).clamp_min(1e-6)
                    _lo_fix = (_lo - _mu) / _sd * _ref_sd + _ref_mu
                    _fix = _lo_fix + _hi   
                    _commit = (_ca * _fix + (1.0 - _ca) * _cf).to(_commit.dtype)
                _hn = float(os.environ.get("HIST_NOISE", "0") or 0)
                if _hn > 0:


                    _commit = (_commit.float() + _hn * torch.randn_like(_commit.float())).to(_commit.dtype)
                model_input = torch.cat([model_input, _commit], dim=1)
                if k + 1 < args.num_chunks:
                    arg_c, noise, mask2, img, hist_full = prepare_next_chunk(
                        wan_i2v, model_input, _chunk_caption(k + 1), max_area, device)

        def _mean(key, src):
            vals = [c[key] for c in src if c.get(key) is not None]
            return (sum(vals) / len(vals)) if vals else None


        from pwm_rl.metrics import inception_features, start_end_contrast
        gen_vid = torch.cat(frames, dim=0)            
        gt_vid = torch.cat(gt_frames, dim=0)
        drift = start_end_contrast(gen_vid, metric_models.inception)
        drift_gt = start_end_contrast(gt_vid, metric_models.inception)
        if metric_models.inception is not None:
            fg = inception_features(gen_vid.to(device), metric_models.inception)
            fr = inception_features(gt_vid.to(device), metric_models.inception)
            np.save(os.path.join(bench_dir, f"{args.scene}_{split_tag}_{tag}_incfeat_gen.npy"), fg)
            np.save(os.path.join(bench_dir, f"{args.scene}_{split_tag}_{tag}_incfeat_gt.npy"), fr)

        means = {"gt": _mean("gt", comps),       
                 "lpips": _mean("lpips", mets),   
                 "psnr": _mean("psnr", mets),
                 "ssim": _mean("ssim", mets),
                 "action_follow": _mean("action_follow", mets),  
                 "smoothness": _mean("smoothness", mets),        
                 "drift": drift,                  
                 "drift_gt": drift_gt}            
        out_mp4 = os.path.join(args.infer_out, f"{args.scene}_{split_tag}_{tag}.mp4")
        save_mp4(frames, out_mp4, fps=16)
        rec = {"scene": args.scene, "split": split_tag, "tag": tag,
               "num_chunks": args.num_chunks, "means": means,
               "per_chunk": [{**comps[i], **mets[i]} for i in range(len(comps))]}
        out_json = os.path.join(bench_dir, f"{args.scene}_{split_tag}_{tag}.json")
        with open(out_json, "w") as f:
            json.dump(rec, f, indent=2)
        ms = " ".join(f"{k}={v:.4f}" for k, v in means.items() if v is not None)
        print(f"[infer {tag} {split_tag}] {ms}  -> {out_mp4}  (metrics: {out_json})")
        return means


    results = {}
    if have_lora:
        results["lora"] = one_pass("lora", adapter_off=False)
        if not args.no_baseline:
            results["baseline"] = one_pass("baseline", adapter_off=True)
    else:
        results["baseline"] = one_pass("baseline", adapter_off=False)

    if "lora" in results and "baseline" in results:

        lo, ba = results["lora"], results["baseline"]
        if lo.get("lpips") is not None and ba.get("lpips") is not None:
            d = ba["lpips"] - lo["lpips"]  
            print(f"\n[infer {split_tag}] LPIPS  lora={lo['lpips']:.4f}  "
                  f"baseline={ba['lpips']:.4f}  (base-lora={d:+.4f})  "
                  f"{'<- LoRA helps' if d > 0 else '<- no gain'}")
        dg = lo['gt'] - ba['gt']
        print(f"[infer {split_tag}] gt(ref)  lora={lo['gt']:.4f}  "
              f"baseline={ba['gt']:.4f}  (lora-base={dg:+.4f})  [reward metric, reference only]")


def main(args):
    sys.path.insert(0, args.pwm_root)
    from pwm_rl.rewards import compute_chunk_reward, load_chunk_gt, load_reward_models


    if getattr(args, "deterministic", False):
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


        strict = os.environ.get("STRICT_DET") == "1"
        torch.use_deterministic_algorithms(True, warn_only=not strict)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print(f"[determinism] use_deterministic_algorithms(warn_only={not strict}) enabled")

    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)


    cfg = WAN_CONFIGS["ti2v-5B"]
    wan_i2v = wan23.Yume(config=cfg, checkpoint_dir=args.ckpt_dir, device_id=device.index)
    wan_i2v.device = device
    transformer = wan_i2v.model.to(torch.bfloat16).eval().requires_grad_(False)
    vae = wan_i2v.vae


    from peft import LoraConfig, get_peft_model

    lcfg = LoraConfig(r=args.lora_rank, lora_alpha=args.lora_alpha, lora_dropout=0.0,
                      bias="none", task_type=None,
                      target_modules=["q", "k", "v", "o", "ffn.0", "ffn.2"])
    transformer = get_peft_model(transformer, lcfg)
    for _, p in transformer.named_parameters():
        if p.requires_grad:
            p.data = p.data.to(torch.float32)
    transformer = transformer.to(device)  


    opt = torch.optim.AdamW([p for p in transformer.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=args.weight_decay)
    n_train = sum(p.numel() for p in transformer.parameters() if p.requires_grad) / 1e6
    print(f"[pwm-grpo] trainable LoRA params: {n_train:.1f} M")


    if args.from_h5:
        pixel_values_vid = load_start_frame_from_h5(args.h5_path, args.scene, args.split)
        chunk_caps = load_captions_from_h5(args.h5_path, args.scene, args.split)
        if args.num_chunks > len(chunk_caps):
            print(f"[pwm-grpo] num_chunks {args.num_chunks} > {len(chunk_caps)} H5 captions; "
                  f"clamping to {len(chunk_caps)}")
            args.num_chunks = len(chunk_caps)
    else:
        assert args.jpg_dir and args.caption_path, "need --jpg_dir + --caption_path (or use --from_h5)"
        dataset_ddp, _ = create_scaled_videos(args.jpg_dir, total_frames=33, H1=480, W1=832)
        pixel_values_vid = dataset_ddp[0][0]  
        with open(args.caption_path, "r", encoding="utf-8") as f:
            captions = [ln.rstrip("\n") for ln in f if ln.strip()]

        chunk_caps = captions[1:] if len(captions) > args.num_chunks else captions
    assert len(chunk_caps) >= args.num_chunks, f"need >= {args.num_chunks} chunk captions, got {len(chunk_caps)}"

    reward_models = load_reward_models(device=device,
                                       depth_id=(args.depth_id if args.w_geo > 0 else None),
                                       lpips_net=(args.lpips_net if args.w_lpips > 0 else None),
                                       clip_name=(args.clip_name if args.w_clip > 0 else None),
                                       qual_metric=(args.qual_metric if args.w_qual > 0 else None))
    weights = {"gt": args.w_gt, "cross": args.w_cross, "dyn": args.w_dyn, "geo": args.w_geo,
               "action": args.w_action, "temporal": args.w_temporal, "quality": args.w_quality,
               "lpips": args.w_lpips, "motion": args.w_motion, "sharp": args.w_sharp,
               "clip": args.w_clip, "qual": args.w_qual, "precision": args.w_precision}


    if args.diversity_adaptive and args.from_h5:
        import h5py as _h5, numpy as _np
        with _h5.File(args.h5_path, "r") as _f:
            _p = _f[args.scene]["train"]["poses"][:]
        _uw = _np.unwrap(_np.arctan2(_p[:, 1, 0], _p[:, 0, 0]))
        _N = len(_uw); _nb = max(1, _N // 9)
        _nets = [_uw[min((c + 1) * 9, _N - 1)] - _uw[c * 9] for c in range(_nb)]
        _D = float(_np.std(_nets))
        _lo, _hi = 0.003, 0.020
        a = min(1.0, max(0.0, (_D - _lo) / (_hi - _lo)))     
        weights["lpips"] = 0.10 + 0.30 * a                    
        weights["gt"]    = 0.60 - 0.20 * a                    
        weights["cross"] = 0.30 - 0.10 * a                    
        args.kl_coef     = 0.25 - 0.15 * a                    
        print(f"[div-adaptive] D={_D:.4f} alpha={a:.3f} -> "
              f"lpips={weights['lpips']:.2f} gt={weights['gt']:.2f} "
              f"cross={weights['cross']:.2f} kl={args.kl_coef:.3f}", flush=True)


    if args.w_precision > 0:
        import cv2
        import h5py
        from pwm_rl.rewards import build_scene_bank
        with h5py.File(args.h5_path, "r") as _f:
            _vc = _f[args.scene]["train"]["video_clip"]
            _frs = [cv2.cvtColor(cv2.imdecode(np.frombuffer(_vc[i], np.uint8), cv2.IMREAD_COLOR),
                                 cv2.COLOR_BGR2RGB) for i in range(len(_vc))]
        _bt = torch.from_numpy(np.stack(_frs)).float().permute(0, 3, 1, 2) / 255.0
        reward_models.scene_bank = build_scene_bank(_bt.to(device), reward_models.dino)
        print(f"[pwm-grpo] R_precision scene_bank: {tuple(reward_models.scene_bank.shape)} "
              f"from {len(_frs)} real train frames")
    max_area = 480 * 832


    if args.mode == "infer":
        run_infer(args, transformer, vae, wan_i2v, pixel_values_vid, chunk_caps,
                  reward_models, weights, device, load_chunk_gt, compute_chunk_reward)
        return


    train_n = min(args.train_chunks, args.num_chunks)
    print(f"[pwm-grpo] training on chunks 0:{train_n} (of {args.num_chunks}); "
          f"chunks {train_n}:{args.num_chunks} held out for test")


    best_reward, best_epoch, no_improve = float("-inf"), -1, 0


    W = args.support_win if args.support_win > 0 else train_n
    schedule = []  
    for _s in range(0, max(1, train_n - W + 1), max(1, args.support_stride)):
        _wl = min(W, train_n - _s)
        for _kk in range(_wl):
            schedule.append((_s + _kk, _kk == 0, _kk + 1 < _wl))
    print(f"[support-buffer] W={W} stride={args.support_stride} -> {len(schedule)} "
          f"updates/epoch; window starts {[t[0] for t in schedule if t[1]]}", flush=True)
    for epoch in range(args.epochs):
        ep_reward = 0.0
        for (k, is_wstart, prep_next) in schedule:
            sigmas = get_sampling_sigmas(args.num_euler_timesteps, 7.0)
            if is_wstart:
                start_vid = pixel_values_vid if k == 0 else load_start_frame_from_h5(
                    args.h5_path, args.scene, args.split, frame_idx=KEEP_FRAME * k)
                model_input, arg_c, noise, mask2, img = prepare_first_chunk(
                    wan_i2v, start_vid, chunk_caps[k], max_area, device)
                gtk, _, _ = load_chunk_gt(args.h5_path, args.scene, k, device=device, chunk_len=KEEP_FRAME,
                                          offset=args.gt_offset, split=args.split)
                prev_last = gtk[0]
                hist_full = img[0]

            if args.vader:
                gt, poses, K = load_chunk_gt(args.h5_path, args.scene, k, device=device, chunk_len=KEEP_FRAME,
                                             offset=args.gt_offset, split=args.split)
                init_noise = noise if k == 0 else torch.randn_like(hist_full)
                opt.zero_grad()
                final_latent, kl_term = rollout_vader(transformer, arg_c, mask2, init_noise, hist_full,
                                             sigmas, args.vader_kgrad, device, kl_coef=args.kl_coef)
                gen = emit9_grad(vae, final_latent)
                R = r_recon_diff(gen, gt.to(device), reward_models, args.w_gt, args.w_lpips)
                loss = -R + args.kl_coef * kl_term
                loss.backward()
                gnorm = float(torch.nn.utils.clip_grad_norm_(
                    [p for p in transformer.parameters() if p.requires_grad], 1.0))
                opt.step()
                ep_reward += float(R.detach())
                prev_last = gen[KEEP_FRAME - 1].detach()
                with torch.no_grad():
                    commit_latent = final_latent.detach()
                    if k + 1 < args.num_chunks:
                        model_input = torch.cat(
                            [model_input, commit_latent[:, -LATENT_FRAME_ZERO:][:, :KEEP_LATENT]], dim=1)
                        arg_c, noise, mask2, img, hist_full = prepare_next_chunk(
                            wan_i2v, model_input, chunk_caps[k + 1], max_area, device)
                print(f"[vader ep {epoch} chunk {k}] R={float(R.detach()):.4f} kl={float(kl_term.detach()):.4f} gnorm={gnorm:.3e}", flush=True)
                del final_latent, gen, R, loss
                torch.cuda.empty_cache()
                continue


            init_noise = noise if is_wstart else torch.randn_like(hist_full)


            rollouts, rewards, comps = [], [], []
            gt, poses, K = load_chunk_gt(args.h5_path, args.scene, k, device=device, chunk_len=KEEP_FRAME, offset=args.gt_offset, split=args.split)
            for grp in range(args.group_size):


                g_init = torch.randn_like(init_noise) if getattr(args, "unshare_init", False) else init_noise


                hist_g = hist_full
                if getattr(args, "cond_jitter", 0.0) > 0.0:
                    hist_g = hist_full + args.cond_jitter * torch.randn_like(hist_full)
                final_latent, records = rollout_sde(
                    transformer, arg_c, mask2, g_init, hist_g, sigmas,
                    beta=args.beta, seed=args.seed + epoch * 10000 + k * 100 + grp, device=device)
                _fm = records[-1].pop("final_mean_full", None)
                if getattr(args, "score_on_mean", False) and _fm is not None:
                    gen = emit9(vae, _fm.to(device))
                    del _fm
                else:
                    gen = emit9(vae, final_latent)
                rollouts.append({"final_latent": final_latent, "records": records, "gen": gen})
                del final_latent
                torch.cuda.empty_cache()


            patch_w = None
            if getattr(args, "var_gate", 0.0) > 0.0 and len(rollouts) > 1:
                import torch.nn.functional as _F
                _gens = torch.stack([r["gen"] for r in rollouts])            
                _v = _gens.var(dim=0).mean(dim=1, keepdim=True)              
                _v16 = _F.adaptive_avg_pool2d(_v, (16, 16)).flatten(1)       
                _vn = _v16 / (_v16.mean(dim=1, keepdim=True) + 1e-8)
                patch_w = 1.0 / (1.0 + args.var_gate * _vn)
                print(f"[var-gate ep {epoch} chunk {k}] w_mean={float(patch_w.mean()):.3f} "
                      f"w_min={float(patch_w.min()):.3f}", flush=True)

            sp_k = ss_k = None
            if os.environ.get("STATIC_MASK") == "1":
                if not hasattr(args, "_static_masks"):
                    import numpy as _np
                    _mp = os.path.join(args.pwm_root, "masks", f"{args.scene}_train.npz")
                    _z = _np.load(_mp)
                    args._static_masks = {
                        "pix": torch.from_numpy(_z["pix"].astype("float32")),
                        "patch": torch.from_numpy(_z["patch"].astype("float32"))}
                    print(f"[static-mask] loaded {_mp} mean_patch_w="
                          f"{float(args._static_masks['patch'].mean()):.3f}", flush=True)
                _lo = k * KEEP_FRAME + args.gt_offset
                sp_k = args._static_masks["patch"][_lo:_lo + KEEP_FRAME].to(device)
                ss_k = args._static_masks["pix"][_lo:_lo + KEEP_FRAME].to(device)
            for grp, ro in enumerate(rollouts):
                out = compute_chunk_reward(ro["gen"], gt, prev_last, reward_models,
                                           poses=poses, K=K, weights=weights,
                                           captions=(make_caption_views(chunk_caps[min((k*KEEP_FRAME)//9, len(chunk_caps)-1)], args.clip_views)
                                                     if args.w_clip > 0 else None),
                                           patch_w=patch_w,
                                           static_patch=sp_k, static_pix=ss_k)
                rewards.append(out["reward"])
                comps.append(out["components"])


            if os.environ.get("DIAG_DIV") == "1":
                gens = [r["gen"] for r in rollouts]
                df = [float((gens[a] - gens[b]).abs().mean())
                      for a in range(len(gens)) for b in range(a + 1, len(gens))]
                rw = [f"{x:.4f}" for x in rewards]
                print(f"[DIV ep{epoch} chunk{k}] pairwise_pixel_L1={np.mean(df):.5f} "
                      f"max={max(df):.5f} | rewards={rw}")

            adv = group_advantages(comps, weights)

            _rf = float(os.environ.get("RSTD_FLOOR", "0") or 0)
            _gate_skip = False
            if _rf > 0:
                _rs = float(np.std(np.asarray(rewards, dtype=np.float64)))
                if _rs < _rf:
                    _gate_skip = True
                    print(f"[rstd-gate ep {epoch} chunk {k}] rstd={_rs:.4f} < {_rf}"
                          " -> skip update (noise-only chunk)", flush=True)


            micro = args.group_size * args.num_euler_timesteps
            ratios, kls, gnorm = ([1.0] if _gate_skip else []), [], 0.0
            for _ppo in range(0 if _gate_skip else args.ppo_epochs):
                opt.zero_grad()
                ratios = []
                kls = []
                for grp, ro in enumerate(rollouts):
                    a = torch.tensor(adv[grp], device=device, dtype=torch.float32)
                    for rec in ro["records"]:
                        lat = rec["latent_before"].to(device)
                        ts = build_timestep(rec["sigma_now"], mask2, arg_c["seq_len"], device)
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            v_pred = transformer([lat.squeeze(0)], t=ts, **arg_c)[0]
                        mean, std = sde_mean_std(lat[:, -LATENT_FRAME_ZERO:].float(),
                                                 v_pred[:, -LATENT_FRAME_ZERO:].float(),
                                                 rec["sigma_now"], rec["sigma_next"], args.beta)
                        new_lp = gaussian_logprob(rec["sample_gen"].to(device).float(), mean, std)
                        old_lp = rec["old_logprob"].to(device)
                        loss = grpo_loss(new_lp, old_lp, a, args.eps_clip)


                        if args.kl_coef > 0.0:
                            if "mean_base" not in rec:
                                with torch.no_grad(), transformer.disable_adapter(), \
                                        torch.autocast("cuda", dtype=torch.bfloat16):
                                    v_base = transformer([lat.squeeze(0)], t=ts, **arg_c)[0]
                                mean_b, _ = sde_mean_std(lat[:, -LATENT_FRAME_ZERO:].float(),
                                                         v_base[:, -LATENT_FRAME_ZERO:].float(),
                                                         rec["sigma_now"], rec["sigma_next"], args.beta)
                                rec["mean_base"] = mean_b.detach().cpu()
                                del v_base
                            kl = ((mean - rec["mean_base"].to(device)) ** 2).mean()
                            loss = loss + args.kl_coef * kl
                            kls.append(float(kl.detach()))
                        (loss / micro).backward()
                        ratios.append(float(torch.exp(new_lp.detach() - old_lp)))
                        del lat, v_pred
                        torch.cuda.empty_cache()
                gnorm = torch.nn.utils.clip_grad_norm_(
                    [p for p in transformer.parameters() if p.requires_grad], 1.0)
                opt.step()


            best = int(np.argmax(rewards))
            if getattr(args, "commit_deterministic", False):


                with torch.no_grad():
                    det_latent, _ = rollout_sde(transformer, arg_c, mask2, init_noise, hist_full,
                                                sigmas, beta=0.0, seed=args.seed + k, device=device)
                    det_gen = emit9(vae, det_latent)
                commit_latent, prev_last = det_latent, det_gen[KEEP_FRAME - 1]
            else:
                commit_latent = rollouts[best]["final_latent"]
                prev_last = rollouts[best]["gen"][KEEP_FRAME - 1]
            model_input = torch.cat(
                [model_input, commit_latent[:, -LATENT_FRAME_ZERO:-LATENT_FRAME_ZERO + KEEP_LATENT]], dim=1)
            ep_reward += rewards[best]
            comp_str = " ".join(f"{kk}={np.mean([c[kk] for c in comps]):.3f}" for kk in comps[0])
            kl_str = f" kl={np.mean(kls):.2e}" if kls else ""


            rstd = float(np.std(rewards))
            print(f"[ep {epoch} chunk {k}] best_R={rewards[best]:.3f} rstd={rstd:.3f} adv={adv[best]:+.2f} | {comp_str} "
                  f"|| ratio={np.mean(ratios):.3f} gnorm={float(gnorm):.2e}{kl_str}")


            if prep_next:
                arg_c, noise, mask2, img, hist_full = prepare_next_chunk(
                    wan_i2v, model_input, chunk_caps[k + 1], max_area, device)

        ep_mean = ep_reward / len(schedule)
        print(f"[ep {epoch}] mean best reward = {ep_mean:.4f}")
        os.makedirs(args.lora_out, exist_ok=True)
        state = {n: p.detach().cpu() for n, p in transformer.named_parameters() if p.requires_grad}
        if (epoch + 1) % args.save_every == 0 or epoch + 1 == args.epochs:
            torch.save(state, os.path.join(args.lora_out, f"lora_ep{epoch}.pt"))
            print(f"[ep {epoch}] saved LoRA -> {args.lora_out}/lora_ep{epoch}.pt")

        if ep_mean > best_reward + args.early_stop_min_delta:
            best_reward, best_epoch, no_improve = ep_mean, epoch, 0
            torch.save(state, os.path.join(args.lora_out, "lora_best.pt"))
            print(f"[ep {epoch}] new BEST {ep_mean:.4f} -> lora_best.pt")
        else:
            no_improve += 1
            print(f"[ep {epoch}] no improve ({no_improve}/{args.early_stop_patience}); "
                  f"best={best_reward:.4f}@ep{best_epoch}")
            if args.early_stop_patience > 0 and no_improve >= args.early_stop_patience:
                print(f"[early-stop] no improvement for {args.early_stop_patience} epochs; "
                      f"stopping. BEST={best_reward:.4f}@ep{best_epoch} (lora_best.pt)")
                break
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_dir", default="./Yume-5B-720P")
    p.add_argument("--jpg_dir", default=None, help="start-frame folder (file mode; omit with --from_h5)")
    p.add_argument("--caption_path", default=None, help="caption_re.txt (file mode; omit with --from_h5)")
    p.add_argument("--from_h5", action="store_true",
                   help="read start frame + per-chunk captions straight from the H5 scene/split "
                        "(no jpg/ + caption_re.txt needed) -> sweep the dataset by looping --scene")
    p.add_argument("--h5_path", required=True)
    p.add_argument("--teacher_forced", action="store_true",
                   help="GT-anchored sliding-window support buffer")
    p.add_argument("--eval_real_starts", action="store_true",
                   help="infer-only diagnostic: reset every chunk from its real GT start frame")
    p.add_argument("--eval_legacy_caption_bug", action="store_true",
                   help="infer-only diagnostic: reproduce retired sequential caption indexing")
    p.add_argument("--support_win", type=int, default=0,
                   help="chunks per support-buffer window (0=single trajectory)")
    p.add_argument("--color_anchor", type=float, default=0.0,
                   help="infer: blend committed latents toward first-frame channel stats (0=off, 1=full)")
    p.add_argument("--support_stride", type=int, default=1,
                   help="window start stride in chunks")
    p.add_argument("--scene", required=True)
    p.add_argument("--split", default=None,
                   help="H5 subgroup: 'train' or 'test'. Leave unset when frames are stored directly under the scene.")
    p.add_argument("--pwm_root", default=".")
    p.add_argument("--num_chunks", type=int, default=11)
    p.add_argument("--mode", choices=["train", "infer"], default="train")
    p.add_argument("--train_chunks", type=int, default=9999,
                   help="train GRPO on chunks 0:train_chunks; the rest are the test region")
    p.add_argument("--lora_ckpt", default=None, help="(infer) LoRA .pt to load; omit for baseline")
    p.add_argument("--infer_out", default="./outputs/infer", help="(infer) dir for the saved mp4")
    p.add_argument("--bench_json", default=None,
                   help="(infer) dir for per-scene benchmark JSON (default: --infer_out). "
                        "Use a separate output directory for each scene.")
    p.add_argument("--lpips_net", default="alex", choices=["alex", "vgg"],
                   help="(infer) LPIPS backbone; alex=fast default, vgg=slower")
    p.add_argument("--group_size", type=int, default=4)
    p.add_argument("--num_euler_timesteps", type=int, default=4)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--beta", type=float, default=0.5)          
    p.add_argument("--eps_clip", type=float, default=0.2)
    p.add_argument("--ppo_epochs", type=int, default=4,
                   help="inner PPO passes over each chunk's rollouts; >1 makes the "
                        "clip trust region active (was effectively 1 = pure REINFORCE)")
    p.add_argument("--weight_decay", type=float, default=1e-2,
                   help="AdamW weight decay on the LoRA weights (weak/cheap shrink-to-base; "
                        "the real anchor is --kl_coef)")
    p.add_argument("--vader", action="store_true", help="direct reward-gradient training (VADER/DRaFT)")
    p.add_argument("--vader_kgrad", type=int, default=1, help="last K denoising steps kept in the backprop graph")
    p.add_argument("--diversity_adaptive", action="store_true",
                   help="scale reward (lpips/gt/cross) + kl by train action diversity D")
    p.add_argument("--kl_coef", type=float, default=0.1,
                   help="KL-to-base coefficient: penalise the policy mean drifting from the "
                        "frozen base mean; 0 disables regularization")
    p.add_argument("--lr", type=float, default=5e-5)           
    p.add_argument("--lora_rank", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--gt_offset", type=int, default=0)         
    p.add_argument("--w_gt", type=float, default=0.70)         
    p.add_argument("--w_cross", type=float, default=0.20)
    p.add_argument("--w_dyn", type=float, default=0.10)        
    p.add_argument("--w_precision", type=float, default=0.0)   
    p.add_argument("--w_geo", type=float, default=0.0)         
    p.add_argument("--w_action", type=float, default=0.0)      
    p.add_argument("--w_temporal", type=float, default=0.0)    
    p.add_argument("--w_quality", type=float, default=0.0)     
    p.add_argument("--w_lpips", type=float, default=0.0)       
    p.add_argument("--w_motion", type=float, default=0.0)      
    p.add_argument("--w_sharp", type=float, default=0.0)       
    p.add_argument("--w_clip", type=float, default=0.0)        
    p.add_argument("--clip_name", default="ViT-B/32")          
    p.add_argument("--clip_views", type=int, default=1)        
    p.add_argument("--w_qual", type=float, default=0.0)        
    p.add_argument("--qual_metric", default="clipiqa")         
    p.add_argument("--unshare_init", action="store_true")
    p.add_argument("--score_on_mean", action="store_true") 
    p.add_argument("--cond_jitter", type=float, default=0.0)   
    p.add_argument("--deterministic", action="store_true",
                   help="enable CUDA deterministic algos (warn_only) to cut the "
                        "run-to-run eval noise floor; slower, some ops stay nondet")
    p.add_argument("--commit_deterministic", action="store_true",
                   help="(train) autoregress by committing a beta=0 rollout (matches "
                        "single-shot eval) instead of best-of-G; closes the regime gap")
    p.add_argument("--depth_id", default="depth-anything/Depth-Anything-V2-Small-hf")
    p.add_argument("--var_gate", type=float, default=0.0,
                   help="variance-gated R_gt strength: down-weight patches with high cross-rollout variance (transient/unpredictable content). 0=off")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save_every", type=int, default=5)
    p.add_argument("--early_stop_patience", type=int, default=0)   
    p.add_argument("--early_stop_min_delta", type=float, default=0.002)
    p.add_argument("--no_baseline", action="store_true",
                   help="(infer) skip the auto baseline pass; emit only the LoRA video")
    p.add_argument("--lora_out", default="./outputs/pwm_lora")
    main(p.parse_args())
