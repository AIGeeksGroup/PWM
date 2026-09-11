import json
import os
import sys
import math
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import style_bench as SB  

HORIZON = int(os.environ.get("HORIZON", "145"))


def score_arm(h5, scene, mp4):
    gt = SB.h5_frames(h5, scene, "test", max_n=HORIZON)
    gen = SB.read_mp4(mp4, HORIZON)
    if len(gt) != HORIZON or len(gen) != HORIZON:
        raise ValueError(f"Expected {HORIZON} GT and generated frames, got {len(gt)}, {len(gen)}")
    gt_cls = SB.dino_cls(gt)
    gen_cls = SB.dino_cls(gen)
    d_pair = SB.appear_distance(gen_cls, gt_cls)
    bank = SB.clip_text_bank()
    known = SB.clip_probs(gt[:1], bank)
    known = {k: bool(known[k][0] > SB.T_REF) for k in SB.CATS}
    d_hall, _ = SB.halluc_rate(gen, known, bank)
    return d_pair, d_hall, 1.0 * d_pair + 0.5 * d_hall


def score_pair(h5, scene, lora, base):
    lp, lh, ls = score_arm(h5, scene, lora)
    bp, bh, bs = score_arm(h5, scene, base)
    return dict(dS=ls - bs, d_pair=lp - bp, d_hall=lh - bh)


def is_improved(mean_dS):
    if not math.isfinite(mean_dS):
        raise ValueError("Non-finite S_gt difference")
    return mean_dS < 0


if __name__ == "__main__":
    if sys.argv[1] == "--batch":
        jobs = json.load(open(sys.argv[2]))
        out = {}
        for tag, j in jobs.items():
            if set(j["seeds"]) != {"42", "123", "777"}:
                raise ValueError("Batch evaluation requires seeds 42, 123, and 777")
            per = {}
            for s, (lora, base) in j["seeds"].items():
                per[s] = score_pair(j["h5"], j["scene"], lora, base)
                print(tag, s, {k: round(v, 4) for k, v in per[s].items()}, flush=True)
            ds = [p["dS"] for p in per.values()]
            m = sum(ds) / len(ds)
            out[tag] = dict(
                mean_dS=m,
                mean_dpair=sum(p["d_pair"] for p in per.values()) / len(per),
                mean_dhall=sum(p["d_hall"] for p in per.values()) / len(per),
                improved_seed_count=sum(1 for d in ds if d < 0),
                improved_vs_native=is_improved(m),
                per_seed=per,
            )
            print(tag, "mean_delta=", round(m, 4), "improved_vs_native=",
                  out[tag]["improved_vs_native"], flush=True)
        dst = sys.argv[3] if len(sys.argv) > 3 else "sgt_results.json"
        json.dump(out, open(dst, "w"), indent=1)
        print("SGT_BATCH_DONE", dst, flush=True)
    else:
        h5, scene, lora, base = sys.argv[1:5]
        r = score_pair(h5, scene, lora, base)
        print(json.dumps(r, indent=1))
