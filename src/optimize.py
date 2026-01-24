import os, json, math, time
from dataclasses import dataclass, asdict
from typing import List, Tuple
from PIL import Image

@dataclass
class Candidate:
    lora_scale: float
    strength: float
    cfg: float
    seed: int
    scores: dict
    total: float
    image_path: str

def total_score(s: dict) -> float:
    # weights tuned for "dialing in" (reward match & preservation, punish defects/overbake)
    return (
        0.35 * float(s["identity"]) +
        0.20 * float(s["style"]) +
        0.15 * float(s["outfit_attributes"]) +
        0.30 * float(s["main_preservation"]) -
        0.12 * float(s["artifacts"]) -
        0.18 * float(s["overbaked"])
    )

def linspace(a: float, b: float, n: int) -> List[float]:
    if n <= 1:
        return [a]
    step = (b - a) / (n - 1)
    return [a + i * step for i in range(n)]

def run_search(
    run_dir: str,
    generator,
    judge_fn,
    main_img: Image.Image,
    ref_imgs: List[Image.Image],
    prompt: str,
    negative: str,
    lora_path: str,
    lora_min: float,
    lora_max: float,
    steps: int,
    seeds: List[int],
    denoise_min: float,
    denoise_max: float,
    cfg_min: float,
    cfg_max: float,
    img_w: int,
    img_h: int,
    infer_steps: int,
    extra_judge_text: str = "",
):
    os.makedirs(run_dir, exist_ok=True)
    cand_dir = os.path.join(run_dir, "candidates")
    os.makedirs(cand_dir, exist_ok=True)

    generator.load_lora(lora_path)

    # --- Phase 1: coarse sweep ---
    lora_grid = linspace(lora_min, lora_max, steps)
    strength_grid = linspace(denoise_min, denoise_max, max(3, min(steps//3, 7)))
    cfg_grid = linspace(cfg_min, cfg_max, 3)

    results: List[Candidate] = []
    idx = 0

    for w in lora_grid:
        for st in strength_grid:
            for cfg in cfg_grid:
                # evaluate across seeds and average (stability)
                per_seed = []
                for seed in seeds:
                    img = generator.generate(
                        init_image=main_img,
                        prompt=prompt,
                        negative_prompt=negative,
                        lora_scale=w,
                        strength=st,
                        cfg=cfg,
                        steps=infer_steps,
                        seed=seed,
                        width=img_w,
                        height=img_h,
                    )

                    scores = judge_fn(main_img, ref_imgs, img, extra_text=extra_judge_text)
                    tot = total_score(scores)

                    out_path = os.path.join(cand_dir, f"cand_{idx:05d}_w{w:.3f}_s{st:.3f}_c{cfg:.2f}_seed{seed}.png")
                    img.save(out_path)

                    per_seed.append((scores, tot, out_path))
                    idx += 1

                # average totals; keep representative image as first seed
                avg_total = sum(t for _, t, _ in per_seed) / len(per_seed)
                rep_scores, rep_tot, rep_path = per_seed[0]

                results.append(Candidate(
                    lora_scale=float(w),
                    strength=float(st),
                    cfg=float(cfg),
                    seed=int(seeds[0]),
                    scores=rep_scores,
                    total=float(avg_total),
                    image_path=rep_path
                ))

    results.sort(key=lambda c: c.total, reverse=True)
    top = results[:8]

    # --- Phase 2: refine around best LoRA weight ---
    best = top[0]
    refine_span = (lora_max - lora_min) * 0.12
    rmin = max(lora_min, best.lora_scale - refine_span)
    rmax = min(lora_max, best.lora_scale + refine_span)
    refine_grid = linspace(rmin, rmax, 7)

    refined: List[Candidate] = []
    for w in refine_grid:
        # keep best strength/cfg found
        st = best.strength
        cfg = best.cfg

        per_seed = []
        for seed in seeds:
            img = generator.generate(
                init_image=main_img,
                prompt=prompt,
                negative_prompt=negative,
                lora_scale=w,
                strength=st,
                cfg=cfg,
                steps=infer_steps,
                seed=seed,
                width=img_w,
                height=img_h,
            )
            scores = judge_fn(main_img, ref_imgs, img, extra_text=extra_judge_text)
            tot = total_score(scores)

            out_path = os.path.join(cand_dir, f"refine_w{w:.3f}_s{st:.3f}_c{cfg:.2f}_seed{seed}.png")
            img.save(out_path)
            per_seed.append((scores, tot, out_path))

        avg_total = sum(t for _, t, _ in per_seed) / len(per_seed)
        rep_scores, _, rep_path = per_seed[0]
        refined.append(Candidate(
            lora_scale=float(w),
            strength=float(st),
            cfg=float(cfg),
            seed=int(seeds[0]),
            scores=rep_scores,
            total=float(avg_total),
            image_path=rep_path
        ))

    refined.sort(key=lambda c: c.total, reverse=True)

    payload = {
        "best": asdict(refined[0]),
        "top_coarse": [asdict(x) for x in top],
        "top_refined": [asdict(x) for x in refined[:8]],
        "settings": {
            "prompt": prompt,
            "negative": negative,
            "lora_path": lora_path,
            "lora_range": [lora_min, lora_max],
            "denoise_range": [denoise_min, denoise_max],
            "cfg_range": [cfg_min, cfg_max],
            "seeds": seeds,
            "img_size": [img_w, img_h],
            "infer_steps": infer_steps,
        },
    }

    with open(os.path.join(run_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return payload
