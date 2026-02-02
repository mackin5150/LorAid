import os, time, zipfile, json
import numpy as np
from PIL import Image
import gradio as gr
from dotenv import load_dotenv

from .generator_diffusers import DiffusersImg2ImgGenerator
from .generator_openai import OpenAIImg2ImgGenerator
from .judge_openai import judge_candidate_openai
from .optimize import run_search
from .report import write_report

load_dotenv()

STATE = {"gen": None, "backend": None, "base_model": None, "last_payload": None, "last_run_dir": None}

def _ensure_gen(backend: str, base_model: str):
    if (
        STATE["gen"] is None
        or STATE["backend"] != backend
        or STATE["base_model"] != base_model
    ):
        if backend == "openai":
            STATE["gen"] = OpenAIImg2ImgGenerator(base_model=base_model)
        else:
            STATE["gen"] = DiffusersImg2ImgGenerator(base_model=base_model)
        STATE["backend"] = backend
        STATE["base_model"] = base_model
    return STATE["gen"]

def _to_pil(x):
    """
    For main_image from gr.Image(type="pil") this is usually already PIL.
    Kept robust anyway.
    """
    if x is None:
        return None
    if isinstance(x, Image.Image):
        return x
    if isinstance(x, np.ndarray):
        return Image.fromarray(x)
    if isinstance(x, str) and os.path.exists(x):
        return Image.open(x)
    if hasattr(x, "name") and isinstance(x.name, str) and os.path.exists(x.name):
        return Image.open(x.name)
    # Unwrap tuples/lists if any
    if isinstance(x, (tuple, list)) and len(x) > 0:
        return _to_pil(x[0])
    raise TypeError(f"Unsupported image type: {type(x)}  value={repr(x)[:200]}")

def _parse_seeds(seeds_csv: str):
    seeds = []
    for part in (seeds_csv or "").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            seeds.append(int(part))
        except ValueError:
            pass
    return seeds or [12345]

def _load_lora_metadata(lora_path: str) -> dict:
    if not lora_path or not os.path.exists(lora_path):
        return {}

    metadata = {}
    if lora_path.endswith(".safetensors"):
        try:
            from safetensors import safe_open
            with safe_open(lora_path, framework="pt", device="cpu") as f:
                metadata.update(f.metadata() or {})
        except Exception:
            metadata = metadata or {}

    sidecar = os.path.splitext(lora_path)[0] + ".json"
    if os.path.exists(sidecar):
        try:
            with open(sidecar, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                metadata.update(data)
        except Exception:
            pass

    return metadata

def _suggest_lora_range(metadata: dict):
    def _get_float(*keys):
        for key in keys:
            if key in metadata:
                try:
                    return float(metadata[key])
                except (TypeError, ValueError):
                    continue
        return None

    alpha = _get_float("ss_network_alpha", "lora_alpha", "network_alpha")
    dim = _get_float("ss_network_dim", "lora_dim", "network_dim")
    explicit = _get_float("ss_lora_scale", "lora_scale", "preferred_scale")

    if explicit is not None:
        recommended = explicit
    elif alpha is not None and dim:
        recommended = alpha / dim
    else:
        recommended = 1.0

    recommended = max(0.05, min(2.0, recommended))
    lora_min = max(0.0, recommended * 0.5)
    lora_max = min(2.0, recommended * 1.5)

    return lora_min, lora_max, recommended

def load_lora_and_suggest(lora_path: str):
    metadata = _load_lora_metadata(lora_path)
    lora_min, lora_max, recommended = _suggest_lora_range(metadata)

    meta_lines = []
    if metadata:
        for k in sorted(metadata.keys()):
            if k.startswith("__"):
                continue
            meta_lines.append(f"{k}: {metadata[k]}")
    meta_text = "\n".join(meta_lines) if meta_lines else "No metadata found."
    summary = f"Suggested LoRA scale: {recommended:.3f}"

    return (
        gr.update(value=lora_min),
        gr.update(value=lora_max),
        gr.update(value=summary),
        gr.update(value=meta_text),
    )

def generate_lora_preview(
    backend,
    base_model,
    lora_path,
    main_image,
    prompt,
    negative,
    lora_min,
    lora_max,
    denoise_min,
    denoise_max,
    cfg_min,
    cfg_max,
    img_w,
    img_h,
    infer_steps,
    seeds_csv,
):
    if backend != "diffusers":
        raise gr.Error("LoRA preview generation is only available for the diffusers backend.")
    if not lora_path or not os.path.exists(lora_path):
        raise gr.Error(f"LoRA path does not exist: {lora_path}")
    if main_image is None:
        raise gr.Error("Please upload a MAIN image to generate a preview.")

    seed = _parse_seeds(seeds_csv)[0]
    lora_scale = (float(lora_min) + float(lora_max)) / 2.0
    strength = (float(denoise_min) + float(denoise_max)) / 2.0
    cfg = (float(cfg_min) + float(cfg_max)) / 2.0

    gen = _ensure_gen(backend, base_model)
    gen.load_lora(lora_path)

    main_pil = _to_pil(main_image).convert("RGB")
    preview = gen.generate(
        init_image=main_pil,
        prompt=prompt,
        negative_prompt=negative,
        lora_scale=lora_scale,
        strength=strength,
        cfg=cfg,
        steps=int(infer_steps),
        seed=int(seed),
        width=int(img_w),
        height=int(img_h),
    )
    return preview

def save_tuned_settings(lora_path: str, filename: str):
    if STATE["last_payload"] is None or STATE["last_run_dir"] is None:
        raise gr.Error("Run the auto dialer first to generate tuned settings.")

    payload = STATE["last_payload"]
    run_dir = STATE["last_run_dir"]

    name = (filename or "").strip() or "tuned_settings.json"
    if not name.endswith(".json"):
        name = f"{name}.json"
    if os.path.isabs(name):
        out_path = name
    else:
        out_path = os.path.join(run_dir, name)

    tuned = {
        "lora_path": lora_path,
        "best": payload.get("best", {}),
        "settings": payload.get("settings", {}),
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(tuned, f, indent=2)

    return out_path

def run_dial(
    backend,
    base_model,
    lora_path,
    main_image,
    ref_files,          # from gr.Files
    lora_preview_image,
    include_preview,
    prompt,
    negative,
    lora_min,
    lora_max,
    steps,
    seeds_csv,
    denoise_min,
    denoise_max,
    cfg_min,
    cfg_max,
    img_w,
    img_h,
    infer_steps,
    judge_notes,
):
    # Validate
    if main_image is None:
        raise gr.Error("Please upload a MAIN image.")
    if not ref_files or len(ref_files) < 3:
        raise gr.Error("Please upload at least 3 REFERENCE images.")
    if backend == "diffusers":
        if not lora_path or not os.path.exists(lora_path):
            raise gr.Error(f"LoRA path does not exist: {lora_path}")

    # Convert images
    main_pil = _to_pil(main_image).convert("RGB")
    refs_pil = [Image.open(f.name).convert("RGB") for f in ref_files]
    if include_preview and lora_preview_image is not None:
        refs_pil.append(_to_pil(lora_preview_image).convert("RGB"))

    seeds = _parse_seeds(seeds_csv)

    # Run directory
    run_id = time.strftime("run_%Y%m%d_%H%M%S")
    run_dir = os.path.join("runs", run_id)
    os.makedirs(run_dir, exist_ok=True)

    # Generator
    if backend == "openai" and not base_model:
        base_model = "gpt-image-1"

    gen = _ensure_gen(backend, base_model)

    # Search
    payload = run_search(
        run_dir=run_dir,
        generator=gen,
        judge_fn=judge_candidate_openai,
        main_img=main_pil,
        ref_imgs=refs_pil,
        prompt=prompt,
        negative=negative,
        lora_path=lora_path if backend == "diffusers" else "",
        lora_min=float(lora_min),
        lora_max=float(lora_max),
        steps=int(steps),
        seeds=seeds,
        denoise_min=float(denoise_min),
        denoise_max=float(denoise_max),
        cfg_min=float(cfg_min),
        cfg_max=float(cfg_max),
        img_w=int(img_w),
        img_h=int(img_h),
        infer_steps=int(infer_steps),
        extra_judge_text=judge_notes or "",
    )

    report_path = write_report(run_dir, payload)
    STATE["last_payload"] = payload
    STATE["last_run_dir"] = run_dir

    # Gallery images
    gallery_imgs = [c["image_path"] for c in payload["top_refined"]]

    # Bundle zip
    zip_path = os.path.join(run_dir, "run_bundle.zip")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.write(os.path.join(run_dir, "results.json"), arcname="results.json")
        z.write(report_path, arcname="report.html")
        for c in payload["top_refined"] + payload["top_coarse"]:
            p = c["image_path"]
            if os.path.exists(p):
                z.write(p, arcname=os.path.join("candidates", os.path.basename(p)))

    best = payload["best"]
    best_summary = (
        f"BEST\\n"
        f"LoRA scale: {best['lora_scale']}\\n"
        f"Denoise: {best['strength']}\\n"
        f"CFG: {best['cfg']}\\n"
        f"Total: {best['total']}\\n"
        f"Notes: {best['scores'].get('notes','')}\\n"
        f"Run dir: {run_dir}"
    )

    return best_summary, best["image_path"], gallery_imgs, report_path, zip_path


with gr.Blocks(title="LoRA Auto Dialer") as demo:
    gr.Markdown("# LoRA Auto Dialer (main image + 3+ refs → auto dial LoRA weight)")

    with gr.Row():
        backend = gr.Dropdown(
            label="Generator backend",
            choices=["diffusers", "openai"],
            value="diffusers",
        )
        base_model = gr.Textbox(
            label="Base model (Diffusers path/HF repo or OpenAI model)",
            value="stabilityai/stable-diffusion-xl-base-1.0",
        )
        lora_path = gr.Textbox(label="LoRA file path (.safetensors)", value="")
        load_lora_btn = gr.Button("Load LoRA metadata")

    with gr.Row():
        main_image = gr.Image(label="Main image (composition anchor)", type="pil")
        ref_preview = gr.Gallery(label="Reference preview", columns=3, height=260, object_fit="contain")

    ref_upload = gr.Files(label="Upload reference images (3+)", file_types=["image"])
    with gr.Row():
        lora_preview = gr.Image(label="LoRA preview (generated or upload)", type="pil")
        include_preview = gr.Checkbox(label="Include preview in judging", value=True)
    preview_btn = gr.Button("Generate LoRA preview")

    def files_to_preview(files):
        if not files:
            return []
        imgs = []
        for f in files:
            imgs.append(Image.open(f.name).convert("RGB"))
        return imgs

    ref_upload.change(files_to_preview, inputs=[ref_upload], outputs=[ref_preview])

    prompt = gr.Textbox(label="Prompt", value="high quality, detailed, cinematic lighting")
    negative = gr.Textbox(label="Negative prompt", value="lowres, blurry, deformed, extra fingers, bad anatomy")

    with gr.Accordion("Tuning ranges", open=True):
        with gr.Row():
            lora_min = gr.Slider(0.0, 2.0, value=0.3, step=0.01, label="LoRA min")
            lora_max = gr.Slider(0.0, 2.0, value=1.2, step=0.01, label="LoRA max")
            steps = gr.Slider(5, 21, value=11, step=1, label="LoRA steps")
        with gr.Row():
            denoise_min = gr.Slider(0.05, 0.95, value=0.35, step=0.01, label="Denoise min (img2img strength)")
            denoise_max = gr.Slider(0.05, 0.95, value=0.65, step=0.01, label="Denoise max")
        with gr.Row():
            cfg_min = gr.Slider(1.0, 12.0, value=4.0, step=0.1, label="CFG min")
            cfg_max = gr.Slider(1.0, 12.0, value=7.0, step=0.1, label="CFG max")
        with gr.Row():
            img_w = gr.Slider(512, 1536, value=1024, step=64, label="Width")
            img_h = gr.Slider(512, 1536, value=1024, step=64, label="Height")
            infer_steps = gr.Slider(10, 60, value=30, step=1, label="Inference steps")

    seeds_csv = gr.Textbox(label="Seeds (comma-separated)", value="12345, 54321, 999")

    judge_notes = gr.Textbox(
        label="Judge notes (optional)",
        value="Prioritize face identity and hairstyle; preserve pose and silhouette; penalize plastic skin/over-sharpening.",
        lines=2,
    )

    run_btn = gr.Button("Run Auto Dial", variant="primary")
    lora_suggestion = gr.Textbox(label="LoRA suggestion", lines=1)
    lora_metadata = gr.Textbox(label="LoRA metadata", lines=6)

    best_summary = gr.Textbox(label="Best settings", lines=7)
    best_img = gr.Image(label="Best image", type="filepath")
    gallery = gr.Gallery(label="Top refined candidates", columns=4, height=360, object_fit="contain")
    report_file = gr.File(label="report.html")
    zip_file = gr.File(label="Download bundle (images+json+report)")
    save_name = gr.Textbox(label="Save tuned settings as", value="tuned_settings.json")
    save_btn = gr.Button("Save tuned settings (JSON preset)")
    tuned_file = gr.File(label="Saved tuned settings")

    run_btn.click(
        fn=run_dial,
        inputs=[
            backend, base_model, lora_path, main_image, ref_upload, lora_preview, include_preview,
            prompt, negative,
            lora_min, lora_max, steps, seeds_csv,
            denoise_min, denoise_max, cfg_min, cfg_max,
            img_w, img_h, infer_steps, judge_notes
        ],
        outputs=[best_summary, best_img, gallery, report_file, zip_file],
    )

    load_lora_btn.click(
        fn=load_lora_and_suggest,
        inputs=[lora_path],
        outputs=[lora_min, lora_max, lora_suggestion, lora_metadata],
    )

    preview_btn.click(
        fn=generate_lora_preview,
        inputs=[
            backend, base_model, lora_path, main_image, prompt, negative,
            lora_min, lora_max, denoise_min, denoise_max, cfg_min, cfg_max,
            img_w, img_h, infer_steps, seeds_csv,
        ],
        outputs=[lora_preview],
    )

    save_btn.click(
        fn=save_tuned_settings,
        inputs=[lora_path, save_name],
        outputs=[tuned_file],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
