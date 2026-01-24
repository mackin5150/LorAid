import os, time, zipfile
import numpy as np
from PIL import Image
import gradio as gr
from dotenv import load_dotenv

from .generator_diffusers import DiffusersImg2ImgGenerator
from .judge_openai import judge_candidate_openai
from .optimize import run_search
from .report import write_report

load_dotenv()

STATE = {"gen": None}

def _ensure_gen(base_model: str):
    if STATE["gen"] is None or getattr(STATE["gen"], "base_model", None) != base_model:
        STATE["gen"] = DiffusersImg2ImgGenerator(base_model=base_model)
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

def run_dial(
    base_model,
    lora_path,
    main_image,
    ref_files,          # from gr.Files
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
    if not lora_path or not os.path.exists(lora_path):
        raise gr.Error(f"LoRA path does not exist: {lora_path}")

    # Convert images
    main_pil = _to_pil(main_image).convert("RGB")
    refs_pil = [Image.open(f.name).convert("RGB") for f in ref_files]

    seeds = _parse_seeds(seeds_csv)

    # Run directory
    run_id = time.strftime("run_%Y%m%d_%H%M%S")
    run_dir = os.path.join("runs", run_id)
    os.makedirs(run_dir, exist_ok=True)

    # Generator
    gen = _ensure_gen(base_model)

    # Search
    payload = run_search(
        run_dir=run_dir,
        generator=gen,
        judge_fn=judge_candidate_openai,
        main_img=main_pil,
        ref_imgs=refs_pil,
        prompt=prompt,
        negative=negative,
        lora_path=lora_path,
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
        base_model = gr.Textbox(
            label="Base model (Diffusers path or HF repo)",
            value="stabilityai/stable-diffusion-xl-base-1.0",
        )
        lora_path = gr.Textbox(label="LoRA file path (.safetensors)", value="")

    with gr.Row():
        main_image = gr.Image(label="Main image (composition anchor)", type="pil")
        ref_preview = gr.Gallery(label="Reference preview", columns=3, height=260, object_fit="contain")

    ref_upload = gr.Files(label="Upload reference images (3+)", file_types=["image"])

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

    best_summary = gr.Textbox(label="Best settings", lines=7)
    best_img = gr.Image(label="Best image", type="filepath")
    gallery = gr.Gallery(label="Top refined candidates", columns=4, height=360, object_fit="contain")
    report_file = gr.File(label="report.html")
    zip_file = gr.File(label="Download bundle (images+json+report)")

    run_btn.click(
        fn=run_dial,
        inputs=[
            base_model, lora_path, main_image, ref_upload, prompt, negative,
            lora_min, lora_max, steps, seeds_csv,
            denoise_min, denoise_max, cfg_min, cfg_max,
            img_w, img_h, infer_steps, judge_notes
        ],
        outputs=[best_summary, best_img, gallery, report_file, zip_file],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
