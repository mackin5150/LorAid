import os, base64, io, json
from PIL import Image

def _img_to_b64(img: Image.Image, fmt="PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def _pack_image(img: Image.Image) -> dict:
    return {"type": "input_image", "image_url": f"data:image/png;base64,{_img_to_b64(img)}"}

def judge_candidate_openai(main_img, ref_imgs, candidate_img, extra_text: str = "") -> dict:
    """
    Returns JSON dict with normalized scores [0..1].
    This uses OpenAI's Responses API shape (conceptual).
    If you already have an OpenAI client wrapper, plug it in here.
    """
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY in environment (.env).")

    # We avoid importing openai SDK here to keep file minimal.
    # Install: pip install openai
    from openai import OpenAI
    client = OpenAI(api_key=api_key)

    system = (
        "You are a strict image judge for a LoRA auto-dialer. "
        "Score how well a candidate matches multiple reference images and preserves the main image composition. "
        "Return ONLY valid JSON matching the schema."
    )

    # Prompt: explain scoring dimensions
    user_text = f"""
Evaluate the candidate image using:
- identity: likeness to the reference set (face/hair/character identity) [0..1]
- style: similarity to rendering style of references [0..1]
- outfit_attributes: clothing/props/traits alignment to references [0..1]
- main_preservation: how well candidate preserves main image pose/composition/silhouette [0..1]
- artifacts: visible defects (extra limbs, melted features, bad text, etc.) [0..1] where 1=very bad
- overbaked: LoRA overpower signs (plastic skin, oversharp, same-face syndrome, heavy imprint) [0..1] where 1=very overbaked

Return JSON:
{{
  "identity": float,
  "style": float,
  "outfit_attributes": float,
  "main_preservation": float,
  "artifacts": float,
  "overbaked": float,
  "notes": "short reason"
}}

Extra notes/context from user:
{extra_text}
""".strip()

    # Build multimodal content: main + refs + candidate
    content = [{"type": "input_text", "text": user_text}]
    content.append({"type": "input_text", "text": "MAIN IMAGE (composition anchor):"})
    content.append(_pack_image(main_img))

    content.append({"type": "input_text", "text": "REFERENCE IMAGES (identity/style/outfit anchors):"})
    for r in ref_imgs:
        content.append(_pack_image(r))

    content.append({"type": "input_text", "text": "CANDIDATE IMAGE (to score):"})
    content.append(_pack_image(candidate_img))

    resp = client.responses.create(
        model="gpt-4.1-mini",  # you can swap to a stronger vision model if desired
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": content},
        ],
        temperature=0,
    )

    # Extract text JSON
    text = resp.output_text.strip()
    try:
        data = json.loads(text)
    except Exception as e:
        raise RuntimeError(f"Judge returned non-JSON: {text[:500]}") from e

    # Clamp & ensure keys
    keys = ["identity","style","outfit_attributes","main_preservation","artifacts","overbaked","notes"]
    for k in keys:
        if k not in data:
            raise RuntimeError(f"Judge missing key: {k}. Got: {data}")
    return data
