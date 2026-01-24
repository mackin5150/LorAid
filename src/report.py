import os, json
from jinja2 import Template

TPL = Template("""
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>LoRA Auto Dialer Report</title>
  <style>
    body { font-family: sans-serif; margin: 24px; }
    .best { padding: 12px; border: 1px solid #ddd; border-radius: 10px; margin-bottom: 20px; }
    .grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; }
    .card { border: 1px solid #eee; border-radius: 10px; padding: 10px; }
    img { width: 100%; height: auto; border-radius: 10px; }
    code { background: #f6f6f6; padding: 2px 6px; border-radius: 6px; }
    .small { color: #555; font-size: 12px; }
  </style>
</head>
<body>
  <h1>LoRA Auto Dialer Report</h1>

  <div class="best">
    <h2>Best</h2>
    <p>
      <b>LoRA scale:</b> <code>{{ best.lora_scale }}</code> &nbsp;
      <b>Denoise:</b> <code>{{ best.strength }}</code> &nbsp;
      <b>CFG:</b> <code>{{ best.cfg }}</code> &nbsp;
      <b>Total:</b> <code>{{ best.total }}</code>
    </p>
    <p class="small">{{ best.scores.notes }}</p>
    <img src="{{ best.image_rel }}" />
  </div>

  <h2>Top refined</h2>
  <div class="grid">
    {% for c in top_refined %}
      <div class="card">
        <img src="{{ c.image_rel }}" />
        <div class="small">
          w={{ c.lora_scale }} s={{ c.strength }} cfg={{ c.cfg }}<br/>
          total={{ c.total }}<br/>
          id={{ c.scores.identity }} style={{ c.scores.style }} outfit={{ c.scores.outfit_attributes }} main={{ c.scores.main_preservation }}<br/>
          art={{ c.scores.artifacts }} over={{ c.scores.overbaked }}
        </div>
      </div>
    {% endfor %}
  </div>

  <h2>Top coarse</h2>
  <div class="grid">
    {% for c in top_coarse %}
      <div class="card">
        <img src="{{ c.image_rel }}" />
        <div class="small">
          w={{ c.lora_scale }} s={{ c.strength }} cfg={{ c.cfg }}<br/>
          total={{ c.total }}
        </div>
      </div>
    {% endfor %}
  </div>
</body>
</html>
""")

def write_report(run_dir: str, payload: dict) -> str:
    def rel(p): return os.path.relpath(p, run_dir).replace("\\", "/")

    best = payload["best"]
    best["image_rel"] = rel(best["image_path"])

    top_refined = payload["top_refined"]
    for c in top_refined:
        c["image_rel"] = rel(c["image_path"])

    top_coarse = payload["top_coarse"]
    for c in top_coarse:
        c["image_rel"] = rel(c["image_path"])

    html = TPL.render(best=best, top_refined=top_refined, top_coarse=top_coarse)
    out_path = os.path.join(run_dir, "report.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    return out_path
