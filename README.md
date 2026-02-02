# LorAid

## API-based image generation (no local model downloads)

This app can run image generation in two modes:

1. **diffusers (local)** – downloads a base model and uses your GPU/CPU.
2. **openai (API)** – calls OpenAI's Images API so you don't need to download models.

To use the OpenAI API backend:

1. Install the OpenAI SDK: `pip install openai`.
2. Set your API key: `export OPENAI_API_KEY="your-key"` (or add it to a `.env` file).
3. In the UI, set **Generator backend** to `openai`.
4. Set **Base model** to `gpt-image-1` (or another supported OpenAI image model).
5. Leave **LoRA file path** empty (LoRA parameters are ignored in OpenAI mode).

The judge already uses OpenAI's vision models via the Responses API, so you can also keep only the judge on OpenAI while using local diffusers for generation if desired.
