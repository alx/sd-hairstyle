# Haircut Transfer API — Quick Reference

## Setup

```bash
pip install flask pillow

# Set your paths
export SD_BINARY=./bin/sd-cli
export SD_MODEL=./models/v1-5-pruned-emaonly.safetensors
export SD_VAE=./models/vae.safetensors   # optional

python app.py
```

## POST /transfer-haircut

| Field       | Type   | Default | Notes                                       |
|-------------|--------|---------|---------------------------------------------|
| selfie      | file   | —       | Subject photo (any common format)           |
| hairstyle   | file   | —       | Reference haircut photo                     |
| prompt      | str    | ""      | Extra descriptive text appended to prompt   |
| strength    | float  | 0.65    | Deviation from original (0.1–0.95)          |
| steps       | int    | 25      | Denoising steps                             |
| cfg_scale   | float  | 7.0     | Classifier-free guidance scale              |
| seed        | int    | -1      | -1 = random                                 |
| width       | int    | 512     | Output width (rounded to nearest 64)        |
| height      | int    | 512     | Output height (rounded to nearest 64)       |

Returns: `image/jpeg`

## curl examples

### Basic usage
```bash
curl -X POST http://localhost:5000/transfer-haircut \
  -F "selfie=@/path/to/selfie.jpg" \
  -F "hairstyle=@/path/to/reference_haircut.jpg" \
  -o result.jpg
```

### With all parameters
```bash
curl -X POST http://localhost:5000/transfer-haircut \
  -F "selfie=@selfie.jpg" \
  -F "hairstyle=@target_hair.jpg" \
  -F "prompt=natural lighting, skin texture preserved" \
  -F "strength=0.60" \
  -F "steps=30" \
  -F "cfg_scale=7.5" \
  -F "seed=42" \
  -F "width=512" \
  -F "height=512" \
  -o result.jpg
```

### Health check
```bash
curl http://localhost:5000/health
```

## Python client example

```python
import requests

with open("selfie.jpg", "rb") as selfie, \
     open("hairstyle.jpg", "rb") as hairstyle:
    resp = requests.post(
        "http://localhost:5000/transfer-haircut",
        files={
            "selfie":    ("selfie.jpg",    selfie,    "image/jpeg"),
            "hairstyle": ("hairstyle.jpg", hairstyle, "image/jpeg"),
        },
        data={
            "strength":  0.65,
            "steps":     25,
            "cfg_scale": 7.0,
            "seed":      -1,
        },
        timeout=360,
    )

if resp.status_code == 200:
    with open("result.jpg", "wb") as f:
        f.write(resp.content)
    print("Saved result.jpg")
else:
    print("Error:", resp.json())
```

## Tuning notes

| Goal                              | Adjustment                          |
|-----------------------------------|-------------------------------------|
| Keep more of the original face    | Lower `strength` (0.4–0.55)         |
| More dramatic hair change         | Raise `strength` (0.70–0.85)        |
| Faster inference                  | Lower `steps` (15–20), LCM model    |
| Better quality                    | Raise `steps` (40–50), lower CFG    |
| SDXL / 1024 px output             | Use SDXL model, set width/height    |

## Using ControlNet for better face preservation (advanced)

If you need the face to stay more faithful, add ControlNet (depth or
canny) by appending these flags inside `run_sd()` in app.py:

```
"--control-net",  "/path/to/control_net_model.safetensors",
"--control-image", selfie_resized_path,
"--control-strength", "0.8",
```

ControlNet keeps the pose/structure while img2img rewrites the hair.

## Using PhotoMaker instead (SDXL only)

For stronger identity preservation, use PhotoMaker mode:

```
"--photo-maker",       "/path/to/photomaker-v1.safetensors",
"--pm-id-images-dir",  "/dir/containing/selfies/",
"--pm-style-strength", "15",
```

And adapt the prompt to include the class word + trigger:
`"a woman img with <hairstyle_desc> hair"`
