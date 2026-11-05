# load_local_clip.py
import os, sys
sys.path.append("/workdir1.8t/fei27/CGT/GNNMD/BACH/updated_X-MVLP")

import torch
import open_clip
import json


def load_local_biomedclip(model_dir, device="cuda"):
    """
    Load BioMedCLIP entirely from local files, no internet required.
    """
    config_path = os.path.join(model_dir, "open_clip_config.json")
    weights_path = os.path.join(model_dir, "open_clip_pytorch_model.bin")

    # Load config
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights not found: {weights_path}")

    with open(config_path, 'r') as f:
        model_config = json.load(f)

    # Extract architecture info
    model_name = model_config.get("model", "ViT-B-16")  # e.g., "ViT-B-16"
    embed_dim = model_config.get("embed_dim", 512)
    vision_cfg = model_config.get("vision_cfg", {})
    text_cfg = model_config.get("text_cfg", {})

    # Print model info
    print(f"Loading {model_name} with embed_dim={embed_dim}")
    print(f"Vision: {vision_cfg}")
    print(f"Text: {text_cfg}")

    # Create model architecture
    model = open_clip.create_model(
        model_name=model_name,
        pretrained=None,
        embed_dim=embed_dim,
        **vision_cfg,
        **text_cfg
    )

    # Load state dict
    state_dict = torch.load(weights_path, map_location="cpu")

    # OpenCLIP saves as `state_dict` inside a dict sometimes
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    elif "model" in state_dict:
        state_dict = state_dict["model"]

    # Remove "module." prefix if present
    new_state_dict = {}
    for k, v in state_dict.items():
        k = k.replace("module.", "")
        new_state_dict[k] = v

    # Load
    incompatible = model.load_state_dict(new_state_dict, strict=False)
    print("Loaded weights with:")
    print("  Missing keys:", incompatible.missing_keys)
    print("  Unexpected keys:", incompatible.unexpected_keys)

    # Move to device
    model.to(device)
    model.eval()

    # Get tokenizer
    tokenizer = open_clip.get_tokenizer(model_name)

    return model, tokenizer, model.visual.image_size

# Usage
# load_local_clip.py (fix for img_size)

if __name__ == "__main__":
    model_dir = "/workdir1.8t/fei27/CGT/GNNMD/BACH/biomedclip_local"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, tokenizer, img_size = load_local_biomedclip(model_dir, device)
    print("✅ Model loaded successfully in offline mode!")

    # --- �� Fix: Handle img_size whether it's int or tuple ---
    if isinstance(img_size, (list, tuple)):
        H, W = img_size[0], img_size[1]
    else:
        H = W = img_size

    # Create dummy image input
    dummy_img = torch.randn(1, 3, H, W).to(device)

    # Tokenize text
    text = tokenizer(["a histology image of cancer"]).to(device)

    # Run inference
    with torch.no_grad():
        image_features = model.encode_image(dummy_img)
        text_features = model.encode_text(text)

    # Output shapes
    print("Image features shape:", image_features.shape)      # e.g., [1, 512]
    print("Text features shape:", text_features.shape)        # e.g., [1, 512]
    print("Image size used:", (H, W))
