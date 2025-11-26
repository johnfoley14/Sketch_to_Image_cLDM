from pathlib import Path

import numpy as np
from PIL import Image
import torch


def _device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def _pil_to_tensor(img: Image.Image, device: torch.device) -> torch.Tensor:
    arr = (np.array(img).astype(np.float32) / 255.0)[None]  # (1,H,W,3)
    arr = np.transpose(arr, (0, 3, 1, 2))  # (1,3,H,W)
    x = torch.from_numpy(arr).to(device=device, dtype=torch.float32)
    x = x * 2.0 - 1.0  # [-1,1]
    return x


def _tensor_to_pil(x: torch.Tensor) -> Image.Image:
    x = x.detach().cpu().clamp(-1, 1)
    x = (x + 1.0) / 2.0
    arr = (x[0].permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    return Image.fromarray(arr)

def main():
    from diffusers import AutoencoderKL

    out_path = Path("images/recon.png")
    device = _device()

    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema", torch_dtype=torch.float32)
    vae.to(device).eval()

    img = Image.open(Path("images/original.jpg")).convert("RGB")
    x = _pil_to_tensor(img, device)

    with torch.no_grad():
        enc = vae.encode(x)
        z = enc.latent_dist.sample()  # sample from posterior
        x_rec = vae.decode(z).sample

    _tensor_to_pil(x_rec).save(out_path)
    print(f"Saved: {out_path.resolve()}")


if __name__ == "__main__":
    main()
