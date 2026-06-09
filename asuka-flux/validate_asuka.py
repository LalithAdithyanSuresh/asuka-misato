import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
from einops import rearrange
from omegaconf import OmegaConf

# Import repo modules
from src.flux.util import load_ae, load_flow_model
from src.flux.sampling import denoise, get_noise, get_schedule, unpack
from MAE.util import misc
from datas.misato_dataset import prepare_data

# Set tokenizer parallelism environment variable to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def calc_psnr(gt, pre):
    # gt and pre are numpy arrays of shape (H, W, 3) in range 0-255
    mse = np.mean((gt.astype(np.float64) - pre.astype(np.float64)) ** 2)
    if mse == 0:
        return 100.0
    return 20 * np.log10(255.0 / np.sqrt(mse))

def get_alignment_clip():
    from src.flux.modules.layers import SingleStreamBlockAsuka
    dim = 768
    head_num = dim // 64
    alignment = nn.Sequential(
        nn.Linear(512, dim),
        SingleStreamBlockAsuka(dim, head_num),
        SingleStreamBlockAsuka(dim, head_num),
        SingleStreamBlockAsuka(dim, head_num),
        SingleStreamBlockAsuka(dim, head_num),
        nn.LayerNorm(dim),
    )
    return alignment

def get_alignment_flant():
    from src.flux.modules.layers import SingleStreamBlockAsuka
    dim = 4096
    head_num = dim // 64
    alignment = nn.Sequential(
        nn.Linear(512, dim),
        SingleStreamBlockAsuka(dim, head_num),
        SingleStreamBlockAsuka(dim, head_num),
        SingleStreamBlockAsuka(dim, head_num),
        SingleStreamBlockAsuka(dim, head_num),
        nn.LayerNorm(dim),
    )
    return alignment

def get_visual_learned_conditioning(visual_condition_extractor, alignment_clip, alignment_flant, x, mask):
    with torch.no_grad():
        x = visual_condition_extractor.forward_return_feature(x, mask, decoder_layer=6).detach()
        if torch.any(torch.isnan(x)):
            print('[!] Warning: NaN found in MAE feature, setting to zeros')
            x = torch.zeros_like(x)

    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        clip_fea = alignment_clip(x)
        flant_fea = alignment_flant(x)
    clip_fea = torch.mean(clip_fea, dim=1)
    return clip_fea, flant_fea

def prepare_mae_model(chkpt_dir, arch='mae_vit_base_patch16', random_mask=False, finetune=False, mae_mask_concat=False):
    model = misc.get_mae_model(arch, random_mask=random_mask, finetune=finetune, mae_mask_concat=mae_mask_concat)
    checkpoint = torch.load(chkpt_dir, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    print(f"Loaded MAE checkpoint from {chkpt_dir}. Status: {msg}")
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, default='/tmp/cks/SEM-Net/datasets/places365/test_256', help='Directory with source images')
    parser.add_argument('--mask_dir', type=str, default='/tmp/cks/SEM-Net/datasets/testing_mask_dataset', help='Directory with masks')
    parser.add_argument('--output_dir', type=str, default='./results', help='Directory to save results')
    parser.add_argument('--decoder_ckpt_path', type=str, default='ckpt/asuka_decoder.ckpt', help='Decoder checkpoint path')
    parser.add_argument('--condition_weight', type=float, default=0.5, help='Condition weight for ASUKA CFG')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    weight_dtype = torch.bfloat16
    print(f"Using device: {device} | dtype: {weight_dtype}")

    # Set random seeds
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # 1. Load flow model and VAE (AutoEncoder)
    print("[*] Loading FLUX.1-Fill-dev base models...")
    flow_transformer = load_flow_model("flux-dev-fill", device=device)
    vae = load_ae("flux-dev-fill", device=device)

    # 2. Modify VAE decoder to ASUKA custom decoder
    print("[*] Replacing VAE decoder with ASUKA conditional decoder...")
    from ldm.modules.diffusionmodules.asuka_decoder import Decoder
    ddconfig = OmegaConf.load("./configs/condition_decoder.yaml")
    decoder = Decoder(**ddconfig)
    
    if not os.path.exists(args.decoder_ckpt_path):
        raise FileNotFoundError(f"ASUKA decoder checkpoint not found at {args.decoder_ckpt_path}")
        
    state_dict = torch.load(args.decoder_ckpt_path, map_location='cpu')['state_dict']
    flux_ae = {}
    for k, v in state_dict.items():
        if k.startswith("encoder.") or k.startswith("decoder."):
            flux_ae[k] = v

    vae.decoder = decoder
    vae.load_state_dict(flux_ae, strict=False)

    # 3. Load alignment models
    print("[*] Loading ASUKA alignment models...")
    alignment_clip = get_alignment_clip()
    alignment_flant = get_alignment_flant()
    
    alignment_clip.load_state_dict(torch.load("./ckpt/asuka_alignment_clip.pt", map_location='cpu'))
    alignment_flant.load_state_dict(torch.load("./ckpt/asuka_alignment_t5.pt", map_location='cpu'))

    # 4. Load MAE visual condition extractor
    print("[*] Loading MAE visual condition extractor...")
    visual_condition_extractor = prepare_mae_model('ckpt/mae_300.pth', random_mask=False, finetune=True, mae_mask_concat=False)

    # Move everything to device and weight_dtype
    flow_transformer.to(device, dtype=weight_dtype).eval()
    vae.to(device, dtype=weight_dtype).eval()
    alignment_clip.to(device, dtype=weight_dtype).eval()
    alignment_flant.to(device, dtype=weight_dtype).eval()
    visual_condition_extractor.to(device, dtype=weight_dtype).eval()

    # Load non-conditional features (needed for CFG alignment)
    print("[*] Loading non-conditional features...")
    none_clip_fea = torch.load("./ckpt/vec.pt").to(device, dtype=weight_dtype)
    none_flant_fea = torch.load("./ckpt/txt_256.pt").to(device, dtype=weight_dtype)

    # Targets to evaluate
    targets = [
        # (category, image_id, mask_id)
        ('LARGE', 'Places365_test_00032703', '08398'),
        ('LARGE', 'Places365_test_00081016', '08986'),
        ('LARGE', 'Places365_test_00014790', '08180'),
        ('LARGE', 'Places365_test_00217740', '10650'),
        ('MEDIUM', 'Places365_test_00018734', '04228'),
    ]

    os.makedirs(args.output_dir, exist_ok=True)
    for cat in ['LARGE', 'MEDIUM']:
        os.makedirs(os.path.join(args.output_dir, cat), exist_ok=True)

    print("\n--- Starting ASUKA Validation on 5 Target Images ---")
    results = []

    for cat, img_id, mask_id in targets:
        print(f"\nProcessing {img_id} with mask {mask_id} ({cat})...")

        # Locate image
        img_path = None
        for ext in ['.jpg', '.png', '.jpeg', '.JPG', '.PNG', '.JPEG']:
            p = os.path.join(args.image_dir, f"{img_id}{ext}")
            if os.path.exists(p):
                img_path = p
                break
        
        if not img_path:
            print(f"Error: Source image {img_id} not found in {args.image_dir}. Skipping.")
            continue

        # Locate mask
        mask_path = None
        for ext in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']:
            p = os.path.join(args.mask_dir, f"{mask_id}{ext}")
            if os.path.exists(p):
                mask_path = p
                break

        if not mask_path:
            print(f"Error: Mask image {mask_id} not found in {args.mask_dir}. Skipping.")
            continue

        # Load original image and mask
        pil_img = Image.open(img_path).convert('RGB')
        pil_mask = Image.open(mask_path).convert('L')
        orig_w, orig_h = pil_img.size

        # ASUKA resolves pre-extracted items at 512, so we resize target to 512x512
        img_512 = pil_img.resize((512, 512), Image.Resampling.LANCZOS)
        mask_512 = pil_mask.resize((512, 512), Image.Resampling.NEAREST)

        img_512_np = np.array(img_512)
        mask_512_np = np.array(mask_512) / 255.0
        mask_512_np = mask_512_np.astype(np.float32)

        # Prepare MAE data (expects 256x256 normalized with ImageNet statistics)
        mask_mae, unmasked_img_mae = prepare_data(img_512_np, mask_512_np)
        mae_tensor = unmasked_img_mae.unsqueeze(0).to(device, dtype=weight_dtype)
        mask_mae_tensor = mask_mae.unsqueeze(0).to(device, dtype=weight_dtype)

        # Extract visual condition features
        clip_fea, flant_fea = get_visual_learned_conditioning(
            visual_condition_extractor, alignment_clip, alignment_flant, mae_tensor, mask_mae_tensor
        )
        clip_fea = clip_fea.to(dtype=weight_dtype)
        flant_fea = flant_fea.to(dtype=weight_dtype)

        # Apply Classifier-Free Guidance (CFG) alignment scale
        clip_fea = none_clip_fea + args.condition_weight * (clip_fea - none_clip_fea)
        flant_fea = none_flant_fea + args.condition_weight * (flant_fea - none_flant_fea)

        # Prepare FLUX-Fill inputs (orig_img in [-1, 1], mask_cond packed, img_cond encoded)
        orig_img_tensor = torch.from_numpy(img_512_np).float() / 127.5 - 1.0
        orig_img_tensor = rearrange(orig_img_tensor, "h w c -> c h w").unsqueeze(0).to(device, dtype=weight_dtype)

        mask_tensor = torch.from_numpy(mask_512_np).float().unsqueeze(0).unsqueeze(0).to(device, dtype=weight_dtype)

        # Build mask_cond
        mask_cond = mask_512_np.copy()
        mask_cond = mask_cond[None] # Shape (1, 512, 512)
        mask_cond = rearrange(mask_cond, "b (h ph) (w pw) -> b (ph pw) h w", ph=8, pw=8)
        mask_cond = rearrange(mask_cond, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)[0]
        mask_cond_tensor = torch.from_numpy(mask_cond).float().unsqueeze(0).to(device, dtype=weight_dtype)

        # Encode conditioning image (known region, where mask is 0)
        img_cond = orig_img_tensor * (1 - mask_tensor)
        with torch.no_grad():
            img_cond_encoded = vae.encode(img_cond)
        img_cond_encoded = img_cond_encoded.to(torch.bfloat16)
        img_cond_encoded = rearrange(img_cond_encoded, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
        img_cond_input = torch.cat((img_cond_encoded, mask_cond_tensor), dim=-1)

        # Load static assets
        img_ids = torch.load("./ckpt/img_ids_512.pt").repeat((1, 1, 1)).to(device, dtype=weight_dtype)
        txt = torch.load("./ckpt/txt.pt").repeat((1, 1, 1)).to(device, dtype=weight_dtype)

        # Get initial noise
        noise = get_noise(
            1,
            512,
            512,
            device=device,
            dtype=torch.bfloat16,
            seed=42
        )
        img_input = rearrange(noise, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)

        # Setup input dictionary
        inp = {
            'img': img_input,
            'img_ids': img_ids,
            'txt': flant_fea,
            'txt_ids': torch.zeros(1, flant_fea.shape[1], 3).to(device, dtype=weight_dtype),
            'vec': clip_fea,
            'img_cond': img_cond_input
        }

        # Timesteps schedule
        timesteps = get_schedule(50, inp["img"].shape[1], shift=True)

        # Inference loop
        print("[*] Running denoising loop (50 steps)...")
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
            denoised_x = denoise(flow_transformer, **inp, timesteps=timesteps, guidance=30.0)
            denoised_x = unpack(denoised_x, 512, 512)
            
            # Decode using custom decoder (haromonization with background)
            preds = vae.my_decode(denoised_x, orig_img_tensor, mask_tensor)
            preds = (preds.permute(0, 2, 3, 1) + 1.0) / 2.0
            pred_np = (preds.float().cpu().numpy()[0] * 255.0).clip(0, 255).astype(np.uint8)

        # Downscale prediction to original size (256x256)
        pred_pil = Image.fromarray(pred_np).resize((orig_w, orig_h), Image.Resampling.BILINEAR)
        pred_resized_np = np.array(pred_pil)

        # Blend with original image using the mask
        gt_np = np.array(pil_img)
        mask_np = np.array(pil_mask) / 255.0
        mask_np = mask_np[:, :, None] # broadcastable shape
        merged_np = (pred_resized_np * mask_np + gt_np * (1.0 - mask_np)).clip(0, 255).astype(np.uint8)

        # Calculate PSNR
        psnr = calc_psnr(gt_np, merged_np)
        print(f"Successfully inpainted. PSNR: {psnr:.2f}")

        # Save output
        save_name = f"{img_id}_{mask_id}_{psnr:.2f}.png"
        save_path = os.path.join(args.output_dir, cat, save_name)
        Image.fromarray(merged_np).save(save_path)
        print(f"Saved output to: {save_path}")

        results.append({
            'Category': cat,
            'Image ID': img_id,
            'Mask ID': mask_id,
            'PSNR': f"{psnr:.2f}"
        })

    print("\n" + "="*60)
    print("ASUKA MODEL EVALUATION SUMMARY")
    print("="*60)
    for r in results:
        print(f"[{r['Category']}] Image: {r['Image ID']} | Mask: {r['Mask ID']} | PSNR: {r['PSNR']}")
    print("="*60)

if __name__ == '__main__':
    main()
