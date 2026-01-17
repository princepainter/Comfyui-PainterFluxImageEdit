import node_helpers
import comfy.utils
import math
import torch
import comfy.model_management

class PainterFluxImageEdit:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True}),
                "width": ("INT", {"default": 1024, "min": 512, "max": 4096, "step": 8}),
                "height": ("INT", {"default": 1024, "min": 512, "max": 4096, "step": 8}),
            },
            "optional": {
                "vae": ("VAE",),
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "image1_mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "negative", "latent")
    FUNCTION = "encode"
    CATEGORY = "advanced/conditioning"
    DESCRIPTION = "Flux2 image editing with up to 3 images support. Text-to-image when no images provided."

    def encode(self, clip, prompt, width=1024, height=1024, vae=None, image1=None, image2=None, image3=None, image1_mask=None):
        ref_latents = []
        vl_images = []
        noise_mask = None
        
        # 处理输入图片
        images = [image1, image2, image3]
        image_prompt_prefix = ""
        has_images = any(img is not None for img in images)
        
        # 确保文生图模式下也有 VAE
        if not has_images and vae is None:
            raise RuntimeError("VAE is required for text-to-image mode. Please connect a VAE loader.")
        
        for i, image in enumerate(images):
            if image is not None:
                samples = image.movedim(-1, 1)
                current_total = samples.shape[3] * samples.shape[2]
                
                # VL processing for CLIP vision (384x384)
                vl_total = int(384 * 384)
                vl_scale_by = math.sqrt(vl_total / current_total)
                vl_width = round(samples.shape[3] * vl_scale_by)
                vl_height = round(samples.shape[2] * vl_scale_by)
                
                s_vl = comfy.utils.common_upscale(samples, vl_width, vl_height, "area", "center")
                vl_image = s_vl.movedim(1, -1)
                vl_images.append(vl_image)
                
                image_prompt_prefix += f"image{i+1}: <|vision_start|><|image_pad|><|vision_end|> "
                
                # Encode to latent
                if vae is not None:
                    ori_longest_edge = max(samples.shape[2], samples.shape[3])
                    target_longest_edge = max(width, height)
                    scale_by = ori_longest_edge / target_longest_edge
                    
                    scaled_width = int(round(samples.shape[3] / scale_by))
                    scaled_height = int(round(samples.shape[2] / scale_by))
                    
                    vae_width = round(scaled_width / 8.0) * 8
                    vae_height = round(scaled_height / 8.0) * 8
                    
                    canvas_width = math.ceil(vae_width / 8.0) * 8
                    canvas_height = math.ceil(vae_height / 8.0) * 8
                    
                    canvas = torch.zeros(
                        (samples.shape[0], samples.shape[1], canvas_height, canvas_width),
                        dtype=samples.dtype,
                        device=samples.device
                    )
                    
                    resized_samples = comfy.utils.common_upscale(samples, vae_width, vae_height, "lanczos", "center")
                    resized_width = resized_samples.shape[3]
                    resized_height = resized_samples.shape[2]
                    canvas[:, :, :resized_height, :resized_width] = resized_samples
                    
                    image_for_vae = canvas.movedim(1, -1)
                    ref_latent = vae.encode(image_for_vae[:, :, :, :3])
                    ref_latents.append(ref_latent)
                    
                    # Process mask for image1
                    if i == 0 and image1_mask is not None:
                        mask = image1_mask
                        if mask.dim() == 2:
                            mask_samples = mask.unsqueeze(0).unsqueeze(0)
                        elif mask.dim() == 3:
                            mask_samples = mask.unsqueeze(1)
                        else:
                            print(f"Warning: Unexpected mask shape {mask.shape}, skipping mask processing")
                            mask_samples = None
                        
                        if mask_samples is not None:
                            latent_width = canvas_width // 8
                            latent_height = canvas_height // 8
                            resized_mask = comfy.utils.common_upscale(mask_samples, latent_width, latent_height, "area", "center")
                            noise_mask = resized_mask.squeeze(1)
        
        # Combine prompt
        full_prompt = image_prompt_prefix + prompt
        
        # Encode conditioning
        tokens = clip.tokenize(full_prompt, images=vl_images)
        positive_conditioning = clip.encode_from_tokens_scheduled(tokens)
        
        if len(ref_latents) > 0:
            positive_conditioning = node_helpers.conditioning_set_values(positive_conditioning, {"reference_latents": ref_latents}, append=True)
        
        negative_tokens = clip.tokenize("")
        negative_conditioning = clip.encode_from_tokens_scheduled(negative_tokens)
        
        if len(ref_latents) > 0:
            negative_conditioning = node_helpers.conditioning_set_values(negative_conditioning, {"reference_latents": ref_latents}, append=True)
        
        # **关键修复**：使用VAE创建空latent，确保尺寸一致
        device = comfy.model_management.get_torch_device()
        dummy_pixels = torch.zeros(1, height, width, 3, device=device)
        # VAE编码时会自动处理尺寸和通道
        empty_latent = vae.encode(dummy_pixels)
        
        latent = {"samples": empty_latent}
        if len(ref_latents) > 0:
            latent["samples"] = ref_latents[0]
            if noise_mask is not None:
                latent["noise_mask"] = noise_mask
        
        return (positive_conditioning, negative_conditioning, latent)


# Register the node
NODE_CLASS_MAPPINGS = {
    "PainterFluxImageEdit": PainterFluxImageEdit,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PainterFluxImageEdit": "Painter Flux Image Edit",
}
