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
                "mode": (["1_image", "2_image", "3_image", "4_image", "5_image", 
                         "6_image", "7_image", "8_image", "9_image", "10_image"],),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 64, "step": 1}),
                "width": ("INT", {"default": 1024, "min": 512, "max": 4096, "step": 8}),
                "height": ("INT", {"default": 1024, "min": 512, "max": 4096, "step": 8}),
            },
            "optional": {
                "vae": ("VAE",),
                "image1_mask": ("MASK",),
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "image4": ("IMAGE",),
                "image5": ("IMAGE",),
                "image6": ("IMAGE",),
                "image7": ("IMAGE",),
                "image8": ("IMAGE",),
                "image9": ("IMAGE",),
                "image10": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "negative", "latent")
    FUNCTION = "encode"
    CATEGORY = "advanced/conditioning"
    DESCRIPTION = "Flux image editing with dynamic image inputs"

    def encode(self, clip, prompt, mode, batch_size, width, height, vae=None, 
               image1_mask=None, image1=None, image2=None, image3=None, image4=None, 
               image5=None, image6=None, image7=None, image8=None, image9=None, image10=None):
        
        if vae is None:
            raise RuntimeError("VAE is required. Please connect a VAE loader.")
        
        all_images = [image1, image2, image3, image4, image5, 
                      image6, image7, image8, image9, image10]
        count = int(mode.split("_")[0])
        images = [img for i, img in enumerate(all_images[:count]) if img is not None]
        
        ref_latents = []
        vl_images = []
        noise_mask = None
        
        image_prompt_prefix = ""
        
        for i, image in enumerate(images):
            samples = image.movedim(-1, 1)
            current_total = samples.shape[3] * samples.shape[2]
            
            vl_total = int(384 * 384)
            vl_scale_by = math.sqrt(vl_total / current_total)
            vl_width = round(samples.shape[3] * vl_scale_by)
            vl_height = round(samples.shape[2] * vl_scale_by)
            
            s_vl = comfy.utils.common_upscale(samples, vl_width, vl_height, "area", "center")
            vl_image = s_vl.movedim(1, -1)
            vl_images.append(vl_image)
            
            image_prompt_prefix += f"image{i+1}: <|vision_start|><|image_pad|><|vision_end|> "
            
            vae_input_canvas = torch.zeros(
                (samples.shape[0], height, width, 3),
                dtype=samples.dtype,
                device=samples.device
            )
            
            resized_img = comfy.utils.common_upscale(samples, width, height, "lanczos", "center")
            resized_img = resized_img.movedim(1, -1)
            
            img_h, img_w = resized_img.shape[1], resized_img.shape[2]
            vae_input_canvas[:, :img_h, :img_w, :] = resized_img
            
            ref_latent = vae.encode(vae_input_canvas)
            ref_latents.append(ref_latent)
            
            if i == 0 and image1_mask is not None:
                mask = image1_mask
                if mask.dim() == 2:
                    mask_samples = mask.unsqueeze(0).unsqueeze(0)
                elif mask.dim() == 3:
                    mask_samples = mask.unsqueeze(1)
                else:
                    mask_samples = None
                
                if mask_samples is not None:
                    latent_width = width // 8
                    latent_height = height // 8
                    m = comfy.utils.common_upscale(mask_samples, latent_width, latent_height, "area", "center")
                    noise_mask = m.squeeze(1)
        
        full_prompt = image_prompt_prefix + prompt
        
        tokens = clip.tokenize(full_prompt, images=vl_images)
        positive_conditioning = clip.encode_from_tokens_scheduled(tokens)
        
        if len(ref_latents) > 0:
            positive_conditioning = node_helpers.conditioning_set_values(positive_conditioning, {"reference_latents": ref_latents}, append=True)
        
        negative_tokens = clip.tokenize("")
        negative_conditioning = clip.encode_from_tokens_scheduled(negative_tokens)
        
        if len(ref_latents) > 0:
            negative_conditioning = node_helpers.conditioning_set_values(negative_conditioning, {"reference_latents": ref_latents}, append=True)
        
        device = comfy.model_management.get_torch_device()
        dummy_pixels = torch.zeros(1, height, width, 3, device=device)
        empty_latent = vae.encode(dummy_pixels)
        
        latent = {"samples": empty_latent}
        
        if len(ref_latents) > 0:
            latent["samples"] = ref_latents[0]
            
        if noise_mask is not None:
            latent["noise_mask"] = noise_mask
            
        if batch_size > 1:
            positive_conditioning = positive_conditioning * batch_size
            negative_conditioning = negative_conditioning * batch_size
            
            samples = latent["samples"]
            if samples.shape[0] != batch_size:
                target_shape = [batch_size] + [1] * (samples.dim() - 1)
                samples = samples.repeat(*target_shape)
            latent["samples"] = samples
            
            if "noise_mask" in latent and noise_mask is not None:
                if latent["noise_mask"].shape[0] == 1 and batch_size > 1:
                    latent["noise_mask"] = latent["noise_mask"].repeat(batch_size, 1, 1)
        
        return (positive_conditioning, negative_conditioning, latent)


NODE_CLASS_MAPPINGS = {
    "PainterFluxImageEdit": PainterFluxImageEdit,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PainterFluxImageEdit": "Painter Flux Image Edit",
}
