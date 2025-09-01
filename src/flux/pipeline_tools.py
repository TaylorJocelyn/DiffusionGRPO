import torch
from torch import Tensor
from diffusers.pipelines import FluxPipeline
from diffusers.utils import logging
import diffusers
from diffusers.pipelines.flux.pipeline_flux import logger
from diffusers.image_processor import VaeImageProcessor

def encode_images(pipeline: FluxPipeline, images: Tensor):
    images = pipeline.image_processor.preprocess(images)
    images = images.to(pipeline.device).to(pipeline.dtype)
    images = pipeline.vae.encode(images).latent_dist.sample()
    images = (
        images - pipeline.vae.config.shift_factor
    ) * pipeline.vae.config.scaling_factor
    images_tokens = pipeline._pack_latents(images, *images.shape)
    images_ids = pipeline._prepare_latent_image_ids(
        images.shape[0],
        images.shape[2],
        images.shape[3],
        pipeline.device,
        pipeline.dtype,
    )
    if images_tokens.shape[1] != images_ids.shape[0]:
        images_ids = pipeline._prepare_latent_image_ids(
            images.shape[0],
            images.shape[2] // 2,
            images.shape[3] // 2,
            pipeline.device,
            pipeline.dtype,
        )
    return images_tokens, images_ids



def resize_position_encoding(prepare_img_ids_func, batch_size, original_height, original_width, target_height, target_width, device, dtype):
    """
    original_height/width: noise image size / target size
    target_height/width: condition image size
    """
    latent_image_ids = prepare_img_ids_func(batch_size, original_height, original_width, device, dtype)

    scale_h = original_height / target_height
    scale_w = original_width / target_width
    bias_h, bias_w = scale_h / 2 - 0.5, scale_w / 2 - 0.5
    latent_image_ids_resized = torch.zeros(target_height//2, target_width//2, 3, device=device, dtype=dtype)
    latent_image_ids_resized[..., 1] = latent_image_ids_resized[..., 1] + torch.arange(target_height//2, device=device)[:, None] * scale_h + bias_h
    latent_image_ids_resized[..., 2] = latent_image_ids_resized[..., 2] + torch.arange(target_width//2, device=device)[None, :] * scale_w + bias_w
    
    cond_latent_image_id_height, cond_latent_image_id_width, cond_latent_image_id_channels = latent_image_ids_resized.shape
    cond_latent_image_ids = latent_image_ids_resized.reshape(
            cond_latent_image_id_height * cond_latent_image_id_width, cond_latent_image_id_channels
        )
    return cond_latent_image_ids 

def _pack_latents(latents, batch_size, num_channels_latents, height, width):
        latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)

        return latents

def _prepare_latent_image_ids(batch_size, height, width, device, dtype):
        latent_image_ids = torch.zeros(height, width, 3)
        latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height)[:, None]
        latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width)[None, :]

        latent_image_id_height, latent_image_id_width, latent_image_id_channels = latent_image_ids.shape

        latent_image_ids = latent_image_ids.reshape(
            latent_image_id_height * latent_image_id_width, latent_image_id_channels
        )

        return latent_image_ids.to(device=device, dtype=dtype)

# def encode_cond_images(pipeline: FluxPipeline, images: Tensor, height:int, width:int, vae_scale_factor:int):
#     """
#     height: noise image height / target size
#     width: noise image width / target size
#     """

#     images = pipeline.image_processor.preprocess(images)
#     images = images.to(pipeline.device).to(pipeline.dtype)
#     images = pipeline.vae.encode(images).latent_dist.sample()
#     images = (
#         images - pipeline.vae.config.shift_factor
#     ) * pipeline.vae.config.scaling_factor
#     images_tokens = _pack_latents(images, *images.shape)

#     batch_size, height_cond, width_cond = images.shape[0], images.shape[2], images.shape[3]
#     height = 2 * (int(height) // vae_scale_factor)  
#     width = 2 * (int(width) // vae_scale_factor)

#     images_ids = resize_position_encoding(
#         _prepare_latent_image_ids,
#         batch_size,
#         height,
#         width,
#         height_cond, # height_cond
#         width_cond, # width_cond
#         pipeline.device,
#         pipeline.dtype
#     )

#     if images_tokens.shape[1] != images_ids.shape[0]:
#         raise NotImplementedError("Recheck code")
#         image_ids = resize_position_encoding(
#             _prepare_latent_image_ids,
#             batch_size,
#             height,
#             width,
#             height_cond // 2, # height_cond
#             width_cond // 2, # width_cond
#             pipeline.device,
#             pipeline.dtype
#         )
#     return images_tokens, images_ids

def encode_cond_images(vae: diffusers.models.autoencoders.autoencoder_kl.AutoencoderKL, images: torch.Tensor, height:int, width:int, device: torch.device, dtype: torch.dtype):
    """
    height: noise image height / target size
    width: noise image width / target size
    """
    vae_scale_factor = (
                2 ** (len(vae.config.block_out_channels))
            )
    image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor * 2)
    images = image_processor.preprocess(images)
    images = images.to(device).to(dtype)
    images = vae.encode(images).latent_dist.sample()
    images = (
        images - vae.config.shift_factor
    ) * vae.config.scaling_factor
    images_tokens = _pack_latents(images, *images.shape)

    batch_size, height_cond, width_cond = images.shape[0], images.shape[2], images.shape[3]
    height = 2 * (int(height) // vae_scale_factor)  
    width = 2 * (int(width) // vae_scale_factor)

    images_ids = resize_position_encoding(
        _prepare_latent_image_ids,
        batch_size,
        height,
        width,
        height_cond, # height_cond
        width_cond, # width_cond
        device,
        dtype
    )

    if images_tokens.shape[1] != images_ids.shape[0]:
        raise NotImplementedError("Recheck code")
        image_ids = resize_position_encoding(
            pipeline._prepare_latent_image_ids,
            batch_size,
            height,
            width,
            height_cond // 2, # height_cond
            width_cond // 2, # width_cond
            pipeline.device,
            pipeline.dtype
        )
    return images_tokens, images_ids


def prepare_text_input(pipeline: FluxPipeline, prompts, max_sequence_length=512):
    # Turn off warnings (CLIP overflow)
    logger.setLevel(logging.ERROR)
    (
        prompt_embeds,
        pooled_prompt_embeds,
        text_ids,
    ) = pipeline.encode_prompt(
        prompt=prompts,
        prompt_2=None,
        prompt_embeds=None,
        pooled_prompt_embeds=None,
        device=pipeline.device,
        num_images_per_prompt=1,
        max_sequence_length=max_sequence_length,
        lora_scale=None,
    )
    # Turn on warnings
    logger.setLevel(logging.WARNING)
    return prompt_embeds, pooled_prompt_embeds, text_ids
