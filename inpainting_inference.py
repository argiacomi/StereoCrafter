import gc
import os

import cv2
import numpy as np
import torch
from decord import VideoReader, cpu
from diffusers import AutoencoderKLTemporalDecoder, UNetSpatioTemporalConditionModel
from fire import Fire
from pipelines.stereo_video_inpainting import (
    StableVideoDiffusionInpaintingPipeline,
    tensor2vid,
)
from torch_runtime_utils import (
    configure_cuda_performance_flags,
    is_cuda_oom,
    mark_torch_compile_step_begin,
)
from transformers import CLIPVisionModelWithProjection

configure_cuda_performance_flags()


def spatial_tile_shape(height, width, tile_num, tile_overlap=(128, 128)):
    tile_size = (
        int((height + tile_overlap[0] * (tile_num - 1)) / tile_num),
        int((width + tile_overlap[1] * (tile_num - 1)) / tile_num),
    )
    tile_stride = (tile_size[0] - tile_overlap[0], tile_size[1] - tile_overlap[1])
    return tile_size, tile_stride


def max_supported_tile_num(height, width, tile_overlap=(128, 128)):
    tile_num = 1
    while True:
        _, tile_stride = spatial_tile_shape(height, width, tile_num, tile_overlap)
        if tile_stride[0] <= 0 or tile_stride[1] <= 0:
            return max(1, tile_num - 1)
        tile_num += 1


def blend_h(a: torch.Tensor, b: torch.Tensor, overlap_size: int) -> torch.Tensor:
    weight_b = (torch.arange(overlap_size).view(1, 1, 1, -1) / overlap_size).to(
        b.device
    )
    b[:, :, :, :overlap_size] = (1 - weight_b) * a[
        :, :, :, -overlap_size:
    ] + weight_b * b[:, :, :, :overlap_size]
    return b


def blend_v(a: torch.Tensor, b: torch.Tensor, overlap_size: int) -> torch.Tensor:
    weight_b = (torch.arange(overlap_size).view(1, 1, -1, 1) / overlap_size).to(
        b.device
    )
    b[:, :, :overlap_size, :] = (1 - weight_b) * a[
        :, :, -overlap_size:, :
    ] + weight_b * b[:, :, :overlap_size, :]
    return b


def spatial_tiled_process(
    cond_frames,
    mask_frames,
    process_func,
    tile_num,
    spatial_n_compress=8,
    **kargs,
):
    height = cond_frames.shape[2]
    width = cond_frames.shape[3]

    tile_overlap = (128, 128)
    tile_size, tile_stride = spatial_tile_shape(height, width, tile_num, tile_overlap)
    if tile_stride[0] <= 0 or tile_stride[1] <= 0:
        raise ValueError(
            f"`tile_num={tile_num}` is too large for frame size {(height, width)} "
            f"with tile overlap {tile_overlap}."
        )

    cols = []
    for i in range(0, tile_num):
        rows = []
        for j in range(0, tile_num):
            cond_tile = cond_frames[
                :,
                :,
                i * tile_stride[0] : i * tile_stride[0] + tile_size[0],
                j * tile_stride[1] : j * tile_stride[1] + tile_size[1],
            ]
            mask_tile = mask_frames[
                :,
                :,
                i * tile_stride[0] : i * tile_stride[0] + tile_size[0],
                j * tile_stride[1] : j * tile_stride[1] + tile_size[1],
            ]

            tile = process_func(
                frames=cond_tile,
                frames_mask=mask_tile,
                height=cond_tile.shape[2],
                width=cond_tile.shape[3],
                num_frames=len(cond_tile),
                output_type="latent",
                **kargs,
            ).frames[0]

            rows.append(tile)
        cols.append(rows)

    latent_stride = (
        tile_stride[0] // spatial_n_compress,
        tile_stride[1] // spatial_n_compress,
    )
    latent_overlap = (
        tile_overlap[0] // spatial_n_compress,
        tile_overlap[1] // spatial_n_compress,
    )

    results_cols = []
    for i, rows in enumerate(cols):
        results_rows = []
        for j, tile in enumerate(rows):
            if i > 0:
                tile = blend_v(cols[i - 1][j], tile, latent_overlap[0])
            if j > 0:
                tile = blend_h(rows[j - 1], tile, latent_overlap[1])
            results_rows.append(tile)
        results_cols.append(results_rows)

    pixels = []
    for i, rows in enumerate(results_cols):
        for j, tile in enumerate(rows):
            if i < len(results_cols) - 1:
                tile = tile[:, :, : latent_stride[0], :]
            if j < len(rows) - 1:
                tile = tile[:, :, :, : latent_stride[1]]
            rows[j] = tile
        pixels.append(torch.cat(rows, dim=3))
    x = torch.cat(pixels, dim=2)
    return x


def create_video_writer(output_video_path, fps, width, height):
    return cv2.VideoWriter(
        output_video_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )


def write_video_chunk(writer, input_frames):
    for frame in input_frames:
        writer.write(frame[:, :, ::-1])


def load_inpainting_chunk(video_reader, start, stop, crop_height, crop_width):
    frames = video_reader.get_batch(list(range(start, stop))).asnumpy()
    frames = torch.from_numpy(frames).permute(0, 3, 1, 2).contiguous().float()
    frames = frames[:, :, : crop_height * 2, : crop_width * 2]
    frames = frames / 255.0

    frames_left = frames[:, :, :crop_height, :crop_width]
    frames_mask = frames[:, :, crop_height : crop_height * 2, :crop_width]
    frames_warpped = frames[:, :, crop_height : crop_height * 2, crop_width : crop_width * 2]
    frames_mask = frames_mask.mean(dim=1, keepdim=True)
    return frames_warpped, frames_left, frames_mask


def write_output_chunk(sbs_writer, anaglyph_writer, frames_left, frames_right):
    left_rgb = (
        frames_left.mul(255)
        .clamp_(0, 255)
        .permute(0, 2, 3, 1)
        .to(dtype=torch.uint8)
        .cpu()
        .numpy()
    )
    right_rgb = (
        frames_right.mul(255)
        .clamp_(0, 255)
        .permute(0, 2, 3, 1)
        .to(dtype=torch.uint8)
        .cpu()
        .numpy()
    )

    sbs_rgb = np.concatenate([left_rgb, right_rgb], axis=2)
    anaglyph_rgb = np.zeros_like(left_rgb)
    anaglyph_rgb[..., 0] = left_rgb[..., 0]
    anaglyph_rgb[..., 1] = right_rgb[..., 1]
    anaglyph_rgb[..., 2] = right_rgb[..., 2]

    write_video_chunk(sbs_writer, sbs_rgb)
    write_video_chunk(anaglyph_writer, anaglyph_rgb)


def main(
    pre_trained_path,
    unet_path,
    input_video_path,
    save_dir,
    frames_chunk=23,
    overlap=3,
    tile_num=1,
    guidance_scale=1.0,
    num_inference_steps=8,
    decode_chunk_size=None,
    vae_encode_chunk_size=None,
    noise_aug_strength=0.0,
    cpu_offload=None,
):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for stereo video inpainting.")
    if frames_chunk <= overlap:
        raise ValueError(
            f"`frames_chunk` must be larger than `overlap`, but got {frames_chunk=} and {overlap=}."
        )
    if tile_num < 1:
        raise ValueError(f"`tile_num` must be at least 1, but got {tile_num}.")

    decode_chunk_size = min(decode_chunk_size or frames_chunk, frames_chunk)

    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        pre_trained_path,
        subfolder="image_encoder",
        variant="fp16",
        torch_dtype=torch.float16,
    )

    vae = AutoencoderKLTemporalDecoder.from_pretrained(
        pre_trained_path, subfolder="vae", variant="fp16", torch_dtype=torch.float16
    )

    unet = UNetSpatioTemporalConditionModel.from_pretrained(
        unet_path,
        subfolder="unet_diffusers",
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
    )

    image_encoder.requires_grad_(False)
    vae.requires_grad_(False)
    unet.requires_grad_(False)

    pipeline = StableVideoDiffusionInpaintingPipeline.from_pretrained(
        pre_trained_path,
        image_encoder=image_encoder,
        vae=vae,
        unet=unet,
        torch_dtype=torch.float16,
    )

    if cpu_offload is None:
        pipeline = pipeline.to("cuda")
        pipeline.unet = torch.compile(pipeline.unet, mode="max-autotune")
        pipeline.vae.decoder = torch.compile(pipeline.vae.decoder, mode="max-autotune")
    elif cpu_offload == "sequential":
        pipeline.enable_sequential_cpu_offload()
    elif cpu_offload == "model":
        pipeline.enable_model_cpu_offload()
    else:
        raise ValueError(
            f"Unknown cpu_offload mode '{cpu_offload}'. Expected None, 'sequential', or 'model'."
        )

    if tile_num > 1 and hasattr(pipeline.vae, "enable_tiling"):
        pipeline.vae.enable_tiling()

    os.makedirs(save_dir, exist_ok=True)
    video_name = (
        input_video_path.split("/")[-1]
        .replace(".mp4", "")
        .replace("_splatting_results", "")
        + "_inpainting_results"
    )

    video_reader = VideoReader(input_video_path, ctx=cpu(0))
    fps = video_reader.get_avg_fps()
    num_frames = len(video_reader)
    sample_frame = video_reader[0].asnumpy()
    height = (sample_frame.shape[0] // 2) // 128 * 128
    width = (sample_frame.shape[1] // 2) // 128 * 128
    if height == 0 or width == 0:
        raise ValueError(
            f"Input video is too small after 128-alignment: derived output size is {(height, width)}."
        )
    max_tile_num = max_supported_tile_num(height, width)
    if tile_num > max_tile_num:
        raise ValueError(
            f"`tile_num={tile_num}` is too large for frame size {(height, width)}. "
            f"Maximum supported tile_num is {max_tile_num}."
        )

    frames_sbs_path = os.path.join(save_dir, f"{video_name}_sbs.mp4")
    vid_anaglyph_path = os.path.join(save_dir, f"{video_name}_anaglyph.mp4")
    sbs_writer = create_video_writer(frames_sbs_path, fps, width * 2, height)
    anaglyph_writer = create_video_writer(vid_anaglyph_path, fps, width, height)

    generated_context = None
    step = frames_chunk - overlap
    current_tile_num = tile_num

    try:
        with torch.inference_mode():
            for i in range(0, num_frames, step):
                if i + overlap >= num_frames:
                    break

                if generated_context is not None and i + frames_chunk > num_frames:
                    cur_i = max(num_frames + overlap - frames_chunk, 0)
                    cur_overlap = i - cur_i + overlap
                else:
                    cur_i = i
                    cur_overlap = overlap

                stop_i = min(cur_i + frames_chunk, num_frames)
                frames_warpped, frames_left, frames_mask = load_inpainting_chunk(
                    video_reader,
                    cur_i,
                    stop_i,
                    height,
                    width,
                )

                input_frames_i = (
                    frames_warpped.clone()
                    if generated_context is not None
                    else frames_warpped
                )

                if generated_context is not None:
                    try:
                        input_frames_i[:cur_overlap] = generated_context[
                            -cur_overlap:
                        ].to(dtype=input_frames_i.dtype)
                    except Exception as e:
                        print(e)
                        print(
                            f"i: {i}, cur_i: {cur_i}, cur_overlap: {cur_overlap}, input_frames_i: {input_frames_i.shape}, generated_context: {generated_context.shape}"
                        )

                while True:
                    try:
                        if cpu_offload is None:
                            mark_torch_compile_step_begin()
                        video_latents = spatial_tiled_process(
                            input_frames_i,
                            frames_mask,
                            pipeline,
                            current_tile_num,
                            spatial_n_compress=8,
                            min_guidance_scale=guidance_scale,
                            max_guidance_scale=guidance_scale,
                            fps=7,
                            motion_bucket_id=127,
                            noise_aug_strength=noise_aug_strength,
                            num_inference_steps=num_inference_steps,
                            vae_encode_chunk_size=vae_encode_chunk_size,
                        )
                        break
                    except Exception as exc:
                        if is_cuda_oom(exc) and current_tile_num < max_tile_num:
                            next_tile_num = current_tile_num + 1
                            print(
                                "Spatial tiling hit CUDA OOM; "
                                f"retrying with tile_num={next_tile_num}."
                            )
                            current_tile_num = next_tile_num
                            if hasattr(pipeline.vae, "enable_tiling"):
                                pipeline.vae.enable_tiling()
                            gc.collect()
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            continue
                        raise

                video_latents = video_latents.unsqueeze(0)

                # Under offload, maybe_free_model_hooks() has moved the VAE
                # off-device after the pipeline call. Move it back for decoding.
                if cpu_offload is not None:
                    pipeline.vae.to(device=pipeline._execution_device, dtype=video_latents.dtype)
                elif pipeline.vae.dtype != video_latents.dtype:
                    pipeline.vae.to(dtype=video_latents.dtype)

                if cpu_offload is None:
                    mark_torch_compile_step_begin()
                video_frames = pipeline.decode_latents(
                    video_latents,
                    num_frames=video_latents.shape[1],
                    decode_chunk_size=decode_chunk_size,
                )
                video_frames = tensor2vid(
                    video_frames, pipeline.image_processor, output_type="np"
                )[0]
                video_frames = (
                    torch.from_numpy(video_frames)
                    .permute(0, 3, 1, 2)
                    .contiguous()
                    .to(dtype=torch.float32)
                )

                output_frames = video_frames if i == 0 else video_frames[cur_overlap:]
                output_left = frames_left if i == 0 else frames_left[cur_overlap:]
                write_output_chunk(
                    sbs_writer,
                    anaglyph_writer,
                    output_left,
                    output_frames,
                )

                context_size = min(frames_chunk - 1, video_frames.shape[0])
                generated_context = video_frames[-context_size:].clone()
    finally:
        sbs_writer.release()
        anaglyph_writer.release()


if __name__ == "__main__":
    Fire(main)
