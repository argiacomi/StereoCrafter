import gc
import os
from pathlib import Path

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
    configure_compile_cache,
    configure_cuda_performance_flags,
    is_cuda_invalid_argument,
    is_cuda_oom,
    is_torch_compile_failure,
    load_compile_artifacts,
    mark_torch_compile_step_begin,
    save_compile_artifacts,
)
from tqdm import tqdm
from transformers import CLIPVisionModelWithProjection

configure_cuda_performance_flags()


def try_enable_memory_feature(method_owner, method_name, feature_name, component_name):
    method = getattr(method_owner, method_name, None)
    if not callable(method):
        return False

    try:
        method()
        return True
    except NotImplementedError:
        print(
            f"Skipping {component_name} {feature_name}: "
            f"{component_name} does not implement it."
        )
        return False


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
    compile_cache_dir=None,
    compile_warmup=True,
):
    repo_root = Path(__file__).resolve().parent
    compile_cache_dir = compile_cache_dir or os.environ.get(
        "TORCHINDUCTOR_CACHE_DIR",
        str(repo_root / ".torch_compile_cache" / "inpainting"),
    )
    compile_cache_dir = configure_compile_cache(compile_cache_dir)
    compile_artifact_path = os.path.join(
        compile_cache_dir, "inpainting_compile_artifacts.bin"
    )
    load_compile_artifacts(compile_artifact_path)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for stereo video inpainting.")
    if frames_chunk <= overlap:
        raise ValueError(
            f"`frames_chunk` must be larger than `overlap`, but got {frames_chunk=} and {overlap=}."
        )
    if tile_num < 1:
        raise ValueError(f"`tile_num` must be at least 1, but got {tile_num}.")
    if decode_chunk_size is not None and decode_chunk_size < 1:
        raise ValueError(
            f"`decode_chunk_size` must be at least 1, but got {decode_chunk_size}."
        )
    if vae_encode_chunk_size is not None and vae_encode_chunk_size < 1:
        raise ValueError(
            "`vae_encode_chunk_size` must be at least 1, "
            f"but got {vae_encode_chunk_size}."
        )

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

    eager_unet = pipeline.unet
    eager_vae_decoder = pipeline.vae.decoder
    compiled_unet = None
    compiled_vae_decoder = None
    compiled_unet_available = False
    compiled_vae_decoder_available = False
    if cpu_offload is None:
        pipeline = pipeline.to("cuda")
        pipeline.unet = torch.compile(pipeline.unet, mode="default")
        compiled_unet = pipeline.unet
        compiled_unet_available = True
        pipeline.vae.decoder = torch.compile(pipeline.vae.decoder, mode="default")
        compiled_vae_decoder = pipeline.vae.decoder
        compiled_vae_decoder_available = True
    elif cpu_offload == "sequential":
        pipeline.enable_sequential_cpu_offload()
    elif cpu_offload == "model":
        pipeline.enable_model_cpu_offload()
    else:
        raise ValueError(
            f"Unknown cpu_offload mode '{cpu_offload}'. Expected None, 'sequential', or 'model'."
        )

    vae_name = pipeline.vae.__class__.__name__
    try_enable_memory_feature(pipeline.vae, "enable_slicing", "slicing", vae_name)
    if tile_num > 1:
        try_enable_memory_feature(pipeline.vae, "enable_tiling", "tiling", vae_name)

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
    current_decode_chunk_size = decode_chunk_size
    current_vae_encode_chunk_size = max(1, int(vae_encode_chunk_size or 5))
    chunk_infos = []
    for chunk_index, start_i in enumerate(range(0, num_frames, step)):
        if start_i + overlap >= num_frames:
            break

        if chunk_index > 0 and start_i + frames_chunk > num_frames:
            cur_i = max(num_frames + overlap - frames_chunk, 0)
            cur_overlap = start_i - cur_i + overlap
        else:
            cur_i = start_i
            cur_overlap = overlap

        stop_i = min(cur_i + frames_chunk, num_frames)
        chunk_infos.append(
            {
                "start_i": start_i,
                "cur_i": cur_i,
                "stop_i": stop_i,
                "cur_overlap": cur_overlap,
                "frame_count": stop_i - cur_i,
            }
        )

    compiled_chunk_length = (
        chunk_infos[0]["frame_count"] if chunk_infos else None
    )
    tail_chunk_warned = False
    warmed_chunk = None

    def run_inpainting_chunk(
        input_frames_i,
        frames_mask,
        use_compiled_unet_for_chunk,
        use_compiled_vae_decoder_for_chunk,
    ):
        nonlocal compiled_unet_available
        nonlocal compiled_vae_decoder_available
        nonlocal current_decode_chunk_size
        nonlocal current_tile_num
        nonlocal current_vae_encode_chunk_size

        use_compiled_unet = (
            compiled_unet_available and use_compiled_unet_for_chunk
        )
        use_compiled_vae_decoder = (
            compiled_vae_decoder_available and use_compiled_vae_decoder_for_chunk
        )
        restore_compiled_unet = (
            compiled_unet_available and not use_compiled_unet
        )
        restore_compiled_vae_decoder = (
            compiled_vae_decoder_available and not use_compiled_vae_decoder
        )

        if restore_compiled_unet:
            pipeline.unet = eager_unet
        if restore_compiled_vae_decoder:
            pipeline.vae.decoder = eager_vae_decoder

        try:
            while True:
                try:
                    if use_compiled_unet:
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
                        vae_encode_chunk_size=current_vae_encode_chunk_size,
                    )
                    break
                except Exception as exc:
                    if use_compiled_unet and is_torch_compile_failure(exc):
                        print(
                            "torch.compile failed for the inpainting UNet; "
                            "falling back to eager execution."
                        )
                        pipeline.unet = eager_unet
                        compiled_unet_available = False
                        use_compiled_unet = False
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        continue

                    if use_compiled_unet and is_cuda_invalid_argument(exc):
                        print(
                            "Compiled inpainting UNet hit CUDA invalid argument; "
                            "falling back to eager execution."
                        )
                        pipeline.unet = eager_unet
                        compiled_unet_available = False
                        use_compiled_unet = False
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        continue

                    if is_cuda_oom(exc):
                        if use_compiled_unet:
                            print(
                                "Inpainting hit CUDA OOM with a compiled UNet; "
                                "retrying in eager mode."
                            )
                            pipeline.unet = eager_unet
                            compiled_unet_available = False
                            use_compiled_unet = False
                            gc.collect()
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            continue

                        if current_vae_encode_chunk_size > 1:
                            next_vae_encode_chunk_size = max(
                                1, current_vae_encode_chunk_size // 2
                            )
                            if next_vae_encode_chunk_size == current_vae_encode_chunk_size:
                                next_vae_encode_chunk_size = (
                                    current_vae_encode_chunk_size - 1
                                )
                            print(
                                "Inpainting hit CUDA OOM; "
                                "retrying with "
                                f"vae_encode_chunk_size={next_vae_encode_chunk_size}."
                            )
                            current_vae_encode_chunk_size = next_vae_encode_chunk_size
                            gc.collect()
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            continue

                        if current_tile_num < max_tile_num:
                            next_tile_num = current_tile_num + 1
                            print(
                                "Inpainting hit CUDA OOM after exhausting "
                                "VAE encode chunk retries; "
                                f"retrying with tile_num={next_tile_num}."
                            )
                            current_tile_num = next_tile_num
                            try_enable_memory_feature(
                                pipeline.vae,
                                "enable_tiling",
                                "tiling",
                                vae_name,
                            )
                            gc.collect()
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            continue

                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    raise

            video_latents = video_latents.unsqueeze(0)

            # Under offload, maybe_free_model_hooks() has moved the VAE
            # off-device after the pipeline call. Move it back for decoding.
            if cpu_offload is not None:
                pipeline.vae.to(
                    device=pipeline._execution_device, dtype=video_latents.dtype
                )
            elif pipeline.vae.dtype != video_latents.dtype:
                pipeline.vae.to(dtype=video_latents.dtype)

            while True:
                try:
                    if use_compiled_vae_decoder:
                        mark_torch_compile_step_begin()
                    video_frames = pipeline.decode_latents(
                        video_latents,
                        num_frames=video_latents.shape[1],
                        decode_chunk_size=current_decode_chunk_size,
                    )
                    return video_frames
                except Exception as exc:
                    if use_compiled_vae_decoder and is_torch_compile_failure(exc):
                        print(
                            "torch.compile failed for the inpainting VAE decoder; "
                            "falling back to eager execution."
                        )
                        pipeline.vae.decoder = eager_vae_decoder
                        compiled_vae_decoder_available = False
                        use_compiled_vae_decoder = False
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        continue

                    if use_compiled_vae_decoder and is_cuda_invalid_argument(exc):
                        print(
                            "Compiled inpainting VAE decoder hit CUDA invalid argument; "
                            "falling back to eager execution."
                        )
                        pipeline.vae.decoder = eager_vae_decoder
                        compiled_vae_decoder_available = False
                        use_compiled_vae_decoder = False
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        continue

                    if is_cuda_oom(exc):
                        if use_compiled_vae_decoder:
                            print(
                                "Inpainting hit CUDA OOM with a compiled VAE decoder; "
                                "retrying in eager mode."
                            )
                            pipeline.vae.decoder = eager_vae_decoder
                            compiled_vae_decoder_available = False
                            use_compiled_vae_decoder = False
                            gc.collect()
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            continue

                        if current_decode_chunk_size > 1:
                            next_decode_chunk_size = max(
                                1, current_decode_chunk_size // 2
                            )
                            if next_decode_chunk_size == current_decode_chunk_size:
                                next_decode_chunk_size = (
                                    current_decode_chunk_size - 1
                                )
                            print(
                                "Inpainting hit CUDA OOM during VAE decode; "
                                f"retrying with decode_chunk_size={next_decode_chunk_size}."
                            )
                            current_decode_chunk_size = next_decode_chunk_size
                            gc.collect()
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            continue

                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    raise
        finally:
            if restore_compiled_unet and compiled_unet_available:
                pipeline.unet = compiled_unet
            if restore_compiled_vae_decoder and compiled_vae_decoder_available:
                pipeline.vae.decoder = compiled_vae_decoder

    try:
        with torch.inference_mode():
            pipeline.set_progress_bar_config(disable=True)
            if (
                compile_warmup
                and compiled_chunk_length is not None
                and (compiled_unet_available or compiled_vae_decoder_available)
            ):
                warmup_info = chunk_infos[0]
                print(
                    "Warming up torch.compile on the canonical inpainting chunk shape "
                    f"({warmup_info['frame_count']} frames)."
                )
                warmup_frames_warpped, warmup_frames_left, warmup_frames_mask = (
                    load_inpainting_chunk(
                        video_reader,
                        warmup_info["cur_i"],
                        warmup_info["stop_i"],
                        height,
                        width,
                    )
                )
                warmup_video_frames = run_inpainting_chunk(
                    warmup_frames_warpped,
                    warmup_frames_mask,
                    use_compiled_unet_for_chunk=compiled_unet_available,
                    use_compiled_vae_decoder_for_chunk=compiled_vae_decoder_available,
                )
                warmed_chunk = (
                    warmup_info["start_i"],
                    warmup_frames_left,
                    tensor2vid(
                        warmup_video_frames,
                        pipeline.image_processor,
                        output_type="np",
                    )[0],
                )

            for chunk_info in tqdm(
                chunk_infos, desc="Stereo inpainting", unit="chunk"
            ):
                start_i = chunk_info["start_i"]
                cur_overlap = chunk_info["cur_overlap"]

                if warmed_chunk is not None and warmed_chunk[0] == start_i:
                    frames_left = warmed_chunk[1]
                    video_frames = warmed_chunk[2]
                    warmed_chunk = None
                else:
                    frames_warpped, frames_left, frames_mask = load_inpainting_chunk(
                        video_reader,
                        chunk_info["cur_i"],
                        chunk_info["stop_i"],
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
                                f"i: {start_i}, cur_i: {chunk_info['cur_i']}, cur_overlap: {cur_overlap}, input_frames_i: {input_frames_i.shape}, generated_context: {generated_context.shape}"
                            )

                    use_compiled_chunk = (
                        compiled_chunk_length is not None
                        and chunk_info["frame_count"] == compiled_chunk_length
                    )
                    if (
                        (compiled_unet_available or compiled_vae_decoder_available)
                        and not use_compiled_chunk
                        and not tail_chunk_warned
                    ):
                        print(
                            "Skipping torch.compile for the undersized tail inpainting chunk "
                            f"({chunk_info['frame_count']} vs {compiled_chunk_length} frames) "
                            "to avoid recompilation/autotuning."
                        )
                        tail_chunk_warned = True

                    video_frames = run_inpainting_chunk(
                        input_frames_i,
                        frames_mask,
                        use_compiled_unet_for_chunk=use_compiled_chunk,
                        use_compiled_vae_decoder_for_chunk=use_compiled_chunk,
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

                output_frames = (
                    video_frames if start_i == 0 else video_frames[cur_overlap:]
                )
                output_left = (
                    frames_left if start_i == 0 else frames_left[cur_overlap:]
                )
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
        save_compile_artifacts(compile_artifact_path)


if __name__ == "__main__":
    Fire(main)
