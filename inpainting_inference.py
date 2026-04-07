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
)
from torch_runtime_utils import (
    configure_compile_cache,
    configure_cuda_performance_flags,
    force_math_sdpa,
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
    # Round tile sizes up to the nearest multiple of 8 so they satisfy the
    # VAE's spatial compression requirement (height % 8 == 0, width % 8 == 0).
    tile_size = (
        ((tile_size[0] + 7) // 8) * 8,
        ((tile_size[1] + 7) // 8) * 8,
    )
    tile_stride = (tile_size[0] - tile_overlap[0], tile_size[1] - tile_overlap[1])
    return tile_size, tile_stride


def max_supported_tile_num(height, width, tile_overlap=(128, 128)):
    # Each tile contributes tile_stride pixels of unique content. When the
    # stride shrinks below the minimum, additional tiles are degenerate —
    # mostly overlap with negligible new information and wasted pipeline calls.
    min_stride = (
        max(1, tile_overlap[0] // 2),
        max(1, tile_overlap[1] // 2),
    )
    tile_num = 1
    while True:
        tile_size, tile_stride = spatial_tile_shape(height, width, tile_num, tile_overlap)
        if tile_stride[0] < min_stride[0] or tile_stride[1] < min_stride[1]:
            return max(1, tile_num - 1)
        # After rounding up, tiles must still fit within the frame.
        if tile_size[0] > height or tile_size[1] > width:
            return max(1, tile_num - 1)
        # The last tile's start must fall within the frame.
        last_start_h = (tile_num - 1) * tile_stride[0]
        last_start_w = (tile_num - 1) * tile_stride[1]
        if last_start_h >= height or last_start_w >= width:
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

    latent_stride = (
        tile_stride[0] // spatial_n_compress,
        tile_stride[1] // spatial_n_compress,
    )
    latent_overlap = (
        tile_overlap[0] // spatial_n_compress,
        tile_overlap[1] // spatial_n_compress,
    )

    # Pre-encode the chunk-global CLIP image once so we don't repeat it per tile.
    if tile_num > 1 and "image_embeddings" not in kargs:
        clip_image = kargs.pop("clip_image", None)
        clip_source = clip_image if clip_image is not None else cond_frames[0:1]
        kargs["image_embeddings"] = process_func._encode_image(
            clip_source,
            process_func._execution_device,
            1,  # num_videos_per_prompt
            max(kargs.get("min_guidance_scale", 1.0), kargs.get("max_guidance_scale", 1.0)) > 1.0,
        )

    # Generate, blend, and trim tiles row-by-row so that at most two rows of
    # latents are resident at once (current + previous for vertical blending).
    # Blending always uses already-blended neighbors so 4-way overlap corners
    # are weight-normalized consistently.
    prev_row = None  # blended tiles from the previous row (needed for blend_v)
    row_strips = []
    for i in range(tile_num):
        cur_row = []
        for j in range(tile_num):
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

            # Blend against already-blended neighbors (progressive, not raw).
            if prev_row is not None:
                tile = blend_v(prev_row[j], tile, latent_overlap[0])
            if j > 0:
                tile = blend_h(cur_row[j - 1], tile, latent_overlap[1])

            cur_row.append(tile)

        # Trim each tile's contribution to its stride region (except last tile
        # in each dimension which keeps its full extent) and concatenate the row.
        trimmed = []
        for j, tile in enumerate(cur_row):
            if j < tile_num - 1:
                tile = tile[:, :, :, : latent_stride[1]]
            trimmed.append(tile)
        row_strip = torch.cat(trimmed, dim=3)

        if i < tile_num - 1:
            row_strip = row_strip[:, :, : latent_stride[0], :]

        row_strips.append(row_strip)
        prev_row = cur_row  # keep for next row's vertical blending

    return torch.cat(row_strips, dim=2)


def create_video_writer(output_video_path, fps, width, height):
    """Create a VideoWriter preferring HEVC, falling back to mp4v."""
    for codec in ("HEVC", "mp4v"):
        writer = cv2.VideoWriter(
            output_video_path,
            cv2.VideoWriter_fourcc(*codec),
            fps,
            (width, height),
        )
        if writer.isOpened():
            return writer
        writer.release()
    raise RuntimeError(
        f"Failed to open VideoWriter for {output_video_path} with any codec."
    )


def write_video_chunk(writer, input_frames):
    for frame in input_frames:
        writer.write(frame[:, :, ::-1])


def _pad_to_model_res(tensor, pad_h, pad_w):
    """Replicate-pad the bottom/right edges to reach the model's 128-aligned resolution."""
    if pad_h > 0 or pad_w > 0:
        # F.pad order: (left, right, top, bottom)
        tensor = torch.nn.functional.pad(tensor, (0, pad_w, 0, pad_h), mode="replicate")
    return tensor


def load_inpainting_chunk(video_reader, start, stop, orig_height, orig_width, pad_h, pad_w):
    """Legacy loader: parse the 2x2 debug-grid MP4 into left/mask/warped, then pad."""
    frames = video_reader.get_batch(list(range(start, stop))).asnumpy()
    frames = torch.from_numpy(frames).permute(0, 3, 1, 2).contiguous().float()
    frames = frames[:, :, : orig_height * 2, : orig_width * 2]
    frames = frames / 255.0

    frames_left = frames[:, :, :orig_height, :orig_width]
    frames_mask = frames[:, :, orig_height : orig_height * 2, :orig_width]
    frames_warpped = frames[:, :, orig_height : orig_height * 2, orig_width : orig_width * 2]
    frames_mask = frames_mask.mean(dim=1, keepdim=True)

    frames_left = _pad_to_model_res(frames_left, pad_h, pad_w)
    frames_warpped = _pad_to_model_res(frames_warpped, pad_h, pad_w)
    frames_mask = _pad_to_model_res(frames_mask, pad_h, pad_w)
    return frames_warpped, frames_left, frames_mask


def load_inpainting_chunk_raw(mmap_left, mmap_right, mmap_mask, start, stop, pad_h, pad_w):
    """Loader for compact memmap arrays written by the splatting stage.

    Supports both legacy float32 sidecars and the newer compact layout
    (uint8 left/mask, float16 right).  Everything is promoted to float32
    tensors in [0, 1] for the inpainting pipeline.
    """
    raw_left = np.array(mmap_left[start:stop], copy=True)
    if raw_left.dtype == np.uint8:
        raw_left = raw_left.astype(np.float32) / 255.0
    left = torch.from_numpy(raw_left).permute(0, 3, 1, 2).contiguous()

    raw_right = np.array(mmap_right[start:stop], copy=True)
    if raw_right.dtype != np.float32:
        raw_right = raw_right.astype(np.float32)
    right = torch.from_numpy(raw_right).permute(0, 3, 1, 2).contiguous()

    raw_mask = np.array(mmap_mask[start:stop], copy=True)
    if raw_mask.dtype == np.uint8:
        raw_mask = raw_mask.astype(np.float32) / 255.0
    mask = torch.from_numpy(raw_mask).permute(0, 3, 1, 2).contiguous()

    left = _pad_to_model_res(left, pad_h, pad_w)
    right = _pad_to_model_res(right, pad_h, pad_w)
    mask = _pad_to_model_res(mask, pad_h, pad_w)
    return right, left, mask


def decode_to_frames(video_tensor):
    """Convert decode_latents output [B, C, F, H, W] to [F, C, H, W] in [0, 1]."""
    frames = video_tensor[0]  # [C, F, H, W]
    frames = (frames / 2 + 0.5).clamp(0, 1)  # denormalize from [-1, 1] to [0, 1]
    return frames.permute(1, 0, 2, 3).contiguous().float()  # [F, C, H, W]


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

    # Detect lossless raw splatting outputs by checking for the meta file
    # produced by the splatting stage alongside the grid MP4.
    # Sidecars are keyed off the splatting output stem (e.g. foo_splatting_results_raw_meta.npz).
    input_stem = Path(input_video_path).stem
    raw_meta_path = os.path.join(
        os.path.dirname(input_video_path) or ".", f"{input_stem}_raw_meta.npz"
    )
    use_raw = False
    if os.path.isfile(raw_meta_path):
        try:
            raw_meta = np.load(raw_meta_path, allow_pickle=True)
            mmap_left = np.load(str(raw_meta["left_path"]), mmap_mode="r")
            mmap_right = np.load(str(raw_meta["right_path"]), mmap_mode="r")
            mmap_mask = np.load(str(raw_meta["mask_path"]), mmap_mode="r")
            fps = float(raw_meta["fps"])
            num_frames = int(raw_meta["num_frames"])
            raw_height = int(raw_meta["height"])
            raw_width = int(raw_meta["width"])
            video_reader = None
            video_name = (
                input_stem
                .replace("_splatting_results", "")
                + "_inpainting_results"
            )
            use_raw = True
        except Exception as e:
            print(
                f"Warning: raw sidecar files found but unusable ({e}); "
                "falling back to MP4 grid loader."
            )
            mmap_left = mmap_right = mmap_mask = None

    if not use_raw:
        mmap_left = mmap_right = mmap_mask = None
        raw_height = raw_width = None
        video_reader = VideoReader(input_video_path, ctx=cpu(0))
        fps = video_reader.get_avg_fps()
        num_frames = len(video_reader)
        video_name = (
            input_video_path.split("/")[-1]
            .replace(".mp4", "")
            .replace("_splatting_results", "")
            + "_inpainting_results"
        )

    # Pad to next 128-multiple instead of cropping, so we don't lose border pixels.
    # The model output is cropped back to orig_height x orig_width before writing.
    def _ceil128(x):
        return ((x + 127) // 128) * 128

    if use_raw:
        orig_height, orig_width = raw_height, raw_width
    else:
        sample_frame = video_reader[0].asnumpy()
        orig_height = sample_frame.shape[0] // 2
        orig_width = sample_frame.shape[1] // 2

    height = _ceil128(orig_height)
    width = _ceil128(orig_width)
    if orig_height == 0 or orig_width == 0:
        raise ValueError(
            f"Input video is too small: derived output size is {(orig_height, orig_width)}."
        )
    pad_h = height - orig_height
    pad_w = width - orig_width

    max_tile_num = max_supported_tile_num(height, width)
    if tile_num > max_tile_num:
        raise ValueError(
            f"`tile_num={tile_num}` is too large for frame size {(height, width)}. "
            f"Maximum supported tile_num is {max_tile_num}."
        )

    frames_sbs_path = os.path.join(save_dir, f"{video_name}_sbs.mp4")
    vid_anaglyph_path = os.path.join(save_dir, f"{video_name}_anaglyph.mp4")
    sbs_writer = create_video_writer(frames_sbs_path, fps, orig_width * 2, orig_height)
    anaglyph_writer = create_video_writer(vid_anaglyph_path, fps, orig_width, orig_height)

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
    math_sdpa_forced = False

    def run_inpainting_chunk(
        input_frames_i,
        frames_mask,
        use_compiled_unet_for_chunk,
        use_compiled_vae_decoder_for_chunk,
        clip_image=None,
    ):
        nonlocal compiled_unet_available
        nonlocal compiled_vae_decoder_available
        nonlocal current_decode_chunk_size
        nonlocal current_tile_num
        nonlocal current_vae_encode_chunk_size
        nonlocal math_sdpa_forced

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
                    # fps and motion_bucket_id are SVD micro-conditioning values
                    # baked into the checkpoint, not the actual output framerate.
                    # The pipeline shifts fps to fps-1 internally.
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
                        clip_image=clip_image,
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

                    if is_cuda_invalid_argument(exc) and not math_sdpa_forced:
                        if force_math_sdpa():
                            print(
                                "Inpainting UNet SDPA hit CUDA invalid argument; "
                                "retrying with math attention kernels."
                            )
                            math_sdpa_forced = True
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
                if use_raw:
                    warmup_frames_warpped, warmup_frames_left, warmup_frames_mask = (
                        load_inpainting_chunk_raw(
                            mmap_left, mmap_right, mmap_mask,
                            warmup_info["cur_i"], warmup_info["stop_i"],
                            pad_h, pad_w,
                        )
                    )
                else:
                    warmup_frames_warpped, warmup_frames_left, warmup_frames_mask = (
                        load_inpainting_chunk(
                            video_reader,
                            warmup_info["cur_i"],
                            warmup_info["stop_i"],
                            orig_height,
                            orig_width,
                            pad_h,
                            pad_w,
                        )
                    )
                warmup_video_frames = run_inpainting_chunk(
                    warmup_frames_warpped,
                    warmup_frames_mask,
                    use_compiled_unet_for_chunk=compiled_unet_available,
                    use_compiled_vae_decoder_for_chunk=compiled_vae_decoder_available,
                    clip_image=warmup_frames_warpped[0:1],
                )
                warmed_chunk = (
                    warmup_info["start_i"],
                    warmup_frames_left,
                    decode_to_frames(warmup_video_frames),
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
                    if use_raw:
                        frames_warpped, frames_left, frames_mask = load_inpainting_chunk_raw(
                            mmap_left, mmap_right, mmap_mask,
                            chunk_info["cur_i"], chunk_info["stop_i"],
                            pad_h, pad_w,
                        )
                    else:
                        frames_warpped, frames_left, frames_mask = load_inpainting_chunk(
                            video_reader,
                            chunk_info["cur_i"],
                            chunk_info["stop_i"],
                            orig_height,
                            orig_width,
                            pad_h,
                            pad_w,
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

                    # Use the original warped frame (before overlap replacement)
                    # for CLIP conditioning to prevent drift over long clips.
                    video_frames = run_inpainting_chunk(
                        input_frames_i,
                        frames_mask,
                        use_compiled_unet_for_chunk=use_compiled_chunk,
                        use_compiled_vae_decoder_for_chunk=use_compiled_chunk,
                        clip_image=frames_warpped[0:1],
                    )
                    video_frames = decode_to_frames(video_frames)

                output_frames = (
                    video_frames if start_i == 0 else video_frames[cur_overlap:]
                )
                output_left = (
                    frames_left if start_i == 0 else frames_left[cur_overlap:]
                )
                # Crop back to original resolution after model inference at padded size.
                if pad_h > 0 or pad_w > 0:
                    output_frames = output_frames[:, :, :orig_height, :orig_width]
                    output_left = output_left[:, :, :orig_height, :orig_width]
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
