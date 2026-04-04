import gc
import os
import shutil
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from decord import VideoReader, cpu
from dependency.DepthCrafter.depthcrafter.depth_crafter_ppl import DepthCrafterPipeline
from dependency.DepthCrafter.depthcrafter.unet import (
    DiffusersUNetSpatioTemporalConditionModelDepthCrafter,
)
from dependency.DepthCrafter.depthcrafter.utils import vis_sequence_depth
from diffusers.training_utils import set_seed
from fire import Fire

FORWARD_WARP_ROOT = Path(__file__).resolve().parent / "dependency" / "Forward-Warp"
if FORWARD_WARP_ROOT.is_dir() and str(FORWARD_WARP_ROOT) not in sys.path:
    sys.path.insert(0, str(FORWARD_WARP_ROOT))

from Forward_Warp import forward_warp
from tqdm import tqdm
from torch_runtime_utils import (
    configure_compile_cache,
    configure_cuda_performance_flags,
    is_cuda_oom,
    is_torch_compile_failure,
    load_compile_artifacts,
    mark_torch_compile_step_begin,
    save_compile_artifacts,
)

configure_cuda_performance_flags()


def from_pretrained_with_dtype_compat(
    cls, *args, dtype=None, prefer_torch_dtype=False, **kwargs
):
    if dtype is None:
        return cls.from_pretrained(*args, **kwargs)

    dtype_kwargs = (
        ({"torch_dtype": dtype}, {"dtype": dtype})
        if prefer_torch_dtype
        else ({"dtype": dtype}, {"torch_dtype": dtype})
    )

    last_exc = None
    for dtype_kwarg in dtype_kwargs:
        try:
            return cls.from_pretrained(*args, **dtype_kwarg, **kwargs)
        except TypeError as exc:
            if not any(
                f"unexpected keyword argument '{key}'" in str(exc)
                for key in dtype_kwarg
            ):
                raise
            last_exc = exc

    if last_exc is not None:
        raise last_exc

    raise RuntimeError("Failed to load model with either dtype or torch_dtype.")


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


def build_video_plan(video_path, process_length, target_fps, max_res, dataset="open"):
    if dataset == "open":
        print("==> processing video: ", video_path)
        vid = VideoReader(video_path, ctx=cpu(0))
        original_height, original_width = vid[0].shape[:2]
        print("==> original video shape: ", (len(vid), original_height, original_width))
        height = round(original_height / 64) * 64
        width = round(original_width / 64) * 64
        if max(height, width) > max_res:
            scale = max_res / max(original_height, original_width)
            height = round(original_height * scale / 64) * 64
            width = round(original_width * scale / 64) * 64
    else:
        height = dataset_res_dict[dataset][0]
        width = dataset_res_dict[dataset][1]

    vid = VideoReader(video_path, ctx=cpu(0), width=width, height=height)
    resized_height, resized_width = vid[0].shape[:2]

    fps = vid.get_avg_fps() if target_fps == -1 else target_fps
    stride = round(vid.get_avg_fps() / fps)
    stride = max(stride, 1)
    frames_idx = list(range(0, len(vid), stride))
    print(
        f"==> downsampled shape: {(len(frames_idx), resized_height, resized_width, 3)}, with stride: {stride}"
    )
    if process_length != -1 and process_length < len(frames_idx):
        frames_idx = frames_idx[:process_length]
    print(
        f"==> final processing shape: {(len(frames_idx), resized_height, resized_width, 3)}"
    )

    return {
        "video_path": video_path,
        "frame_indices": frames_idx,
        "fps": fps,
        "original_height": original_height,
        "original_width": original_width,
        "resized_height": resized_height,
        "resized_width": resized_width,
    }


def iter_window_ranges(total_frames, chunk_size, overlap):
    if chunk_size <= overlap:
        raise ValueError(
            f"`chunk_size` must be larger than `overlap`, but got {chunk_size=} and {overlap=}."
        )

    step = chunk_size - overlap
    for start in range(0, total_frames, step):
        stop = min(start + chunk_size, total_frames)
        keep_from = 0 if start == 0 else overlap
        write_start = start + keep_from
        if write_start >= stop:
            continue
        yield start, stop, keep_from, write_start
        if stop == total_frames:
            break


def iter_batch_ranges(total_frames, batch_size):
    for start in range(0, total_frames, batch_size):
        yield start, min(start + batch_size, total_frames)


def shrink_temporal_window(window_size, overlap):
    if window_size <= 1:
        return None, None

    next_window_size = max(1, window_size // 2)
    if next_window_size == window_size:
        next_window_size = window_size - 1

    if next_window_size <= 1:
        next_overlap = 0
    else:
        scaled_overlap = int(round(overlap * next_window_size / window_size))
        next_overlap = min(next_window_size - 1, max(0, scaled_overlap))

    return next_window_size, next_overlap


def write_video_opencv(input_frames, fps, output_video_path):
    num_frames = len(input_frames)
    height, width, _ = input_frames[0].shape

    out = cv2.VideoWriter(
        output_video_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    for i in range(num_frames):
        out.write(input_frames[i, :, :, ::-1])

    out.release()


def create_video_writer(output_video_path, fps, width, height):
    return cv2.VideoWriter(
        output_video_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )


def resize_depth_to_original(depth_chunk, original_height, original_width, device):
    resized_chunks = []
    resize_batch_size = 64
    for start, stop in iter_batch_ranges(len(depth_chunk), resize_batch_size):
        chunk = torch.from_numpy(depth_chunk[start:stop]).unsqueeze(1).to(
            device=device, dtype=torch.float32
        )
        chunk = F.interpolate(
            chunk,
            size=(original_height, original_width),
            mode="bilinear",
            align_corners=False,
        )
        resized_chunks.append(chunk[:, 0].cpu().numpy())
    return np.concatenate(resized_chunks, axis=0)


def normalize_depth_batch(depth_batch, depth_min, depth_max):
    denom = max(depth_max - depth_min, 1e-6)
    return np.clip((depth_batch - depth_min) / denom, 0.0, 1.0).astype(np.float32)


class DepthCrafterDemo:
    def __init__(
        self,
        unet_path: str,
        pre_trained_path: str,
        cpu_offload: str = None,
    ):
        unet = from_pretrained_with_dtype_compat(
            DiffusersUNetSpatioTemporalConditionModelDepthCrafter,
            unet_path,
            low_cpu_mem_usage=True,
            dtype=torch.float16,
        )
        # load weights of other components from the provided checkpoint
        self.pipe = from_pretrained_with_dtype_compat(
            DepthCrafterPipeline,
            pre_trained_path,
            unet=unet,
            dtype=torch.float16,
            prefer_torch_dtype=True,
            variant="fp16",
        )
        self.pipe = self.pipe.to(dtype=torch.float16)
        vae_name = self.pipe.vae.__class__.__name__
        if not try_enable_memory_feature(
            self.pipe,
            "enable_vae_slicing",
            "slicing",
            vae_name,
        ):
            try_enable_memory_feature(
                self.pipe.vae,
                "enable_slicing",
                "slicing",
                vae_name,
            )
        if not try_enable_memory_feature(
            self.pipe,
            "enable_vae_tiling",
            "tiling",
            vae_name,
        ):
            try_enable_memory_feature(
                self.pipe.vae,
                "enable_tiling",
                "tiling",
                vae_name,
            )
        self._eager_unet = self.pipe.unet
        self._compiled_unet = False

        if cpu_offload is None:
            self.pipe.to("cuda")
            # Only compile when fully on GPU — offload moves modules dynamically
            self.pipe.unet = torch.compile(self.pipe.unet, mode="default")
            self._compiled_unet = True
        elif cpu_offload == "sequential":
            self.pipe.enable_sequential_cpu_offload()
        elif cpu_offload == "model":
            self.pipe.enable_model_cpu_offload()
        else:
            raise ValueError(
                f"Unknown cpu_offload mode '{cpu_offload}'. "
                "Expected None, 'sequential', or 'model'."
            )

        # PyTorch 2.x uses scaled_dot_product_attention (SDPA) by default,
        # which supports all GPU architectures including Blackwell (sm_120).
        # xformers kernels do not support compute capability >= 12.0.

    def _run_depthcrafter_chunk(
        self,
        frames,
        guidance_scale,
        num_denoising_steps,
        window_size,
        overlap,
        decode_chunk_size,
        track_time,
        use_compiled_unet,
    ):
        restore_compiled_unet = None
        if self._compiled_unet and not use_compiled_unet:
            restore_compiled_unet = self.pipe.unet
            self.pipe.unet = self._eager_unet

        current_window_size = max(1, int(window_size))
        current_overlap = max(0, int(overlap))
        current_decode_chunk_size = decode_chunk_size
        try:
            with torch.inference_mode():
                while True:
                    if use_compiled_unet:
                        mark_torch_compile_step_begin()
                    try:
                        chunk_depth = self.pipe(
                            frames,
                            height=frames.shape[1],
                            width=frames.shape[2],
                            output_type="np",
                            guidance_scale=guidance_scale,
                            num_inference_steps=num_denoising_steps,
                            window_size=current_window_size,
                            overlap=current_overlap,
                            decode_chunk_size=current_decode_chunk_size,
                            track_time=track_time,
                        ).frames[0]
                        return (
                            chunk_depth,
                            current_decode_chunk_size,
                            current_window_size,
                            current_overlap,
                        )
                    except Exception as exc:
                        if use_compiled_unet and is_torch_compile_failure(exc):
                            print(
                                "torch.compile failed for the DepthCrafter UNet; "
                                "falling back to eager execution."
                            )
                            self.pipe.unet = self._eager_unet
                            self._compiled_unet = False
                            use_compiled_unet = False
                            gc.collect()
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            continue

                        if is_cuda_oom(exc):
                            if use_compiled_unet:
                                print(
                                    "DepthCrafter hit CUDA OOM with a compiled UNet; "
                                    "retrying in eager mode."
                                )
                                self.pipe.unet = self._eager_unet
                                self._compiled_unet = False
                                use_compiled_unet = False
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
                                    "DepthCrafter hit CUDA OOM during VAE decode; "
                                    f"retrying with decode_chunk_size={next_decode_chunk_size}."
                                )
                                current_decode_chunk_size = next_decode_chunk_size
                                gc.collect()
                                if torch.cuda.is_available():
                                    torch.cuda.empty_cache()
                                continue

                            next_window_size, next_overlap = shrink_temporal_window(
                                current_window_size, current_overlap
                            )
                            if next_window_size is not None:
                                print(
                                    "DepthCrafter hit CUDA OOM after exhausting "
                                    "decode chunk retries; retrying with "
                                    f"window_size={next_window_size}, "
                                    f"overlap={next_overlap}."
                                )
                                current_window_size = next_window_size
                                current_overlap = next_overlap
                                gc.collect()
                                if torch.cuda.is_available():
                                    torch.cuda.empty_cache()
                                continue

                        raise
        finally:
            if restore_compiled_unet is not None and self._compiled_unet:
                self.pipe.unet = restore_compiled_unet

    def infer(
        self,
        video_plan,
        output_video_path: str,
        scratch_dir: str,
        num_denoising_steps: int = 8,
        guidance_scale: float = 1.2,
        window_size: int = 70,
        overlap: int = 25,
        decode_chunk_size: int = 8,
        seed: int = 42,
        track_time: bool = False,
        save_depth: bool = False,
        compile_warmup: bool = True,
    ):
        set_seed(seed)
        num_frames = len(video_plan["frame_indices"])
        if num_frames == 0:
            raise ValueError(
                "No frames selected for depth estimation. "
                "Check `process_length`, `target_fps`, and the input video."
            )
        original_height = video_plan["original_height"]
        original_width = video_plan["original_width"]
        resized_height = video_plan["resized_height"]
        resized_width = video_plan["resized_width"]
        target_fps = video_plan["fps"]
        save_path = os.path.join(
            os.path.dirname(output_video_path), os.path.splitext(os.path.basename(output_video_path))[0]
        )
        raw_depth_path = os.path.join(scratch_dir, "depth_raw.npy")
        raw_depth = np.lib.format.open_memmap(
            raw_depth_path,
            mode="w+",
            dtype=np.float32,
            shape=(num_frames, original_height, original_width),
        )

        resized_reader = VideoReader(
            video_plan["video_path"],
            ctx=cpu(0),
            width=resized_width,
            height=resized_height,
        )
        device = self.pipe._execution_device
        chunk_size = max(window_size, overlap + 1)
        chunk_ranges = list(iter_window_ranges(num_frames, chunk_size, overlap))
        current_window_size = max(1, int(window_size))
        current_window_overlap = max(0, int(overlap))
        current_decode_chunk_size = max(1, int(decode_chunk_size))
        compiled_chunk_length = chunk_ranges[0][1] - chunk_ranges[0][0]
        tail_chunk_warned = False
        warmed_chunk = None

        if compile_warmup and self._compiled_unet and chunk_ranges:
            warmup_start, warmup_stop, _, _ = chunk_ranges[0]
            warmup_indices = video_plan["frame_indices"][warmup_start:warmup_stop]
            warmup_frames = (
                resized_reader.get_batch(warmup_indices).asnumpy().astype(np.float32)
                / 255.0
            )
            print(
                "Warming up torch.compile on the canonical depth chunk shape "
                f"({len(warmup_indices)} frames)."
            )
            (
                warmup_depth,
                current_decode_chunk_size,
                current_window_size,
                current_window_overlap,
            ) = self._run_depthcrafter_chunk(
                warmup_frames,
                guidance_scale=guidance_scale,
                num_denoising_steps=num_denoising_steps,
                window_size=current_window_size,
                overlap=current_window_overlap,
                decode_chunk_size=current_decode_chunk_size,
                track_time=track_time,
                use_compiled_unet=True,
            )
            warmed_chunk = ((warmup_start, warmup_stop), warmup_depth)

        # Silence per-chunk denoising bars; show one outer progress bar instead
        self.pipe.set_progress_bar_config(disable=True)
        for start, stop, keep_from, write_start in tqdm(
            chunk_ranges, desc="Depth estimation", unit="chunk"
        ):
            chunk_key = (start, stop)
            if warmed_chunk is not None and warmed_chunk[0] == chunk_key:
                chunk_depth = warmed_chunk[1]
                warmed_chunk = None
            else:
                chunk_indices = video_plan["frame_indices"][start:stop]
                frames = (
                    resized_reader.get_batch(chunk_indices).asnumpy().astype(np.float32)
                    / 255.0
                )
                use_compiled_unet = self._compiled_unet and (
                    len(chunk_indices) == compiled_chunk_length
                )
                if self._compiled_unet and not use_compiled_unet:
                    if not tail_chunk_warned:
                        print(
                            "Skipping torch.compile for the undersized tail depth chunk "
                            f"({len(chunk_indices)} vs {compiled_chunk_length} frames) "
                            "to avoid recompilation/autotuning."
                        )
                        tail_chunk_warned = True

                (
                    chunk_depth,
                    current_decode_chunk_size,
                    current_window_size,
                    current_window_overlap,
                ) = self._run_depthcrafter_chunk(
                    frames,
                    guidance_scale=guidance_scale,
                    num_denoising_steps=num_denoising_steps,
                    window_size=current_window_size,
                    overlap=current_window_overlap,
                    decode_chunk_size=current_decode_chunk_size,
                    track_time=track_time,
                    use_compiled_unet=use_compiled_unet,
                )

            chunk_depth = chunk_depth.sum(-1) / chunk_depth.shape[-1]
            chunk_depth = resize_depth_to_original(
                chunk_depth, original_height, original_width, device
            )

            # Blend overlap region with previously written depth to avoid
            # hard temporal cuts between independently-processed chunks.
            if keep_from > 0:
                overlap_new = chunk_depth[:keep_from]
                overlap_old = np.array(raw_depth[start : start + keep_from])
                weights = np.linspace(1.0, 0.0, keep_from, dtype=np.float32)[
                    :, None, None
                ]
                blended = overlap_old * weights + overlap_new * (1.0 - weights)
                raw_depth[start : start + keep_from] = blended

            unique_depth = chunk_depth[keep_from:]
            write_stop = write_start + len(unique_depth)
            raw_depth[write_start:write_stop] = unique_depth

        del resized_reader

        del raw_depth

        # Compute min/max from the finalized memmap so blended overlap
        # regions are included and no stale pre-blend extrema leak through.
        raw_depth = np.load(raw_depth_path, mmap_mode="r+")
        depth_min = np.inf
        depth_max = -np.inf
        with tqdm(total=num_frames, desc="Depth extrema scan", unit="frame") as progress_bar:
            for start, stop in iter_batch_ranges(num_frames, 64):
                batch = raw_depth[start:stop]
                depth_min = min(depth_min, float(batch.min()))
                depth_max = max(depth_max, float(batch.max()))
                progress_bar.update(stop - start)

        depth_vis_writer = None
        if save_depth:
            depth_vis_writer = create_video_writer(
                save_path + "_depth_vis.mp4",
                target_fps,
                original_width,
                original_height,
            )

        with tqdm(total=num_frames, desc="Depth normalization", unit="frame") as progress_bar:
            for start, stop in iter_batch_ranges(num_frames, 64):
                normalized_batch = normalize_depth_batch(
                    raw_depth[start:stop], depth_min, depth_max
                )
                raw_depth[start:stop] = normalized_batch

                if depth_vis_writer is not None:
                    depth_vis = np.clip(
                        vis_sequence_depth(normalized_batch) * 255.0, 0, 255
                    ).astype(np.uint8)
                    for frame in depth_vis:
                        depth_vis_writer.write(frame[:, :, ::-1])

                progress_bar.update(stop - start)

        if depth_vis_writer is not None:
            depth_vis_writer.release()

        del raw_depth

        if save_depth:
            shutil.copy2(raw_depth_path, save_path + "_depth.npy")

        return {
            "depth_path": raw_depth_path,
            "num_frames": num_frames,
            "height": original_height,
            "width": original_width,
        }


class ForwardWarpStereo(nn.Module):
    def __init__(self, eps=1e-6, occlu_map=False):
        super(ForwardWarpStereo, self).__init__()
        self.eps = eps
        self.occlu_map = occlu_map
        self.fw = forward_warp()

    def forward(self, im, disp):
        """
        :param im: BCHW
        :param disp: B1HW
        :return: BCHW
        detach will lead to unconverge!!
        """
        im = im.contiguous()
        disp = disp.contiguous()
        # weights_map = torch.abs(disp)
        weights_map = disp - disp.min()
        weights_map = (
            1.414
        ) ** weights_map  # using 1.414 instead of EXP for avoding numerical overflow.
        flow = -disp.squeeze(1)
        dummy_flow = torch.zeros_like(flow, requires_grad=False)
        flow = torch.stack((flow, dummy_flow), dim=-1)
        res_accum = self.fw(im * weights_map, flow)
        # mask = self.fw(weights_map, flow.detach())
        mask = self.fw(weights_map, flow)
        mask.clamp_(min=self.eps)
        res = res_accum / mask
        if not self.occlu_map:
            return res
        else:
            ones = torch.ones_like(disp, requires_grad=False)
            occlu_map = self.fw(ones, flow)
            occlu_map.clamp_(0.0, 1.0)
            occlu_map = 1.0 - occlu_map
            return res, occlu_map


def DepthSplatting(
    video_plan,
    output_video_path,
    depth_result,
    max_disp,
    batch_size,
):
    """
    Depth-Based Video Splatting Using the Video Depth.
    Args:
        video_plan: Processed video sampling plan.
        output_video_path: Path to the output video.
        depth_result: Streamed normalized depth metadata.
        batch_size: The batch size for splatting to save GPU memory.
    """
    vid_reader = VideoReader(video_plan["video_path"], ctx=cpu(0))
    video_depth = np.load(depth_result["depth_path"], mmap_mode="r")

    # Forward-Warp is a custom CUDA extension that Dynamo cannot trace
    # cleanly, so keep this stage eager and manage memory via batch sizing.
    stereo_projector = ForwardWarpStereo(occlu_map=True).cuda()

    num_frames = depth_result["num_frames"]
    height = depth_result["height"]
    width = depth_result["width"]
    current_batch_size = max(1, int(batch_size))

    # Initialize OpenCV VideoWriter
    out = cv2.VideoWriter(
        output_video_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        video_plan["fps"],
        (width * 2, height * 2),
    )

    try:
        with torch.inference_mode():
            frame_offset = 0
            with tqdm(total=num_frames, desc="Stereo splatting", unit="frame") as progress_bar:
                while frame_offset < num_frames:
                    left_video = None
                    disp_map = None
                    right_video = None
                    occlusion_mask = None
                    try:
                        batch_indices = video_plan["frame_indices"][
                            frame_offset : frame_offset + current_batch_size
                        ]
                        batch_frames = (
                            vid_reader.get_batch(batch_indices).asnumpy().astype(np.float32)
                            / 255.0
                        )
                        # Copy the memmap slice so torch.from_numpy receives writable storage.
                        batch_depth = np.array(
                            video_depth[frame_offset : frame_offset + current_batch_size],
                            dtype=np.float32,
                            copy=True,
                        )
                        batch_depth_vis = vis_sequence_depth(batch_depth)

                        left_video = (
                            torch.from_numpy(batch_frames)
                            .permute(0, 3, 1, 2)
                            .contiguous()
                            .pin_memory()
                            .cuda(non_blocking=True)
                        )
                        disp_map = (
                            torch.from_numpy(batch_depth)
                            .float()
                            .unsqueeze(1)
                            .pin_memory()
                            .cuda(non_blocking=True)
                        )

                        disp_map = disp_map * 2.0 - 1.0
                        disp_map = disp_map * max_disp

                        right_video, occlusion_mask = stereo_projector(
                            left_video, disp_map
                        )

                        right_video = right_video.cpu().permute(0, 2, 3, 1).numpy()
                        occlusion_mask = (
                            occlusion_mask.cpu()
                            .permute(0, 2, 3, 1)
                            .numpy()
                            .repeat(3, axis=-1)
                        )

                        for j in range(len(batch_frames)):
                            video_grid_top = np.concatenate(
                                [batch_frames[j], batch_depth_vis[j]], axis=1
                            )
                            video_grid_bottom = np.concatenate(
                                [occlusion_mask[j], right_video[j]], axis=1
                            )
                            video_grid = np.concatenate(
                                [video_grid_top, video_grid_bottom], axis=0
                            )

                            video_grid_uint8 = np.clip(video_grid * 255.0, 0, 255).astype(
                                np.uint8
                            )
                            video_grid_bgr = cv2.cvtColor(video_grid_uint8, cv2.COLOR_RGB2BGR)
                            out.write(video_grid_bgr)

                        processed_frames = len(batch_indices)
                        frame_offset += processed_frames
                        progress_bar.update(processed_frames)
                    except Exception as exc:
                        if is_cuda_oom(exc) and current_batch_size > 1:
                            next_batch_size = max(1, current_batch_size // 2)
                            if next_batch_size == current_batch_size:
                                next_batch_size = current_batch_size - 1
                            print(
                                "Stereo splatting hit CUDA OOM; "
                                f"retrying with batch_size={next_batch_size}."
                            )
                            current_batch_size = next_batch_size
                            gc.collect()
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            continue
                        raise
                    finally:
                        del left_video, disp_map, right_video, occlusion_mask
    finally:
        out.release()


def main(
    input_video_path: str,
    output_video_path: str,
    unet_path: str,
    pre_trained_path: str,
    max_disp: float = 20.0,
    process_length: int = -1,
    batch_size: int = 64,
    cpu_offload: str = None,
    num_denoising_steps: int = 8,
    guidance_scale: float = 1.2,
    window_size: int = 70,
    overlap: int = 25,
    decode_chunk_size: int = 4,
    max_res: int = 1024,
    dataset: str = "open",
    target_fps: int = -1,
    seed: int = 42,
    track_time: bool = False,
    save_depth: bool = False,
    compile_cache_dir: str = None,
    compile_warmup: bool = True,
):
    repo_root = Path(__file__).resolve().parent
    compile_cache_dir = compile_cache_dir or os.environ.get(
        "TORCHINDUCTOR_CACHE_DIR",
        str(repo_root / ".torch_compile_cache" / "depth_splatting"),
    )
    compile_cache_dir = configure_compile_cache(compile_cache_dir)
    compile_artifact_path = os.path.join(
        compile_cache_dir, "depth_splatting_compile_artifacts.bin"
    )
    load_compile_artifacts(compile_artifact_path)

    video_plan = build_video_plan(
        input_video_path,
        process_length,
        target_fps,
        max_res,
        dataset,
    )

    depthcrafter_demo = DepthCrafterDemo(
        unet_path=unet_path,
        pre_trained_path=pre_trained_path,
        cpu_offload=cpu_offload,
    )

    scratch_root = os.path.dirname(output_video_path) or "."
    os.makedirs(scratch_root, exist_ok=True)
    try:
        with tempfile.TemporaryDirectory(
            prefix="stereocrafter_depth_", dir=scratch_root
        ) as scratch_dir:
            depth_result = depthcrafter_demo.infer(
                video_plan,
                output_video_path,
                scratch_dir,
                num_denoising_steps=num_denoising_steps,
                guidance_scale=guidance_scale,
                window_size=window_size,
                overlap=overlap,
                decode_chunk_size=decode_chunk_size,
                seed=seed,
                track_time=track_time,
                save_depth=save_depth,
                compile_warmup=compile_warmup,
            )

            # Release DepthCrafter before splatting so the forward-warp stage
            # does not compete with the diffusion pipeline for VRAM.
            del depthcrafter_demo
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            DepthSplatting(
                video_plan,
                output_video_path,
                depth_result,
                max_disp,
                batch_size,
            )
    finally:
        save_compile_artifacts(compile_artifact_path)


if __name__ == "__main__":
    Fire(main)
