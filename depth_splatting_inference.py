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

# Performance flags for CUDA
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("medium")

if hasattr(torch.backends.cuda.matmul, "fp32_precision"):
    torch.backends.cuda.matmul.fp32_precision = "tf32"
else:
    torch.backends.cuda.matmul.allow_tf32 = True

if hasattr(torch.backends.cudnn, "conv") and hasattr(
    torch.backends.cudnn.conv, "fp32_precision"
):
    torch.backends.cudnn.conv.fp32_precision = "tf32"
else:
    torch.backends.cudnn.allow_tf32 = True


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


def is_torch_compile_failure(exc):
    needles = (
        "torch._dynamo",
        "Dynamo failed",
        "TorchRuntimeError",
        "fake tensors",
        "BackendCompilerFailed",
    )
    current = exc
    while current is not None:
        message = str(current)
        if any(needle in message for needle in needles):
            return True
        current = current.__cause__ or current.__context__
    return False


def mark_torch_compile_step_begin():
    if hasattr(torch, "compiler") and hasattr(
        torch.compiler, "cudagraph_mark_step_begin"
    ):
        torch.compiler.cudagraph_mark_step_begin()


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
        self._eager_unet = self.pipe.unet
        self._compiled_unet = False

        if cpu_offload is None:
            self.pipe.to("cuda")
            # Only compile when fully on GPU — offload moves modules dynamically
            self.pipe.unet = torch.compile(self.pipe.unet, mode="max-autotune")
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

    def infer(
        self,
        video_plan,
        output_video_path: str,
        scratch_dir: str,
        num_denoising_steps: int = 8,
        guidance_scale: float = 1.2,
        window_size: int = 70,
        overlap: int = 25,
        seed: int = 42,
        track_time: bool = False,
        save_depth: bool = False,
    ):
        set_seed(seed)
        num_frames = len(video_plan["frame_indices"])
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

        # Silence per-chunk denoising bars; show one outer progress bar instead
        self.pipe.set_progress_bar_config(disable=True)
        chunk_ranges = list(iter_window_ranges(num_frames, chunk_size, overlap))

        for start, stop, keep_from, write_start in tqdm(
            chunk_ranges, desc="Depth estimation", unit="chunk"
        ):
            chunk_indices = video_plan["frame_indices"][start:stop]
            frames = (
                resized_reader.get_batch(chunk_indices).asnumpy().astype(np.float32) / 255.0
            )

            with torch.inference_mode():
                if self._compiled_unet:
                    mark_torch_compile_step_begin()
                try:
                    chunk_depth = self.pipe(
                        frames,
                        height=frames.shape[1],
                        width=frames.shape[2],
                        output_type="np",
                        guidance_scale=guidance_scale,
                        num_inference_steps=num_denoising_steps,
                        window_size=window_size,
                        overlap=overlap,
                        track_time=track_time,
                    ).frames[0]
                except Exception as exc:
                    if not (self._compiled_unet and is_torch_compile_failure(exc)):
                        raise

                    print(
                        "torch.compile failed for the DepthCrafter UNet; "
                        "falling back to eager execution."
                    )
                    self.pipe.unet = self._eager_unet
                    self._compiled_unet = False
                    chunk_depth = self.pipe(
                        frames,
                        height=frames.shape[1],
                        width=frames.shape[2],
                        output_type="np",
                        guidance_scale=guidance_scale,
                        num_inference_steps=num_denoising_steps,
                        window_size=window_size,
                        overlap=overlap,
                        track_time=track_time,
                    ).frames[0]

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

        del raw_depth

        # Compute min/max from the finalized memmap so blended overlap
        # regions are included and no stale pre-blend extrema leak through.
        raw_depth = np.load(raw_depth_path, mmap_mode="r+")
        depth_min = np.inf
        depth_max = -np.inf
        for start, stop in iter_batch_ranges(num_frames, 64):
            batch = raw_depth[start:stop]
            depth_min = min(depth_min, float(batch.min()))
            depth_max = max(depth_max, float(batch.max()))

        depth_vis_writer = None
        if save_depth:
            depth_vis_writer = create_video_writer(
                save_path + "_depth_vis.mp4",
                target_fps,
                original_width,
                original_height,
            )

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

    stereo_projector = ForwardWarpStereo(occlu_map=True).cuda()
    stereo_projector = torch.compile(stereo_projector, mode="max-autotune")

    num_frames = depth_result["num_frames"]
    height = depth_result["height"]
    width = depth_result["width"]

    # Initialize OpenCV VideoWriter
    out = cv2.VideoWriter(
        output_video_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        video_plan["fps"],
        (width * 2, height * 2),
    )

    with torch.inference_mode():
        for i in range(0, num_frames, batch_size):
            batch_indices = video_plan["frame_indices"][i : i + batch_size]
            batch_frames = (
                vid_reader.get_batch(batch_indices).asnumpy().astype(np.float32) / 255.0
            )
            batch_depth = np.asarray(video_depth[i : i + batch_size], dtype=np.float32)
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

            mark_torch_compile_step_begin()
            right_video, occlusion_mask = stereo_projector(left_video, disp_map)

            right_video = right_video.cpu().permute(0, 2, 3, 1).numpy()
            occlusion_mask = occlusion_mask.cpu().permute(0, 2, 3, 1).numpy().repeat(3, axis=-1)

            for j in range(len(batch_frames)):
                video_grid_top = np.concatenate([batch_frames[j], batch_depth_vis[j]], axis=1)
                video_grid_bottom = np.concatenate([occlusion_mask[j], right_video[j]], axis=1)
                video_grid = np.concatenate([video_grid_top, video_grid_bottom], axis=0)

                video_grid_uint8 = np.clip(video_grid * 255.0, 0, 255).astype(np.uint8)
                video_grid_bgr = cv2.cvtColor(video_grid_uint8, cv2.COLOR_RGB2BGR)
                out.write(video_grid_bgr)

            del left_video, disp_map, right_video, occlusion_mask

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
    max_res: int = 1024,
    dataset: str = "open",
    target_fps: int = -1,
    seed: int = 42,
    track_time: bool = False,
    save_depth: bool = False,
):
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
    with tempfile.TemporaryDirectory(prefix="stereocrafter_depth_", dir=scratch_root) as scratch_dir:
        depth_result = depthcrafter_demo.infer(
            video_plan,
            output_video_path,
            scratch_dir,
            num_denoising_steps=num_denoising_steps,
            guidance_scale=guidance_scale,
            window_size=window_size,
            overlap=overlap,
            seed=seed,
            track_time=track_time,
            save_depth=save_depth,
        )

        DepthSplatting(
            video_plan,
            output_video_path,
            depth_result,
            max_disp,
            batch_size,
        )


if __name__ == "__main__":
    Fire(main)
