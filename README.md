
<div align="center">
<h2>StereoCrafter: Diffusion-based Generation of Long and High-fidelity Stereoscopic 3D from Monocular Videos</h2>

Sijie Zhao*&emsp;
Wenbo Hu*&emsp;
Xiaodong Cun*&emsp;
Yong Zhang&dagger;&emsp;
Xiaoyu Li&dagger;&emsp;<br>
Zhe Kong&emsp;
Xiangjun Gao&emsp;
Muyao Niu&emsp;
Ying Shan

&emsp;* equal contribution &emsp; &dagger; corresponding author 

<h3>Tencent AI Lab&emsp;&emsp;ARC Lab, Tencent PCG</h3>

<a href='https://arxiv.org/abs/2409.07447'><img src='https://img.shields.io/badge/arXiv-PDF-a92225'></a> &emsp;
<a href='https://stereocrafter.github.io/'><img src='https://img.shields.io/badge/Project_Page-Page-64fefe' alt='Project Page'></a> &emsp;
<a href='https://huggingface.co/TencentARC/StereoCrafter'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Weights-yellow'></a>
</div>

## 💡 Abstract

We propose a novel framework to convert any 2D videos to immersive stereoscopic 3D ones that can be viewed on different display devices, like 3D Glasses, Apple Vision Pro and 3D Display. It can be applied to various video sources, such as movies, vlogs, 3D cartoons, and AIGC videos.

![teaser](assets/teaser.jpg)

## 📣 News
- `2024/12/27` We released our inference code and model weights.
- `2024/09/11` We submitted our technical report on arXiv and released our project page.

## 🎞️ Showcases
Here we show some examples of input videos and their corresponding stereo outputs in Anaglyph 3D format.
<div align="center">
    <img src="assets/demo.gif">
</div>


## 🛠️ Installation

#### 1. Set up the environment
We run our code on Python 3.8 and Cuda 11.8.
You can use Anaconda or Docker to build this basic environment.

#### 2. Clone the repo
```bash
# use --recursive to clone the dependent submodules
git clone --recursive https://github.com/TencentARC/StereoCrafter
cd StereoCrafter
```

#### 3. Install the requirements
```bash
pip install -r requirements.txt
```

Optional: install `ffmpeg` only if you plan to enable final delivery transcoding
with `--final_video_codec` during stage 2. The default workflow does not require
it.


#### 4. Install customized 'Forward-Warp' package for forward splatting
```
cd ./dependency/Forward-Warp
chmod a+x install.sh
./install.sh
```


## 📦 Model Weights

#### 1. Download the [SVD img2vid model](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt-1-1) for the image encoder and VAE.

```bash
# in StereoCrafter project root directory
mkdir weights
cd ./weights
git lfs install
git clone https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt-1-1
```

#### 2. Download the [DepthCrafter model](https://huggingface.co/tencent/DepthCrafter) for the video depth estimation.
```bash
git clone https://huggingface.co/tencent/DepthCrafter
```

#### 3. Download the [StereoCrafter model](https://huggingface.co/TencentARC/StereoCrafter) for the stereo video generation.
```bash
git clone https://huggingface.co/TencentARC/StereoCrafter
```


## 🔄 Inference

Script:

```bash
# in StereoCrafter project root directory
sh run_inference.sh
```

There are two main steps in this script for generating stereo video.

#### 1. Depth-Based Video Splatting Using the Video Depth from DepthCrafter
Execute the following command:
```bash
python depth_splatting_inference.py \
    --pre_trained_path [PATH] \
    --unet_path [PATH] \
    --input_video_path [PATH] \
    --output_video_path [PATH] \
    --target_fps [FPS] \
    --compile_cache_dir [PATH]
```
Arguments:
- `--pre_trained_path`: Path to the SVD img2vid model weights (e.g., `./weights/stable-video-diffusion-img2vid-xt-1-1`).
- `--unet_path`: Path to the DepthCrafter model weights (e.g., `./weights/DepthCrafter`).
- `--input_video_path`: Path to the input video (e.g., `./source_video/camel.mp4`).
- `--output_video_path`: Path to the output video (e.g., `./outputs/camel_splatting_results.mp4`).
- `--target_fps`: Optional output FPS used for frame sampling. Default is `-1`, which keeps the source video FPS.
- `--max_disp`: Parameter controlling the maximum disparity between the generated right video and the input left video. Default value is `20` pixels.
- `--compile_cache_dir`: Optional persistent `torch.compile` cache directory. Default is `./.torch_compile_cache/depth_splatting`.
- `--compile_warmup`: Warm up the canonical depth chunk shape before the main loop and reuse it. Default is `True`.

Depth inference keeps the compiled path on the dominant full-size chunk shape and runs the undersized tail chunk eagerly to avoid a second recompilation/autotuning pass.

The first step generates:
- the video grid at `--output_video_path`, containing the input video, visualized depth map, occlusion mask, and splatting right video
- an additional side-by-side video named `<input basename>_sbs.mp4` in the output directory, containing the left frame and warped frame

The grid output looks like this:

<img src="assets/camel_splatting_results.jpg" alt="camel_splatting_results" width="800"/> 

#### 2. Stereo Video Inpainting of the Splatting Video
Execute the following command:
```bash
python inpainting_inference.py \
    --pre_trained_path [PATH] \
    --unet_path [PATH] \
    --input_video_path [PATH] \
    --save_dir [PATH] \
    --target_fps [FPS] \
    --max_res [RES] \
    --final_video_codec [CODEC] \
    --compile_cache_dir [PATH]
```
Arguments:
- `--pre_trained_path`: Path to the SVD img2vid model weights (e.g., `./weights/stable-video-diffusion-img2vid-xt-1-1`).
- `--unet_path`: Path to the StereoCrafter model weights (e.g., `./weights/StereoCrafter`).
- `--input_video_path`: Path to the splatting video result generated by the first stage (e.g., `./outputs/camel_splatting_results.mp4`).
- `--save_dir`: Directory for the output stereo video (e.g., `./outputs`).
- `--target_fps`: Optional inpainting-only FPS cap. Values below the source FPS reduce the number of processed chunks and write the output video at the lower FPS. Default is `-1` (keep source FPS).
- `--max_res`: Optional inpainting-only cap on the longest output edge. The inpainting stage downsamples before padding/model execution, so lower values reduce per-chunk cost. Default is disabled.
- `--tile_num`: The number of tiles in width and height dimensions for tiled processing, which allows for handling high resolution input without requiring more GPU memory. The default value is `1` (1 $\times$ 1 tile). For input videos with a resolution of 2K or higher, you could use more tiles to avoid running out of memory.
- `--decode_chunk_size`: The number of frames to decode through the inpainting VAE at once. Lower values reduce peak GPU memory and can avoid unstable compiled-decoder paths during VAE decode. The default value is `frames_chunk`.
- `--vae_encode_chunk_size`: The number of frames to encode through the inpainting VAE at once. Lower values reduce peak GPU memory during VAE encoding when spatial tiling alone is not enough. The default value is `5`.
- `--final_video_codec`: Optional `ffmpeg` codec used to transcode the final SBS/anaglyph deliverables after the reliable OpenCV `mp4v` write completes. Recommended value is `libx264`. Default is disabled, including in the checked-in helper script.
- `--final_video_preset`: `ffmpeg` preset used when `--final_video_codec` is `libx264` or `libx265`. Default is `medium`.
- `--final_video_crf`: `ffmpeg` CRF used when `--final_video_codec` is `libx264` or `libx265`. Lower is higher quality. Default is `18`.
- `--compile_cache_dir`: Optional persistent `torch.compile` cache directory. Default is `./.torch_compile_cache/inpainting`.
- `--compile_warmup`: Warm up the canonical inpainting chunk shape before the main loop and reuse it. Default is `True`.

Inpainting keeps `torch.compile` on the dominant full-size chunk shape and runs undersized tail chunks eagerly to avoid a second recompilation/autotuning pass.

If inpainting is falling into repeated OOM retries, a practical speed-oriented invocation is:
```bash
python inpainting_inference.py \
    --pre_trained_path ./weights/stable-video-diffusion-img2vid-xt-1-1 \
    --unet_path ./weights/StereoCrafter \
    --input_video_path ./outputs/camel_splatting_results.mp4 \
    --save_dir ./outputs \
    --target_fps 12 \
    --max_res 768 \
    --num_inference_steps 4
```
This typically cuts runtime by reducing chunk count first, then avoiding the expensive high-resolution OOM fallback path.

For maximum compatibility inside constrained containers, OpenCV still writes with `mp4v` internally. If `--final_video_codec libx264` is enabled, StereoCrafter writes temporary `_opencv.mp4` files and then lets `ffmpeg` transcode the final deliverables.

The stereo video inpainting generates the stereo video result in side-by-side format and anaglyph 3D format, as shown below:

<img src="assets/camel_sbs.jpg" alt="camel_sbs" width="800"/> 

<img src="assets/camel_anaglyph.jpg" alt="camel_anaglyph" width="400"/>

## 🤝 Acknowledgements

We would like to express our gratitude to the following open-source projects:
- [Stable Video Diffusion](https://github.com/Stability-AI/generative-models): A latent diffusion model trained to generate video clips from an image or text conditioning.
- [DepthCrafter](https://github.com/Tencent/DepthCrafter): A novel method to generate temporally consistent depth sequences from videos.


## 📚 Citation

```bibtex
@article{zhao2024stereocrafter,
  title={Stereocrafter: Diffusion-based generation of long and high-fidelity stereoscopic 3d from monocular videos},
  author={Zhao, Sijie and Hu, Wenbo and Cun, Xiaodong and Zhang, Yong and Li, Xiaoyu and Kong, Zhe and Gao, Xiangjun and Niu, Muyao and Shan, Ying},
  journal={arXiv preprint arXiv:2409.07447},
  year={2024}
}
```
