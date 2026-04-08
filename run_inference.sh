# python depth_splatting_inference.py \
#     --pre_trained_path ./weights/stable-video-diffusion-img2vid-xt-1-1 \
#     --unet_path ./weights/DepthCrafter \
#     --input_video_path ./source_video/camel.mp4 \
#     --output_video_path ./outputs/camel_splatting_results.mp4 \
#     --target_fps 24


python inpainting_inference.py \
    --pre_trained_path ./weights/stable-video-diffusion-img2vid-xt-1-1 \
    --unet_path ./weights/StereoCrafter \
    --input_video_path ./outputs/camel_splatting_results.mp4 \
    --save_dir ./outputs \
    --tile_num 2

# Faster stage-2 fallback for long/high-FPS videos:
#     --target_fps 12 \
#     --max_res 768 \
#     --num_inference_steps 4 \
#
# Optional final delivery transcode via ffmpeg:
#     --final_video_codec libx264 \
#     --final_video_preset medium \
#     --final_video_crf 18
