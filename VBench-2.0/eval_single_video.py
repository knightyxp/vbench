#!/usr/bin/env python
"""
Simple script to evaluate camera motion for a single video.

Usage:
    python eval_single_video.py /path/to/video.mp4
    
    # 使用不同的GPU (通过环境变量)
    CUDA_VISIBLE_DEVICES=1 python eval_single_video.py /path/to/video.mp4
"""

import cv2
import torch
import decord
decord.bridge.set_bridge('torch')
import argparse
import sys
from vbench2.camera_motion import CameraPredict
from vbench2.utils import split_video_into_scenes


def evaluate_video(video_path, device='cuda:0'):
    """Evaluate camera motion for a single video."""
    print(f"Loading video: {video_path}")
    
    # Initialize camera motion predictor
    print(f"Initializing model on {device}...")
    device = torch.device(device)
    submodules_config = {
        "repo": "facebookresearch/co-tracker",
        "model": "cotracker2"
    }
    camera_predictor = CameraPredict(device, submodules_config)
    
    # Detect scene changes
    end_frame = -1
    scene_list = split_video_into_scenes(video_path, 5.0)
    if len(scene_list) != 0:
        end_frame = int(scene_list[0][1].get_frames())
        print(f"Scene detection: Using first {end_frame} frames")
    
    # Load video
    print("Loading video frames...")
    video_reader = decord.VideoReader(video_path)
    video = video_reader.get_batch(range(len(video_reader)))
    frame_count, height, width = video.shape[0], video.shape[1], video.shape[2]
    
    # Prepare video tensor: B T C H W
    video = video.permute(0, 3, 1, 2)[None].float().to(device)
    
    # Get FPS
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    
    print(f"Video info: {frame_count} frames, {width}x{height}, {fps} fps")
    
    # Predict camera motion
    print("Analyzing camera motion...")
    camera_motion_types = camera_predictor.predict(video, fps, end_frame)
    
    # Print results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Video: {video_path}")
    print(f"Camera Motion Types: {camera_motion_types}")
    print("="*60)
    
    return camera_motion_types


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate camera motion for a single video',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python eval_single_video.py video.mp4
  CUDA_VISIBLE_DEVICES=1 python eval_single_video.py /path/to/video.mp4
  
Camera Motion Types:
  - pan_left / pan_right  : 左右摇镜
  - tilt_up / tilt_down   : 上下摇镜
  - zoom_in / zoom_out    : 推拉镜头
  - orbits                : 环绕运动
  - oblique               : 斜向运动
  - static                : 静止镜头
        """
    )
    parser.add_argument(
        'video_path',
        type=str,
        help='Path to the video file'
    )
    
    args = parser.parse_args()
    
    try:
        evaluate_video(args.video_path)
    except Exception as e:
        print(f"\nError: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

