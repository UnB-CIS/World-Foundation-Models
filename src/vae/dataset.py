"""Dataset Class and utils for VAE training."""

import os
import cv2

from PIL import Image
from torch.utils.data import Dataset


def extract_frames(video_path: str, frames_dir: str, frame_step: int = 5):
    """Extract frames from video at specified interval."""
    os.makedirs(frames_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    saved_count = 0

    while True:
        success, frame = cap.read()
        if not success:
            break

        if frame_count % frame_step == 0:
            frame_path = os.path.join(frames_dir, f"frame_{saved_count:04d}.jpg")
            cv2.imwrite(frame_path, frame)
            saved_count += 1

        frame_count += 1

    cap.release()
    # print(f"Extraídos {saved_count} frames (de {frame_count} totais) para {frames_dir}")


def process_videos(
    videos_folder: str, frames_root: str = "./frames", frame_step: int = 5
):
    """Process all videos in folder and extract frames."""
    for video_file in os.listdir(videos_folder):
        if video_file.lower().endswith((".mp4", ".avi", ".mov")):
            video_path = os.path.join(videos_folder, video_file)
            video_name = os.path.splitext(video_file)[0]

            frames_dir = os.path.join(frames_root, video_name)

            if not os.path.exists(frames_dir) or len(os.listdir(frames_dir)) == 0:
                extract_frames(video_path, frames_dir, frame_step=frame_step)


class VideoFramesDataset(Dataset):
    """Dataset for video frames."""

    def __init__(self, frames_dir, transform=None):
        # Recursively find all .jpg frames
        self.frame_files = sorted(
            [
                os.path.join(root, f)
                for root, _, files in os.walk(frames_dir)
                for f in files
                if f.lower().endswith(".jpg")
            ]
        )
        self.transform = transform

    def __len__(self):
        return len(self.frame_files)

    def __getitem__(self, idx):
        img_path = self.frame_files[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image
