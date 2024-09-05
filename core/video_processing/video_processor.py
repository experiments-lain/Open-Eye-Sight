import cv2
import os
import numpy as np
import torch
import torchvision
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor, to_pil_image
# CV2 READS HWC
# 

class VideoProccessor:
    """
    Class for the initial video proccessing
    
    STANDART : USING RGB VIDEOS & IMAGES WITH SHAPE [*, C, H, W] (AND AS TENSOR | NUMPY NDARRAY) 
    """
    def __init__(self, cap_freq = 4):
        self.cap_freq = cap_freq
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.transform = torchvision.transforms.Resize(
            (640, 640), 
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR, 
            max_size=None, 
            antialias=False,
        ).to(self.device) # Transfrom for YOLO, maybe change later.
    
    def read_video(self, video_path):
        """Read video from the given path, sample it and convert it to RGB Tensor[F, H, W, C]."""
        try:
            capturer = cv2.VideoCapture(video_path)
            video, step = [], 0
            while(True):
                ret, frame = capturer.read()
                if not ret:
                    break
                if step % self.cap_freq == 0:
                    frame = torch.tensor(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    frame = self.transform(torch.tensor(frame).permute((2, 0, 1))) # permuting for transform HWC->CHW
                    video.append(frame.unsqueeze(0))
                step += 1
            video = torch.cat(video, axis=0).cpu().permute((0, 2, 3, 1))
            capturer.release()
            return video
        except Exception as e:
            print(f"Error reading video from {video_path}: {e}")
            return None
        
    @staticmethod
    def read_image(image_path):
        """Read image from the given path and convert it to RGB Tensor[H, W, C]."""
        try:
            os.makedirs(os.path.dirname(os.getcwd() + image_path), exist_ok=True)
            return pil_to_tensor(Image.open(image_path).convert("RGB" )).permute(1, 2, 0)
        except Exception as e:
            print(f"Error reading image from {image_path}: {e}")
    @staticmethod
    def save_video(video, output_path):
        """Save the given video(RGB) Tensor[F, H, W, C] to the specified output path. If the video is saved, returns true."""
        try:
            directory_path, _ = os.path.split(output_path)    
            os.makedirs(os.path.dirname(os.getcwd() + directory_path), exist_ok=True)
            video = video.cpu()
            frames, height, width, _ = video.shape
            out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 24, (width, height))
            for frame in range(frames):
                out.write(np.uint8(video[frame]))
            out.release()
            return True
        except Exception as e:
            print(f"Error saving video to {output_path}: {e}")
            return False
    @staticmethod
    def save_image(image, output_path):
        """Save the given image(RGB) Tensor[H, W, C] to the specified output path. If the image is saved, returns true."""
        try:
            directory_path, _ = os.path.split(output_path)    
            os.makedirs(os.path.dirname(os.getcwd() + directory_path), exist_ok=True)
            image = to_pil_image(image.permute(2, 0, 1))
            image.save(output_path)
            return True
        except Exception as e:
            print(f"Error saving image to {output_path}: {e}")
            return False

        return 
    @staticmethod
    def show_image(image):
        """
        Show image.

        Attributes:
        - image (Tensor, shape[H, W, C]):
        Returns:
        - boolean: If video was saved succesfully return True
        """
        VideoProccessor.get_pil_image(image).show()
    @staticmethod
    def get_pil_image(image):
        """Convert the given image in standart format to PIL Image"""
        return to_pil_image(image.permute((2, 0, 1)))
    