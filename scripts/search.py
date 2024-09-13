import os
import sys
sys.path.append(os.getcwd()) 
from core.pipelines import SearchPipeline
from core.search_algos.searchv2 import EntititesBucketCLIP, EntititesBucketDINO

def __main__():
    search_pipeline = SearchPipeline(
        search_cls=EntititesBucketDINO, 
        sampling_rate=24, 
        yolo_path="models/yolov8x.pt",
        model_path="models/dinov2_vitl14_pretrain.pth",
    ) 
    # Search by image
    search_pipeline.add_video("assets/search/citycam.mp4")
    search_pipeline.search_by_image("assets/search/white_truck.jpg")

__main__()

