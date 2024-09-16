import os
import sys
sys.path.append(os.getcwd()) 
from core.pipelines.search_pipelinev2 import SearchPipeline
from core.search_algos.searchv3 import BucketOperationsCLIP, BucketOperationsDINO
import time

def __main__():
    search_pipeline = SearchPipeline(
        search_cls=BucketOperationsDINO,
        yolo_path="models/yolov8x.pt",
        model_path="models/dinov2_vitl14_pretrain.pth",
    ) 
    # Search by image
    time_start = time.time()
    search_pipeline.add_video("assets/search/city_camera.mp4")
    search_pipeline.search_by_image("assets/search/search_long.jpg")
    print(time.time() - time_start)

__main__()

