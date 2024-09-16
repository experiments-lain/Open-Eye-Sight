import torch
import os
import sys
sys.path.append(os.getcwd())
from core.video_processing import ObjectRetriever, VideoProccessor
from core.search_algos.searchv3 import BucketManagerV2, BucketOperationsDINO
from PIL import Image
import time

# logger
class SearchPipeline():
    """Class that connect separate modules into one pipeline for searching objects on videos."""
    def __init__(
        self, 
        search_cls, 
        yolo_path,
        model_path,
        time_block_size=60,
        debug_mode=False,
    ):
        self.video_processor = VideoProccessor()
        self.object_retriever = ObjectRetriever(yolo_path)
        self.bucket_manager = BucketManagerV2(search_cls, model_path, time_block_size)
    def add_video(self, video_path):
        video = self.video_processor.read_video(video_path)
        timestamps = [i for i in range(video.shape[0])]
        _entities, _timestamps, _entity_ids = self.object_retriever.retrieve_objects(video, timestamps)
        self.bucket_manager.add_batch(_entities, _timestamps, _entity_ids)
    def search_by_image(self, image_path, min_time=0, max_time=2000000000):
        img = VideoProccessor.read_image(image_path)
        results = self.bucket_manager.image_search(VideoProccessor.get_pil_image(img), min_time, max_time)
        if len(results) == 0:
            print("No such objects on video.")
        else:
            print(f"Score : {results[0][0]}")
            print(f"Sampled frame : {results[0][1]}")
            print(f"Entity id : {results[0][2]}")
            self.object_retriever._get_entity(results[0][2]).show()
    def search_by_text(self, text, min_time=0, max_time=2000000000):
        results = self.bucket_manager.text_search(text, min_time, max_time)
        if len(results) == 0:
            print("No such objects on video.")
        else:
            print(f"Score : {results[0][0]}")
            print(f"Sampled frame : {results[0][1]}")
            print(f"Entity id : {results[0][2]}")
            self.object_retriever._get_entity(results[0][2]).show()

