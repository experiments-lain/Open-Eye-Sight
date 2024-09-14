import os
import sys
sys.path.append(os.getcwd()) 
import torch
import numpy as np
from torch import nn
from ultralytics import YOLO
from PIL import Image
from core.video_processing.video_processor import VideoProccessor
from core.database.database_manager import DataBaseManager

class ObjectRetriever:
    """
    A class for entity/object retrieving from videos and maintaining the retrieved entitites effectively.
    This class provides functionality to retrieve and store entity images using YOLO model. 

    Attributes:
    - model : The YOLO model for object detection.
    - iou_thresh (float): Treshold of intersection over union value for YOLO model.
    - conf_thresh (float): Treshold of confidence score for YOLO model.
    - device (str): The device to use for computations ('cuda' or 'cpu').
    - num_entities (int): The number of entities that was retrieved at the moment.
    - entity_images (list): List of entity images in PIL Image format. 
    """
    def __init__(self, yolo_weights_path="models/yolov8x.pt"):
        """
        Initialize the object retriever.

        Parameters:
        - yolo_weights_path (str): The path to your YOLO model weights.
        """
        self.model = YOLO(yolo_weights_path)
        self.iou_thresh = 0.7
        self.conf_thresh = 0.5
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.db_manager = DataBaseManager()
        self.num_entities = 0
        self.entity_images = []
    def _add_entity(self, entity):
        """
        Add entity to the object retriever storage.

        Parameteres:
        - entity (PIL Image [H, W, C]): The entity to be added. 
        """
        entity_dbid = self.db_manager.add_entity(entity_id=1337)
        self.db_manager.upd_entity(entity_dbid, {"img" :  (np.array(entity)).tolist()})
        return entity_dbid

    def _get_entity(self, entity_dbid):
        x = np.array(self.db_manager.get_entity(entity_dbid)['value']['img']).astype(np.uint8)
        return Image.fromarray(x)

    def retrieve_objects(self, source, timestamps):
        """
        Retrieve objects from source video/image.

        Parameteres:
        - source (Tensor [F, H, W, C]) : height, width, channels(RGB) (Image can be provided (shape : [C, H, W]))
        - timestamps : timestamp for each frame
        """
        if len(source.shape) == 3:
            # image -> video
            timestamps = np.array([timestamps])
            source = source.reshape((1, source.shape[0], source.shape[1], source.shape[2]))
        
        assert source.shape[3] == 3, f"The last dimension of the source must represent channels, the source shape : {source.shape}"
        
        _entities, _timestamps, _entity_ids = [], [], []
        
        for frame in range(source.shape[0]):
            img = VideoProccessor.get_pil_image(source[frame])
            with torch.no_grad():
                boxes = ((self.model(source=img, iou=self.iou_thresh, conf=self.conf_thresh, verbose=False))[0].boxes.xyxy)
            for obj in boxes:
                lx, rx = int(obj[0] - 0.5), int(obj[2] + 0.5)
                ly, ry = int(obj[1] - 0.5), int(obj[3] + 0.5)
                entity = img.crop((lx, ly, rx, ry))
                entity_id = self._add_entity(entity)
                _entities.append(entity)
                _timestamps.append(timestamps[frame])
                _entity_ids.append(entity_id)
        return (_entities, _timestamps, _entity_ids)