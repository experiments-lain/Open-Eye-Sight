from abc import ABC, abstractmethod
import sys
import os
import torch
from torch import nn
sys.path.append(os.getcwd())
from core.database.database_manager import DataBaseManager
import tools.clip as clip
import tools.dinov2.models as dino
import numpy as np
        

class BucketOperations(ABC):
    """
    Static abstract class for performing operations on a collection of objects efficiently.

    Attributes:
    - device (str): The device to use for computations ('cuda' or 'cpu').
    - cos_sim (nn.CosineSimilarity): Cosine similarity function for semantic search.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cos_sim = nn.CosineSimilarity(dim=1, eps=1e-8)

    @staticmethod    
    def _gather(min_time: int, max_time: int, db_manager: DataBaseManager, bucket_id: int):
        """
        Gather all visual embeddings of entities that appears in [min_time, max_time] range

        Parameteres:
        - min_time (int): The start of the search interval.  
        - max_time (int): The end of the search interval.
        - db_manager (DataBaseManager): Database manager.
        - bucket_id (int): Bucket id.
        
        Returns:
        - tupple: A tupple containing
            - list: Visiual embeddings of entities within the time range.
            - list: Indices of the corresponding entities in the self.entities list.

        """
        entities = db_manager.get_bucket(bucket_id)
        vis_embs, idx = [], []
        for entity_dbid in entities:
            entity_data = db_manager.get_entity(entity_dbid)
            if min_time <= entity_data['value']['timestamp'] and entity_data['value']['timestamp'] <= max_time:
                vis_embs.append(torch.from_numpy(np.array(entity_data['value']['vis_embs'])).unsqueeze(0))
                idx.append(entity_data['_id'])
        return (vis_embs, idx)
    
    @staticmethod  
    def add_batch(bucket_id, entity_images, timestamps, entity_dbids, db_manager, search_cls):
        """
        Add batch of images to the bucket.
        
        Parameteres:
        - bucket_id(int): Bucket id.
        - entity_images (list): List of PIL Images of entities (RGB).  
        - timestamps (list): List of timestamps when the object was captured.
        - entity_dbids (list): List of entity dbids.
        - db_manager (DataBaseManager): Database manager.
        - search_cls: Used search class.
        """
        vis_embeddings = search_cls.encode_image(entity_images)
        for vis_emb, timestamp, entity_dbid in zip(vis_embeddings, timestamps, entity_dbids):
            db_manager.upd_entity(entity_dbid, {"vis_embs" : vis_emb.numpy().tolist(), "timestamp" : timestamp})
            db_manager.add_entity_to_bucket(bucket_id, entity_dbid)
    
    @staticmethod
    def search(target_embedding, min_time, max_time, db_manager, bucket_id, search_cls, score_mul=0):
        """
        Calculate and return a distance between all embeddings stored in the bucket and target embedding.
        [IMPROVE THAT PART LATER, USE K-ANN OR ANOTHER METHOD]

        Parameteres:
        - target_embedding (torch.Tensor [1, embeddings_dim]): Target entity embeddings
        - min_time (int): min time
        - max_time (int): max_time
        - db_manager (DatabaseManager): Database manager.
        - bucket_id (int): Bucket ID
        - search_cls: Used search class.
        - score_mul (int): Use scaling factor for the score calculation. If 1, uses the prewritten scaling factor.
        
        Returns:
        - list: A list of tuples containing (score, timestamp, entity_id) for matching entities.
        """
        score_mul = (search_cls._score_mul if score_mul > 0 else 1)
        data_embeddings, data_idx = __class__._gather(min_time=min_time, max_time=max_time, db_manager=db_manager, bucket_id=bucket_id)
        data_size = len(data_idx)
        if data_size == 0:
            return []
        target_embedding = target_embedding.repeat(len(data_embeddings), 1)
        scores = __class__.cos_sim(torch.cat(data_embeddings, dim=0), target_embedding)
        results = []
        for idx in range(data_size):
            if score_mul * scores[idx] > search_cls._threshold:
                timestamp = db_manager.get_entity(data_idx[idx])['value']['timestamp']
                results.append((score_mul * (scores[idx].numpy()), timestamp, data_idx[idx]))
        return results

    @staticmethod
    @abstractmethod
    def load_model(path):
        """
        Load model from path.

        Parameters:
        - path (str): path to the model.
        
        Returns:
        - tuple: A tuple containing the loaded model and preprocessing function.
        """
        pass
    @staticmethod
    @abstractmethod
    def encode_image(image):
        """
        Extract vision embeddings from the image(s) using the model.

        Parameters:
        - image (PIL.Image or list): Input image(s) to encode.

        Returns:
        - torch.Tensor: Vision embeddings.
        """
        pass
    @staticmethod
    @abstractmethod
    def encode_text(text):
        """
        Extract text embeddings using the model.

        Parameters:
        - text (str or list): Input text(s) to encode.

        Returns:
        - torch.Tensor: Text embeddings.
        """
        pass
        

class BucketOperationsDINO(BucketOperations):
    """
    A subclass of EntitiesBucket that uses the DINOv2 model for entity encoding and search.

    Attributes:
    - _threshold (float): Similarity threshold for entity matching.
    - _score_mul (float): Scaling factor for similarity score.
    - _model: Visual features extracting model.
    - _preprocess: Preprocess to use the model.
    """
    _threshold = 0.3
    _score_mul = 1.0
    _model = None
    _preprocess = None

    @staticmethod
    def load_model(path='models/dinov2_vitl14_pretrain.pth'):
        """ Refer to the description in parent class. """
        __class__._model = dino.vits.vit_large(
            patch_size=14,
            img_size=518,
            init_values=1.0,
            block_chunks=0
        )
        __class__._preprocess = dino.make_classification_eval_transform()
        __class__._model.load_state_dict(torch.load(path))
        __class__._model = __class__._model.to("cuda")
    @staticmethod
    def encode_image(images):
        """ Refer to the description in parent class. """
        images = (images if type(images) == list else [images])
        batch_size = len(images)
        processed_images = torch.cat(
            [__class__._preprocess(images[idx]).unsqueeze(0) for idx in range(batch_size)], dim=0
        )
        with torch.no_grad():
            vis_embeddings = __class__._model.encode_image(processed_images.to("cuda")).to("cpu")
        return vis_embeddings
    @staticmethod
    def encode_text(texts):
        """ DINOv2 contains only vision transformer. """
        raise NotImplementedError("Text encoding is not supported in the DINOv2 model")


class BucketOperationsCLIP(BucketOperations):
    """
    A subclass of EntitiesBucket that uses the CLIP/ViT model for entity encoding and search.

    Attributes:
    - _threshold (float): Similarity threshold for entity matching.
    - _score_mul (float): Scaling factor for similarity score.
    - _model: Visual features extracting model.
    - _preprocess: Preprocess to use the model.
    """
    _threshold = 0.3
    _score_mul = 1.0
    _model = None
    _preporcess = None

    @staticmethod
    def load_model(path='models/ViT-L-14.pt'):
        """ Refer to the description in parent class. """
        __class__._model, __class__._preporcess = clip.load(path)
    @staticmethod
    def encode_image(images):
        """ Refer to the description in parent class. """
        images = (images if type(images) == list else [images])
        batch_size = len(images)
        processed_images = torch.cat(
            [__class__._preporcess(images[idx]).unsqueeze(0) for idx in range(batch_size)], dim=0
        )
        with torch.no_grad():
            vis_embeddings = __class__._model.encode_image(processed_images.to("cuda")).to("cpu")
        return vis_embeddings
    @staticmethod
    def encode_text(texts):
        """ Refer to the description in parent class. """
        texts = (texts if type(texts) == list else [texts])
        with torch.no_grad():
            target_embeddings = __class__._model.encode_text(clip.tokenize(texts).to("cuda")).to("cpu")
        return target_embeddings


class BucketManagerV2():
    """
    A class for managing buckets and perform operations on them.

    Attributes:
    - search_class: Search class.
    - db_manager (DatabaseManager): Database manager.
    - block (int): Time block size.
    
    Why do we need BucketManager?
    - Control the buckets and make search on them more convenient.

    Also we can try to fixate the bucket capacity,  
    """
    def __init__(self, search_class, path, block_size=60):
        self.search_class = search_class
        self.db_manager = DataBaseManager()
        self.search_class.load_model(path)
        self.block = block_size

    def _get_bucket_idx(self, time_stamp):
        return self.block*(time_stamp//self.block)
    def _get_bucket_range(self, bucket_id):
        return (bucket_id, bucket_id + self.block)
    
    def add_batch(self, obj_batch, time_stamps, entity_dbids):
        """
        Releases objects from buffer

        Parameters:
        - obj_batch (Tensor):
        - time_stamps (int): Timestamp. 
        - entity_ids (int): Timestamp.
        """
        for obj, time_stamp, entity_dbid in zip(obj_batch, time_stamps, entity_dbids):
            self.search_class.add_batch(
                self._get_bucket_idx(time_stamp), 
                [obj], [time_stamp], [entity_dbid], 
                self.db_manager, self.search_class
            )
    def text_search(self, caption, min_time = 0, max_time = 2000000000):
        """
        Perform search on entities by text description.

        Parameters:
        - caption (str): The text caption to search for.
        - min_time (int): 
        - max_time (int):

        Returns:
        - torch.Tensor: top 5 results
        """
        results = []
        target_embeddings = self.search_class.encode_text(caption)
        buckets = self.db_manager.get_buckets()

        for bucket_id in buckets:
            bucket_timerange = self._get_bucket_range(bucket_id)
            if max(bucket_timerange[0], min_time) >  min(bucket_timerange[1], max_time):
                continue
            results.extend(self.bucket.search(target_embeddings, min_time, max_time, self.db_manager, bucket_id, self.search_class, 0))
        results = sorted(results, reverse=True)
        return results[0:5] # return top 5 results
    def image_search(self, image, min_time = 0, max_time = 2000000000):
        """
        Perform search on entities by image.

        Parameters:
        - image (PIL Image): The image to search for.
        - min_time (int): 
        - max_time (int):

        Returns:
        - torch.Tensor: top 5 results
        """
        results = []
        target_embeddings = self.search_class.encode_image(image)
        buckets = self.db_manager.get_buckets()

        for bucket_id in buckets:
            bucket_timerange = self._get_bucket_range(bucket_id)
            if max(bucket_timerange[0], min_time) >  min(bucket_timerange[1], max_time):
                continue
            results.extend(self.search_class.search(target_embeddings, min_time, max_time, self.db_manager, bucket_id, self.search_class, 0))
        results = sorted(results, reverse=True)
        return results[0:5] # return top 5 results