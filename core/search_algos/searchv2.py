from abc import ABC, abstractmethod
import sys
import os
import torch
from torch import nn
sys.path.append(os.getcwd()) 
import tools.clip as clip
import tools.dinov2.models as dino

class EntityObject():
    """
    A class with entity data.

    Atrributes:
    - vis_embs (torch.Tensor): Tensor that stores the vision embeddings of entity.
    - entity_id (int): The assigned entity id.
    - timestamp (int): Timestamp that indicates the moment when entity was captured.
    """
    def __init__(self, vis_embedding, entity_id : int, timestamp : int):
        self.vis_embedding = vis_embedding
        self.entity_id = entity_id
        self.timestamp = timestamp
    def get_entity_id(self,):
        return self.entity_id
    def get_time(self,):
        return self.timestamp
    def get_vis_embs(self,):
        return self.vis_embedding        

class EntitiesBucket(ABC):
    """
    An abstract class for managing and performing operations on a collection of objects efficiently and perform search.

    Attributes:
    - entities (list): A list of EntityObject instances.
    - device (str): The device to use for computations ('cuda' or 'cpu').
    - cos_sim (nn.CosineSimilarity): Cosine similarity function for semantic search.
    """
    def __init__(self, ):
        self.entities = []  
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.cos_sim = nn.CosineSimilarity(dim=1, eps=1e-8)

    def _gather(self, min_time, max_time):
        """
        Gather all vision embeddings of entities that appears in [min_time, max_time] range

        Parameteres:
        - min_time (int): The start of the search interval.  
        - max_time (int): The end of the search interval.
        
        Returns:
        - tupple: A tupple containing
            - list: Vision embeddings of entities within the time range.
            - list: Indices of the corresponding entities in the self.entities list.

        """
        vis_embs, idx = [], []
        for entity_idx in range(len(self.entities)):
            if min_time <= self.entities[entity_idx].get_time() and self.entities[entity_idx].get_time() <= max_time:
                vis_embs.append(self.entities[entity_idx].get_vis_embs().unsqueeze(0))
                idx.append(entity_idx)
        return (vis_embs, idx)
    
    def add_batch(self, entity_images, timestamps, entity_ids, model, preprocess):
        """
        Add batch of images to the bucket.
        
        Parameteres:
        - entity_images (list): List of PIL Images of entities (RGB).  
        - timestamps (list): List of timestamps when the object was captured.
        - entity_ids (list): List of entity ids.
        - model (nn.Module): Encoding model.
        - preprocess (Transofrm): Preprocessing transform for images.  
        """
        batch_size = len(entity_images)
        vis_embeddings = self.encode_image(entity_images, model, preprocess)
        for idx in range(batch_size):
            self.entities.append(
                EntityObject(vis_embedding=vis_embeddings[idx], timestamp=timestamps[idx], entity_id=entity_ids[idx])
            ) 
    
    def search(self, target_embedding, min_time, max_time, score_mul=0):
        """
        Calculate and return a distance between all embeddings stored in the bucket and target embedding.
        [IMPROVE THAT PART LATER, USE K-ANN OR ANOTHER METHOD]

        Parameteres:
        - target_embedding (torch.Tensor [1, embeddings_dim]): Target entity embeddings
        - min_time (int): min time
        - max_time (int): max_time
        - score_mul (int): Use scaling factor for the score calculation. If 1, uses the prewritten scaling factor.
        
        Returns:
        - list: A list of tuples containing (score, timestamp, entity_id) for matching entities.
        """
        score_mul = (self.score_mul if score_mul > 0 else 1)
        data_embeddings, data_idx = self._gather(min_time=min_time, max_time=max_time)
        data_size = len(data_idx)
        target_embedding = target_embedding.repeat(len(data_embeddings), 1)
        scores = self.cos_sim(torch.cat(data_embeddings, dim=0), target_embedding)
        results = []
        for idx in range(data_size):
            if score_mul * scores[idx] > self.threshold:
                results.append(
                    (score_mul * (scores[idx].numpy()), self.entities[data_idx[idx]].get_time(), self.entities[data_idx[idx]].get_entity_id())
                )
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
    def encode_image(image, model, preprocess):
        """
        Extract vision embeddings from the image(s) using the model.

        Parameters:
        - image (PIL.Image or list): Input image(s) to encode.
        - model (nn.Module): Encoding model.
        - preprocess (callable): Preprocessing transform for images.

        Returns:
        - torch.Tensor: Vision embeddings.
        """
        pass
    @staticmethod
    @abstractmethod
    def encode_text(text, model, preprocess):
        """
        Extract text embeddings using the model.

        Parameters:
        - text (str or list): Input text(s) to encode.
        - model (nn.Module): Encoding model.
        - preprocess (callable): Preprocessing transform for text.

        Returns:
        - torch.Tensor: Text embeddings.
        """
        pass
        

class EntititesBucketDINO(EntitiesBucket):
    """
    A subclass of EntitiesBucket that uses the DINOv2 model for entity encoding and search.

    Attributes:
        threshold (float): Similarity threshold for entity matching.
        score_mul (float): Scaling factor for similarity score.
    """
    def __init__(self,):
        super().__init__()
        self.threshold = 0.3
        self.score_mul = 1.0
    @staticmethod
    def load_model(path='models/dinov2_vitl14_pretrain.pth'):
        """ Refer to the description in parent class. """
        model = dino.vits.vit_large(
            patch_size=14,
            img_size=518,
            init_values=1.0,
            block_chunks=0
        )
        preprocess = dino.make_classification_eval_transform()
        model.load_state_dict(torch.load(path))
        model = model.to("cuda")
        return (model, preprocess)
    @staticmethod
    def encode_image(images, model, preprocess):
        """ Refer to the description in parent class. """
        images = (images if type(images) == list else [images])
        batch_size = len(images)
        processed_images = torch.cat(
            [preprocess(images[idx]).unsqueeze(0) for idx in range(batch_size)], dim=0
        )
        with torch.no_grad():
            vis_embeddings = model.encode_image(processed_images.to("cuda")).to("cpu")
        return vis_embeddings
    @staticmethod
    def encode_text(texts, model, preprocess):
        """ DINOv2 contains only vision transformer. """
        raise NotImplementedError("Text encoding is not supported in the DINOv2 model")


class EntititesBucketCLIP(EntitiesBucket):
    """
    A subclass of EntitiesBucket that uses the CLIP/ViT model for entity encoding and search.

    Attributes:
    - threshold (float): Similarity threshold for entity matching.
    - score_mul (float): Scaling factor for similarity score.
    """
    def __init__(self,):
        super().__init__()
        self.threshold = 0.65
        self.score_mul = 3.125
    @staticmethod
    def load_model(path='models/ViT-L-14.pt'):
        """ Refer to the description in parent class. """
        return clip.load(path)
    @staticmethod
    def encode_image(images, model, preprocess):
        """ Refer to the description in parent class. """
        images = (images if type(images) == list else [images])
        batch_size = len(images)
        processed_images = torch.cat(
            [preprocess(images[idx]).unsqueeze(0) for idx in range(batch_size)], dim=0
        )
        with torch.no_grad():
            vis_embeddings = model.encode_image(processed_images.to("cuda")).to("cpu")
        return vis_embeddings
    @staticmethod
    def encode_text(texts, model, preprocess):
        """ Refer to the description in parent class. """
        texts = (texts if type(texts) == list else [texts])
        with torch.no_grad():
            target_embeddings = model.encode_text(clip.tokenize(texts).to("cuda")).to("cpu")
        return target_embeddings


class BucketManagerV2():
    """
    A class for managing buckets and perform operations on them.

    Attributes:
    - buckets (list): List of ObjectsBucket objects
    - model (nn.Module): Encoding model.
    - preprocess (callable): Preprocessing transform for text.
    - block (int): Time block size.
    
    Why do we need BucketManager?
    - Control the buckets and make search on them more convenient.

    Also we can try to fixate the bucket capacity,  
    """
    def __init__(self, search_class, path, block_size=60):
        self.search_class = search_class
        self.buckets = {} # 
        self.model, self.preprocess = self.search_class.load_model(path)
        self.block = block_size

    def _get_bucket_idx(self, time_stamp):
        return self.block*(time_stamp//self.block)
    def _get_bucket_range(self, bucket_id):
        return (bucket_id, bucket_id + self.block)
    
    def add_batch(self, obj_batch, time_stamps, entity_ids):
        """
        Releases objects from buffer

        Parameters:
        - obj_batch (Tensor):
        - time_stamps (int): Timestamp. 
        - entity_ids (int): Timestamp.
        """
        for obj, time_stamp, entity_id in zip(obj_batch, time_stamps, entity_ids):
            if self._get_bucket_idx(time_stamp) not in self.buckets:
                self.buckets.update({self._get_bucket_idx(time_stamp) : self.search_class()})
            self.buckets[self._get_bucket_idx(time_stamp)].add_batch([obj], [time_stamp], [entity_id], self.model, self.preprocess)
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
        target_embeddings = self.search_class.encode_text(caption, self.model, self.preprocess)
        for bucket_id, bucket in self.buckets.items():
            bucket_timerange = self._get_bucket_range(bucket_id)
            if max(bucket_timerange[0], min_time) >  min(bucket_timerange[1], max_time):
                continue
            results.extend(bucket.search(target_embeddings, min_time, max_time, 1))
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
        target_embeddings = self.search_class.encode_image(image, self.model, self.preprocess)
        for bucket_id, bucket in self.buckets.items():
            bucket_timerange = self._get_bucket_range(bucket_id)
            if max(bucket_timerange[0], min_time) >  min(bucket_timerange[1], max_time):
                continue
            results.extend(bucket.search(target_embeddings, min_time, max_time, 0))
        results = sorted(results, reverse=True)
        return results[0:5] # return top 5 results