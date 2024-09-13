import sys
import os
import torch
from torch import nn
sys.path.append(os.getcwd()) 
import tools.clip as clip

  

class ObjectsBucket():
    """
    A class for managing and performing operations on a collection of objects efficiently and perform semantic search.

    Attributes:
        clip_feats (int): The dimensionality of CLIP feature vectors.
        objs (torch.Tensor): Tensor storing the CLIP embeddings of objects.
        obj_features (torch.Tensor): Normalized CLIP embeddings for efficient similarity computation.
        size (int): Current number of objects in the bucket.
        capacity (int): Current capacity of the bucket.
        entity_ids (list): List of the entity ids for the objects. 
        time_stamps (list): List of time stamps for the objects.
        device (str): The device to use for computations ('cuda' or 'cpu').
        cos_sim (nn.CosineSimilarity): Cosine similarity function for semantic search.
        buffer_capacity (int): Maximum capacity of the buffer.
        buffer_size (int): Current number of objects in the buffer.
        buffer (list): Temporary storage for objects before processing.
    
    Why do we need buckets?
    - We can distribute resources easier by using buckets
    """
    def __init__(self, clip_feats = 768):
        """
        Initialize the ObjectsBucket.

        Parameters:
            clip_feats (int, optional): The dimensionality of CLIP feature vectors. Defaults to 768.
        """
        self.clip_feats = clip_feats
        self.objs = torch.empty((1, clip_feats), dtype=torch.float32, device="cpu")
        self.obj_features = torch.empty((1, clip_feats), dtype=torch.float32, device="cpu")
        self.size, self.capacity = 0, 1
        self.entity_ids = []
        self.timestamps = []  
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.cos_sim = nn.CosineSimilarity(dim=1, eps=1e-8)
        self.buffer_capacity, self.buffer_size = 8, 0
        self.buffer = [(None, None) for i in range(self.buffer_capacity)]


    def _adjust_capacity(self, additional_size):
        """
        Ensure that the bucket has enough capacity for new objects
        We're using c++ vector capacity mainatain approach.
        
        Parameters:
        - additional_size (int): Number of new objects to add.
        
        Why we just don't use cat(x, 1 element) instead of this?
        - Each time it will copy objs Tensor for cat, later each add will take a lot of time
        
        Why we don't initialize tensor with big size?
        - Too much memory consumption
        
        """
        while self.size + additional_size > self.capacity:
            self.objs = torch.cat([
                self.objs, torch.empty((self.capacity, self.clip_feats), 
                dtype=torch.float32)
            ])
            self.obj_features = torch.cat([
                self.obj_features, torch.empty((self.capacity, self.clip_feats),
                dtype=torch.float32)
            ])
            self.capacity = 2 * self.capacity                 
    def add_batch(self, obj_batch, timestamps, entity_ids, clip_model, clip_preprocess):
        """
        Add a batch of objects to the buffer, release buffer if it's necessary

        Parameters:
        - obj_batch (list): List of object images to be added.
        - timestamps (int): timestamp for each object.
        - clip_model: CLIP model
        - clip_preprocess: CLIP preprocessing transform
        """
        batch_size = len(obj_batch)
        start_idx = 0
        while start_idx < batch_size:
            additional_size = min(batch_size - start_idx, self.buffer_capacity - self.buffer_size)
            for idx in range(additional_size):
                self.buffer[self.buffer_size] = (obj_batch[start_idx], timestamps[start_idx], entity_ids[start_idx])
                start_idx += 1
                self.buffer_size += 1
            if additional_size == 0:
                self.release_buffer(clip_model=clip_model, clip_preprocess=clip_preprocess)
        return
    def release_buffer(self, clip_model, clip_preprocess):
        """
        Releases objects from buffer

        Parameters:
        - clip_model: CLIP model
        - clip_preprocess: CLIP preprocessing transform


        Why do we need buffers? 
        - Calculating a CLIP embeddings for a batch of objects is better
        - More control over GPU memory, you can contorl the batch size through buffer capacity  
        """
        if self.buffer_size == 0:
            return
        timestamps = [self.buffer[idx][1] for idx in range(self.buffer_size)]
        entity_ids = [self.buffer[idx][2] for idx in range(self.buffer_size)]
        processed_batch  = torch.cat([
            clip_preprocess(self.buffer[idx][0]).unsqueeze(0)
            for idx in range(self.buffer_size)
        ])
        self._adjust_capacity(self.buffer_size)
        
        with torch.no_grad():
            encoded_batch = clip_model.encode_image(processed_batch.to("cuda")).to("cpu")
        
        final_size = self.size + self.buffer_size
        
        self.objs[self.size:final_size] = encoded_batch
        self.obj_features[self.size:final_size] = (encoded_batch / encoded_batch.norm(dim=1, keepdim=True))
        
        self.timestamps.extend(timestamps)
        self.entity_ids.extend(entity_ids)
        
        self.buffer_size, self.size = 0, final_size

    def semantic_search(self, caption, clip_model, clip_preprocess, start_timestamp = 0, end_timestamp = 2000000000):
        """
        Perform semantic search on objects in bucket using CLIP model.

        Parameters:
        - caption (str): The text caption to search for.
        - clip_model (): CLIP Model

        Returns:
        - torch.Tensor: Logits per text, representing similarity scores.

        TODO:
        - Search in range [x, y]
        """
        self.release_buffer(clip_model, clip_preprocess)
        with torch.no_grad():
            text = clip.tokenize(caption).to(self.device)
            text = clip_model.encode_text(text)
            text_features = text.to(self.device, dtype=torch.float32)
            clipscore_per_text = self.cos_sim(text_features.expand(self.size, self.clip_feats), self.obj_features[0:self.size].to(self.device))
        results = []
        for i in range(clipscore_per_text.shape[0]):
            clip_score = 3.125*clipscore_per_text[i].to("cpu").numpy()
            if start_timestamp <= self.timestamps[i] and self.timestamps[i] <= end_timestamp and clip_score > 0.65:
                results.append((clip_score, self.timestamps[i], self.entity_ids[i]))
        return results

class BucketManager():
    """
    A class for managing buckets and perform operations on them.

    Attributes:
        buckets (list): List of ObjectsBucket objects
        clip_model (CLIP): CLIP model
        clip_preprocess (torchvision.transforms.Compose): preprocessing transform to CLIP format
    
    Why do we need BucketManager?
    - Control the buckets and make search on them more convenient.

    Also we can try to fixate the bucket capacity,  
    """
    def __init__(self,):
        self.buckets = {} # (timestamp(seconds)//60) * 60 : Bucket
        self.clip_model, self.clip_preprocess = clip.load("models/ViT-L-14.pt")
    def _get_bucket_idx(self, time_stamp):
        """
        Parameters:
        - time_stamp (int): Timestamp
        Returns:
        - int: Index of bucket for time_stamp.
        """
        return 60*(time_stamp//60)
    def _get_bucket_range(self, bucket_id):
        """
        returns range of timestamps in the bucket
        """
        return (bucket_id, bucket_id + 60)
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
                self.buckets.update({self._get_bucket_idx(time_stamp) : ObjectsBucket()})
            self.buckets[self._get_bucket_idx(time_stamp)].add_batch([obj], [time_stamp], [entity_id], self.clip_model, self.clip_preprocess)
    def search_object(self, caption, start_timestamp = 0, end_timestamp = 2000000000):
        """
        Perform semantic search on buckets.

        Parameters:
        - caption (str): The text caption to search for.

        Returns:
        - torch.Tensor: Logits per text, representing similarity scores.
        """
        results = []
        for bucket_id, bucket in self.buckets.items():
            bucket_timerange = self._get_bucket_range(bucket_id)
            if max(bucket_timerange[0], start_timestamp) >  min(bucket_timerange[1], end_timestamp):
                continue
            results.extend(bucket.semantic_search(caption, self.clip_model, self.clip_preprocess, start_timestamp, end_timestamp))
        results = sorted(results, reverse=True)
        return results[0:5] # return top 5 results