from pymongo import MongoClient
from bson.objectid import ObjectId


class DataBaseManager:
    """
    Class for managing database that stores data about processed videos.

    Atrributes:
    - bucket_manager : Collection that stores bucket_manager's data.
    - stored_entities : Collection that stores information about entities.  
    """
    def __init__(self, ):
        """Manager initialization"""
        client = MongoClient('mongodb://localhost:27017/')
        database = client['open_eye_sight']
        self.bucket_manager = database["bucket_manager"]
        self.stored_entities = database["stored_entities"]
    
    def add_entity(self, entity_id):
        """Add entity to the database"""
        insert_entity = self.stored_entities.insert_one({
            'name': entity_id,
            'value' : {'vis_embs' : None, 'img' : None, 'timestamp' : None}
        })
        return insert_entity.inserted_id
    def upd_entity(self, entity_dbid, new_vals):
        """Update entity values in the database"""
        new_vals = {'value.' + key : val for key, val in new_vals.items()}
        #print(new_vals)
        self.stored_entities.update_one(
            {'_id': ObjectId(entity_dbid)},
            {'$set' : new_vals}
        )
    def get_entity_by_id(self, entity_id):
        """Get entity by id."""
        return self.stored_entities.find_one({"name" : entity_id})
    def get_entity(self, entity_dbid):
        """Get entity by Database id."""
        return self.stored_entities.find_one({"_id" : ObjectId(entity_dbid)})
    def add_entity_to_bucket(self, bucket_id, entity_dbid):
        """Add entity to the bucket list"""
        self.bucket_manager.update_one(
            {'name' : bucket_id},
            {'$push' : {'entities' : entity_dbid}},
            upsert=True
        )
    def get_bucket(self, bucket_id):
        """Get list of entities in bucket"""
        return self.bucket_manager.find_one({'name' : bucket_id})['entities']
    def get_buckets(self):
        """Get all buckets id"""
        all_lists = self.bucket_manager.distinct('name')
        return all_lists
    



    


