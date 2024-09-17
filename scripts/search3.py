import os
import sys
sys.path.append(os.getcwd()) 
from core.pipelines.search_pipelinev2 import SearchPipeline
from core.search_algos.searchv3 import BucketOperationsCLIP, BucketOperationsDINO
from tools.parser import ArgsParser
import time

def __main__(**kwargs):
    search_pipeline = SearchPipeline(
        search_cls=BucketOperationsDINO,
        yolo_path=kwargs['yolo_ckpt_path'],
        model_path=kwargs['extractor_ckpt_path'],
    ) 
    # Search by image
    time_start = time.time()
    search_pipeline.search_by_image(kwargs['image_path'])
    print(time.time() - time_start)


kwargs = ArgsParser().parse_args()
print(kwargs)
__main__(**kwargs)

