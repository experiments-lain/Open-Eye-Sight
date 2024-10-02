# Repo Structure

```plaintext
Open-Eye-Sight
├── README.md
├── assets
│   ├── reports                    -> Images for the reports
│   ├── search                     -> Images used for demo
│   └── readme                     -> Assets used in README
├── configs                        -> Configs for searchs
├── models                         -> Models
├── core
│   ├── data_mining                -> Kinda useless, probably I'll move it to tools later
│   ├── pipelines                  -> Module that contains pipelines
│   │   └── search_pipeline.py     -> Pipeline for the search function
│   ├── search_algos               -> The searching algorithms 
│   │   ├── searchv1.py            -> The first implementation of the searching algorithms
│   │   └── searchv2.py            -> Last implementation of the searching algorithms (buckets + pluggable image encoding models system)
│   └── video_processing           -> Video processing algorithms & functions
│       ├── obj_retr.py            -> Object retriever implementation
│       └── video_processor.py     -> Video processor implementation            
├── docs
│   ├── structure.md               -> This file
│   └── report_01.md               -> Report for Open-Eye-Sight first implementation
├── scripts
│   └── search.py                  -> Search script.
├── tests                          -> Tests for the project
└── tools                          -> Tools for models
    ├── clip                       -> code for CLIP model
    └── dinov2                     -> code for DINOv2 model
```
