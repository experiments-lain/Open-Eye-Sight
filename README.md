<p align="center">
    <img src="./assets/readme/icon.png" width="256"/>
</p>
<div align="center">
    <a href="https://github.com/experiments-lain/Open-Eye-Sight/stargazers"><img src="https://img.shields.io/github/stars/experiments-lain/Open-Eye-Sight?style=social"></a>
   
</div>

## Open-Eye-Sight: Multi-Camera and Video Perception

**Open-Eye-Sight** is an initiative dedicated to **efficiently** run entities and events search on the video by textual description/image. 
Open-Eye-Sight aims to provide open-source solution for multi-camera semantic and image search problem and try out the capabilities of 
modern vision models. My method is capable of finding the objects and events over custom time interval on the video/multi-camera stream by using textual description or target image.

## 📰 News

- **[2024.09.04]** The first implementation released.
  [[Technical Report]](docs/report_01.md)

## 🎥 Demo


<p align="center">
  <img src="assets/readme/demo.gif" alt="Demo GIF" class="fast-gif">
</p>

## 🔆 New Features/Updates

- [x] Image search based on DINOv2 visual embeddings extraction

- [x] Semantic search based on CLIP visual&text embeddings extraction, the text and image queries are available

- [x] Pluggable image encoding model system that allows to easily connect other image encoding models for this task.

### TODO list sorted by priority

<details>
<summary>View more</summary>

- [ ] Asynchronized data loading from multiple sources (EarthCam Videos/Streams).

- [ ] Change the architecture of the BucketManagerV2 to connect the MongoDB, and suddenly connect the MongoDB

</details>

## Contents

- [Installation](#installation)
- [Search](#search)
- [Evaluation](#evaluation)
- [Acknowledgement](#acknowledgement)

Other useful documents and links are listed below.

- Technical Reports:
  - [report 1.0](docs/report_01.md): architecture, implementation details, etc.
- Repo structure: [structure](docs/structure.md)

## Installation

### Install from Source

For CUDA 12.1, you can install the dependencies with the following commands. Otherwise, please refer to [TODO](TODO) for more instructions on different cuda version, and additional dependency.

```bash
# create a virtual env and activate (conda as an example)
conda create -n camera_search python=3.12.4
conda activate camera_search

# download the repo
git clone https://github.com/experiments-lain/Open-Eye-Sight
cd Open-Eye-Sight

# install required libraries (torch, torchvision, ultralytics and xformers)
pip install -r requirements/requirements.txt # TODO TODO TODO
```

### Use Docker

TODO

## Search

### Search using script

Fill the data(video path & query) to the search.py code and run the scripts/search.py.

### Search using config

TODO : CONFIG DESCRIPTION, ADD PARSING WITHOUT CONFIG ETC.

```bash
python scripts/search3.py --config configs/sample_vss.yaml
```


## Evaluation

TODO

## Acknowledgement

Here we list a projects that were used in the Open-Eye-Sight implementation.

- [CLIP](https://github.com/openai/CLIP): A powerful text-image embedding model that uses contrastive learning approach.
- [YOLOv8](https://github.com/ultralytics/ultralytics): A powerful computer vision model for object detection.
- [DinoV2](https://github.com/facebookresearch/dinov2/tree/main): A powerful vision transformer model from Meta.


