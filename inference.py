import random
import cv2
import os 
import numpy as np
import pandas as pd  
import copy
import torch
import base64   
import json
import math
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sys
import argparse

from detectron2.data.datasets import register_coco_instances
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_train_loader
from detectron2.utils.visualizer import Visualizer
from detectron2.engine import DefaultTrainer
from adet.config import get_cfg
from detectron2 import model_zoo
from detectron2.utils.visualizer import ColorMode
from detectron2.engine.defaults import DefaultPredictor
from detectron2.data import detection_utils as utils
import detectron2.data.transforms as T
from detectron2.utils.visualizer import ColorMode

sys.path.insert(1, 'AdelaiDet')


parser = argparse.ArgumentParser()
parser.add_argument('--im', type=str, required=True)
args = parser.parse_args()

for d in ["test"]:
    register_coco_instances(f"grocery_{d}", {}, f"instances_{d}.json", f"ShelfImages/{d}/")

## Loading the data
dataset_dicts = DatasetCatalog.get("grocery_test")
grocery_metadata = MetadataCatalog.get("grocery_test")

cfg = get_cfg()
cfg.merge_from_file("AdelaiDet/configs/FCOS-Detection/MS_R_50_2x.yaml")
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
cfg.MODEL.WEIGHTS = "model_grocery_20k.pth"  # path to the model we trained
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.FCOS.INFERENCE_TH_TEST = 0.4
cfg.MODEL.DEVICE='cpu'
predictor = DefaultPredictor(cfg)   
    
im = cv2.imread(args.im)
outputs = predictor(im) 
v = Visualizer(im[:, :, ::-1],
                   metadata=grocery_metadata, 
                   scale=0.5, 
                   instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. Only available for segmentation models
    )
out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
cv2.imwrite("output.jpg",out.get_image()[:, :, ::-1][..., ::-1])
