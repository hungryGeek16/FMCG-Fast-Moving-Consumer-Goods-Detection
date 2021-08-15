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
from detectron2.evaluation import (
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    verify_results,
)
from detectron2.evaluation import inference_on_dataset

parser = argparse.ArgumentParser()
parser.add_argument('--cpu', action='store_true')

args = parser.parse_args()

sys.path.insert(1, 'AdelaiDet')
## Registering data in COCO format
for d in ["train","test"]:
    register_coco_instances(f"grocery_{d}", {}, f"instances_{d}.json", f"ShelfImages/{d}/")

## Loading the data
dataset_dicts = DatasetCatalog.get("grocery_train")
grocery_metadata = MetadataCatalog.get("grocery_train")


## Making a Cutsom Mapper to perform augmentations

def custom_mapper(dataset_dict):
    dataset_dict = copy.deepcopy(dataset_dict)  
    image = utils.read_image(dataset_dict["file_name"], format="BGR")
    transform_list = [T.Resize((800,800)),
                      T.RandomFlip(prob=0.5, horizontal=False, vertical=True),
                      T.RandomFlip(prob=0.5, horizontal=True, vertical=False),
                      T.RandomApply(T.RandomRotation(angle=[-30,30], expand=True, center=None, sample_style="range", interp=None), prob=0.25),
                      T.RandomApply(T.RandomCrop(crop_type="relative_range", crop_size=(0.4, 0.4)), 
                      prob=0.20)
                      ]
   
    image, transforms = T.apply_transform_gens(transform_list, image)
    dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))

    annos = [
        utils.transform_instance_annotations(obj, transforms, image.shape[:2])
        for obj in dataset_dict.pop("annotations")
        if obj.get("iscrowd", 0) == 0
    ]
    instances = utils.annotations_to_instances(annos, image.shape[:2])
    dataset_dict["instances"] = utils.filter_empty_instances(instances)
    return dataset_dict

## Class which can call the custom mapper we made

class GroceryTrainer(DefaultTrainer):
    
    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=custom_mapper)

## Training

cfg = get_cfg()
cfg.merge_from_file("AdelaiDet/configs/FCOS-Detection/MS_R_50_2x.yaml")
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
cfg.DATASETS.TRAIN = ("grocery_train",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = "detectron2://ImageNetPretrained/MSRA/R-50.pkl"
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 20000
if args.cpu == True:
    cfg.MODEL.DEVICE='cpu'
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = GroceryTrainer(cfg) 
trainer.resume_or_load(resume=True)
trainer.train()


cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we trained
cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.3
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3
cfg.MODEL.FCOS.INFERENCE_TH_TEST = 0.3
cfg.MODEL.MEInst.INFERENCE_TH_TEST = 0.3
cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = 0.3
predictor = DefaultPredictor(cfg)

### Testing
output_folder="output/"        
output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
evaluator_list = []
evaluator_type = MetadataCatalog.get("grocery_test").evaluator_type
evaluator_list.append(COCOEvaluator("grocery_test", cfg, True, output_folder))

eval = DatasetEvaluators(evaluator_list)
from detectron2.data import build_detection_test_loader
val_loader = build_detection_test_loader(cfg, "grocery_test")
print(inference_on_dataset(trainer.model, val_loader, eval))
