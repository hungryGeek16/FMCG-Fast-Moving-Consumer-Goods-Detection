import numpy as np
import json
import pandas as pd
import cv2
import pandas as pd
import os
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--train_dir', type=str, required=True)
parser.add_argument('--test_dir', type=str, required=True)
parser.add_argument('--csv_file', type=str, required=True)

args = parser.parse_args()

test = os.listdir(args.test_dir)
train = os.listdir(args.train_dir)

df = pd.read_csv(args.csv_file,names=["filename","xmin","ymin","xmax","ymax","class"]) 

df['class'] = 1 # Class has been set 1, since detection is on product or no product level

train_anno=df[df['filename'].isin(train)] # Seprating csv files in to train-test splits
test_anno=df[df['filename'].isin(test)]

di = {"instances_train.json":train_anno,"instances_test.json":test_anno}

for i in ["instances_train.json","instances_test.json"]: # Conversion of csv files in to json forma starts from here 
  save_json_path = i #output path to the json i.e. jsons will be saved in current working directory.
  data = di[i]
  images = []
  categories = []
  annotations = []

  category = {}
  category["supercategory"] = 'none'
  category["id"] = 0
  category["name"] = 'None'
  categories.append(category)

  data['fileid'] = data['filename'].astype('category').cat.codes
  data['categoryid']= pd.Categorical(data['class'],ordered= True).codes
  data['categoryid'] = data['categoryid']+1
  data['annid'] = data.index

  def image(row): # Gets image size into JSON formats
    image = {}
    im = cv2.imread("ShelfImages/"+i[i.index("_")+1:i.index(".")]+"/"+row.filename)
    image["height"] = im.shape[0]
    image["width"] = im.shape[1]
    image["id"] = row.fileid
    image["file_name"] = row.filename
    return image

  def category(row): # Gets categories into JSON formats
    category = {}
    category["supercategory"] = 'Product'
    category["id"] = row.categoryid
    category["name"] = 'Product'
    return category

  def annotation(row): # Gets annotations into json format
    annotation = {}
    area = (row.xmax -row.xmin)*(row.ymax - row.ymin)
    annotation["segmentation"] = []
    annotation["iscrowd"] = 0
    annotation["area"] = area
    annotation["image_id"] = row.fileid
    annotation["bbox"] = [row.xmin, row.ymin, row.xmax -row.xmin,row.ymax-row.ymin ]
    annotation["category_id"] = row.categoryid
    annotation["id"] = row.annid
    return annotation

  for row in data.itertuples(): #Iterates over every frame
    annotations.append(annotation(row))

  imagedf = data.drop_duplicates(subset=['fileid']).sort_values(by='fileid')
  for row in imagedf.itertuples():
    images.append(image(row))

  catdf = data.drop_duplicates(subset=['categoryid']).sort_values(by='categoryid')
  for row in catdf.itertuples():
    categories.append(category(row))

  data_coco = {}
  data_coco["images"] = images
  data_coco["categories"] = categories
  data_coco["annotations"] = annotations
  json.dump(data_coco, open(save_json_path, "w"), indent=4) # Saves the file
