
import torch
import pandas as pd
import numpy as np
import json
import os
import sys
from sklearn.metrics import f1_score
from tqdm import tqdm
import logging
from helper_functions import *

sys.path.append('../')
import datasets
import models
import utils

import argparse

parser = argparse.ArgumentParser(description="Script to process thresholds and perform an experiment.")
parser.add_argument("--model_path", type=str, default='model_an_full_input_enc_sin_cos_hard_cap_num_per_class_1000.pt', help="Model path.")
parser.add_argument("--exp_name", type=str, default='test', help="Experiment name")
parser.add_argument(
        '--evaluation_set',
        type=str,
        choices=['iucn', 'snt'],
        default='iucn',
        help="Choose the species set to evaluate against."
    )
args = parser.parse_args()

iucn_json_path = "./sinr/data/eval/iucn/iucn_res_5.json"
snt_npy_path = "./sinr/data/eval/snt/snt_res_5.npy"

DEVICE = torch.device('cpu')

# Set up logging to file
log_file_path = "results/" + args.exp_name + "/log.out"
logging.basicConfig(filename=log_file_path, filemode='a', level=logging.INFO,
                    format='%(levelname)s: %(message)s')
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger('').addHandler(console)

logging.info(f"Model used for experiment: {args.model_path}")

# load model
train_params = torch.load(args.model_path, map_location='cpu')
model = models.get_model(train_params['params'])
model.load_state_dict(train_params['state_dict'], strict=True)
model = model.to(DEVICE)
model.eval()

#load species ids to evaluate
if args.evaluation_set == "iucn":
    with open(iucn_json_path, 'r') as f:
                data = json.load(f)
    species_ids = list((data['taxa_presence'].keys()))
elif args.evaluation_set == "snt":
    #load reference from snt
    data = np.load(snt_npy_path, allow_pickle=True)
    data = data.item()
    species_ids = data['taxa']
    loc_indices_per_species = data['loc_indices_per_species']
    labels_per_species = data['labels_per_species']

if train_params['params']['input_enc'] in ['env', 'sin_cos_env']:
    raster = datasets.load_env()
else:
    raster = None
enc = utils.CoordEncoder(train_params['params']['input_enc'], raster=raster)

if args.evaluation_set == "iucn": obs_locs = np.array(data['locs'], dtype=np.float32)
elif args.evaluation_set == "snt": obs_locs = np.array(data['obs_locs'], dtype=np.float32)
obs_locs = torch.from_numpy(obs_locs).to('cpu')
loc_feat = enc.encode(obs_locs)

classes_of_interest = torch.zeros(len(species_ids), dtype=torch.int64)
for tt_id, tt in enumerate(species_ids):
    class_of_interest = np.array([train_params['params']['class_to_taxa'].index(int(tt))])
    classes_of_interest[tt_id] = torch.from_numpy(class_of_interest)

with torch.no_grad():
    loc_emb = model(loc_feat, return_feats=True)
    wt = model.class_emb.weight[classes_of_interest, :]

num_points = 19

# Generate linearly spaced numbers in the range [0, 1]
linspace_values = np.linspace(0.05, 1, num=num_points, endpoint=False)
threshs = linspace_values

per_species_f1 = np.zeros((len(species_ids),len(threshs)))
for tt_id, class_id in tqdm(enumerate(classes_of_interest), total=len(classes_of_interest)):
    taxa = species_ids[tt_id]
    wt_1 = wt[tt_id,:]
    preds = torch.sigmoid(torch.matmul(loc_emb, wt_1)).cpu().numpy()
    if args.evaluation_set == "iucn":
        species_locs = data['taxa_presence'].get(str(taxa))
        y_test = np.zeros(preds.shape, int)
        y_test[species_locs] = 1

        for i, thresh in enumerate(threshs):
            per_species_f1[tt_id][i] = f1_at_thresh(y_test, preds, thresh, type='binary')
    elif args.evaluation_set == "snt":
        cur_loc_indices = np.array(loc_indices_per_species[tt_id])
        cur_labels = np.array(labels_per_species[tt_id])
        pred = preds[cur_loc_indices]
        for i, thresh in enumerate(threshs):
            per_species_f1[tt_id][i] = f1_at_thresh(y_test, preds, thresh, type='binary')


f1score = per_species_f1.mean(axis=0).max()
best_thres = linspace_values[per_species_f1.mean(axis=0).argmax()]

logging.info(f"Best unified threshold: {best_thres}")
logging.info(f"Mean f1 score: {f1score}")
