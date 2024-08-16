import pandas as pd
import numpy as np
import json
import os
import sys
import torch
import argparse
from tqdm import tqdm
from eval_functions import *

sys.path.append("./sinr")
import datasets
import models
import utils
import setup
import logging

parser = argparse.ArgumentParser(description="Script to process thresholds and perform an experiment.")
parser.add_argument("--model_path", type=str, required=True, help="Model path.")
parser.add_argument("--exp_name", type=str, default='test', help="Experiment name, also the dir where thresholds will be collected from.")
parser.add_argument(
        '--evaluation_set',
        type=str,
        choices=['iucn', 'snt'],
        default='iucn',
        help="Choose the species set to evaluate against."
    )
parser.add_argument(
        '--eval_type',
        type=str,
        default='thresholds',
        choices=['thresholds','upper_bound', 'single_best_thres', 'subsample_expert_data'],
        help="Choose to use the evaluation set to create one of the baselines."
    )
parser.add_argument(
    '--subsample_test_size',
    type=float,
    help="Subsample size for the expert data. Required if --eval_type is subsample_expert_data."
)
args = parser.parse_args()

# Validate conditional requirement
if args.eval_type == 'subsample_expert_data' and args.subsample_test_size is None:
    parser.error("--subsample_test_size is required when --eval_type is 'subsample_expert_data'.")

iucn_json_path = "./sinr/data/eval/iucn/iucn_res_5.json"
snt_npy_path = "./sinr/data/eval/snt/snt_res_5.npy"

if args.eval_type == "thresholds":
    threshs = pd.read_csv("results/" + args.exp_name + "/thresholds.csv")
    log_file_path = "results/" + args.exp_name + "/log.out"
elif args.eval_type == "upper_bound":
    model_name = args.model_path.split("/")[-2]
    output_dir = os.path.join("results/upper_bounds/", model_name, args.evaluation_set)
    os.makedirs(output_dir, exist_ok=True)
    log_file_path = output_dir + "/log.out"
elif args.eval_type == "single_best_thres":
    output_dir = "results/single_best_thres"
    os.makedirs(output_dir, exist_ok=True)
    log_file_path = output_dir + "/log.out"

# Set up logging to file
logging.basicConfig(filename=log_file_path, filemode='a', level=logging.INFO,
                    format='%(levelname)s: %(message)s')
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger('').addHandler(console)

logging.info(f"Model used for experiment: {args.model_path}")

DEVICE = torch.device('cpu')
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

if args.eval_type == "thresholds": taxon_ids = threshs.taxon_id
else: taxon_ids = species_ids

if train_params['params']['input_enc'] in ['env', 'sin_cos_env']:
    raster = datasets.load_env()
else:
    raster = None
enc = utils.CoordEncoder(train_params['params']['input_enc'], raster=raster)

if args.evaluation_set == "iucn": obs_locs = np.array(data['locs'], dtype=np.float32)
elif args.evaluation_set == "snt": obs_locs = np.array(data['obs_locs'], dtype=np.float32)
obs_locs = torch.from_numpy(obs_locs).to('cpu')
loc_feat = enc.encode(obs_locs)

classes_of_interest = torch.zeros(len(taxon_ids), dtype=torch.int64)
for tt_id, tt in enumerate(taxon_ids):
    class_of_interest = np.array([train_params['params']['class_to_taxa'].index(int(tt))])
    classes_of_interest[tt_id] = torch.from_numpy(class_of_interest)

with torch.no_grad():
    loc_emb = model(loc_feat, return_feats=True)
    wt = model.class_emb.weight[classes_of_interest, :]

if args.eval_type == "single_best_thres":
    num_points = 19
    # Generate linearly spaced numbers in the range [0, 1]
    linspace_values = np.linspace(0.05, 1, num=num_points, endpoint=False)
    lin_threshs = linspace_values
    per_species_f1 = np.zeros((len(species_ids),len(lin_threshs)))
else:
    per_species_f1 = np.zeros((len(taxon_ids)))
    per_species_thres = np.zeros((len(taxon_ids)))

for tt_id, taxa in tqdm(enumerate(taxon_ids), total=len(taxon_ids)):
    wt_1 = wt[tt_id,:]
    preds = torch.sigmoid(torch.matmul(loc_emb, wt_1)).cpu().numpy()

    if args.evaluation_set == "iucn":
        species_locs = data['taxa_presence'].get(taxa)
        y_test = np.zeros(preds.shape, int)
        y_test[species_locs] = 1
        pred = preds

    elif args.evaluation_set == "snt":
        cur_loc_indices = np.array(loc_indices_per_species[tt_id])
        y_test = np.array(labels_per_species[tt_id])
        pred = preds[cur_loc_indices]

    if args.eval_type == "thresholds":
        thresh = threshs['thres'][tt_id]
        per_species_f1[tt_id] = f1_at_thresh(y_test, pred, thresh, type='binary')
        per_species_thres[tt_id] = thresh
    elif args.eval_type == "upper_bound":
        per_species_thres[tt_id], per_species_f1[tt_id] = upper_bound_f1(y_test, preds)
    elif args.eval_type == "single_best_thres":
         for i, lin_thresh in enumerate(lin_threshs):
            per_species_f1[tt_id][i] = f1_at_thresh(y_test, preds, lin_thresh, type='binary')

#Save results differently depending on method
if args.eval_type == "single_best_thres":
    f1score = per_species_f1.mean(axis=0).max()
    best_thres = linspace_values[per_species_f1.mean(axis=0).argmax()]
    logging.info(f"Best unified threshold: {best_thres}")
    logging.info(f"Mean f1 score: {f1score}")
else:
    mean_f1 = np.mean(per_species_f1)
    logging.info(f"Mean f1 score: {mean_f1}")   #output mean f1 score to log file
    if args.eval_type == "upper_bound":
        np.save(output_dir+"/f1_scores.npy", per_species_f1)
        np.save(output_dir+"/opt_thres.npy", per_species_thres)
    elif args.eval_type == "thresholds":
        np.save(f'results/{args.exp_name}/f1_scores.npy', per_species_f1)   #save scores
