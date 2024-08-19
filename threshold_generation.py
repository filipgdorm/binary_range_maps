import pandas as pd
import numpy as np
import json
import os
import sys
import h3pandas
import torch
import argparse
from tqdm import tqdm
from threshold_methods import *
from args_parser import parse_args

sys.path.append("./sinr")
import datasets
import models
import utils

args = parse_args()

# Construct the absolute paths to the data files
gdfk_path = "./threshold_dataset/gdfk.csv"
train_df_h3_path = "./threshold_dataset/train_df_h3.csv"
iucn_json_path = "./sinr/data/eval/iucn/iucn_res_5.json"
snt_npy_path = "./sinr/data/eval/snt/snt_res_5.npy"

gdfk = pd.read_csv(gdfk_path, index_col=0)  #make sure if required resolution, default 4
#Load PA data if method requires it:
if args.method in ['lpt_x', 'rdm_sampling', 'tgt_sampling',]:
    train_df_h3 = pd.read_csv(train_df_h3_path, index_col=0)
    all_spatial_grid_counts = train_df_h3.index.value_counts()
    presence_absence = pd.DataFrame({
        "background": all_spatial_grid_counts,
    })
    presence_absence = presence_absence.fillna(0)

DEVICE = torch.device('cpu')
train_params = torch.load(args.model_path, map_location='cpu')
model = models.get_model(train_params['params'])
model.load_state_dict(train_params['state_dict'], strict=True)
model = model.to(DEVICE)
model.eval()

if train_params['params']['input_enc'] in ['env', 'sin_cos_env']:
    raster = datasets.load_env()
else:
    raster = None
enc = utils.CoordEncoder(train_params['params']['input_enc'], raster=raster)

obs_locs = np.array(gdfk[['lng', 'lat']].values, dtype=np.float32)

#load species:ids to generate thresholds for
if args.species_set == "iucn":
    with open(iucn_json_path, 'r') as f:
        data = json.load(f)
    species_ids = list((data['taxa_presence'].keys()))
    if args.method in ["mlp_classifier", "rf_classifier"]: obs_locs = np.array(data['locs'], dtype=np.float32)  #in this case use same resolution as test set
elif args.species_set == "snt":
    #load reference from snt
    data = np.load(snt_npy_path, allow_pickle=True)
    data = data.item()
    species_ids = data['taxa']
    if args.method in ["mlp_classifier", "rf_classifier"]: obs_locs = np.array(data['obs_locs'], dtype=np.float32) #in this case use same resolution as test set
elif args.species_set == "all":
    species_ids = train_params['params']['class_to_taxa']
elif args.species_set == "custom":
    species_ids = np.array(args.species_ids)

obs_locs = torch.from_numpy(obs_locs).to('cpu')
loc_feat = enc.encode(obs_locs)

classes_of_interest = torch.zeros(len(species_ids), dtype=torch.int64)
for tt_id, tt in enumerate(species_ids):
    try:
        class_of_interest = np.array([train_params['params']['class_to_taxa'].index(int(tt))])
    except ValueError:
        # Inform the user about the invalid species ID and exit
        raise ValueError(f"Invalid species ID {tt}: not found in the class_to_taxa list.")
    classes_of_interest[tt_id] = torch.from_numpy(class_of_interest)

with torch.no_grad():
    loc_emb = model(loc_feat, return_feats=True)
    wt = model.class_emb.weight[classes_of_interest, :]

#If thresholding method relies on classifier we generate the thresholds in one go. Otherwise species by species.
if args.method == "mlp_classifier":
    upper_b_dir = os.path.join("results/upper_bounds",args.model_path.split("/")[-2],args.species_set)
    taxa, thres = mlp_classifier(upper_b_dir, wt, species_ids)
    output_pd = pd.DataFrame({'taxon_id': taxa, 'thres': thres})
elif args.method == "rf_classifier":
    upper_b_dir = os.path.join("results/upper_bounds",args.model_path.split("/")[-2],args.species_set)
    taxa, thres = rf_classifier(upper_b_dir, wt, species_ids)
    output_pd = pd.DataFrame({'taxon_id': taxa, 'thres': thres})
else:
    output = []
    for i, class_id in tqdm(enumerate(classes_of_interest), total=len(classes_of_interest)):
        wt_1 = wt[i,:]
        preds = torch.sigmoid(torch.matmul(loc_emb, wt_1)).cpu().numpy()

        if args.method == "lpt_x":
            thres = lpt_x(gdfk ,train_df_h3, presence_absence, class_id, preds, args.lpt_level)
        elif args.method == "tgt_sampling":
            thres = tgt_sampling(gdfk ,train_df_h3, presence_absence, class_id, preds)
        elif args.method == "rdm_sampling":
            thres = rdm_sampling(gdfk ,train_df_h3, presence_absence, class_id, preds, args.raw_number, args.factor_presences)
        elif args.method == "single_fixed_thres":
            thres=args.threshold
        elif args.method == "mean_pred_thres":
            thres = preds.mean()
        row = {
            "taxon_id": species_ids[i],
            "thres": thres,
        }
        row_dict = dict(row)
        output.append(row_dict)
    output_pd = pd.DataFrame(output)
        
# Construct the output directory and file path
output_dir = os.path.join("results/", args.exp_name)
os.makedirs(output_dir, exist_ok=True)
output_file_path = os.path.join(output_dir, "thresholds.csv")

# Save the DataFrame to the CSV file
output_pd.to_csv(output_file_path)