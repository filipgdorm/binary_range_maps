import pandas as pd
import numpy as np
import json
import os
import sys
import h3pandas
import torch
import argparse
from tqdm import tqdm

sys.path.append('../')
import datasets
import models
import utils

parser = argparse.ArgumentParser(description="Script to process thresholds and perform an experiment.")
parser.add_argument("--model_path", type=str, help="Model path.")
parser.add_argument("--result_dir", type=str, default='test', help="Output directory.")
parser.add_argument("--counter", type=int, default=0, help="Counter, for when using several iterations.")
parser.add_argument("--lpt_level", type=float, default=5.0, help="Robustness level for LPT-R, if set to 0 same as LPT.")
args = parser.parse_args()

#load PA dataset
train_df_h3 = pd.read_csv("../pa_data/train_df_h3.csv", index_col=0)
gdfk = pd.read_csv("../pa_data/gdfk.csv", index_col=0)
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

#load reference from iucn
with open(os.path.join('../data/eval/iucn/', 'iucn_res_5.json'), 'r') as f:
            data = json.load(f)
species_ids = list((data['taxa_presence'].keys()))

obs_locs = np.array(gdfk[['lng', 'lat']].values, dtype=np.float32)
obs_locs = torch.from_numpy(obs_locs).to('cpu')
loc_feat = enc.encode(obs_locs)

classes_of_interest = torch.zeros(len(species_ids), dtype=torch.int64)
for tt_id, tt in enumerate(species_ids):
    class_of_interest = np.array([train_params['params']['class_to_taxa'].index(int(tt))])
    classes_of_interest[tt_id] = torch.from_numpy(class_of_interest)

with torch.no_grad():
    loc_emb = model(loc_feat, return_feats=True)
    wt = model.class_emb.weight[classes_of_interest, :]

output = []
for class_index, class_id in tqdm(enumerate(classes_of_interest), total=len(classes_of_interest)):
    wt_1 = wt[class_index,:]
    preds = torch.sigmoid(torch.matmul(loc_emb, wt_1)).cpu().numpy()
    gdfk["pred"] = preds

    target_spatial_grid_counts = train_df_h3[train_df_h3.label==class_id.item()].index.value_counts()
         
    presence_absence["forground"] = target_spatial_grid_counts
    presence_absence["predictions"] = gdfk["pred"]
    presence_absence.forground = presence_absence.forground.fillna(0)

    presences = presence_absence[(presence_absence["forground"]>0)]["predictions"]

    thres = np.percentile(presences, args.lpt_level)
    
    row = {
        "taxon_id": species_ids[class_index],
        "thres": thres,
    }
    row_dict = dict(row)
    output.append(row_dict)

output_pd = pd.DataFrame(output)
    
output_pd.to_csv(args.result_dir+f"/thresholds_{args.counter}.csv")
