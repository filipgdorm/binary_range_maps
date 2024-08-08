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

sys.path.append('../')
import datasets
import models
import utils
from args_parser import parse_args


args = parse_args()

gdfk = pd.read_csv("../pa_data/gdfk.csv", index_col=0)  #make sure if required resolution, default 4
#Load PA data if method requires it:
if args.method in ['lpt_x', 'rdm_sampling', 'tgt_sampling',]:
    train_df_h3 = pd.read_csv("../pa_data/train_df_h3.csv", index_col=0)
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
obs_locs = torch.from_numpy(obs_locs).to('cpu')
loc_feat = enc.encode(obs_locs)

#load species:ids to generate thresholds for
if args.species_set == "iucn":
    with open(os.path.join('../data/eval/iucn/', 'iucn_res_5.json'), 'r') as f:
                data = json.load(f)
    species_ids = list((data['taxa_presence'].keys()))
elif args.species_set == "snt":
    #load reference from snt
    data = np.load(os.path.join('../data/eval/snt/', 'snt_res_5.npy'), allow_pickle=True)
    data = data.item()
    species_ids = data['taxa']
elif args.species_set == "custom":
     pass

classes_of_interest = torch.zeros(len(species_ids), dtype=torch.int64)
for tt_id, tt in enumerate(species_ids):
    class_of_interest = np.array([train_params['params']['class_to_taxa'].index(int(tt))])
    classes_of_interest[tt_id] = torch.from_numpy(class_of_interest)

with torch.no_grad():
    loc_emb = model(loc_feat, return_feats=True)
    wt = model.class_emb.weight[classes_of_interest, :]

output = []
for i, class_id in tqdm(enumerate(classes_of_interest), total=len(classes_of_interest)):
    wt_1 = wt[i,:]
    preds = torch.sigmoid(torch.matmul(loc_emb, wt_1)).cpu().numpy()

    if args.method == "lpt_x":
        thres = lpt_x(gdfk ,train_df_h3, presence_absence, class_id, preds, args.lpt_level)
    elif args.method == "tgt_sampling":
        thres = tgt_sampling(gdfk ,train_df_h3, presence_absence, class_id, preds)
    elif args.method == "rdm_sampling":
        thres = rdm_sampling(gdfk ,train_df_h3, presence_absence, class_id, preds, num_absences)
    elif args.method == "single_fixed_thres":
         thres=0.5  #change single fixed thres to allow for arbitrary value.
    elif args.method == "mlp_classifier":
         pass
    elif args.method == "rf_classifier":
         pass
    
    row = {
        "taxon_id": species_ids[i],
        "thres": thres,
    }
    row_dict = dict(row)
    output.append(row_dict)

output_pd = pd.DataFrame(output)
    
output_pd.to_csv(args.result_dir+f"/thresholds_{args.counter}.csv")