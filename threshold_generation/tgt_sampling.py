import pandas as pd
import numpy as np
import json
import os
import sys
from sklearn.metrics import f1_score
import h3pandas
import torch
import h3
from sklearn.metrics import precision_recall_curve
import argparse
from tqdm import tqdm

sys.path.append('../')
import datasets
import models
import utils
import setup

parser = argparse.ArgumentParser(description="Script to process thresholds and perform an experiment.")
parser.add_argument("--model_path", type=str, default='model_an_full_input_enc_sin_cos_hard_cap_num_per_class_1000.pt', help="Model path.")
parser.add_argument("--result_dir", type=str, default='test', help="Experiment name")
parser.add_argument("--counter", type=int, default='test', help="Experiment name")

args = parser.parse_args()

print(args.counter, args.result_dir, args.model_path)

#load relevant data
train_df_h3 = pd.read_csv("../pseudo_absence_generation_data/train_df_h3.csv", index_col=0)
gdfk = pd.read_csv("../pseudo_absence_generation_data/gdfk_res3.csv", index_col=0)
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
    yield_cutoff = np.percentile((presence_absence["background"]/presence_absence["forground"])[presence_absence["forground"]>0], 95)
    absences = presence_absence[(presence_absence["forground"]==0) & (presence_absence["background"] > yield_cutoff)]["predictions"]
    presences = presence_absence[(presence_absence["forground"]>0)]["predictions"]
    df_x = pd.DataFrame({'predictions': presences, 'test': 1})
    df_y = pd.DataFrame({'predictions': absences, 'test': 0})
    for_thres = pd.concat([df_x, df_y], ignore_index=False)
    precision, recall, thresholds = precision_recall_curve(for_thres.test, for_thres.predictions)
    p1 = (2 * precision * recall)
    p2 = (precision + recall)
    out = np.zeros( (len(p1)) )
    fscore = np.divide(p1,p2, out=out, where=p2!=0)
    index = np.argmax(fscore)
    thres = thresholds[index]
    max_fscore = fscore[index]
    
    row = {
        "taxon_id": species_ids[class_index],
        "thres": thres,
    }
    row_dict = dict(row)
    output.append(row_dict)

output_pd = pd.DataFrame(output)
    
output_pd.to_csv(args.result_dir+f"/thresholds_{args.counter}.csv")