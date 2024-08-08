
import pandas as pd
import numpy as np
import json
import os
import sys
from sklearn.metrics import f1_score
import torch
import argparse
from tqdm import tqdm
import logging
from sklearn.metrics import precision_recall_curve
from tqdm import tqdm

sys.path.append('../')
import datasets
import models
import utils

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

parser = argparse.ArgumentParser(description="Script to process thresholds and perform an experiment.")
parser.add_argument("--model_path", type=str, default='model_an_full_input_enc_sin_cos_hard_cap_num_per_class_1000.pt', help="Model path.")
parser.add_argument("--result_dir", type=str, default='test', help="Experiment name")
parser.add_argument("--counter", type=int, default='test', help="Experiment name")

args = parser.parse_args()

THRES_MODEL = "mlp"

print(args.counter, args.result_dir, args.model_path)

DEVICE = torch.device('cpu')

# Set up logging to file
log_file_path = args.result_dir + f"/results/log_{args.counter}.out"
logging.basicConfig(filename=log_file_path, filemode='a', level=logging.INFO,
                    format='%(levelname)s: %(message)s')
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger('').addHandler(console)

logging.info(f"Model used for experiment: {args.model_path}")

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

classes_of_interest = torch.zeros(len(species_ids), dtype=torch.int64)
taxa_ids = torch.zeros(len(species_ids), dtype=torch.int64)
for tt_id, tt in enumerate(species_ids):
    class_of_interest = np.array([train_params['params']['class_to_taxa'].index(int(tt))])
    classes_of_interest[tt_id] = torch.from_numpy(class_of_interest)
    taxa_ids[tt_id] = int(tt)

obs_locs = np.array(data['locs'], dtype=np.float32)
obs_locs = torch.from_numpy(obs_locs).to('cpu')
loc_feat = enc.encode(obs_locs)

with torch.no_grad():
    loc_emb = model(loc_feat, return_feats=True)
    wt = model.class_emb.weight[classes_of_interest, :]

model_type = args.result_dir.split("/")[-1]
y_path = f'./upper_bound/{model_type}/results/opt_thres_{args.counter}.npy'

X = wt.numpy()
y = np.load(y_path)

# Define the number of categories
num_categories = 20

# Calculate bin edges
bin_edges = np.linspace(0,1, num_categories + 1)

# Assign categories
y_cat = np.digitize(y, bin_edges) - 1
y = y_cat

np.random.seed(42)
num_samples = len(X)
random_indices = np.random.choice(num_samples, size=int(num_samples * 0.75), replace=False)

# Creating a boolean mask
mask = np.full(num_samples, False)
mask[random_indices] = True

# Split the dataset into training and testing sets based on the random indices
X_train_thres, X_test_thres = X[mask], X[~mask]
y_train_thres, y_test_thres = y[mask], y[~mask]

logging.info(f"Threshold classifier model used {THRES_MODEL}")

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_thres)
X_test_scaled = scaler.transform(X_test_thres)

# Create an MLP Classifier object
mlp_classifier = MLPClassifier(hidden_layer_sizes=(200,100), random_state=42)

# Train the MLP Classifier on the training data
mlp_classifier.fit(X_train_scaled, y_train_thres)

# Predict categories for the testing data
predictions = mlp_classifier.predict(X_test_scaled)

# Compute accuracy
accuracy = accuracy_score(y_test_thres, predictions)

logging.info(f"Accuracy of threshold model: {accuracy}")

class_to_thres = np.arange(0.025, 1, 0.05)

def f1_at_thresh(y_true, y_pred, thresh, type = 'binary'):
    y_thresh = y_pred > thresh
    return f1_score(y_true, y_thresh, average=type)

wt_subset = wt[~mask]
taxa_ids_subset = taxa_ids[~mask]

output = list()
for tt_id, taxa in tqdm(enumerate(taxa_ids_subset),total=len(taxa_ids_subset)):
    wt_1 = wt_subset[tt_id,:]
    preds = torch.sigmoid(torch.matmul(loc_emb, wt_1)).cpu().numpy()
    taxa = taxa.item()
    species_locs = data['taxa_presence'].get(str(taxa))
    y_test = np.zeros(preds.shape, int)
    y_test[species_locs] = 1

    thres = class_to_thres[predictions[tt_id]]

    f1 = f1_at_thresh(y_test, preds, thres)
    
    row = {
        "taxon_id": taxa,
        "thres": thres,
        "fscore": f1
    }
    row_dict = dict(row)
    output.append(row_dict)

output_pd = pd.DataFrame(output)
output_pd.to_csv(args.result_dir+f"/scores.csv")

mean_f1 = output_pd.fscore.mean()
logging.info(f"Mean threshold: {output_pd.thres.mean()}")
logging.info(f"Mean F1 score: {mean_f1}")
logging.info("")

# Append the mean F1 score to a CSV file
results_file = args.result_dir + '/mean_f1_scores.csv'
results_data = pd.DataFrame({'counter': [args.counter], 'mean_f1': [mean_f1], 'model_acc': [accuracy]})

if os.path.isfile(results_file):
    results_data.to_csv(results_file, mode='a', header=False, index=False)
else:
    results_data.to_csv(results_file, mode='w', header=True, index=False)
