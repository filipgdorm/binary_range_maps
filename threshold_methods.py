import numpy as np
import pandas as pd
import os
from sklearn.metrics import precision_recall_curve
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

def lpt_x (gdfk ,train_df_h3, presence_absence, class_id, preds, lpt_level):
    gdfk["pred"] = preds
    target_spatial_grid_counts = train_df_h3[train_df_h3.label==class_id.item()].index.value_counts()  
    presence_absence["forground"] = target_spatial_grid_counts
    presence_absence["predictions"] = gdfk["pred"]
    presence_absence.forground = presence_absence.forground.fillna(0)
    presences = presence_absence[(presence_absence["forground"]>0)]["predictions"]
    thres = np.percentile(presences, lpt_level)
    return thres

def tgt_sampling(gdfk ,train_df_h3, presence_absence, class_id, preds):        
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
    return thres

def rdm_sampling(gdfk ,train_df_h3, presence_absence, class_id, preds, num_absences, raw_number, factor_presences):        
    gdfk["pred"] = preds
    target_spatial_grid_counts = train_df_h3[train_df_h3.label==class_id.item()].index.value_counts()
    presence_absence["forground"] = target_spatial_grid_counts
    presence_absence["predictions"] = gdfk["pred"]
    presence_absence.forground = presence_absence.forground.fillna(0)
    if raw_number is not None:
        num_absences = raw_number
    elif factor_presences is not None:  
        num_absences = int(factor_presences * len(presences))
    absences = presence_absence[(presence_absence["forground"]==0)].sample(n=num_absences, random_state=42)["predictions"]
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
    return thres

def mlp_classifier(upper_b_dir, wt, species_ids):
    y_path = upper_b_dir + "/opt_thres.npy"
    if not os.path.exists(y_path):
        print(f"Error: {y_path} not found. Please run the script upper_bound.py for this model that generates 'opt_thres.npy' first.")
        raise SystemExit("Execution stopped due to missing required file.")
    
    X = wt.numpy()
    y = np.load(y_path)
    # Define the number of categories
    num_categories = 20
    # Calculate bin edges
    bin_edges = np.linspace(0,1, num_categories + 1)
    # Assign categories
    y = np.digitize(y, bin_edges) - 1
    np.random.seed(42)
    num_samples = len(X)
    random_indices = np.random.choice(num_samples, size=int(num_samples * 0.75), replace=False)
    # Creating a boolean mask
    mask = np.full(num_samples, False)
    mask[random_indices] = True
    # Split the dataset into training and testing sets based on the random indices
    X_train_thres, X_test_thres = X[mask], X[~mask]
    y_train_thres, y_test_thres = y[mask], y[~mask]
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
    class_to_thres = np.arange(0.025, 1, 0.05)
    thresolds = class_to_thres[predictions]
    species_ids_subset = np.array(species_ids)[~mask]

    return species_ids_subset, thresolds

def rf_classifier(upper_b_dir, wt, species_ids):
    y_path = upper_b_dir + "/opt_thres.npy"
    # Check if y_path exists
    if not os.path.exists(y_path):
        print(f"Error: {y_path} not found. Please run the script upper_bound.py for this model that generates 'opt_thres.npy' first.")
        raise SystemExit("Execution stopped due to missing required file.")
    
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
    # Create a Random Forest Classifier object
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    # Train the Random Forest Classifier on the training data
    rf_classifier.fit(X_train_thres, y_train_thres)
    # Predict categories for the testing data
    predictions = rf_classifier.predict(X_test_thres)
    class_to_thres = np.arange(0.025, 1, 0.05)
    thresolds = class_to_thres[predictions]
    species_ids_subset = np.array(species_ids)[~mask]

    return species_ids_subset, thresolds