import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

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

def rdm_sampling(gdfk ,train_df_h3, presence_absence, class_id, preds, num_absences):        
    gdfk["pred"] = preds
    target_spatial_grid_counts = train_df_h3[train_df_h3.label==class_id.item()].index.value_counts()
    presence_absence["forground"] = target_spatial_grid_counts
    presence_absence["predictions"] = gdfk["pred"]
    presence_absence.forground = presence_absence.forground.fillna(0)
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

def mlp_classifier():
    pass

def rf_classifier():
    pass