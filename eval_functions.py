from sklearn.metrics import f1_score
import numpy as np
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import train_test_split

def f1_at_thresh(y_true, y_pred, thresh, type = 'binary'):
    y_thresh = y_pred > thresh
    return f1_score(y_true, y_thresh, average=type)

def upper_bound_f1(y_test, preds):
    precision, recall, thresholds = precision_recall_curve(y_test, preds)
    p1 = (2 * precision * recall)
    p2 = (precision + recall)
    out = np.zeros( (len(p1)) )
    fscore = np.divide(p1,p2, out=out, where=p2!=0)
    index = np.argmax(fscore)
    thres = thresholds[index]
    max_fscore = fscore[index]
    return thres, max_fscore

def subsample_expert_data(y_test, preds, test_size):
    eval_preds, thresh_preds, eval_y_test, thresh_y_test = train_test_split(
            preds, y_test, test_size=test_size, random_state=42
        )

    #calculate thresholds 
    precision, recall, thresholds = precision_recall_curve(thresh_y_test, thresh_preds)
    p1 = (2 * precision * recall)
    p2 = (precision + recall)
    out = np.zeros( (len(p1)) )
    fscore = np.divide(p1,p2, out=out, where=p2!=0)
    index = np.argmax(fscore)
    thres = thresholds[index]

    #evaluate performance
    return f1_at_thresh(eval_y_test, eval_preds, thres)