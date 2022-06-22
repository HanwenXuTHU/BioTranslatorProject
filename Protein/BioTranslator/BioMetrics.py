'''
This file contains functions used for evaluating our method
'''
import numpy as np
from sklearn.metrics import roc_curve, auc


def compute_roc(labels, preds):
    # Compute ROC curve and ROC area for each class
    fpr, tpr, _ = roc_curve(labels.flatten(), preds.flatten())
    roc_auc = auc(fpr, tpr)
    return roc_auc


def auroc_metrics(labels, preds):
    roc_matrix = np.zeros(np.size(preds, 1))
    for i in range(len(roc_matrix)):
        pred_i, label_i = preds[:, i], labels[:, i]
        auroc = compute_roc(label_i, pred_i)
        roc_matrix[i] = auroc

    auroc_percentage = {0.65: 0, 0.7: 0, 0.75: 0, 0.8: 0, 0.85: 0, 0.95: 0}
    for j in range(np.size(preds, 1)):
        preds_id = preds[:, j].reshape([-1, 1])
        label_id = labels[:, j].reshape([-1, 1])
        roc_auc_j = compute_roc(label_id, preds_id)
        for T in auroc_percentage.keys():
            if T <= roc_auc_j:
                auroc_percentage[T] += 100.0 / np.size(preds, 1)
    return np.mean(roc_matrix), auroc_percentage