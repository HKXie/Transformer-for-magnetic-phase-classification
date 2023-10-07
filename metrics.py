import numpy as np
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score, f1_score, recall_score

def Accuracy_sk(pred, true):
    return accuracy_score(true, pred)


def Recall_sk(pred, true):
    return recall_score(true, pred, average='micro')

def f1_score_sk(pred, true):
    return f1_score(true, pred, average='micro')