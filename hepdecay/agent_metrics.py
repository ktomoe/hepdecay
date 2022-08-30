"""Collection of pre-defined Metric classes."""

import numpy as np
import torch.nn
import torch.nn as nn
from multiml.agent.metric import BaseMetric


class MyAUCMetric(BaseMetric):
    """A metric class to return ACC."""
    def __init__(self, **kwargs):
        """Initialize ACCMetric."""
        super().__init__(**kwargs)
        self._name = 'acc'
        self._type = 'max'

    def calculate(self):
        """Calculate AUC."""
        y_true, y_pred = self.get_true_pred_data()

        softmax = nn.Softmax(dim=1) 
        y_pred = softmax(torch.tensor(y_pred))

        y_pred = y_pred[:, 1]

        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        from sklearn.metrics import auc
        roc_auc = auc(fpr, tpr)

        return roc_auc
