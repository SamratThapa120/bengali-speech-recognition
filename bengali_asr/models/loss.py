
import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskedCrossEntropyLoss(nn.Module):
    def __init__(self,ignore_index=-1):
        super(MaskedCrossEntropyLoss, self).__init__()
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')  # We will handle reduction ourselves
        self.ignore_index = ignore_index
    def forward(self, predictions, ground_truth):
        
        # Compute the raw cross-entropy loss values
        _,_,cls = predictions.shape
        predictions = predictions.view(-1,cls)
        ground_truth = ground_truth.view(-1)
        mask = (ground_truth != self.ignore_index)

        loss = self.cross_entropy(predictions[mask], ground_truth[mask])
        
        return loss
