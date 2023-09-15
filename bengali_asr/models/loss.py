
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
    
class GreedyCTCDecoder(torch.nn.Module):
    def __init__(self, labels, blank=0):
        super().__init__()
        self.labels = labels
        self.blank = blank
        
    def forward(self, emission: torch.Tensor):
        """Given a sequence emission over labels, get the best path
        Args:
          emission (Tensor): Logit tensors. Shape `[num_seq, num_label]`.

        Returns:
          List[str]: The resulting transcript
        """
        indices = torch.argmax(emission, dim=-1)  # [num_seq,]
        original = "".join([self.labels[i.item()] for i in indices])
        indices = torch.unique_consecutive(indices, dim=-1)
        indices = [i for i in indices.numpy() if i != self.blank]
        joined = "".join([self.labels[i] for i in indices])
        return joined,original
    
class CTCLossBatchFirst(torch.nn.Module):
    def __init__(self,blank=0,ignore_index=-1,zero_infinity=False):
        super().__init__()
        self.ign_index=ignore_index
        self.lossf = torch.nn.CTCLoss(blank=blank,zero_infinity=zero_infinity)
        
    def forward(self, input_features:torch.Tensor,labels):
        input_features = torch.nn.functional.log_softmax(input_features,dim=2)
        b,s,d = input_features.size()
        inplen = torch.tensor([s for _ in range(b)]).long().to(input_features.device)
        target_length = (labels!=self.ign_index).sum(1).long()
        return self.lossf(torch.transpose(input_features,0,1),labels,inplen,target_length)
