"""
Implemented from https://github.com/mathiaszinnen/focal_loss_torch
"""

from torch import nn
import torch
from torch.nn import functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, num_classes=2, reduction="mean"):
        """
        focal_loss
        """
        super(FocalLoss,self).__init__()
        self.reduction = reduction
        if isinstance(alpha, list):
            assert len(alpha)==num_classes   
            print(" --- Focal_loss alpha = {} --- ".format(alpha))
            self.alpha = torch.Tensor(alpha)
        else:
            assert alpha<1  
            print(" --- Focal_loss alpha = {}  --- ".format(alpha))
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] = alpha
            for n in range(1, num_classes):
                self.alpha[n] = 1-alpha

        self.gamma = gamma

    def forward(self, preds, labels):

        preds = preds.view(-1, preds.size(-1))
        self.alpha = self.alpha.to(preds.device)
        preds_logsoft = F.log_softmax(preds, dim=1) # log_softmax
        preds_softmax = torch.exp(preds_logsoft)    # softmax

        # gather : get prob from certain label-column
        preds_softmax = preds_softmax.gather(1, labels.view(-1,1))
        preds_logsoft = preds_logsoft.gather(1, labels.view(-1,1))

        alpha = self.alpha.gather(0, labels.view(-1))


        loss = -torch.mul(torch.pow((1-preds_softmax), self.gamma), preds_logsoft)

        loss = torch.mul(alpha, loss.t())
        if self.reduction =="mean":
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss

_criterion = nn.BCEWithLogitsLoss()
class BCEWithLogitsLossWrapper(torch.nn.Module):
    def forward(self, preds, labels):
        return _criterion(preds, labels.float())
