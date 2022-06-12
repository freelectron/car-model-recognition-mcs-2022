import torch.nn as nn
import torch
from torch.nn import PairwiseDistance

PAIRWISEDISTANCE_P2 = PairwiseDistance(p=2.0)


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin: float = 2.0, euclidean_distance: torch.nn = PAIRWISEDISTANCE_P2):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.distance_metric = euclidean_distance

    def forward(self, output1, output2, label):
        euclidean_distance = self.distance_metric(output1, output2)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        return loss_contrastive