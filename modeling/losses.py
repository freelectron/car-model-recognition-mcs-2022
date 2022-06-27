import torch.nn as nn
import torch
from torch.nn import PairwiseDistance

from preprocessing.dataset import Config

PAIRWISEDISTANCE_P2 = PairwiseDistance(p=2.0)


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(
        self, margin: float = 1.0, euclidean_distance: torch.nn = PAIRWISEDISTANCE_P2
    ):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.distance_metric = euclidean_distance

    def forward(self, output1, output2, label):
        euclidean_distance = self.distance_metric(output1, output2)
        zero_vec = torch.tensor([0] * len(euclidean_distance)).cuda()
        # From keras docs
        squared_distance = torch.square(euclidean_distance)
        squared_margin = torch.square(
            torch.maximum(self.margin - euclidean_distance, zero_vec)
        )
        loss_contrastive = torch.mean(
            label * squared_distance + (1 - label) * squared_margin
        )

        return loss_contrastive
