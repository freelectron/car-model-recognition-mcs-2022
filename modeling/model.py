import logging

from torchvision import models
import torch
import torch.nn as nn


logging.getLogger().setLevel(logging.INFO)


class SiameseNetwork(nn.Module):
    """
    Pretrained model from
      https://debuggercafe.com/stanford-cars-classification-using-efficientnet-pytorch/ .
    """
    def __init__(self, load_model_from_state_dictionary: str = None):
        super(SiameseNetwork, self).__init__()
        # First load standard pretrained model
        self.cnn1 = models.efficientnet_b1(pretrained=False)
        logging.info("Loading pretrained parameters from EfficientNet B1.")

        # Downloaded Pretrained model was outputting for 196 classes
        # self.cnn1.classifier[1] = nn.Linear(in_features=1280, out_features=196)
        # self.cnn1.load_state_dict(
        #     torch.load('./modeling/pretrained_stanford_cars_model_params.pth')['model_state_dict']
        # )
        # logging.info(
        #     "Loading pretrained parameters from EfficientNet B1 trained on Stanford dataset for classification."
        # )

        # Replace classification layer Identity (from 1280 to 1280) to output embeddings
        self.cnn1.classifier[1] = nn.Identity()

        # Make the convolutions stage tunable
        for params in self.cnn1.parameters():
            params.requires_grad = True

        # Defining the fully connected layers
        self.fc1 = nn.Sequential(
            nn.Linear(1280, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 128),
        )

        if load_model_from_state_dictionary:
            self.load_state_dict(torch.load(load_model_from_state_dictionary))

    def forward_once(self, x):
        # Forward pass
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        # Forward pass of input 1
        output1 = self.forward_once(input1)
        # Forward pass of input 2
        output2 = self.forward_once(input2)
        return output1, output2


class ClassificationNetwork(nn.Module):
    """
    Pretrained model from
      https://debuggercafe.com/stanford-cars-classification-using-efficientnet-pytorch/ .
    """
    def __init__(self, load_model_from_state_dictionary: str = None):
        super(ClassificationNetwork, self).__init__()
        # First load standard pretrained model
        self.cnn1 = models.efficientnet_b1(pretrained=True)
        logging.info("Loading pretrained parameters from EfficientNet B1.")

        # Downloaded Pretrained model was outputting for 196 classes
        # self.cnn1.classifier[1] = nn.Linear(in_features=1280, out_features=196)
        # self.cnn1.load_state_dict(
        #     torch.load('./modeling/pretrained_stanford_cars_model_params.pth')['model_state_dict']
        # )
        # logging.info(
        #     "Loading pretrained parameters from EfficientNet B1 trained on Stanford dataset for classification."
        # )

        # Replace classification layer Identity (from 1280 to 1280) to output embeddings
        self.cnn1.classifier[1] = nn.Identity()

        # Make the convolutions stage tunable
        for params in self.cnn1.parameters():
            params.requires_grad = True

        # Defining the fully connected layers
        self.fc1 = nn.Sequential(
            nn.Linear(1280, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, 1),
            nn.Sigmoid(),
        )

        if load_model_from_state_dictionary:
            self.load_state_dict(torch.load(load_model_from_state_dictionary))

    def forward_once(self, x):
        # Forward pass
        output = self.cnn1(x)
        return output

    def forward(self, input1, input2):
        # Forward pass of input 1
        output1 = self.forward_once(input1)
        # Forward pass of input 2
        output2 = self.forward_once(input2)
        diffs = output1 - output2
        out = self.fc1(diffs)

        return out


if __name__ == "__main__":
    from datetime import datetime
    nn_test = SiameseNetwork().cuda()
    with open(f"siamese_nn_layers_test_{datetime.now().isoformat()}.txt", "w") as f:
        f.write(str(dict(nn_test.named_modules())))

