import os
from dataclasses import dataclass

import torch
import torchvision as tv
from PIL import Image
import numpy as np
import pandas as pd


@dataclass
class Config:
    """
    Specs for training.
    """
    training_dir = "../stanford_cars_dataset/dataset_siamese_nn/training_data/images"
    training_csv = "../stanford_cars_dataset/dataset_siamese_nn/training_data/df_training_siamese.csv"
    testing_dir = "../stanford_cars_dataset/dataset_siamese_nn/testing_data/images"
    testing_csv = "../stanford_cars_dataset/dataset_siamese_nn/testing_data/df_testing_siamese.csv"
    train_batch_size = 8
    train_number_epochs = 20
    image_input_size = 224


NORMALISATION_1 = tv.transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)
TRAINING_TRANSFORMATION_SEQUENCE_1 = tv.transforms.Compose([
    # TODO: try with cropping
    # tv.transforms.RandomResizedCrop(Config.image_input_size),
    tv.transforms.Resize((Config.image_input_size, Config.image_input_size)),
    tv.transforms.ToTensor(),
    NORMALISATION_1
])

TESTING_TRANSFORMATION_SEQUENCE_1 = tv.transforms.Compose([
    tv.transforms.Resize((Config.image_input_size, Config.image_input_size)),
    tv.transforms.CenterCrop(Config.image_input_size),
    tv.transforms.ToTensor(),
    NORMALISATION_1
])


class SiameseNetworkDataset:
    """
    Define how to load and transform images.
    Two images-pair as one training example.
    """

    def __init__(self, image_pair_label_csv=None, image_directory=None, transform=None):
        # used to prepare the labels and images path
        self.image_pair_label_csv = pd.read_csv(image_pair_label_csv)[["image1", "image2", "label"]]
        self.image_directory = image_directory
        self.transform = transform

    def __getitem__(self, index):
        # getting the image path
        image1_path = os.path.join(self.image_directory, self.image_pair_label_csv.iat[index, 0])
        image2_path = os.path.join(self.image_directory, self.image_pair_label_csv.iat[index, 1])

        # Loading the image
        img0 = Image.open(image1_path)
        img1 = Image.open(image2_path)
        # "L" means gray scale
        # TODO: try gray scale
        img0 = img0.convert("RGB")
        img1 = img1.convert("RGB")

        # Apply image transformations
        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)

        return img0, img1, torch.from_numpy(np.array([int(self.image_pair_label_csv.iat[index, 2])], dtype=np.float32))

    def __len__(self):
        return len(self.image_pair_label_csv)


class SiameseNetworkDatasetTesting(SiameseNetworkDataset):
    """
    Define how to load and transform images.
    Two images-pair as one training example.
    """

    def __init__(self, image_pair_label_csv=None, image_directory=None, transform=None):
        super().__init__(
            image_pair_label_csv=image_pair_label_csv,
            image_directory=image_directory,
            transform=transform
        )

    def __getitem__(self, index):
        # getting the image path
        image1_path = os.path.join(self.image_directory, self.image_pair_label_csv.iat[index, 0])
        image2_path = os.path.join(self.image_directory, self.image_pair_label_csv.iat[index, 1])

        # Loading the image
        img0 = Image.open(image1_path)
        img1 = Image.open(image2_path)
        # "L" means gray scale
        # TODO: try gray scale
        img0 = img0.convert("RGB")
        img1 = img1.convert("RGB")

        # Apply image transformations
        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)

        return img0,\
               img1,\
               torch.from_numpy(np.array([int(self.image_pair_label_csv.iat[index, 2])], dtype=np.float32)),\
               image1_path,\
               image2_path


# FIXME: delete testing code
if __name__ == "__main__":
    import torchvision as tv
    from torch.utils.data import DataLoader

    # Added siamese_network to PYTHON PATH
    from visualisation_utils import imshow

    training_csv = "../stanford_cars_dataset/dataset_siamese_nn/training_data/df_training_siamese.csv"
    training_dir = "../stanford_cars_dataset/dataset_siamese_nn/training_data/images"

    siamese_dataset = SiameseNetworkDataset(training_csv,
                                            training_dir,
                                            transform=TRAINING_TRANSFORMATION_SEQUENCE_1
                                            )
    vis_dataloader = DataLoader(siamese_dataset,
                                shuffle=True,
                                batch_size=8)

    dataiter = iter(vis_dataloader)

    example_batch = next(dataiter)
    concatenated = torch.cat((example_batch[0], example_batch[1]), 0)
    # print(example_batch)
    imshow(tv.utils.make_grid(concatenated))
    print(example_batch[2].numpy())


