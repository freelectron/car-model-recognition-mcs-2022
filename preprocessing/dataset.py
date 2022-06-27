import os
from dataclasses import dataclass

import torch
import torchvision as tv
from PIL import Image
import numpy as np
import pandas as pd

from stanford_cars_dataset.create_siamese_training_dataset import process_image


@dataclass
class Config:
    """
    Specs for training.
    """
    training_dir = "../comp_cars_dataset/dataset_siamese_classification_nn/images"
    # training_dir =  "../stanford_cars_dataset/dataset_siamese_nn/images"
    training_csv = "../comp_cars_dataset/dataset_siamese_classification_nn/df_training_classification.csv"
    # training_csv = "../stanford_cars_dataset/dataset_siamese_nn/df_training_siamese.csv"
    testing_dir = "../comp_cars_dataset/dataset_siamese_classification_nn/images"
    # testing_dir = "../stanford_cars_dataset/dataset_siamese_nn/images"
    testing_csv = "../comp_cars_dataset/dataset_siamese_classification_nn/df_testing_classification.csv"
    # testing_csv = "../stanford_cars_dataset/dataset_siamese_nn/df_testing_siamese.csv"
    train_batch_size = 8
    train_number_epochs = 10
    image_input_size = 224


NORMALISATION_1 = tv.transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)
TRAINING_TRANSFORMATION_SEQUENCE_1 = tv.transforms.Compose([
    tv.transforms.RandomResizedCrop(Config.image_input_size),
    tv.transforms.RandomHorizontalFlip(),
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


class SiameseNetworkDatasetValidation(SiameseNetworkDataset):
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


class SiameseNetworkDatasetTesting:
    """
    Preprocess images from ODS competition, make them usable for the Siamese network.
    """

    @staticmethod
    def preprocess_image_pair_df(image_pair_df: pd.DataFrame) -> pd.DataFrame:
        return image_pair_df[["img1", "img2"]]

    @staticmethod
    def preprocess_annotation_df(annotation_df: pd.DataFrame) -> pd.DataFrame:
        result_df = annotation_df.set_index("img_path")
        result_df["car_model"] = "Unknown"
        return result_df

    def __init__(
            self,
            image_pair_csv,
            annotation_csv,
            image_directory,
            transform,
    ):
        # Used to prepare the labels and images path
        self.image_pair_df = self.preprocess_image_pair_df(pd.read_csv(image_pair_csv))
        self.annotation_df = self.preprocess_annotation_df(pd.read_csv(annotation_csv))
        self.image_directory = image_directory
        self.transform = transform

    def __getitem__(self, index):

        image1_file_name = self.image_pair_df.iat[index, 0]
        image2_file_name = self.image_pair_df.iat[index, 1]

        # Loading and processing as for the training dataset
        image1 = process_image(
            image_file_name=image1_file_name,
            car_model_folder=self.image_directory,
            annotation_df=self.annotation_df,
            images_folder_path="",
            save_cropped_image=False
        )
        image2 = process_image(
            image_file_name=image2_file_name,
            car_model_folder=self.image_directory,
            annotation_df=self.annotation_df,
            images_folder_path="",
            save_cropped_image=False
        )

        # See what conversion was applied for training
        image1 = image1.convert("RGB")
        image2 = image2.convert("RGB")

        # Apply image transformations
        if self.transform is not None:
            image1 = self.transform(image1)
            image2 = self.transform(image2)

        return image1, image2

    def __len__(self):
        return len(self.image_pair_df)


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


