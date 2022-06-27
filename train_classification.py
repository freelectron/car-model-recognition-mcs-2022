"""
Train and validate a classification NN that can be used for the recognition task (by using embeddings).
"""

import logging
from datetime import datetime
import os
from typing import List
import pickle

import pandas as pd
from pandas import DataFrame
from tqdm import tqdm
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import roc_auc_score

import torch
from torch import optim
from torch.optim.lr_scheduler import StepLR

from preprocessing.dataset import (
    Config,
    SiameseNetworkDataset,
    TRAINING_TRANSFORMATION_SEQUENCE_1,
    TESTING_TRANSFORMATION_SEQUENCE_1,
    SiameseNetworkDatasetValidation,
)
from modeling.model import ClassificationNetwork

START_OF_TRAIN = datetime.now().isoformat()[:16]
MODEL_CHECKPOINT_STATE_DICTS_FOLDER = (
    f"./modeling/model_store/classification_cc_v0_{START_OF_TRAIN}"
)
if not os.path.isdir(MODEL_CHECKPOINT_STATE_DICTS_FOLDER):
    os.makedirs(MODEL_CHECKPOINT_STATE_DICTS_FOLDER)

logging.getLogger().setLevel(logging.INFO)
# Create file handler which logs even debug messages
fh = logging.FileHandler(
    os.path.join(MODEL_CHECKPOINT_STATE_DICTS_FOLDER, f"train_{START_OF_TRAIN}.log")
)
fh.setLevel(logging.DEBUG)
logging.getLogger().addHandler(fh)


def train(
    model: torch.nn.Module,
    dataloader_instance: ClassificationNetwork,
    epoch: int,
    log_after_n_iterations: int = 50,
) -> (torch.nn.Module, List[List[float]], int):
    """
    Runs the main train loop.
    """
    # Set to train mode
    model.train()
    loss_history = list()
    true_labels = list()
    sigmoid_output_vectors = list()
    total_number_of_iterations = 0

    # Hold average loss over several pairs
    epoch_loss_values = list()
    for i, data in enumerate(tqdm(dataloader_instance)):
        img0, img1, label = data
        img0, img1, label = img0.cuda(), img1.cuda(), label.cuda()
        optimizer.zero_grad()
        sigmoid_output_difference_vector = model(img0, img1)
        loss_bce = criterion(sigmoid_output_difference_vector, label)
        loss_bce.backward()
        # Apply gradient clipping to prevent nan loss after n epochs ?
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0, norm_type=2)
        optimizer.step()
        # Save to calculate metrics on
        sigmoid_output_vectors += (
            sigmoid_output_difference_vector.squeeze().detach().cuda().tolist()
        )
        true_labels += label.squeeze().int().detach().cpu().tolist()
        epoch_loss_values.append(loss_bce.detach().cpu().numpy())

        if i % log_after_n_iterations == 0:
            loses_mean = np.array(epoch_loss_values).mean()
            logging.info(
                "Epoch number {}| Iteration {}\nCurrent epoch's average train loss is {}\n".format(
                    epoch, i, loses_mean
                )
            )
            total_number_of_iterations += log_after_n_iterations
            loss_history.append(epoch_loss_values)

    # Calculate eval metrics
    roc_score = roc_auc_score(true_labels, sigmoid_output_vectors)
    logging.info(
        "{} epoch finished. Train data ROC-AUC score based on euclidean similarity is {}".format(
            epoch, roc_score
        )
    )

    return model, loss_history


def validate(
    model: torch.nn.Module,
    dataloader_instance: SiameseNetworkDatasetValidation,
    epoch: int,
    device: str = "cuda",
) -> tuple[DataFrame, float]:
    true_label_values = list()
    sigmoid_outputs_difference_vector = list()
    image1_paths = list()
    image2_paths = list()

    with torch.no_grad():
        # Since batch size is one
        model.cuda()
        model.eval()
        for i, data_image_pair in enumerate(dataloader_instance, 0):
            # Run model inference
            x0, x1, label, image1_path, image2_path = data_image_pair
            sigmoid_output_difference_vector = model(x0.to(device), x1.to(device))

            sigmoid_outputs_difference_vector += (
                sigmoid_output_difference_vector.detach().cuda().tolist()
            )

            # Calculate Euclidean distances
            true_label_values += label.squeeze().int().tolist()

            # Save images for later inspection
            image1_paths += list(image1_path)
            image2_paths += list(image2_path)

    # Calculate ROC-AUC Score
    roc_auc_score_epoch = roc_auc_score(
        true_label_values, sigmoid_outputs_difference_vector
    )
    logging.info(
        "{} epoch finished. Test data ROC-AUC score based on euclidean similarity is {}".format(
            epoch, roc_auc_score_epoch
        )
    )

    df_epoch_validation = pd.DataFrame(
        {
            "true_label_values": true_label_values,
            "sigmoid_outputs_difference_vector": sigmoid_outputs_difference_vector,
            "image1_paths": image1_paths,
            "image2_paths": image2_paths,
        }
    )

    return df_epoch_validation, roc_auc_score_epoch


if __name__ == "__main__":
    training_dataset = SiameseNetworkDataset(
        Config.training_csv,
        Config.training_dir,
        transform=TRAINING_TRANSFORMATION_SEQUENCE_1,
    )
    # Load the test dataset
    test_dataset = SiameseNetworkDatasetValidation(
        image_pair_label_csv=Config.testing_csv,
        image_directory=Config.testing_dir,
        transform=TESTING_TRANSFORMATION_SEQUENCE_1,
    )

    # Load the dataset as pytorch tensors using dataloader
    train_dataloader = DataLoader(
        training_dataset,
        shuffle=True,
        num_workers=12,
        batch_size=Config.train_batch_size,
    )
    test_dataloader = DataLoader(
        test_dataset,
        num_workers=12,
        batch_size=Config.train_batch_size,
        # No shuffling for easier comparison
        shuffle=False,
    )

    # Declare Siamese Network
    load_model_folder = "./modeling/model_store/classification_cc_v0_2022-06-24T08:03/"
    load_model_from_state_dictionary = os.path.join(
        load_model_folder, "model_0_epoch.pt"
    )
    net = ClassificationNetwork(load_model_from_state_dictionary).cuda()
    # Declare Loss Function
    criterion = torch.nn.BCELoss()
    # Declare Optimizer
    optimizer = optim.SGD(net.parameters(), lr=0.00001, momentum=0.9)
    # Set learning rate schedule for the optimizer
    scheduler = StepLR(optimizer, step_size=Config.train_number_epochs, gamma=0.1)

    for epoch in range(0, Config.train_number_epochs):
        logging.info(f"\nEPOCH: {epoch}")
        # Train on both datasets
        model_epoch, loss_history_epoch = train(
            model=net, dataloader_instance=train_dataloader, epoch=epoch
        )

        torch.save(
            net.state_dict(),
            os.path.join(
                MODEL_CHECKPOINT_STATE_DICTS_FOLDER, f"model_{epoch}_epoch.pt"
            ),
        )
        pickle.dump(
            loss_history_epoch,
            open(
                os.path.join(
                    MODEL_CHECKPOINT_STATE_DICTS_FOLDER, f"loss_history_epoch_{epoch}"
                ),
                "wb",
            ),
        )

        # # Only use when testing validation
        # output_classifier = pickle.load(
        #     open(os.path.join(
        #         load_model_folder,
        #         f"output_classifier_epoch_{epoch}"), "rb")
        # )

        # Validate
        df_epoch_validation, roc_auc_score_epoch = validate(
            model=net,
            dataloader_instance=test_dataloader,
            epoch=epoch,
        )
        df_epoch_validation.to_csv(
            os.path.join(MODEL_CHECKPOINT_STATE_DICTS_FOLDER, f"validation_df_{epoch}")
        )

        # Reduce learning rate for the next epoch
        scheduler.step()
