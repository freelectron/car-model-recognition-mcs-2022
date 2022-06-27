"""
Train and validate a Siamese NN that directly outputs whether a pair of images are same/similar.
"""

import logging
from datetime import datetime
import os
from typing import List

import pandas as pd
from pandas import DataFrame
from tqdm import tqdm
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

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
from modeling.model import SiameseNetwork
from modeling.losses import ContrastiveLoss, PAIRWISEDISTANCE_P2

START_OF_TRAIN = datetime.now().isoformat()[:16]
MODEL_CHECKPOINT_STATE_DICTS_FOLDER = (
    f"./modeling/model_store/siamese_v0_{START_OF_TRAIN}"
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
    dataloader_instance: SiameseNetworkDataset,
    epoch: int,
    log_after_n_iterations: int = 50,
) -> (torch.nn.Module, List[List[float]], int):
    """
    Runs the main train loop.
    """
    # Set to train mode
    model.train()
    loss_history = list()
    euclidean_distances = list()
    true_labels = list()
    difference_embedding_vector = list()
    total_number_of_iterations = 0

    # Hold average loss over several pairs
    epoch_loss_values = list()
    for i, data in enumerate(tqdm(dataloader_instance)):
        img0, img1, label = data
        img0, img1, label = img0.cuda(), img1.cuda(), label.cuda()
        optimizer.zero_grad()
        output1, output2 = model(img0, img1)
        loss_contrastive = criterion(output1, output2, label)
        loss_contrastive.backward()
        # Apply gradient clipping to prevent nan loss after n epochs ?
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0, norm_type=2)
        optimizer.step()
        # Save to calculate metrics on
        difference_embedding_vector += (output1 - output2).cuda().tolist()
        euclidean_distances += (
            PAIRWISEDISTANCE_P2(output1, output2).detach().cpu().tolist()
        )
        true_labels += label.squeeze().int().detach().cpu().tolist()
        epoch_loss_values.append(loss_contrastive.detach().cpu().numpy())

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
    similarity_scores = np.divide(
        (np.array([2] * len(euclidean_distances)) - np.array(euclidean_distances)),
        np.array([2] * len(euclidean_distances)),
    ).tolist()
    roc_score = roc_auc_score(true_labels, similarity_scores)
    logging.info(
        "{} epoch finished. Train data ROC-AUC score based on euclidean similarity is {}".format(
            epoch, roc_score
        )
    )
    clf = LogisticRegression(random_state=0).fit(
        np.array(euclidean_distances).reshape(-1, 1), true_labels
    )
    similarity_scores_clf = clf.predict_proba(
        np.array(euclidean_distances).reshape(-1, 1)
    )
    roc_auc_score_clf_epoch = roc_auc_score(
        true_labels, np.argmax(similarity_scores_clf, axis=1)
    )
    logging.info(
        "{} epoch finished. Train data ROC-AUC score based on euclidean similarity & logistic regression "
        "is {}".format(epoch, roc_auc_score_clf_epoch)
    )

    # FIXME: delete
    pd.DataFrame(
        {
            "true_label_values": true_labels,
            "euclidean_distance_values": euclidean_distances,
            "similarity_scores": similarity_scores,
        }
    ).to_csv(f"./results/overfitting_testing_{epoch}.csv")

    return model, loss_history, clf


def validate(
    model: torch.nn.Module,
    dataloader_instance: SiameseNetworkDatasetValidation,
    epoch: int,
    output_classifier: LogisticRegression,
    device: str = "cuda",
) -> tuple[DataFrame, float]:
    true_label_values = list()
    output_embedding_img1 = list()
    output_embedding_img2 = list()
    difference_embedding_vector = list()
    euclidean_distance_values = list()
    similarity_scores = list()
    image1_paths = list()
    image2_paths = list()

    with torch.no_grad():
        # Since batch size is one
        model.cuda()
        model.eval()
        for i, data_image_pair in enumerate(dataloader_instance, 0):
            # Run model inference
            x0, x1, label, image1_path, image2_path = data_image_pair
            output1, output2 = model(x0.to(device), x1.to(device))
            difference_embedding_vector += (output1 - output2).cuda().tolist()
            output_embedding_img1 += output1.cuda().tolist()
            output_embedding_img2 += output2.cuda().tolist()

            # Calculate Euclidean distances
            true_label_values += label.squeeze().int().tolist()
            euclidean_distances = PAIRWISEDISTANCE_P2(output1, output2).tolist()
            euclidean_distance_values += euclidean_distances

            # Calculate similarity scores
            similarity_scores += np.divide(
                (
                    np.array([2] * len(euclidean_distances))
                    - np.array(euclidean_distances)
                ),
                np.array([2] * len(euclidean_distances)),
            ).tolist()

            # Save images for later inspection
            image1_paths += list(image1_path)
            image2_paths += list(image2_path)

    # Calculate ROC-AUC Score
    roc_auc_score_epoch = roc_auc_score(true_label_values, similarity_scores)
    logging.info(
        "{} epoch finished. Test data ROC-AUC score based on euclidean similarity is {}".format(
            epoch, roc_auc_score_epoch
        )
    )
    similarity_scores_clf = output_classifier.predict_proba(
        np.array(euclidean_distance_values).reshape(-1, 1)
    )
    roc_auc_score_clf_epoch = roc_auc_score(
        true_label_values, np.argmax(similarity_scores_clf, axis=1)
    )
    logging.info(
        "{} epoch finished. Test data ROC-AUC score based on euclidean similarity & logistic regression "
        "is {}".format(epoch, roc_auc_score_clf_epoch)
    )
    logging.info("=========================================")

    df_epoch_validation = pd.DataFrame(
        {
            "true_label_values": true_label_values,
            "output_embedding_img1": output_embedding_img1,
            "output_embedding_img2": output_embedding_img2,
            "euclidean_distance_values": euclidean_distance_values,
            "similarity_scores": similarity_scores,
            "image1_paths": image1_paths,
            "image2_paths": image2_paths,
        }
    )

    return df_epoch_validation, roc_auc_score_epoch


if __name__ == "__main__":
    from torch.utils.data import SubsetRandomSampler

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
    load_model_folder = "./modeling/model_store/siamese_v0_2022-06-16T18:43/"
    load_model_from_state_dictionary = os.path.join(
        load_model_folder, "model_9_epoch.pt"
    )
    net = (
        SiameseNetwork().cuda()
    )  # FIXME: load previous run ? load_model_from_state_dictionary
    # Declare Loss Function
    criterion = ContrastiveLoss()
    # Declare Optimizer
    optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)
    # Set learning rate schedule for the optimizer
    scheduler = StepLR(optimizer, step_size=Config.train_number_epochs, gamma=0.1)

    for epoch in range(0, Config.train_number_epochs):
        logging.info(f"\nEPOCH: {epoch}")
        # Train on both datasets
        model_epoch, loss_history_epoch, output_classifier = train(
            model=net, dataloader_instance=train_dataloader, epoch=epoch
        )

        torch.save(
            net.state_dict(),
            os.path.join(MODEL_CHECKPOINT_STATE_DICTS_FOLDER, f"model_{epoch}_epoch.pt")
        )
        pickle.dump(
            loss_history_epoch,
            open(os.path.join(
                MODEL_CHECKPOINT_STATE_DICTS_FOLDER,
                f"loss_history_epoch_{epoch}"), "wb")
        )
        pickle.dump(
            output_classifier,
            open(os.path.join(
                MODEL_CHECKPOINT_STATE_DICTS_FOLDER,
                f"output_classifier_epoch_{epoch}"), "wb")
        )

        # Only use when testing validation
        output_classifier = pickle.load(
            open(os.path.join(
                load_model_folder,
                f"output_classifier_epoch_{epoch}"), "rb")
        )

        Validate
        df_epoch_validation, roc_auc_score_epoch = validate(
            model=net, dataloader_instance=test_dataloader, epoch=epoch, output_classifier=output_classifier,
        )
        df_epoch_validation.to_csv(os.path.join(MODEL_CHECKPOINT_STATE_DICTS_FOLDER, f"validation_df_{epoch}"))

        # Reduce learning rate for the next epoch
        scheduler.step()
