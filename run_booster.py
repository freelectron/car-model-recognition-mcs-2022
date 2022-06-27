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
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

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
from modeling.model import SiameseNetwork, ClassificationNetwork
from modeling.losses import ContrastiveLoss, PAIRWISEDISTANCE_P2

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


def extract_embeddings(
    model: torch.nn.Module,
    dataloader_instance: ClassificationNetwork,
    epoch: int,
) -> (torch.nn.Module, List[List[float]], int):
    """
    Runs the main train loop.
    """
    # Set to train mode
    model.eval()
    true_labels = list()
    diff_vectors = np.zeros((1, 1280))

    xgb_model = XGBClassifier()

    # Hold average loss over several pairs
    with torch.no_grad():
        epoch_loss_values = list()
        for i, data in enumerate(tqdm(dataloader_instance)):
            img0, img1, label = data
            img0, img1, label = img0.cuda(), img1.cuda(), label.cuda()
            batch_embedding_vectors = model.extract_features(img0, img1)
            vecs1 = batch_embedding_vectors[:, :1280]
            vecs2 = batch_embedding_vectors[:, 1280:]
            diffs = vecs1 - vecs2
            xgb_features = diffs.detach().cpu().numpy()
            # Save to calculate metrics on
            diff_vectors = np.vstack([diff_vectors, xgb_features])
            true_labels += label.squeeze().int().detach().cpu().tolist()

    # Calculate eval metrics
    xgb_model.fit(diff_vectors[1:, :], np.array(true_labels))
    predictions = xgb_model.predict_proba(diff_vectors[1:, :])[:, 1]

    roc_score = roc_auc_score(true_labels, predictions.squeeze())
    logging.info(
        "{} epoch finished. Train data ROC-AUC score based on euclidean similarity is {}".format(
            epoch, roc_score
        )
    )

    # # FIXME: delete later
    # pd.DataFrame(
    #     {
    #         "true_label_values": true_labels,
    #         "similarity_scores": sigmoid_output_vectors,
    #     }
    # ).to_csv(f"./results/classification/train_similarity_scores_{epoch}.csv")

    return xgb_model


def validate(
    model: torch.nn.Module,
    xgb_model: XGBClassifier,
    dataloader_instance: SiameseNetworkDatasetValidation,
    epoch: int,
    device: str = "cuda",
) -> tuple[DataFrame, float]:
    true_label_values = list()
    image1_paths = list()
    image2_paths = list()
    diff_vectors = np.zeros((1, 1280))

    with torch.no_grad():
        # Since batch size is one
        model.cuda()
        model.eval()
        for i, data_image_pair in enumerate(dataloader_instance, 0):
            # Run model inference
            x0, x1, label, image1_path, image2_path = data_image_pair
            # sigmoid_output_difference_vector = model(x0.to(device), x1.to(device))
            batch_embedding_vectors = model.extract_features(
                x0.to(device), x1.to(device)
            )
            vecs1 = batch_embedding_vectors[:, :1280]
            vecs2 = batch_embedding_vectors[:, 1280:]
            diffs = vecs1 - vecs2
            xgb_features = diffs.detach().cpu().numpy()
            # Save to calculate metrics on
            diff_vectors = np.vstack([diff_vectors, xgb_features])

            # Calculate Euclidean distances
            true_label_values += label.squeeze().int().tolist()

            # Save images for later inspection
            image1_paths += list(image1_path)
            image2_paths += list(image2_path)

    # Calculate ROC-AUC Score
    predictions = xgb_model.predict_proba(diff_vectors[1:, :])[:, 1]
    roc_auc_score_epoch = roc_auc_score(true_label_values, predictions)
    logging.info(
        "{} epoch finished. Test data ROC-AUC score based on euclidean similarity is {}".format(
            epoch, roc_auc_score_epoch
        )
    )
    logging.info("=========================================")

    df_epoch_validation = pd.DataFrame(
        {
            "true_label_values": true_label_values,
            "predictions": predictions,
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
    load_model_folder = "./modeling/model_store/classification_v0_stanford_cars_success_0.75_roc_public_test/"
    load_model_from_state_dictionary = os.path.join(
        load_model_folder, "model_19_epoch.pt"
    )
    net = ClassificationNetwork(load_model_from_state_dictionary).cuda()

    epoch = 19
    # Train on both datasets
    model_xgboost = extract_embeddings(
        model=net, dataloader_instance=train_dataloader, epoch=epoch
    )

    torch.save(
        net.state_dict(),
        os.path.join(MODEL_CHECKPOINT_STATE_DICTS_FOLDER, f"model_{epoch}_epoch.pt"),
    )

    # Save XGBoost
    output_classifier = pickle.dump(
        model_xgboost,
        open(
            os.path.join(MODEL_CHECKPOINT_STATE_DICTS_FOLDER, f"output_classifier"),
            "wb",
        ),
    )

    # Validate
    df_epoch_validation, roc_auc_score_epoch = validate(
        model=net,
        xgb_model=model_xgboost,
        dataloader_instance=test_dataloader,
        epoch=epoch,
    )
    df_epoch_validation.to_csv(
        os.path.join(MODEL_CHECKPOINT_STATE_DICTS_FOLDER, f"validation_df_{epoch}")
    )
