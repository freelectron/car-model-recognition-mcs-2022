# Import all the necessary Library
import logging
from datetime import datetime
import os
from typing import List, Tuple
import pickle

from tqdm import tqdm
from torch.utils.data import DataLoader
import numpy as np
import torch
from torch import optim

from preprocessing.dataset import Config, SiameseNetworkDataset, TRAINING_TRANSFORMATION_SEQUENCE_1, \
    TESTING_TRANSFORMATION_SEQUENCE_1, SiameseNetworkDatasetTesting
from modeling.model import SiameseNetwork
from modeling.losses import ContrastiveLoss, PAIRWISEDISTANCE_P2

logging.getLogger().setLevel(logging.INFO)

START_OF_TRAIN = datetime.now().isoformat()[:16]
MODEL_CHECKPOINT_STATE_DICTS_FOLDER = f"./modeling/model_store/siamese_v0_{START_OF_TRAIN}"


def train(
        model: torch.nn.Module,
        dataloader_instance: SiameseNetworkDataset,
        epoch: int,
        log_after_n_iterations: int = 50
) -> (torch.nn.Module, List[List[float]], int):
    """
    Runs the main train loop.
    """
    loss_history = list()
    total_number_of_iterations = 0

    # Hold average loss over several pairs
    epoch_loss_values = list()
    for i, data in (enumerate(tqdm(dataloader_instance), 0)):
        img0, img1, label = data
        img0, img1, label = img0.cuda(), img1.cuda(), label.cuda()
        optimizer.zero_grad()
        output1, output2 = model(img0, img1)
        loss_contrastive = criterion(output1, output2, label)
        loss_contrastive.backward()
        # Apply gradient clipping to prevent nan loss after n epochs
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0, norm_type=2)
        optimizer.step()
        epoch_loss_values.append(loss_contrastive.detach().cpu().numpy())
        if i % log_after_n_iterations == 0:
            loses_mean = np.array(epoch_loss_values).mean()
            logging.info("Epoch number {}\n Current epoch's average loss {}\n".format(epoch, loses_mean))
            total_number_of_iterations += log_after_n_iterations
            loss_history.append(epoch_loss_values)

    return model, loss_history, total_number_of_iterations


def validate(
    model: torch.nn.Module,
    dataloader_instance: SiameseNetworkDatasetTesting,
    epoch: int,
    n_batches: int = 100,
    device: str = "cuda",
) -> Tuple[List, List[int], List[float], List[Tuple[str, str]]]:
    predicted_label_values = list()
    output_pairs = list()
    eucledian_distance_values = list()
    image_pairs = list()

    with torch.no_grad():
        # Since batch size is one
        model.cuda()
        model.eval()
        counter = 0
        correct = 0
        for i, data_image_pair in enumerate(dataloader_instance, 0):
            # Run model inference
            # TODO: get to know what model outputs/calculates, see loss?
            x0, x1, label, image1_path, image2_path = data_image_pair
            # Colab: "One-shot applies in the output of 128 dense vectors which is then converted to 2 dense vectors"
            output1, output2 = model(x0.to(device), x1.to(device))
            output_pairs.append((output1.cuda().tolist(), output2.cuda().tolist()))
            # Calculate accuracy
            res = torch.abs(output1.cuda() - output2.cuda())
            label = label[0].tolist()
            label = int(label[0])

            # FIXME: WEAK POINT! HOW TO DETERMINE THE THRESHOLD
            # TODO: double check what the output is
            distance_magnitude = torch.sqrt(torch.pow(res[0][0], 2) + torch.pow(res[0][1], 2))
            result = 0 if distance_magnitude > 0.01 else 1
            # result = torch.max(res, 1)[1][0][0][0].data[0].tolist()
            predicted_label_values.append(result)
            if label == result:
                correct += 1
            counter += 1

            # Calculate dissimilarity
            eucledian_distance_values.append(PAIRWISEDISTANCE_P2(output1, output2).item())

            # Save images for later inspection
            image_pairs.append((image1_path, image2_path))
            if counter == n_batches:
                break

        accuracy = (correct / len(test_dataloader)) * 100
        logging.info("=========================================")
        logging.info("After evaluating {}% Accuracy:{}%".format(epoch, accuracy))

    return output_pairs, predicted_label_values, eucledian_distance_values, image_pairs


if __name__ == "__main__":
    siamese_dataset = SiameseNetworkDataset(
        Config.training_csv,
        Config.training_dir,
        transform=TRAINING_TRANSFORMATION_SEQUENCE_1
    )

    # Load the dataset as pytorch tensors using dataloader
    train_dataloader = DataLoader(
        siamese_dataset,
        shuffle=True,
        num_workers=8,
        batch_size=Config.train_batch_size
    )

    # Load the test dataset
    test_dataset = SiameseNetworkDatasetTesting(
        image_pair_label_csv=Config.testing_csv,
        image_directory=Config.testing_dir,
        transform=TESTING_TRANSFORMATION_SEQUENCE_1
    )

    test_dataloader = DataLoader(
        test_dataset,
        num_workers=6,
        batch_size=1,
        # No shuffling for easier comparison?
        shuffle=False,
    )

    # Declare Siamese Network
    net = SiameseNetwork().cuda()
    # Declare Loss Function
    criterion = ContrastiveLoss()
    # Declare Optimizer
    optimizer = optim.RMSprop(net.parameters(), lr=1e-4, alpha=0.99, eps=1e-8, weight_decay=0.0005, momentum=0.9)

    for epoch in tqdm(range(0, Config.train_number_epochs)):
        model_epoch, loss_history_epoch, total_number_of_iterations_epoch = train(
            model=net, dataloader_instance=train_dataloader, epoch=epoch
        )

        if not os.path.isdir(MODEL_CHECKPOINT_STATE_DICTS_FOLDER):
            os.makedirs(MODEL_CHECKPOINT_STATE_DICTS_FOLDER)
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

        output_pairs_epoch, predicted_label_values, eucledian_distance_values, image_pairs = validate(
            model=net, dataloader_instance=test_dataloader, epoch=epoch
        )
        pickle.dump(
            output_pairs_epoch,
            open(os.path.join(
                MODEL_CHECKPOINT_STATE_DICTS_FOLDER,
                f"output_pairs_epoch_{epoch}"), "wb")
        )
