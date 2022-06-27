import os
import pickle

from tqdm import tqdm
import torch
import pandas as pd
import numpy as np

from modeling.model import ClassificationNetwork
from preprocessing.dataset import SiameseNetworkDatasetTesting, TRAINING_TRANSFORMATION_SEQUENCE_1


TESTING_BATCH_SIZE = 32


if __name__ == "__main__":
    # Configure input & output paths
    results_folder = "./results/"
    model_run = "classification_cc_v0_2022-06-24T16:56"
    model_folder = os.path.join("./modeling/model_store/", model_run)
    epoch = 19
    results_model_run_folder = os.path.join(results_folder, model_run)
    if not os.path.isdir(results_model_run_folder):
        os.makedirs(results_model_run_folder)

    xgb_model = pickle.load(open(os.path.join(model_folder,"output_classifier"), "rb"))

    # Load model to evaluate & set ("eval") regime
    net = ClassificationNetwork(
        os.path.join(model_folder, f"model_{epoch}_epoch.pt")
    ).cuda()
    net.eval()

    # Get the dataloader
    test_dataset = SiameseNetworkDatasetTesting(
        image_pair_csv="../ods_public_test_data/submission_list.csv",
        annotation_csv="../ods_public_test_data/images_w_boxes.csv",
        transform=TRAINING_TRANSFORMATION_SEQUENCE_1,
        image_directory="../ods_public_test_data/test_images",
    )

    test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=TESTING_BATCH_SIZE,
            num_workers=11,
            shuffle=False,
            pin_memory=True
    )

    # Dataframe for submission will need to have the same order as the given one
    submit_pairs_df = test_dataset.image_pair_df.copy(deep=True)
    norm_distance_list = list()
    diff_vectors = np.zeros((1, 1280))

    with torch.no_grad():
        for i, (image1, image2) in tqdm(enumerate(test_loader, 0), total=len(test_loader)):
            # This is how it used to be before
            batch_embedding_vectors = net.extract_features(image1.cuda(), image2.cuda())
            vecs1 = batch_embedding_vectors[:, :1280]
            vecs2 = batch_embedding_vectors[:, 1280:]
            diffs = vecs1 - vecs2
            xgb_features = diffs.detach().cpu().numpy()
            # Save to calculate metrics on
            diff_vectors = np.vstack([diff_vectors, xgb_features])
            # TODO: also save features extracted from cnn?

    predictions = xgb_model.predict_proba(diff_vectors[1:, :])[:, 1]
    submit_pairs_df['output'] = predictions
    all_outputs_df_name = f"siamese_classification_ods_dataset_outputs_epoch_{epoch}.csv"
    submit_pairs_df.to_csv(
        os.path.join(results_model_run_folder, all_outputs_df_name)
    )

    # Submission with Euclidean distance
    submit_pairs_df = pd.read_csv(os.path.join(results_model_run_folder, all_outputs_df_name))
    submit_pairs_df['score'] = submit_pairs_df["output"].copy(deep=True)
    submit_pairs_df.rename(columns={"Unnamed: 0": "id"}, inplace=True)
    submit_pairs_df[['id', 'score']].to_csv(
        os.path.join(results_model_run_folder, f"siamese_classification_submission_epoch_{epoch}.csv"),
        index=False
    )
