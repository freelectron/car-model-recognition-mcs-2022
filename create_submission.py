import os

from tqdm import tqdm
import torch
import pandas as pd

from modeling.losses import PAIRWISEDISTANCE_P2
from modeling.model import SiameseNetwork
from preprocessing.dataset import SiameseNetworkDatasetTesting, TRAINING_TRANSFORMATION_SEQUENCE_1


TESTING_BATCH_SIZE = 32


if __name__ == "__main__":
    # Configure input & output paths
    results_folder = "./results/"
    model_run = "siamese_v0_2022-06-15T17:57"
    model_folder = os.path.join("./modeling/model_store/", model_run)
    epoch = 19
    results_model_run_folder = os.path.join(results_folder, model_run)
    if not os.path.isdir(results_model_run_folder):
        os.makedirs(results_model_run_folder)

    # Load model to evaluate & set ("eval") regime
    net = SiameseNetwork(
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
    pairwise_euclidean_distance_list = list()
    output1_list = list()
    output2_list = list()
    with torch.no_grad():
        for i, (image1, image2) in tqdm(enumerate(test_loader, 0), total=len(test_loader)):
            output1, output2 = net(image1.cuda(), image2.cuda())
            output1_list = output1_list + output1.detach().cpu().tolist()
            output2_list = output2_list + output2.detach().cpu().tolist()
            # Calculate dissimilarity
            pairwise_euclidean_distance_list += PAIRWISEDISTANCE_P2(
                output1,
                output2,
            ).detach().cpu().tolist()

    submit_pairs_df['output1'] = output1_list
    submit_pairs_df['output2'] = output2_list
    submit_pairs_df['pairwise_euclidean_distance'] = pairwise_euclidean_distance_list
    all_outputs_df_name = f"siamese_ods_dataset_outputs_epoch_{epoch}.csv"
    submit_pairs_df.to_csv(
        os.path.join(results_model_run_folder, all_outputs_df_name)
    )

    # Submission with Euclidean distance
    submit_pairs_df = pd.read_csv(os.path.join(results_model_run_folder, all_outputs_df_name))
    submit_pairs_df['score'] = submit_pairs_df["pairwise_euclidean_distance"].apply(lambda x: (2 - x) / 2)
    submit_pairs_df.rename(columns={"Unnamed: 0": "id"}, inplace=True)
    submit_pairs_df[['id', 'score']].to_csv(
        os.path.join(results_model_run_folder, f"siamese_submission_epoch_{epoch}.csv"),
        index=False
    )
