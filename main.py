import argparse
from pathlib import Path

import torch
import torch.nn as nn

import models
import datasets as D
from survival_estimator import SurvivalEstimator
from config import EXPERIMENTS_PATH, DATA_PATH


nets = {
    "images": models.Tumor3DNet,
    "clinical": models.ClinicalNet,
    "multimodal": models.MultiModalNet,
}


datasets = {
    "images": D.Images3DDataset,
    "clinical": D.ClinicalDataset,
    "multimodal": D.MultiModalDataset,
}

loss = {
    "mse": nn.MSELoss,
    "l1": nn.L1Loss,
}

parser = argparse.ArgumentParser(description='Launch net training')
parser.add_argument("-exp", "--experiment-name", type=str, help="Name of the experiment")
parser.add_argument("-nn", "--neural-net", type=str, choices=list(nets.keys()),
                    help=f"Should be one of {list(nets.keys())}")
parser.add_argument("-l", "--loss", type=str, choices=list(loss.keys()), default="mse",
                    help="Loss used for backpropagation, default='mse'")
parser.add_argument("-e", "--epochs", type=int, default=100,
                    help="Number of epochs for training, default=100")
parser.add_argument("-mc", "--model-checkpoint", type=int, default=2,
                    help="Number of epochs before saving weights, default=2")
parser.add_argument("-r", "--restore-epochs", type=int, default=0,
                    help="Resume training from specified epochs")
parser.add_argument("-d", "--device", type=str, default="cuda",
                    help="Device used for training, should be one of ['cuda', 'cpu']")
parser.add_argument("-s", "--seed", type=int, default=23,
                    help="Manual random seed")
parser.add_argument("-ts", "--train-split", type=float, default=0.9,
                    help="Train / test split, default=0.9")
parser.add_argument("-bs", "--batch-size", type=int, default=31,
                    help="Batch size, default=31")
parser.add_argument("-i", "--infer", action="store_true", default=False,
                    help="Pass this argument for inference mode")

args = parser.parse_args()

TRAIN_TEST_SPLIT = args.train_split
SEED = args.seed

dataset_kwargs = {
    "images": {
        "images_dir": "~/datasets/tumor/x_train/images/",
        "csv_dir": "~/datasets/tumor/y_train.csv",
        "seed": SEED,
        "train_test_split": TRAIN_TEST_SPLIT,
        "mode": 'concatenate',
        "flip_proba": 0.,
        "noise_magnitude": 2.,
    },
    "clinical": {
        "data_path": "~/datasets/tumor/x_train/features/clinical_data.csv",
        "id_path": "~/datasets/tumor/x_train/images/",
        "csv_dir": "~/datasets/tumor/y_train.csv",
        "seed": SEED,
        "train_test_split": TRAIN_TEST_SPLIT,
    },
}
dataset_kwargs["multimodal"] = {
    "image_dataset_kwargs": dataset_kwargs["images"],
    "clinical_dataset_kwargs": dataset_kwargs["clinical"],
    "train_test_split": TRAIN_TEST_SPLIT,
}

model_kwargs = {
    "images": {
        "linear_in": 1250,
        "conv1_filters": 10,
        "conv2_filters": 10,
        "dropout3d": .5,
        "dropout": .5,
        "conv_bias": True,
        "conv_stride": 2,
        "conv_padding": 0,
        "hidden_layer": 128,
        "linear_bias": True,
        "regress": True,
    },
    "clinical": {
        "hidden_layer": [
                (10, 100),
                (100, 200),
                (200, 100),
            ],
        "dropout": .0,
        "regress": True,
        "linear_bias": True,
    },
    "multimodal": {
        "linear_bias": True,
        "tumornet_transfer": None,
        "clinicalnet_transfer": None,
    },
}
model_kwargs["multimodal"]["tumornet_kwargs"] = model_kwargs["images"]
model_kwargs["multimodal"]["clinicalnet_kwargs"] = model_kwargs["clinical"]
model_kwargs["multimodal"]["linear_in"] = (
    model_kwargs["images"]["hidden_layer"]
    + model_kwargs["clinical"]["hidden_layer"][-1][1])

mode = dataset_kwargs["images"]["mode"].lower()
model_kwargs["images"]["in_channels"] = 2 if mode == 'concatenate' else 1


experiment_name = f"{args.experiment_name}_{args.neural_net}"
experience_parameters = {
    "experiment_name": experiment_name,
    "learning_rate": .0005,
    "batch_size": args.batch_size,  # if last batch is size=1 ClinicalNet does't work
    "num_workers": 6,
    "shuffle": True,
    "nb_epochs_to_save": args.model_checkpoint,
    "device": args.device,
    "loss": loss[args.loss],
}


torch.manual_seed(SEED)
if not args.infer:
    exp_dir = Path(EXPERIMENTS_PATH).expanduser() / experiment_name
    if exp_dir.is_dir():
        assert args.restore_epochs != 0, f"Experiment {exp_dir} already created"

    estimator = SurvivalEstimator(
        nets[args.neural_net],
        model_kwargs[args.neural_net],
        datasets[args.neural_net],
        dataset_kwargs[args.neural_net],
        **experience_parameters,
    )
    estimator.train(args.epochs, epoch_to_restore=args.restore_epochs)

else:
    estimator = SurvivalEstimator(
        nets[args.neural_net],
        model_kwargs[args.neural_net],
        datasets[args.neural_net],
        dataset_kwargs[args.neural_net],
        **experience_parameters,
    )

    predictions = estimator.infer(
        load_epoch=None,
        filename_model=None,
    )
    outpath = Path(DATA_PATH).expanduser() / "predictions.csv"
    predictions.to_csv(outpath)
