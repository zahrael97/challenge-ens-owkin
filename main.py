from train_imagenet import SurvivalEstimator

parameters = {
    "experiment_name": "convergence_test",
    "images_dir": "~/datasets/tumor/x_train/images/",
    "csv_dir": "~/datasets/tumor/y_train.csv",
    "linear_in": 1250,
    "mode": 'concatenate',
    "conv1_filters": 10,
    "conv2_filters": 10,
    "dropout3d": .0,
    "learning_rate": .0005,
    "dataset_seed": 23,
    "train_test_split": .9,
    "flip_proba": .0,
    "batch_size": 30,
    "num_workers": 8,
    "shuffle": True,
    "nb_epochs_to_save": 2,
    "device": 'cuda',
}

estimator = SurvivalEstimator(**parameters)

estimator.train(100)
