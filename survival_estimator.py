from time import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from metrics import cindex_torch
from config import EXPERIMENTS_PATH, INFERENCE_CLINICAL, INFERENCE_IMAGES
from utils import (
    get_hms, AverageMeter,
    count_parameters,
    # create_difference_plot,
    create_txt_config,
)


class SurvivalEstimator:

    def __init__(
        self,
        model,
        model_kwargs,
        dataset,
        dataset_kwargs,
        loss=nn.MSELoss,
        experiment_name=None,
        learning_rate=.0005,
        batch_size=30,
        num_workers=6,
        shuffle=True,
        nb_epochs_to_save=2,
        device='cuda',
    ):
        assert experiment_name, "The experience should be named "
        print(f"Estimator for experience : {experiment_name}")

        self.model = model
        self.model_kwargs = model_kwargs

        self.dataset = dataset
        self.dataset_kwargs = dataset_kwargs

        self.loss = loss
        self.learning_rate = learning_rate

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle

        self.experiment_path = Path(EXPERIMENTS_PATH).expanduser() / experiment_name
        self.models_path = self.experiment_path / 'models'
        self.logs_train_path = self.experiment_path / 'logs_train'
        self.logs_test_path = self.experiment_path / 'logs_test'

        self.nb_epochs_to_save = nb_epochs_to_save
        self.device = device.lower()
        if self.device == 'gpu':
            self.device = 'cuda'
        assert self.device in ['cpu', 'cuda']

    def mkdir(self):
        self.experiment_path.mkdir(parents=True)
        self.models_path.mkdir()
        self.logs_train_path.mkdir()
        self.logs_test_path.mkdir()

    def instantiate_generator(self):
        return self.model(**self.model_kwargs).to(self.device)

    def make_metric_dataframe(self, patient_info, survival_time):
        if len(survival_time.shape) == 1:
            survival_time = np.expand_dims(survival_time, axis=1)
        survival_time = self.dataset_train.scaler.inverse_transform(survival_time)
        survival_time[survival_time <= 1] = 1.1
        df = np.hstack((patient_info, survival_time))
        df = pd.DataFrame(df, columns=["PatientID", "Event", "SurvivalTime"])
        df = df.set_index('PatientID')
        return df

    def to(self, x):
        if x is None:
            return None
        else:
            x = x.float().to(self.device)
        return x

    def train(self, epochs=100, epoch_to_restore=0):
        net = self.instantiate_generator()

        if epoch_to_restore == 0:
            self.mkdir()
        else:
            filename_model = self.models_path / f'epoch_{epoch_to_restore}.pth'
            net.load_state_dict(torch.load(filename_model))

        self.dataset_kwargs["train_test"] = "train"
        self.dataset_train = self.dataset(**self.dataset_kwargs)
        dataloader_train = DataLoader(
            self.dataset_train, self.batch_size, shuffle=self.shuffle,
            num_workers=self.num_workers, pin_memory=True)

        self.dataset_kwargs["train_test"] = "test"
        dataset_test = self.dataset(**self.dataset_kwargs)
        dataloader_test = DataLoader(
            dataset_test, self.batch_size, shuffle=self.shuffle,
            num_workers=self.num_workers, pin_memory=True)

        criterion = self.loss()

        optimizer = optim.Adam(net.parameters())
        writer_train = SummaryWriter(log_dir=str(self.logs_train_path))
        writer_test = SummaryWriter(log_dir=str(self.logs_test_path))

        # x = next(iter(dataloader_train))
        # writer_train.add_graph(net, (x.get("images"), x.get("clinical")))
        config = create_txt_config(net, optimizer, self.dataset_train)
        writer_train.add_text("Experiment summary", config)
        print("Training of net begins, "
              f"optimizing {count_parameters(net)} parameters")

        try:
            for epoch in range(epoch_to_restore + 1, epochs + epoch_to_restore + 1):
                start_time = time()

                net.train()
                for idx_batch, current_batch in enumerate(tqdm(
                    dataloader_train,
                    desc=f'Training for epoch {epoch}')
                ):
                    net.zero_grad()
                    images = self.to(current_batch.get('images'))
                    clinical = self.to(current_batch.get('clinical'))
                    y = self.to(current_batch['y'])
                    prediction = net(images=images, clinical=clinical)
                    loss = criterion(prediction, y)
                    loss.backward()
                    optimizer.step()

                net.eval()
                with torch.no_grad():
                    train_mse_loss = AverageMeter()
                    train_cindex_metric = AverageMeter()
                    for idx_batch, current_batch in enumerate(tqdm(
                        dataloader_train,
                        desc='Evaluating model on training set')
                    ):
                        images = self.to(current_batch.get('images'))
                        clinical = self.to(current_batch.get('clinical'))
                        y = self.to(current_batch['y'])
                        prediction = net(images=images, clinical=clinical)
                        loss = criterion(prediction, y)
                        metric = cindex_torch(
                            y, prediction, current_batch['info'][:, 1].to(self.device))
                        train_mse_loss.update(loss)
                        train_cindex_metric.update(metric)
                    writer_train.add_scalar('metrics/mse_loss', train_mse_loss.avg, epoch)
                    writer_train.add_scalar('metrics/cindex', train_cindex_metric.avg, epoch)

                    test_mse_loss = AverageMeter()
                    test_cindex_metric = AverageMeter()
                    for idx_batch, current_batch in\
                            enumerate(tqdm(dataloader_test, desc='Evaluating model on test set')):
                        images = self.to(current_batch.get('images'))
                        clinical = self.to(current_batch.get('clinical'))
                        y = self.to(current_batch['y'])
                        prediction = net(images=images, clinical=clinical)
                        loss = criterion(prediction, y)
                        metric = cindex_torch(
                            y, prediction, current_batch['info'][:, 1].to(self.device))
                        test_mse_loss.update(loss)
                        test_cindex_metric.update(metric)
                    # fig = create_difference_plot(df_true, df_pred)
                    # writer_test.add_figure("metrics/difference", fig, epoch)
                    writer_test.add_scalar('metrics/mse_loss', test_mse_loss.avg, epoch)
                    writer_test.add_scalar('metrics/cindex', test_cindex_metric.avg, epoch)

                if epoch % self.nb_epochs_to_save == 0:
                    filename = self.models_path / f'epoch_{epoch}.pth'
                    torch.save(net.state_dict(), str(filename))

                end_time = time()
                print(f"[*] Finished epoch {epoch} in {get_hms(end_time - start_time)};\n"
                      "Train :\n"
                      f"\rLoss : {train_mse_loss.avg:.4f}\n"
                      f"\rCindex : {train_cindex_metric.avg:.4f}\n"
                      "Test :\n"
                      f"\rTest Loss : {test_mse_loss.avg:.4f}\n"
                      f"\rCindex : {test_cindex_metric.avg:.4f}\n")

        finally:
            print('[*] Closing Writer.')
            writer_train.close()
            writer_test.close()
            return net

    def infer(self, load_epoch=None, filename_model=None):
        net = self.instantiate_generator()

        if filename_model is None:
            filename_model = self.models_path / f'epoch_{load_epoch}.pth'
        else:
            filename_model = Path(filename_model).expanduser()
        net.load_state_dict(torch.load(filename_model))

        # We must define self.dataset_train for inverse scaling
        self.dataset_kwargs["train_test"] = "train"
        self.dataset_train = self.dataset(**self.dataset_kwargs)

        self.dataset_kwargs["images_dir"] = INFERENCE_IMAGES
        self.dataset_kwargs["infer_path"] = INFERENCE_CLINICAL
        self.dataset_kwargs["train_test"] = "test"
        self.dataset_kwargs["train_test_split"] = 0.
        self.dataset_kwargs["csv_dir"] = None
        dataset_infer = self.dataset(**self.dataset_kwargs)
        dataloader_infer = DataLoader(
            dataset_infer, self.batch_size, shuffle=self.shuffle,
            num_workers=self.num_workers, pin_memory=True)

        infer = []
        net.eval()
        with torch.no_grad():
            for current_batch in tqdm(dataloader_infer, desc='Inference'):
                images = self.to(current_batch.get('images'))
                clinical = self.to(current_batch.get('clinical'))
                prediction = net(images=images, clinical=clinical)
                infer.append(self.make_metric_dataframe(
                    current_batch["info"], prediction.cpu().numpy()))

        infer = pd.concat(infer)
        return infer
