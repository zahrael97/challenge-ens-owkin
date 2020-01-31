import os
from time import time
from pathlib import Path

from tqdm import tqdm
import torch
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from config import EXPERIMENTS_PATH
from datasets import Images3DDataset
from models import Tumor3DConv
from metrics import cindex
from utils import weights_init, get_hms, AverageMeter, make_metric_dataframe


class SurvivalEstimator:

    def __init__(
        self,
        experiment_name,
        images_dir,
        csv_dir,
        linear_in,
        mode='concatenate',
        conv1_filters=10,
        conv2_filters=10,
        dropout3d=.8,
        learning_rate=.0005,
        dataset_seed=23,
        train_test_split=.9,
        flip_proba=.5,
        batch_size=30,
        num_workers=6,
        shuffle=True,
        nb_epochs_to_save=2,
        device='cuda',
    ):
        self.mode = mode.lower()
        self.in_channels = 2 if self.mode == 'concatenate' else 1
        self.linear_in = linear_in
        self.conv1_filters = conv1_filters
        self.conv2_filters = conv2_filters
        self.dropout3d = dropout3d

        self.learning_rate = learning_rate

        self.images_dir = images_dir
        self.csv_dir = csv_dir
        self.dataset_seed = dataset_seed
        self.train_test_split = train_test_split
        self.flip_proba = flip_proba
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
        return Tumor3DConv(
            self.in_channels,
            self.linear_in,
            self.conv1_filters,
            self.conv2_filters,
            self.dropout3d,
            conv_bias=True,
            conv_stride=2,
            conv_padding=0,
            linear_bias=True,
            regress=True,
        ).to(self.device)

    def train(self, epochs=100, epoch_to_restore=0):
        net = self.instantiate_generator()

        if epoch_to_restore == 0:
            self.mkdir()
            net.apply(weights_init)
        else:
            filename_model = self.dir_models / f'epoch_{epoch_to_restore}.pth'
            net.load_state_dict(torch.load(filename_model))

        dataset_train = Images3DDataset(
            self.images_dir, self.csv_dir, seed=self.dataset_seed,
            train_test_split=self.train_test_split, train_test="train",
            mode=self.mode, flip_proba=self.flip_proba)
        dataloader_train = DataLoader(
            dataset_train, self.batch_size, shuffle=self.shuffle,
            num_workers=self.num_workers, pin_memory=True)
        dataset_test = Images3DDataset(
            self.images_dir, self.csv_dir, seed=self.dataset_seed,
            train_test_split=self.train_test_split, train_test="test",
            mode=self.mode, flip_proba=self.flip_proba)
        dataloader_test = DataLoader(
            dataset_test, self.batch_size, shuffle=self.shuffle,
            num_workers=self.num_workers, pin_memory=True)

        criterion = torch.nn.MSELoss()

        optimizer = optim.Adam(net.parameters())
        writer_train = SummaryWriter(log_dir=str(self.logs_train_path))
        writer_test = SummaryWriter(log_dir=str(self.logs_test_path))

        writer_train.add_graph(net, dataset_train[0]['x'].unsqueeze(0).float().to(self.device))

        try:
            for epoch in range(epoch_to_restore + 1, epochs + epoch_to_restore + 1):
                start_time = time()

                net.train()
                for idx_batch, current_batch in enumerate(tqdm(
                    dataloader_train,
                    desc=f'Training for epoch {epoch}')
                ):
                    net.zero_grad()
                    x = Variable(current_batch['x']).float().to(self.device)
                    y = Variable(current_batch['y']).float().to(self.device)
                    prediction = net(x)
                    loss = criterion(prediction, y)
                    loss.backward()
                    optimizer.step()

                net.eval()
                with torch.no_grad():
                    train_mse_loss = AverageMeter()
                    train_cindex_metric = AverageMeter()
                    for idx_batch, current_batch in enumerate(
                        tqdm(dataloader_train,
                             desc='Evaluating model on training set')
                    ):
                        x = current_batch['x'].float().to(self.device)
                        y = current_batch['y'].float().to(self.device)
                        prediction = net(x)
                        loss = criterion(prediction, y)
                        # df_true = make_metric_dataframe(current_batch["info"], y.cpu().numpy())
                        # df_pred = make_metric_dataframe(
                        #     current_batch["info"], prediction.cpu().numpy())
                        # metric = cindex(df_true, df_pred)
                        train_mse_loss.update(loss)
                        # train_cindex_metric.update(metric)
                    writer_train.add_scalar('mse_loss/train', train_mse_loss.avg, epoch)
                    # writer_train.add_scalar('cindex/train', train_cindex_metric.avg, epoch)

                    test_mse_loss = AverageMeter()
                    test_cindex_metric = AverageMeter()
                    # for idx_batch, current_batch in\
                    #         enumerate(tqdm(dataloader_test, desc='Evaluating model on test set')):
                    #     x = current_batch['x'].float().to(self.device)
                    #     y = current_batch['y'].float().to(self.device)
                    #     prediction = net(x)
                    #     loss = criterion(prediction, y)
                    #     df_true = make_metric_dataframe(current_batch["info"], y.cpu().numpy())
                    #     df_pred = make_metric_dataframe(
                    #         current_batch["info"], prediction.cpu().numpy())
                    #     metric = cindex(df_true, df_pred)
                    #     test_mse_loss.update(loss)
                    #     test_cindex_metric.update(metric)
                    # writer_test.add_scalar('mse_loss/test', test_mse_loss.avg, epoch)
                    # writer_test.add_scalar('cindex/test', test_cindex_metric.avg, epoch)

                if epoch % self.nb_epochs_to_save == 0:
                    filename = self.models_path / f'epoch_{epoch}.pth'
                    torch.save(net.state_dict(), str(filename))

                end_time = time()
                print(f"[*] Finished epoch {epoch} in {get_hms(end_time - start_time)};\n"
                      f"Train Loss : {train_mse_loss.avg:.4f}\n")
                      # f"Test Loss : {test_mse_loss.avg:.4f}")

        finally:
            print('[*] Closing Writer.')
            writer_train.close()
            writer_test.close()
