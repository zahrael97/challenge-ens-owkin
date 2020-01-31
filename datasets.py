import random
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from utils import get_id_from_path, Convert


class Images3DDataset(Dataset):

    def __init__(
        self,
        images_dir,
        csv_dir=None,
        seed=23,
        train_test_split=0.9,
        train_test="train",
        mode='concatenate',
        flip_proba=0.5,
    ):
        self.images_dir = Path(images_dir).expanduser()
        self.csv_dir = Path(csv_dir).expanduser()
        self.seed = seed
        self.train_test_split = train_test_split
        self.train_test = train_test.lower()
        assert self.train_test in ["train", "test"]
        self.mode = mode.lower()
        assert self.mode in ["concatenate", "scan", "mask"]
        self.flip_proba = flip_proba
        assert flip_proba <= 1.

        paths = sorted(glob(str(self.images_dir / "*.*")))
        size = len(paths)
        random.seed(self.seed)
        test_idx = random.sample(range(size), int(size * (1 - self.train_test_split)))
        train_idx = list(set(range(size)) - set(test_idx))
        iterator = train_idx if self.train_test == 'train' else test_idx
        self.paths = sorted([paths[i] for i in iterator])
        self.__len = len(self.paths)

        if self.csv_dir:
            self.ground_truth = pd.read_csv(self.csv_dir)

            trains_ids = [int(get_id_from_path(paths[i])) for i in train_idx]
            train_survival_time = self.ground_truth.loc[
                self.ground_truth.PatientID.isin(trains_ids),
                'SurvivalTime']
            self.scaler = StandardScaler().fit(train_survival_time.values.reshape(-1, 1))

        if self.mode != "concatenate":
            transform = [
                Convert(),
                T.RandomHorizontalFlip(self.flip_proba),
                T.RandomVerticalFlip(self.flip_proba),
                T.ToTensor(),
            ]
            if self.mode == 'scan':
                transform.append(T.Lambda(lambda x: x + torch.randn_like(x)),)
            self.transform = T.Compose(transform)

    def __len__(self):
        return self.__len

    def __getitem__(self, idx, seed=None):
        path = self.paths[idx]

        scanner = np.load(path)
        scan, mask = scanner["scan"], scanner["mask"]

        if self.train_test == "train":
            random.seed(seed)
            v_flip = random.random() > self.flip_proba
            h_flip = random.random() > self.flip_proba

            if self.mode == 'concatenate':
                scans = []
                masks = []
                to_pil = Convert()
                to_tens = T.ToTensor()
                for img, msk in zip(scan, mask):
                    img, msk = to_pil(img), to_pil(np.uint8(msk))
                    if v_flip:
                        img = TF.hflip(img)
                        msk = TF.hflip(msk)
                    if h_flip:
                        img = TF.vflip(img)
                        msk = TF.vflip(msk)
                    img, msk = to_tens(img).unsqueeze(0).float(), to_tens(msk).unsqueeze(0).float()
                    noise = torch.randn_like(img)
                    img += noise
                    scans.append(img)
                    masks.append(msk)
                scans = torch.cat(scans, dim=1)
                masks = torch.cat(masks, dim=1)
                output = torch.cat([scans, masks], dim=0)

            elif self.mode == 'mask':
                output = np.expand_dims(mask, axis=1)
                output = torch.cat([self.transform(x) for x in output], dim=0)
            elif self.mode == 'scan':
                output = np.expand_dims(scan, axis=1)
                output = torch.cat([self.transform(x) for x in output], dim=0)

        else:
            if self.mode == 'mask':
                output = torch.Tensor(mask).unsqueeze(1).float()
            elif self.mode == 'scan':
                output = torch.Tensor(scan).unsqueeze(1).float()
            elif self.mode == 'concatenate':
                scan = torch.Tensor(scan).unsqueeze(0).float()
                mask = torch.Tensor(mask).unsqueeze(0).float()
                output = torch.cat([scan, mask], dim=0)

        output = {"x": output}
        if self.csv_dir:
            patient_id = get_id_from_path(path)
            patient_info = self.ground_truth.loc[
                self.ground_truth.PatientID == int(patient_id)]
            output['y'] = self.scaler.transform([[patient_info.iloc[0, 1]]])[0]
            output['info'] = np.array([int(patient_id), int(patient_info.iloc[0, 2])])

        return output
