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

from config import DROP_COLS, CAT_COLS, REG_COLS
from utils import get_id_from_path, Convert
from transformer import PandasOneHotEncoder, PandasScaler


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
        noise_magnitude=None,
        **kwargs,
    ):
        self.images_dir = Path(images_dir).expanduser()
        self.seed = seed
        self.train_test_split = train_test_split
        self.train_test = train_test.lower()
        assert self.train_test in ["train", "test"]
        self.mode = mode.lower()
        assert self.mode in ["concatenate", "scan", "mask"]
        self.flip_proba = flip_proba
        assert flip_proba <= 1.
        self.noise_magnitude = noise_magnitude

        paths = sorted(glob(str(self.images_dir / "*.*")))
        size = len(paths)
        random.seed(self.seed)
        test_idx = random.sample(range(size), int(size * (1 - self.train_test_split)))
        train_idx = list(set(range(size)) - set(test_idx))
        iterator = train_idx if self.train_test == 'train' else test_idx
        self.paths = sorted([paths[i] for i in iterator])
        self.__len = len(self.paths)

        if csv_dir:
            self.csv_dir = Path(csv_dir).expanduser()
            self.ground_truth = pd.read_csv(self.csv_dir)

            train_ids = [int(get_id_from_path(paths[i])) for i in train_idx]
            train_survival_time = self.ground_truth.loc[
                self.ground_truth.PatientID.isin(train_ids),
                'SurvivalTime']
            self.scaler = StandardScaler().fit(train_survival_time.values.reshape(-1, 1))

        if self.mode != "concatenate":
            transform = [
                Convert(),
                T.RandomHorizontalFlip(self.flip_proba),
                T.RandomVerticalFlip(self.flip_proba),
                T.ToTensor(),
            ]
            if self.mode == 'scan' and self.noise_magnitude is not None:
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
            v_flip = random.random() < self.flip_proba
            h_flip = random.random() < self.flip_proba

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
                    if self.noise_magnitude is not None:
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

        output = {"images": output}
        patient_id = int(get_id_from_path(path))
        if self.csv_dir:
            patient_info = self.ground_truth.loc[
                self.ground_truth.PatientID == patient_id]
            output['y'] = self.scaler.transform([[patient_info.iloc[0, 1]]])[0]
            output['info'] = np.array([patient_id, int(patient_info.iloc[0, 2])])
        else:
            output['info'] = np.array([patient_id, 1.])

        return output


class ClinicalDataset(Dataset):

    def __init__(
        self,
        data_path,
        id_path,
        csv_dir=None,
        infer_path=None,
        seed=23,
        train_test_split=0.9,
        train_test="train",
        **kwargs,
    ):
        self.data_path = Path(data_path).expanduser()
        self.id_path = Path(id_path).expanduser()
        self.seed = seed
        self.train_test_split = train_test_split
        self.train_test = train_test.lower()
        assert self.train_test in ["train", "test"]

        paths = sorted(glob(str(self.id_path / "*.*")))
        size = len(paths)
        random.seed(self.seed)
        test_idx = random.sample(range(size), int(size * (1 - self.train_test_split)))
        train_idx = list(set(range(size)) - set(test_idx))
        iterator = train_idx if self.train_test == 'train' else test_idx
        self.ids = [int(get_id_from_path(paths[i])) for i in iterator]
        clinical_df = pd.read_csv(self.data_path).drop(columns=DROP_COLS)
        self.paths = sorted([paths[i] for i in iterator])
        self.__len = len(self.ids)

        train_ids = [int(get_id_from_path(paths[i])) for i in train_idx]
        train_input = clinical_df[clinical_df.PatientID.isin(train_ids)]
        self.encoder = PandasOneHotEncoder(CAT_COLS).fit(train_input)
        self.input_scaler = PandasScaler(REG_COLS).fit(train_input)

        if infer_path is None:
            clinical_df = (
                clinical_df[clinical_df.PatientID.isin(self.ids)] if self.train_test == 'test'
                else train_input)
            clinical_df = self.encoder.transform(clinical_df)
            clinical_df = self.input_scaler.transform(clinical_df)
            self.df = clinical_df.copy()
        else:
            infer_path = Path(infer_path).expanduser()
            infer_df = pd.read_csv(infer_path)
            infer_df = self.encoder.transform(infer_df)
            infer_df = self.input_scaler.transform(infer_df)
            self.df = infer_df.copy()

        if csv_dir:
            self.csv_dir = Path(csv_dir).expanduser()
            ground_truth = pd.read_csv(self.csv_dir)

            train_truth = ground_truth[ground_truth.PatientID.isin(train_ids)]
            self.scaler = StandardScaler().fit(train_truth.SurvivalTime.values.reshape(-1, 1))

            ground_truth = (
                ground_truth[ground_truth.PatientID.isin(self.ids)] if self.train_test == 'test'
                else train_truth)
            self.df = self.df.merge(ground_truth)

    def __len__(self,):
        return self.__len

    def __getitem__(self, idx):
        patient_id = int(get_id_from_path(self.paths[idx]))
        output = dict()

        patient_info = self.df.loc[self.df.PatientID == patient_id]
        output['clinical'] = torch.Tensor(patient_info[REG_COLS + self.encoder.cols].values[0])

        if self.csv_dir:
            output['y'] =\
                self.scaler.transform(patient_info.SurvivalTime.values.reshape(1, 1))[0][0]
            output['info'] = np.array([patient_id, int(patient_info.Event.values[0])])
        else:
            output['info'] = [patient_id, 1.]

        return output


class MultiModalDataset(Dataset):

    def __init__(
        self,
        image_dataset_kwargs,
        clinical_dataset_kwargs,
        train_test="train",
        train_test_split=.9,
        **kwargs,
    ):
        assert image_dataset_kwargs["seed"] == clinical_dataset_kwargs["seed"]
        self.train_test = train_test.lower()
        assert self.train_test in ["train", "test"]
        image_dataset_kwargs["train_test"] = self.train_test
        clinical_dataset_kwargs["train_test"] = self.train_test
        image_dataset_kwargs["train_test_split"] = train_test_split
        clinical_dataset_kwargs["train_test_split"] = train_test_split
        self.image_dataset = Images3DDataset(**image_dataset_kwargs)
        self.clinical_dataset = ClinicalDataset(**clinical_dataset_kwargs)
        assert len(self.image_dataset) == len(self.clinical_dataset)
        self.scaler = self.image_dataset.scaler

    def __len__(self):
        return len(self.image_dataset)

    def __getitem__(self, idx):
        images = self.image_dataset[idx]
        clinical = self.clinical_dataset[idx]
        assert images['info'][0] == clinical['info'][0]
        output = dict()
        output["images"] = images["images"]
        output["clinical"] = clinical["clinical"]
        output["y"] = images["y"]
        output["info"] = images["info"]
        return output


if __name__ == '__main__':
    d = Images3DDataset(
        "~/datasets/tumor/x_train/images/",
        csv_dir="~/datasets/tumor/y_train.csv")
    _ = d[0]

    d = ClinicalDataset(
         "~/datasets/tumor/x_train/features/clinical_data.csv",
         "~/datasets/tumor/x_train/images/",
         csv_dir="~/datasets/tumor/y_train.csv")
    _ = d[0]

    d = MultiModalDataset(
        {"images_dir": "~/datasets/tumor/x_train/images/",
         "csv_dir": "~/datasets/tumor/y_train.csv",
         "seed": 23, "train_test_split": .9, "train_test": "train"},
        {"data_path": "~/datasets/tumor/x_train/features/clinical_data.csv",
         "id_path": "~/datasets/tumor/x_train/images/",
         "csv_dir": "~/datasets/tumor/y_train.csv",
         "seed": 23, "train_test_split": .9, "train_test": "train"}
    )
    _ = d[0]
