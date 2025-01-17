from typing import Any, Iterable, Optional, Tuple

import numpy as np
import yaml
from pydantic import BaseModel, validator

from tools_generate.data import get_skeletonised_ds


class Config(BaseModel):
    # user input in .yaml file
    img_length: int
    input_channels: int
    output_channels: int

    data_path: str
    save_path: str

    img_dims_data_folder: int
    process_image_dim: int
    grey_scale_process_folder: str

    validation_fraction: float
    batch_size: int
    use_small_dataset: bool = False
    max_files: Optional[int] = None

    # generated from user input
    img_dims: Optional[Tuple[int, int]] = None

    dataset: Any = None

    num_labels: Optional[int] = None
    num_validation: Optional[int] = None
    num_train: Optional[int] = None

    train_ids: Optional[Iterable] = None
    validation_ids: Optional[Iterable] = None

    steps_per_epoch: Optional[int]

    @validator("max_files")
    def check_max_files(cls, v, values):
        if values["use_small_dataset"] is True:
            return int(v)
        else:
            return None

    def __init__(self, filepath: str):
        with open(filepath) as f:
            data = yaml.load(f, Loader=yaml.FullLoader)

        super(Config, self).__init__(**data)

        self.img_dims = (self.img_length, self.img_length)

        # training vs validation
        self.dataset = self.create_dataset()
        self.num_labels = len(self.dataset)
        self.num_validation = int(self.validation_fraction * self.num_labels)
        self.num_train = self.num_labels - self.num_validation

        self.train_ids = np.arange(self.num_train)
        self.validation_ids = np.arange(self.num_train, self.num_labels)

        # for training
        self.steps_per_epoch = int(self.num_labels / self.batch_size)

        print(
            f"Total: {self.num_labels} training data --",
            f"[{self.num_train} training]",
            f"[{self.num_validation} validation]",
        )

    def create_dataset(self):
        dataset = get_skeletonised_ds(self.data_path, seed=None)
        if self.use_small_dataset:
            dataset = dataset.take(self.max_files)
        return dataset

    @property
    def training_ds(self):
        return self.dataset.take(self.num_train)

    @property
    def validation_ds(self):
        return self.dataset.skip(self.num_train)
