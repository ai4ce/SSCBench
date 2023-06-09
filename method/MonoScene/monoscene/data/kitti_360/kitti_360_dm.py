from torch.utils.data.dataloader import DataLoader
from monoscene.data.kitti_360.kitti_360_dataset import Kitti360Dataset
import pytorch_lightning as pl
from monoscene.data.kitti_360.collate import collate_fn
from monoscene.data.utils.torch_util import worker_init_fn


class Kitti360DataModule(pl.LightningDataModule):
    def __init__(
        self,
        root, 
        # n_scans, 
        # sequences,
        preprocess_root,
        project_scale=2,
        frustum_size=4,
        batch_size=4, 
        num_workers=6,
    ):

        super().__init__()
        self.root = root
        self.preprocess_root = preprocess_root
        self.project_scale = project_scale
        self.batch_size = batch_size
        self.num_workers = num_workers
        # self.sequences = sequences
        # self.n_scans = n_scans
        self.frustum_size = frustum_size

    def setup(self, stage=None):
        # self.ds = Kitti360Dataset(
        #     root=self.root, sequences=self.sequences, n_scans=self.n_scans
        # )
        self.train_ds = Kitti360Dataset(
            split="train",
            root=self.root,
            preprocess_root=self.preprocess_root,
            project_scale=self.project_scale,
            frustum_size=self.frustum_size,
            fliplr=0.5,
            color_jitter=(0.4, 0.4, 0.4),
        )

        self.val_ds = Kitti360Dataset(
            split="val",
            root=self.root,
            preprocess_root=self.preprocess_root,
            project_scale=self.project_scale,
            frustum_size=self.frustum_size,
            fliplr=0,
            color_jitter=None,
        )

        self.test_ds = Kitti360Dataset(
            split="test",
            root=self.root,
            preprocess_root=self.preprocess_root,
            project_scale=self.project_scale,
            frustum_size=self.frustum_size,
            fliplr=0,
            color_jitter=None,
        )
    # def dataloader(self):
    #     return DataLoader(
    #         self.ds,
    #         batch_size=self.batch_size,
    #         drop_last=False,
    #         num_workers=self.num_workers,
    #         shuffle=False,
    #         pin_memory=True,
    #         worker_init_fn=worker_init_fn,
    #         collate_fn=collate_fn,
    #     )
    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            drop_last=True,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
            worker_init_fn=worker_init_fn,
            collate_fn=collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            drop_last=False,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            worker_init_fn=worker_init_fn,
            collate_fn=collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            drop_last=False,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            worker_init_fn=worker_init_fn,
            collate_fn=collate_fn,
        )
