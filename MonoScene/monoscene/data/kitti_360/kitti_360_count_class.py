"""
Code partly taken from https://github.com/cv-rits/LMSCNet/blob/main/LMSCNet/data/labels_downscale.py
"""
import numpy as np
from tqdm import tqdm
import numpy.matlib
import os
import glob
import hydra
from omegaconf import DictConfig
import monoscene.data.kitti_360.io_data as SemanticKittiIO
from hydra.utils import get_original_cwd
from monoscene.data.NYU.preprocess import _downsample_label


@hydra.main(config_name="../../config/monoscene.yaml")
def main(config: DictConfig):
    scene_size = (256, 256, 32)
    #sequences = ["10"]
    sequences = ["03", "10", "00", "02", "04", "05", "07"]
    remap_lut = SemanticKittiIO.get_remap_lut(
        os.path.join(
            get_original_cwd(),
            "monoscene",
            "data",
            "kitti_360",
            "kitti_360.yaml",
        )
    )
    counts = [0] * (256)
    for sequence in sequences:
        sequence_path = os.path.join(
            config.kitti_360_root, "data_2d_raw", "2013_05_28_drive_00" + sequence + "_sync"
        )
        # print("sequence_path: ", sequence_path)
        label_paths = sorted(
            glob.glob(os.path.join(sequence_path, "voxels", "*.label"))
        )
        invalid_paths = sorted(
            glob.glob(os.path.join(sequence_path, "voxels", "*.invalid"))
        )
        for i in tqdm(range(len(label_paths))):

            frame_id, extension = os.path.splitext(os.path.basename(label_paths[i]))

            LABEL = SemanticKittiIO._read_label_KITTI360(label_paths[i])
            INVALID = SemanticKittiIO._read_invalid_KITTI360(invalid_paths[i])
            nonzero_idx = np.nonzero(LABEL)
            INVALID[nonzero_idx] = 0

            LABEL = remap_lut[LABEL.astype(np.uint16)].astype(
                np.float32
            )  # Remap 20 classes semanticKITTI SSC

            LABEL[np.isclose(INVALID, 1)] = 255  # Setting to unknown all voxels marked on invalid mask...
            for label in LABEL:
                counts[int(label)] += 1
        print("counts: ", counts)
    print("Final_counts_class: ", counts)




if __name__ == "__main__":
    main()
