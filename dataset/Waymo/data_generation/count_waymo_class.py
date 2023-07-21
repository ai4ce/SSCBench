from glob import glob
import os
import numpy as np
import yaml
import time
import argparse
import sys
from tqdm import tqdm

# Append root directory to system path for imports
repo_path, _ = os.path.split(os.path.realpath(__file__))
repo_path, _ = os.path.split(repo_path)
repo_path, _ = os.path.split(repo_path)
sys.path.append(repo_path)

import LMSCNet.data.io_data_waymo as WaymoIO


def parse_args():
  parser = argparse.ArgumentParser(description='LMSCNet labels lower scales creation')
  parser.add_argument(
    '--dset_root',
    dest='dataset_root',
    default='',
    metavar='DATASET',
    help='path to dataset root folder',
    type=str,
  )
  parser.add_argument('--start', type=int, default=0)
  parser.add_argument('--end', type=int, default=500)

  args = parser.parse_args()
  return args

def main():

  args = parse_args()

  dset_root = args.dataset_root
  yaml_path, _ = os.path.split(os.path.realpath(__file__))
  remap_lut = WaymoIO.get_remap_lut(os.path.join(yaml_path, 'waymo.yaml'))
  dataset_config = yaml.safe_load(open(os.path.join(yaml_path, 'waymo.yaml'), 'r'))
  sequences = sorted(glob(os.path.join(dset_root, 'dataset', 'sequences', '*')))
  # Selecting training/validation set sequences only (labels unavailable for test set)
  sequences = sequences[args.start:args.end]

  assert len(sequences) > 0, 'Error, no sequences on selected dataset root path'
  
  counts = [0] * (256)
  for sequence in sequences:

    label_paths  = sorted(glob(os.path.join(sequence, 'voxels', '*.npz')))
    invalid_paths = sorted(glob(os.path.join(sequence, 'voxels', '*.npz')))
    for i in tqdm(range(len(label_paths))):

      LABEL = WaymoIO._read_label_SemKITTI(label_paths[i])
      INVALID = WaymoIO._read_invalid_SemKITTI(invalid_paths[i])
      nonzero_idx = np.nonzero(LABEL)
      INVALID[nonzero_idx] = 0
      LABEL = remap_lut[LABEL.astype(np.uint16)].astype(
        np.float32
      )  # Remap 20 classes semanticKITTI SSC

      LABEL[np.isclose(INVALID, 1)] = 255  # Setting to unknown all voxels marked on invalid mask...
      LABEL = LABEL.reshape(-1)
      for label in LABEL:
        counts[int(label)] += 1
    print(f"sequence: {sequence}")
    print("counts: ", counts)
  print("Final_counts_class: ", counts)



if __name__ == '__main__':
  main()