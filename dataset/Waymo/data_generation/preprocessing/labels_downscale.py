from glob import glob
import os
import numpy as np
import yaml
import time
import argparse
import sys

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
  parser.add_argument('--end', type=int, default=10)
  args = parser.parse_args()
  return args


def majority_pooling(grid, k_size=2):
  result = np.zeros((grid.shape[0] // k_size, grid.shape[1] // k_size, grid.shape[2] // k_size))
  for xx in range(0, int(np.floor(grid.shape[0]/k_size))):
    for yy in range(0, int(np.floor(grid.shape[1]/k_size))):
      for zz in range(0, int(np.floor(grid.shape[2]/k_size))):

        sub_m = grid[(xx*k_size):(xx*k_size)+k_size, (yy*k_size):(yy*k_size)+k_size, (zz*k_size):(zz*k_size)+k_size]
        unique, counts = np.unique(sub_m, return_counts=True)
        if True in ((unique != 0) & (unique != 255)):
          # Remove counts with 0 and 255
          counts = counts[((unique != 0) & (unique != 255))]
          unique = unique[((unique != 0) & (unique != 255))]
        else:
          if True in (unique == 0):
            counts = counts[(unique != 255)]
            unique = unique[(unique != 255)]
        value = unique[np.argmax(counts)]
        result[xx, yy, zz] = value
  return result


def downscale_data(LABEL, downscaling):
    # Majority pooling labels downscaled in 3D
    LABEL = majority_pooling(LABEL, k_size=downscaling)
    # Reshape to 1D
    LABEL = np.moveaxis(LABEL, [0, 1, 2], [0, 2, 1]).reshape(-1)
    # Invalid file downscaled
    INVALID = np.zeros_like(LABEL)
    INVALID[np.isclose(LABEL, 255)] = 1
    return LABEL, INVALID


def main():

  args = parse_args()

  dset_root = args.dataset_root
  yaml_path, _ = os.path.split(os.path.realpath(__file__))
  remap_lut = WaymoIO.get_remap_lut(os.path.join(yaml_path, 'waymo.yaml'))
  dataset_config = yaml.safe_load(open(os.path.join(yaml_path, 'waymo.yaml'), 'r'))
  sequences = sorted(glob(os.path.join(dset_root, 'dataset', 'sequences', '*')))
  # Selecting training/validation set sequences only (labels unavailable for test set)
  sequences = sequences[args.start:args.end]
  grid_dimensions = dataset_config['grid_dims']   # [W, H, D]

  assert len(sequences) > 0, 'Error, no sequences on selected dataset root path'

  for sequence in sequences:

    label_paths  = sorted(glob(os.path.join(sequence, 'voxels', '*.npz')))
    invalid_paths = sorted(glob(os.path.join(sequence, 'voxels', '*.npz')))
    out_dir = os.path.join(sequence, 'voxels')
    downscaling = {'1_2': 2, '1_4': 4, '1_8': 8}

    for i in range(len(label_paths)):

      filename, extension = os.path.splitext(os.path.basename(label_paths[i]))

      LABEL = WaymoIO._read_label_SemKITTI(label_paths[i])
      INVALID = WaymoIO._read_invalid_SemKITTI(invalid_paths[i])
      LABEL = remap_lut[LABEL.astype(np.uint16)].astype(np.float32)  # Remap 20 classes semanticKITTI SSC
      LABEL[np.isclose(INVALID, 1)] = 255  # Setting to unknown all voxels marked on invalid mask...
      LABEL = np.moveaxis(LABEL.reshape([grid_dimensions[0], grid_dimensions[2], grid_dimensions[1]]),
                          [0, 1, 2], [0, 2, 1])   # [256, 32, 256]

      for scale in downscaling:

        label_filename = os.path.join(out_dir, filename + '.label_' + scale)
        invalid_filename = os.path.join(out_dir, filename + '.invalid_' + scale)
        # If files have not been created...
        if not (os.path.isfile(label_filename) & os.path.isfile(invalid_filename)):
          LABEL_ds, INVALID_ds = downscale_data(LABEL, downscaling[scale])
          WaymoIO.pack(INVALID_ds.astype(dtype=np.uint8)).tofile(invalid_filename)
          print(time.strftime('%x %X') + ' -- => File {} - Sequence {} saved...'.format(filename + '.label_' + scale, os.path.basename(sequence)))
          LABEL_ds.astype(np.uint16).tofile(label_filename)
          print(time.strftime('%x %X') + ' -- => File {} - Sequence {} saved...'.format(filename + '.invalid_' + scale, os.path.basename(sequence)))

    print(time.strftime('%x %X') + ' -- => All files saved for Sequence {}'.format(os.path.basename(sequence)))

  print(time.strftime('%x %X') + ' -- => All files saved')

  exit()


if __name__ == '__main__':
  main()