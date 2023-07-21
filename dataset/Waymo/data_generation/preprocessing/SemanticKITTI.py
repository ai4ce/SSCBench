from torch.utils.data import Dataset
from glob import glob
import os
import numpy as np
import yaml
import random
import sys

import LMSCNet.data.io_data_waymo as WaymoIO
import LMSCNet.data.io_data_kitti as SemanticKittiIO

class SemanticKITTI_dataloader(Dataset):

  def __init__(self, dataset, phase):
    '''

    :param dataset: The dataset configuration (data augmentation, input encoding, etc)
    :param phase_tag: To differentiate between training, validation and test phase
    '''

    yaml_path, _ = os.path.split(os.path.realpath(__file__))
    self.dataset_config = yaml.safe_load(open(os.path.join(yaml_path, 'waymo.yaml'), 'r'))
    self.nbr_classes = self.dataset_config['nbr_classes']
    self.grid_dimensions = self.dataset_config['grid_dims']   # [W, H, D]
    self.remap_lut = self.get_remap_lut()
    self.rgb_mean = np.array([0.34749558, 0.36745213, 0.36123651])  # images mean:  [88.61137282 93.70029365 92.11530949]
    self.rgb_std = np.array([0.30599035, 0.3129534 , 0.31933814])   # images std:  [78.02753826 79.80311686 81.43122464]
    self.root_dir = dataset['ROOT_DIR']
    self.modalities = dataset['MODALITIES']
    self.extensions = {'3D_OCCUPANCY': '.npz', '3D_LABEL': '.npz', '3D_OCCLUDED': '.occluded',
                       '3D_INVALID': '.npz'}
    self.data_augmentation = {'FLIPS': dataset['AUGMENTATION']['FLIPS']}

    self.filepaths = {}
    self.phase = phase
    self.class_frequencies = np.array([17296567387, 308138138,808279,56739,34039981,1240016542,267882024,
                                       459304891,740573077,902947363,37827809])
    a=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416, 417, 418, 419, 420, 421, 422, 423, 424, 425, 426, 427, 428, 429, 430, 431, 432, 433, 434, 435, 436, 437, 438, 439, 440, 441, 442, 443, 444, 445, 446, 447, 448, 449, 450, 451, 452, 453, 454, 455, 456, 457, 458, 459, 460, 461, 462, 463, 464, 465, 466, 467, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477, 478, 479, 480, 481, 482, 483, 484, 485, 486, 487, 488, 489, 490, 491, 492, 493, 494, 495, 496, 497, 498, 499]
    b=[500, 501, 502, 503, 504, 505, 506, 507, 508, 509, 510, 511, 512, 513, 514, 515, 516, 517, 518, 519, 520, 521, 522, 523, 524, 525, 526, 527, 528, 529, 530, 531, 532, 533, 534, 535, 536, 537, 538, 539, 540, 541, 542, 543, 544, 545, 546, 547, 548, 549, 550, 551, 552, 553, 554, 555, 556, 557, 558, 559, 560, 561, 562, 563, 564, 565, 566, 567, 568, 569, 570, 571, 572, 573, 574, 575, 576, 577, 578, 579, 580, 581, 582, 583, 584, 585, 586, 587, 588, 589, 590, 591, 592, 593, 594, 595, 596, 597, 598, 599, 600, 601, 602, 603, 604, 605, 606, 607, 608, 609, 610, 611, 612, 613, 614, 615, 616, 617, 618, 619, 620, 621, 622, 623, 624, 625, 626, 627, 628, 629, 630, 631, 632, 633, 634, 635, 636, 637, 638, 639, 640, 641, 642, 643, 644, 645, 646, 647, 648, 649, 650, 651, 652, 653, 654, 655, 656, 657, 658, 659, 660, 661, 662, 663, 664, 665, 666, 667, 668, 669, 670, 671, 672, 673, 674, 675, 676, 677, 678, 679, 680, 681, 682, 683, 684, 685, 686, 687, 688, 689, 690, 691, 692, 693, 694, 695, 696, 697, 698, 699, 700, 701, 702, 703, 704, 705, 706, 707, 708, 709, 710, 711, 712, 713, 714, 715, 716, 717, 718, 719, 720, 721, 722, 723, 724, 725, 726, 727, 728, 729, 730, 731, 732, 733, 734, 735, 736, 737, 738, 739, 740, 741, 742, 743, 744, 745, 746, 747, 748, 749, 750, 751, 752, 753, 754, 755, 756, 757, 758, 759, 760, 761, 762, 763, 764, 765, 766, 767, 768, 769, 770, 771, 772, 773, 774, 775, 776, 777, 778, 779, 780, 781, 782, 783, 784, 785, 786, 787, 788, 789, 790, 791, 792, 793, 794, 795, 796, 797]
    c=[798, 799, 800, 801, 802, 803, 804, 805, 806, 807, 808, 809, 810, 811, 812, 813, 814, 815, 816, 817, 818, 819, 820, 821, 822, 823, 824, 825, 826, 827, 828, 829, 830, 831, 832, 833, 834, 835, 836, 837, 838, 839, 840, 841, 842, 843, 844, 845, 846, 847, 848, 849, 850, 851, 852, 853, 854, 855, 856, 857, 858, 859, 860, 861, 862, 863, 864, 865, 866, 867, 868, 869, 870, 871, 872, 873, 874, 875, 876, 877, 878, 879, 880, 881, 882, 883, 884, 885, 886, 887, 888, 889, 890, 891, 892, 893, 894, 895, 896, 897, 898, 899, 900, 901, 902, 903, 904, 905, 906, 907, 908, 909, 910, 911, 912, 913, 914, 915, 916, 917, 918, 919, 920, 921, 922, 923, 924, 925, 926, 927, 928, 929, 930, 931, 932, 933, 934, 935, 936, 937, 938, 939, 940, 941, 942, 943, 944, 945, 946, 947, 948, 949, 950, 951, 952, 953, 954, 955, 956, 957, 958, 959, 960, 961, 962, 963, 964, 965, 966, 967, 968, 969, 970, 971, 972, 973, 974, 975, 976, 977, 978, 979, 980, 981, 982, 983, 984, 985, 986, 987, 988, 989, 990, 991, 992, 993, 994, 995, 996, 997, 998, 999]

    self.split = {'train': a, 'val': b,
                  'test': c}

    for modality in self.modalities:
      if self.modalities[modality]:
        self.get_filepaths(modality)

    # if self.phase != 'test':
    #   self.check_same_nbr_files()

    self.nbr_files = len(self.filepaths['3D_OCCUPANCY'])  # TODO: Pass to something generic

    return

  def get_filepaths(self, modality):
    '''
    Set modality filepaths with split according to phase (train, val, test)
    '''

    sequences = list(sorted(glob(os.path.join(self.root_dir, 'dataset', 'sequences', '*')))[i] for i in self.split[self.phase])
    if modality == '3D_LABEL':
      self.filepaths['3D_LABEL'] = {'1_1': [], '1_2': [], '1_4': [], '1_8': []}
      self.filepaths['3D_INVALID'] = {'1_1': [], '1_2': [], '1_4': [], '1_8': []}
      for sequence in sequences:
        assert len(os.listdir(sequence)) > 0, 'Error, No files in sequence: {}'.format(sequence)
        # Scale 1:1
        self.filepaths['3D_LABEL']['1_1'] += sorted(glob(os.path.join(sequence, 'voxels', '*.npz')))
        self.filepaths['3D_INVALID']['1_1'] += sorted(glob(os.path.join(sequence, 'voxels', '*.npz')))
        # # Scale 1:2
        # self.filepaths['3D_LABEL']['1_2'] += sorted(glob(os.path.join(sequence, 'voxels', '*.label_1_2')))
        # self.filepaths['3D_INVALID']['1_2'] += sorted(glob(os.path.join(sequence, 'voxels', '*.invalid_1_2')))
        # # Scale 1:4
        # self.filepaths['3D_LABEL']['1_4'] += sorted(glob(os.path.join(sequence, 'voxels', '*.label_1_4')))
        # self.filepaths['3D_INVALID']['1_4'] += sorted(glob(os.path.join(sequence, 'voxels', '*.invalid_1_4')))
        # # Scale 1:8
        # self.filepaths['3D_LABEL']['1_8'] += sorted(glob(os.path.join(sequence, 'voxels', '*.label_1_8')))
        # self.filepaths['3D_INVALID']['1_8'] += sorted(glob(os.path.join(sequence, 'voxels', '*.invalid_1_8')))

    # if modality == '3D_OCCLUDED':
    #   self.filepaths['3D_OCCLUDED'] = []
    #   for sequence in sequences:
    #     assert len(os.listdir(sequence)) > 0, 'Error, No files in sequence: {}'.format(sequence)
    #     self.filepaths['3D_OCCLUDED'] += sorted(glob(os.path.join(sequence, 'voxels', '*.occluded')))

    if modality == '3D_OCCUPANCY':
      self.filepaths['3D_OCCUPANCY'] = []
      for sequence in sequences:
        # assert len(os.listdir(sequence)) > 0, 'Error, No files in sequence: {}'.format(sequence)
        self.filepaths['3D_OCCUPANCY'] += sorted(glob(os.path.join(sequence, 'voxels', '*.npz')))

    # if modality == '2D_RGB':
    #   self.filepaths['2D_RGB'] = []
    #   for sequence in sequences:
    #     assert len(os.listdir(sequence)) > 0, 'Error, No files in sequence: {}'.format(sequence)
    #     self.filepaths['2D_RGB'] += sorted(glob(os.path.join(sequence, 'image_2', '*.png')))[::5]

    return

  def check_same_nbr_files(self):
    '''
    Set modality filepaths with split according to phase (train, val, test)
    '''

    # TODO: Modify for nested dictionaries...
    for i in range(len(self.filepaths.keys()) - 1):
      length1 = len(self.filepaths[list(self.filepaths.keys())[i]])
      length2 = len(self.filepaths[list(self.filepaths.keys())[i+1]])
      assert length1 == length2, 'Error: {} and {} not same number of files'.format(list(self.filepaths.keys())[i],
                                                                                    list(self.filepaths.keys())[i+1])
    return

  def __getitem__(self, idx):
    '''

    '''

    data = {}

    do_flip = 0
    if self.data_augmentation['FLIPS'] and self.phase == 'train':
      do_flip = random.randint(0, 3)

    for modality in self.modalities:
      if (self.modalities[modality]) and (modality in self.filepaths):
        data[modality] = self.get_data_modality(modality, idx, do_flip)

    return data, idx

  def get_data_modality(self, modality, idx, flip):

    if modality == '3D_OCCUPANCY':
      OCCUPANCY = WaymoIO._read_occupancy_SemKITTI(self.filepaths[modality][idx])
      OCCUPANCY = np.moveaxis(OCCUPANCY.reshape([self.grid_dimensions[0],
                                                 self.grid_dimensions[2],
                                                 self.grid_dimensions[1]]), [0, 1, 2], [0, 2, 1])
      OCCUPANCY = WaymoIO.data_augmentation_3Dflips(flip, OCCUPANCY)
      return OCCUPANCY[None, :, :, :]

    elif modality == '3D_LABEL':
      LABEL_1_1 = WaymoIO.data_augmentation_3Dflips(flip, self.get_label_at_scale('1_1', idx))
      # LABEL_1_2 = SemanticKittiIO.data_augmentation_3Dflips(flip, self.get_label_at_scale('1_2', idx))
      # LABEL_1_4 = SemanticKittiIO.data_augmentation_3Dflips(flip, self.get_label_at_scale('1_4', idx))
      # LABEL_1_8 = SemanticKittiIO.data_augmentation_3Dflips(flip, self.get_label_at_scale('1_8', idx))
      return {'1_1': LABEL_1_1}

    # elif modality == '3D_OCCLUDED':
    #   OCCLUDED = SemanticKittiIO._read_occluded_SemKITTI(self.filepaths[modality][idx])
    #   OCCLUDED = np.moveaxis(OCCLUDED.reshape([self.grid_dimensions[0],
    #                                            self.grid_dimensions[2],
    #                                            self.grid_dimensions[1]]), [0, 1, 2], [0, 2, 1])
    #   OCCLUDED = SemanticKittiIO.data_augmentation_3Dflips(flip, OCCLUDED)
    #   return OCCLUDED

    # elif modality == '2D_RGB':
    #   RGB = SemanticKittiIO._read_rgb_SemKITTI(self.filepaths[modality][idx])
    #   # TODO Standarize, Normalize
    #   RGB = SemanticKittiIO.img_normalize(RGB, self.rgb_mean, self.rgb_std)
    #   RGB = np.moveaxis(RGB, (0, 1, 2), (1, 2, 0)).astype(dtype='float32')  # reshaping [3xHxW]
    #   # There is a problem on the RGB images.. They are not all the same size and I used those to calculate the mapping
    #   # for the sketch... I need images all te same size..
    #   return RGB

    else:
      assert False, 'Specified modality not found'

  def get_label_at_scale(self, scale, idx):

    scale_divide = int(scale[-1])

    INVALID = WaymoIO._read_invalid_SemKITTI(self.filepaths['3D_INVALID'][scale][idx])
    LABEL = WaymoIO._read_label_SemKITTI(self.filepaths['3D_LABEL'][scale][idx])

    if scale == '1_1':
      LABEL = self.remap_lut[LABEL.astype(np.uint8)].astype(np.float32)  # Remap 20 classes semanticKITTI SSC
    LABEL[np.isclose(INVALID, 1)] = 255  # Setting to unknown all voxels marked on invalid mask...
    LABEL = np.moveaxis(LABEL.reshape([int(self.grid_dimensions[0] / scale_divide),
                                       int(self.grid_dimensions[2] / scale_divide),
                                       int(self.grid_dimensions[1] / scale_divide)]), [0, 1, 2], [0, 2, 1])

    return LABEL

  def read_semantics_config(self, data_path):

    # get number of interest classes, and the label mappings
    DATA = yaml.safe_load(open(data_path, 'r'))
    self.class_strings = DATA["labels"]
    self.class_remap = DATA["learning_map"]
    self.class_inv_remap = DATA["learning_map_inv"]
    self.class_ignore = DATA["learning_ignore"]
    self.n_classes = len(self.class_inv_remap)

    return

  def get_inv_remap_lut(self):
    '''
    remap_lut to remap classes of semantic kitti for training...
    :return:
    '''

    # make lookup table for mapping
    maxkey = max(self.dataset_config['learning_map_inv'].keys())

    # +100 hack making lut bigger just in case there are unknown labels
    remap_lut = np.zeros((maxkey + 1), dtype=np.int32)
    remap_lut[list(self.dataset_config['learning_map_inv'].keys())] = list(self.dataset_config['learning_map_inv'].values())

    return remap_lut

  def get_remap_lut(self):
    '''
    remap_lut to remap classes of semantic kitti for training...
    :return:
    '''

    # make lookup table for mapping
    maxkey = max(self.dataset_config['learning_map'].keys())

    # +100 hack making lut bigger just in case there are unknown labels
    remap_lut = np.zeros((maxkey + 100), dtype=np.int32)
    remap_lut[list(self.dataset_config['learning_map'].keys())] = list(self.dataset_config['learning_map'].values())

    # in completion we have to distinguish empty and invalid voxels.
    # Important: For voxels 0 corresponds to "empty" and not "unlabeled".
    remap_lut[remap_lut == 0] = 255  # map 0 to 'invalid'
    remap_lut[0] = 0  # only 'empty' stays 'empty'.

    return remap_lut

  def __len__(self):
    """
    Returns the length of the dataset
    """
    # Return the number of elements in the dataset
    return self.nbr_files
