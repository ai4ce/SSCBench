import torch


def collate_fn(batch):
    data = {}
    imgs = []
    frame_ids = []
    img_paths = []
    sequences = []
    CP_mega_matrices = []
    targets = []

    cam_ks = []
    T_velo_2_cams = []

    frustums_masks = []
    frustums_class_dists = []

    scale_3ds = batch[0]["scale_3ds"]
    for scale_3d in scale_3ds:
        data["projected_pix_{}".format(scale_3d)] = []
        data["fov_mask_{}".format(scale_3d)] = []

    for _, input_dict in enumerate(batch):
        if "img_path" in input_dict:
            img_paths.append(input_dict["img_path"])

        for key in data:
            data[key].append(torch.from_numpy(input_dict[key]))

        cam_ks.append(torch.from_numpy(input_dict["cam_k"]).float())
        T_velo_2_cams.append(torch.from_numpy(input_dict["T_velo_2_cam"]).float())

        if "frustums_masks" in input_dict:
            frustums_masks.append(torch.from_numpy(input_dict["frustums_masks"]))
            frustums_class_dists.append(
                torch.from_numpy(input_dict["frustums_class_dists"]).float()
            )

        sequences.append(input_dict["sequence"])

        img = input_dict["img"]
        imgs.append(img)
        target = torch.from_numpy(input_dict["target"])
        targets.append(target)
        CP_mega_matrices.append(torch.from_numpy(input_dict["CP_mega_matrix"]))         
        frame_ids.append(input_dict["frame_id"])

    ret_data = {
        "sequence": sequences,
        "frame_id": frame_ids,
        "frustums_class_dists": frustums_class_dists,
        "frustums_masks": frustums_masks,
        "cam_k": cam_ks,
        "T_velo_2_cam": T_velo_2_cams,
        "img": torch.stack(imgs),
        "img_path": img_paths,
        "CP_mega_matrices": CP_mega_matrices,
        "target": torch.stack(targets)
    }
    for key in data:
        ret_data[key] = data[key]

    return ret_data
