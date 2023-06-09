import torch
import os
import glob
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from torchvision import transforms
from monoscene.data.utils.helpers import (
    vox2pix,
    compute_local_frustums,
    compute_CP_mega_matrix,
)


class WaymoDataset(Dataset):
    def __init__(
        self,
        split,
        root,
        preprocess_root,
        project_scale=2,
        frustum_size=4,
        color_jitter=None,
        fliplr=0.0,
    ):
        super().__init__()
        self.root = root
        self.label_root = os.path.join(preprocess_root, "labels")
        
        self.n_classes = 16
        # a = b = c = []
        # for i in range(0,500):
        #     a.append(str(i).zfill(3))
        # for i in range(500,798):
        #     b.append(str(i).zfill(3))
        # for i in range(798,1000):
        #     c.append(str(i).zfill(3))

        # splits = {'train': a, 'val': b,
        #           'test': c}


        a = ["000", "001", "002", "003", "004", "005", "006", "007", "008", "009", "010", 
             "011", "012", "013", "014", "015", "016", "017", "018", "019", "020", "021", 
             "022", "023", "024", "025", "026", "027", "028", "029", "030", "031", "032", 
             "033", "034", "035", "036", "037", "038", "039", "040", "041", "042", "043", 
             "044", "045", "046", "047", "048", "049", "050", "051", "052", "053", "054", 
             "055", "056", "057", "058", "059", "060", "061", "062", "063", "064", "065", 
             "066", "067", "068", "069", "070", "071", "072", "073", "074", "075", "076", 
             "077", "078", "079", "080", "081", "082", "083", "084", "085", "086", "087", 
             "088", "089", "090", "091", "092", "093", "094", "095", "096", "097", "098", 
             "099", "100", "101", "102", "103", "104", "105", "106", "107", "108", "109", 
             "110", "111", "112", "113", "114", "115", "116", "117", "118", "119", "120", 
             "121", "122", "123", "124", "125", "126", "127", "128", "129", "130", "131", 
             "132", "133", "134", "135", "136", "137", "138", "139", "140", "141", "142", 
             "143", "144", "145", "146", "147", "148", "149", "150", "151", "152", "153", 
             "154", "155", "156", "157", "158", "159", "160", "161", "162", "163", "164", 
             "165", "166", "167", "168", "169", "170", "171", "172", "173", "174", "175", 
             "176", "177", "178", "179", "180", "181", "182", "183", "184", "185", "186", 
             "187", "188", "189", "190", "191", "192", "193", "194", "195", "196", "197", 
             "198", "199", "200", "201", "202", "203", "204", "205", "206", "207", "208", 
             "209", "210", "211", "212", "213", "214", "215", "216", "217", "218", "219", 
             "220", "221", "222", "223", "224", "225", "226", "227", "228", "229", "230", 
             "231", "232", "233", "234", "235", "236", "237", "238", "239", "240", "241", 
             "242", "243", "244", "245", "246", "247", "248", "249", "250", "251", "252", 
             "253", "254", "255", "256", "257", "258", "259", "260", "261", "262", "263", 
             "264", "265", "266", "267", "268", "269", "270", "271", "272", "273", "274", 
             "275", "276", "277", "278", "279", "280", "281", "282", "283", "284", "285", 
             "286", "287", "288", "289", "290", "291", "292", "293", "294", "295", "296", 
             "297", "298", "299", "300", "301", "302", "303", "304", "305", "306", "307", 
             "308", "309", "310", "311", "312", "313", "314", "315", "316", "317", "318", 
             "319", "320", "321", "322", "323", "324", "325", "326", "327", "328", "329", 
             "330", "331", "332", "333", "334", "335", "336", "337", "338", "339", "340",
             "341", "342", "343", "344", "345", "346", "347", "348", "349", "350", "351", 
             "352", "353", "354", "355", "356", "357", "358", "359", "360", "361", "362", 
             "363", "364", "365", "366", "367", "368", "369", "370", "371", "372", "373", 
             "374", "375", "376", "377", "378", "379", "380", "381", "382", "383", "384", 
             "385", "386", "387", "388", "389", "390", "391", "392", "393", "394", "395", 
             "396", "397", "398", "399", "400", "401", "402", "403", "404", "405", "406", 
             "407", "408", "409", "410", "411", "412", "413", "414", "415", "416", "417", 
             "418", "419", "420", "421", "422", "423", "424", "425", "426", "427", "428", 
             "429", "430", "431", "432", "433", "434", "435", "436", "437", "438", "439", 
             "440", "441", "442", "443", "444", "445", "446", "447", "448", "449", "450", 
             "451", "452", "453", "454", "455", "456", "457", "458", "459", "460", "461", 
             "462", "463", "464", "465", "466", "467", "468", "469", "470", "471", "472", 
             "473", "474", "475", "476", "477", "478", "479", "480", "481", "482", "483", 
             "484", "485", "486", "487", "488", "489", "490", "491", "492", "493", "494", 
             "495", "496", "497", "498", "499", "500"]
            
        b = ["501", "502", "503", "504", "505", "506", "507", "508", "509", "510", "511", 
             "512", "513", "514", "515", "516", "517", "518", "519", "520", "521", "522", 
             "523", "524", "525", "526", "527", "528", "529", "530", "531", "532", "533", 
             "534", "535", "536", "537", "538", "539", "540", "541", "542", "543", "544", 
             "545", "546", "547", "548", "549", "550", "551", "552", "553", "554", "555", 
             "556", "557", "558", "559", "560", "561", "562", "563", "564", "565", "566", 
             "567", "568", "569", "570", "571", "572", "573", "574", "575", "576", "577", 
             "578", "579", "580", "581", "582", "583", "584", "585", "586", "587", "588", 
             "589", "590", "591", "592", "593", "594", "595", "596", "597", "598", "599", 
             "600", "601", "602", "603", "604", "605", "606", "607", "608", "609", "610", 
             "611", "612", "613", "614", "615", "616", "617", "618", "619", "620", "621", 
             "622", "623", "624", "625", "626", "627", "628", "629", "630", "631", "632", 
             "633", "634", "635", "636", "637", "638", "639", "640", "641", "642", "643", 
             "644", "645", "646", "647", "648", "649", "650", "651", "652", "653", "654", 
             "655", "656", "657", "658", "659", "660", "661", "662", "663", "664", "665", 
             "666", "667", "668", "669", "670", "671", "672", "673", "674", "675", "676", 
             "677", "678", "679", "680", "681", "682", "683", "684", "685", "686", "687", 
             "688", "689", "690", "691", "692", "693", "694", "695", "696", "697", "698", 
             "699", "700", "701", "702", "703", "704", "705", "706", "707", "708", "709", 
             "710", "711", "712", "713", "714", "715", "716", "717", "718", "719", "720", 
             "721", "722", "723", "724", "725", "726", "727", "728", "729", "730", "731", 
             "732", "733", "734", "735", "736", "737", "738", "739", "740", "741", "742", 
             "743", "744", "745", "746", "747", "748", "749", "750", "751", "752", "753", 
             "754", "755", "756", "757", "758", "759", "760", "761", "762", "763", "764", 
             "765", "766", "767", "768", "769", "770", "771", "772", "773", "774", "775", 
             "776", "777", "778", "779", "780", "781", "782", "783", "784", "785", "786", 
             "787", "788", "789", "790", "791", "792", "793", "794", "795", "796", "797", "798"]
        
        c = ["799", "800", "801", "802", "803", "804", "805", "806", "807", "808", "809",
             "810", "811", "812", "813", "814", "815", "816", "817", "818", "819", "820",
             "821", "822", "823", "824", "825", "826", "827", "828", "829", "830", "831", 
             "832", "833", "834", "835", "836", "837", "838", "839", "840", "841", "842", 
             "843", "844", "845", "846", "847", "848", "849", "850", "851", "852", "853", 
             "854", "855", "856", "857", "858", "859", "860", "861", "862", "863", "864", 
             "865", "866", "867", "868", "869", "870", "871", "872", "873", "874", "875", 
             "876", "877", "878", "879", "880", "881", "882", "883", "884", "885", "886", 
             "887", "888", "889", "890", "891", "892", "893", "894", "895", "896", "897", 
             "898", "899", "900", "901", "902", "903", "904", "905", "906", "907", "908", 
             "909", "910", "911", "912", "913", "914", "915", "916", "917", "918", "919", 
             "920", "921", "922", "923", "924", "925", "926", "927", "928", "929", "930", 
             "931", "932", "933", "934", "935", "936", "937", "938", "939", "940", "941", 
             "942", "943", "944", "945", "946", "947", "948", "949", "950", "951", "952", 
             "953", "954", "955", "956", "957", "958", "959", "960", "961", "962", "963", 
             "964", "965", "966", "967", "968", "969", "970", "971", "972", "973", "974", 
             "975", "976", "977", "978", "979", "980", "981", "982", "983", "984", "985", 
             "986", "987", "988", "989", "990", "991", "992", "993", "994", "995", "996", 
             "997", "998", "999"]
        # for i in range(0,1):
        #     a.append(str(i).zfill(3))
        # for j in range(1,2):
        #     b.append(str(j).zfill(3))
        # for i in range(2,20):
        #     c.append(str(i).zfill(3))
        splits = {"train": a, 
                  "val": b,
                  "test": c}

        self.split = split
        self.sequences = splits[split]
        self.frustum_size = frustum_size
        self.project_scale = project_scale
        self.output_scale = int(self.project_scale / 2)
        self.scene_size = (51.2, 51.2, 6.4)
        self.vox_origin = np.array([0, -25.6, -2])
        self.fliplr = fliplr

        self.voxel_size = 0.2  # 0.2m
        self.img_W = 960
        self.img_H = 640

        self.color_jitter = (
            transforms.ColorJitter(*color_jitter) if color_jitter else None
        )
        self.scans = []
        for sequence in self.sequences:
            calib = self.read_calib(
                os.path.join(self.root, "calib", sequence, "calib.txt")
            )
            P = calib["P2"]
            T_velo_2_cam = calib["Tr"]
            proj_matrix = P @ T_velo_2_cam

            glob_path = os.path.join(
                self.root, "voxels", sequence, "voxels", "*.npz"
            )
            for voxel_path in glob.glob(glob_path):
                self.scans.append(
                    {
                        "sequence": sequence,
                        "P": P,
                        "T_velo_2_cam": T_velo_2_cam,
                        "proj_matrix": proj_matrix,
                        "voxel_path": voxel_path,
                    }
                )

        self.normalize_rgb = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def __getitem__(self, index):
        scan = self.scans[index]
        voxel_path = scan["voxel_path"]
        sequence = scan["sequence"]
        P = scan["P"]
        T_velo_2_cam = scan["T_velo_2_cam"]
        proj_matrix = scan["proj_matrix"]

        filename = os.path.basename(voxel_path)
        frame_id = os.path.splitext(filename)[0]
        rgb_path = os.path.join(
            self.root, "image_2", sequence, str(int(frame_id)).zfill(3) + ".png"
        )

        data = {
            "frame_id": frame_id,
            "sequence": sequence,
            "P": P,
            "T_velo_2_cam": T_velo_2_cam,
            "proj_matrix": proj_matrix,
        }
        scale_3ds = [self.output_scale, self.project_scale]
        data["scale_3ds"] = scale_3ds
        cam_k = P[0:3, 0:3]
        data["cam_k"] = cam_k
        for scale_3d in scale_3ds:

            # compute the 3D-2D mapping
            projected_pix, fov_mask, pix_z = vox2pix(
                T_velo_2_cam,
                cam_k,
                self.vox_origin,
                self.voxel_size * scale_3d,
                self.img_W,
                self.img_H,
                self.scene_size,
            )            

            data["projected_pix_{}".format(scale_3d)] = projected_pix
            data["pix_z_{}".format(scale_3d)] = pix_z
            data["fov_mask_{}".format(scale_3d)] = fov_mask

        target_1_path = os.path.join(self.label_root, sequence, frame_id + "_1_1.npy")
        target = np.load(target_1_path)
        data["target"] = target
        target_8_path = os.path.join(self.label_root, sequence, frame_id + "_1_8.npy")
        target_1_8 = np.load(target_8_path)
        CP_mega_matrix = compute_CP_mega_matrix(target_1_8)
        data["CP_mega_matrix"] = CP_mega_matrix

        # Compute the masks, each indicate the voxels of a local frustum
        if self.split != "test":
            projected_pix_output = data["projected_pix_{}".format(self.output_scale)]
            pix_z_output = data[
                "pix_z_{}".format(self.output_scale)
            ]
            frustums_masks, frustums_class_dists = compute_local_frustums(
                projected_pix_output,
                pix_z_output,
                target,
                self.img_W,
                self.img_H,
                dataset="kitti",
                n_classes=16,
                size=self.frustum_size,
            )
        else:
            frustums_masks = None
            frustums_class_dists = None
        data["frustums_masks"] = frustums_masks
        data["frustums_class_dists"] = frustums_class_dists

        img = Image.open(rgb_path).convert("RGB")

        # Image augmentation
        if self.color_jitter is not None:
            img = self.color_jitter(img)

        # PIL to numpy
        img = np.array(img, dtype=np.float32, copy=False) / 255.0

        from scipy.ndimage import zoom
        img = zoom(img, (0.5, 0.5, 1))

        # img = img[:640, :960, :]  # crop image

        # Fliplr the image
        if np.random.rand() < self.fliplr:
            img = np.ascontiguousarray(np.fliplr(img))
            for scale in scale_3ds:
                key = "projected_pix_" + str(scale)
                data[key][:, 0] = img.shape[1] - 1 - data[key][:, 0]

        data["img"] = self.normalize_rgb(img)
        return data

    def __len__(self):
        return len(self.scans)

    @staticmethod
    def read_calib(calib_path):
        """
        Modify from https://github.com/utiasSTARS/pykitti/blob/d3e1bb81676e831886726cc5ed79ce1f049aef2c/pykitti/utils.py#L68
        :param calib_path: Path to a calibration text file.
        :return: dict with calibration matrices.
        """
        calib_all = {}
        with open(calib_path, "r") as f:
            for line in f.readlines():
                if line == "\n":
                    break
                key, value = line.split(":", 1)
                calib_all[key] = np.array([float(x) for x in value.split()])

        # reshape matrices
        calib_out = {}
        # 3x4 projection matrix for left camera

        calib_all["P2"][0] = calib_all["P2"][0] / 2
        calib_all["P2"][2] = calib_all["P2"][2] / 2
        calib_all["P2"][5] = calib_all["P2"][5] / 2
        calib_all["P2"][6] = calib_all["P2"][6] / 2

        calib_out["P2"] = calib_all["P2"].reshape(3, 4)
        calib_out["Tr"] = np.identity(4)  # 4x4 matrix
        calib_out["Tr"][:3, :4] = calib_all["Tr"].reshape(3, 4)
        return calib_out
