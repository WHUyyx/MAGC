from typing import Dict, Union
import time

import numpy as np
from PIL import Image
import torch.utils.data as data
import torchvision

from utils.file import load_file_list

class CodeformerDataset(data.Dataset):
    def __init__(
        self,
        file_list: str, 
        out_size: int,
        crop_type: str,
        use_hflip: bool,
        file_list_dlg: str,
    ) -> "CodeformerDataset":
        super(CodeformerDataset, self).__init__()
        self.file_list = file_list
        self.paths = load_file_list(file_list)
        self.paths_dlg = load_file_list(file_list_dlg)
        self.out_size = out_size
        self.crop_type = crop_type
        assert self.crop_type in ["none", "center", "random"]
        self.use_hflip = use_hflip

        self.transform = torchvision.transforms.ToTensor()


    def __getitem__(self, index: int) -> Dict[str, Union[np.ndarray, str]]:
        gt_path = self.paths[index]
        ref_path = self.paths_dlg[index]
        success = False
        for _ in range(3):
            try:
                pil_img = Image.open(gt_path).convert("RGB")
                pil_ref = Image.open(ref_path).convert("RGB")
                success = True
                break
            except:
                time.sleep(1)
        assert success, f"failed to load image {gt_path}"
        

        if self.transform is not None:
            img_gt = self.transform(pil_img) # [-1,1]
            img_gt = img_gt * 2 -1
            ref_gt = self.transform(pil_ref) # [0,1] 
        return dict(img_gt=img_gt,ref_gt=ref_gt, txt="")

    def __len__(self) -> int:
        return len(self.paths)
