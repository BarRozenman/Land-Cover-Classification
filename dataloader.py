from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset



class CustomImageDataset(Dataset):
    def __init__(
        self,
        df,
        img_dir,
        loader=None,
        class_rgb_dict=None,
        target_transform=None,
        train=True,
            img_size=None,
            preprocessing_fn=None,
    ):
        self.df = df
        self.sat_paths = self.df["sat_image_path"]  # satellite
        self.mask_paths = self.df["mask_path"]
        self.img_dir = img_dir
        self.loader = loader
        self.target_transform = target_transform
        self.class_rgb_dict = class_rgb_dict
        self.train = train
        self.img_size = img_size
        self.preprocessing_fn = preprocessing_fn

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if self.train:
            mask_image_size = (
                int(self.img_size[0] + self.img_size[0] / 10),
                int(self.img_size[1] + self.img_size[1] / 10),
            )
        else:
            mask_image_size = self.img_size
        sat_path = Path(self.img_dir) / self.sat_paths.iloc[idx]
        mask_path = Path(self.img_dir) / self.mask_paths.iloc[idx]
        rgb_mask = np.array(
            Image.open(mask_path).convert("RGB").resize(mask_image_size)
        )
        sat_image, mask = self.loader(
            sat_path, self.preprocessing_fn, rgb_mask, self.class_rgb_dict, train=self.train
        )
        # mask = self.target_transform(rgb_mask, self.class_rgb_dict)

        # mask =self.loader(mask_path)
        # if self.loader:
        #     image = self.loader(image)
        # # if self.target_transform:
        # #     label = self.target_transform(label)\

        return_dict = {"sat_path": str(sat_path), "sat_image": sat_image, "mask": mask}
        return return_dict
