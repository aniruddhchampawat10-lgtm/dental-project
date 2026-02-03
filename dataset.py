import os, cv2, torch
import numpy as np
from torch.utils.data import Dataset

class DentalDataset(Dataset):
    def __init__(self, image_dir, mask_dir):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = sorted(os.listdir(image_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]

        img = cv2.imread(os.path.join(self.image_dir, img_name), 0)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img = cv2.resize(img, (256,256)) / 255.0

        mask = cv2.imread(os.path.join(self.mask_dir, img_name), 0)
        mask = cv2.resize(mask, (256,256), interpolation=cv2.INTER_NEAREST)
        mask = (mask > 127).astype("float32")
        mask = np.expand_dims(mask, 0)

        return (
            torch.tensor(img).permute(2,0,1).float(),
            torch.tensor(mask).float(),
            img_name
        )
