from os.path import splitext
from os import listdir
import torch
from torch.utils.data import Dataset
import glob
import numpy as np
from PIL import Image
from skimage import io, transform


class LoadDataset(Dataset):

    def __init__(self, img_dir, mask_dir, mask_suffix=''):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.mask_suffix = mask_suffix
        self.ids = [splitext(file)[0] for file in listdir(img_dir)
                    if not file.startswith('.')]

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img):
        w, h = pil_img.size
        newW, newH = int(0.8 * w), int(0.8 * h)
        img_nd = pil_img.resize((newW, newH))
        img_nd = np.array(img_nd)
        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)
        img_trans = img_nd.transpose((2, 0, 1))
        if img_trans.max() > 1:
            img_trans = img_trans / 255
        
        return img_trans
    
    def __getitem__(self, i):
        idx = self.ids[i]
        # mask_file = glob(str(self.mask_dir + idx + self.mask_suffix + '.*'))
        # img_file = glob(str(self.img_dir + idx + '.*'))

        mask = Image.open(str(self.mask_dir + idx + self.mask_suffix + '.gif'))
        img = Image.open(str(self.img_dir + idx + '.jpg'))

        img = self.preprocess(img)
        mask = self.preprocess(mask)

        return {
            'image': torch.from_numpy(img).type(torch.FloatTensor),
            'mask': torch.from_numpy(mask).type(torch.FloatTensor)
        }


class CarvanaDataset(LoadDataset):
    def __init__(self, img_dir, mask_dir):
        super(CarvanaDataset, self).__init__(img_dir, mask_dir, mask_suffix='_mask')