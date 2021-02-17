import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from skimage import io, transform

from model.model import Unet
from load_dataset import LoadDataset

basedir = os.path.abspath(os.path.dirname(__file__))

def load_model(model_name: str = "unet"):
        net = Unet(3, 1)
        try:
            if torch.cuda.is_available():
                filepath = os.path.join(basedir, (model_name + '.pth'))
                net.load_state_dict(torch.load(filepath))
                net.to(torch.device("cuda"))
            else:
                filepath = os.path.join(basedir, (model_name + '.pth'))
                net.load_state_dict(torch.load(filepath, map_location="cpu"))
        except FileNotFoundError:
            raise FileNotFoundError(
                errno.ENOENT, os.strerror(errno.ENOENT), filepath
            )

        net.eval()
        return net

# def predict_img(net, full_img, device):
#     net.eval()
#     img = torch.from_numpy(BasicDataset.preprocess(full_img))
#     img = img.unsqueeze(0)
#     img = img.to(device=device, dtype=torch.float32)

#     with torch.no_grad():
#         output = net(img)

#         if net.n_classes > 1:
#             probs = F.softmax(output, dim=1)
#         else:
#             probs = torch.sigmoid(output)

#         probs = probs.squeeze(0)

#         tf = transforms.Compose(
#             [
#                 transforms.ToPILImage(),
#                 transforms.Resize(full_img.size[1]),
#                 transforms.ToTensor()
#             ]
#         )

#         probs = tf(probs.cpu())
#         full_mask = probs.squeeze().cpu().numpy()

#     return full_mask > out_threshold

def load_image(item):
    w, h  = item.size
    image = np.array(item)
    image = transform.resize(image, (int(w * 0.8), int(h * 0.8)), mode='constant')
    pil_image = item
    if len(image.shape) == 2:
        image = np.expand_dims(image, axis=2)
    image_trans = image.transpose((2, 0, 1))
    if image_trans.max() > 1:
        image_trans = image_trans / 255
    
    return image_trans, pil_image

def norm_pred(predicted):
        ma = torch.max(predicted)
        mi = torch.min(predicted)
        out= (predicted - mi) / (ma - mi)

        return out


def predict(net, item):
    net.eval()
    image, org_image = load_image(item)
    if image is False or org_image is False:
        return False
    image = torch.from_numpy(image).type(torch.FloatTensor)
    image = image.unsqueeze(0)
    image = image.to(device='cpu', dtype=torch.float32)
    with torch.no_grad():
        mask = net(image)
        if net.n_classes > 1:
            mask = F.softmax(mask, dim=1)
        else:
            mask = torch.sigmoid(mask)
            mask = mask[:, 0, :, :]
            mask = norm_pred(mask)
        tf = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.ToTensor()
            ]
        )
        mask = tf(mask.cpu())
        mask_np = mask.squeeze().cpu().detach().numpy()
        mask_np = mask_np > 0.9
        mask = Image.fromarray((mask_np * 255).astype(np.uint8)).convert('L')
        mask = mask.resize(org_image.size, resample=Image.BILINEAR)
        # Apply mask
        # empty = Image.new('RGBA', org_image.size)
        # image = Image.composite(org_image, empty, mask)
        mask.save('mask.jpg')
    
    return mask


