import numpy as np
import torch
import torchio
from torchio.transforms.transform import Transform
from .preprocessing import cut_img
from .preprocessing import rescale_crop_img, rescale_pad_img


class RandomFlip(object):

    def __init__(self, axis=0, p=0.5):
        self.axis = axis
        self.p = p

    def __call__(self, image):

        if np.random.random() < self.p:
            return torch.flip(image, [self.axis])
        return image

    
class ResizeCrop(Transform):
    def __init__(
            self,
            target_dim
            ):
        super().__init__()
        self.target_dim = target_dim

    def apply_transform(self, sample: dict) -> dict:
        for image_dict in sample.values():
            if not torchio.utils.is_image_dict(image_dict):
                continue
            image = image_dict[torchio.DATA][0].numpy()
            image = rescale_crop_img(image, size=self.target_dim)
            tensor = torch.from_numpy(image).unsqueeze(0)
            image_dict[torchio.DATA] = tensor
        return sample


class ResizePad(Transform):
    def __init__(
            self,
            target_dim
            ):
        super().__init__()
        self.target_dim = target_dim

    def apply_transform(self, sample: dict) -> dict:
        for image_dict in sample.values():
            if not torchio.utils.is_image_dict(image_dict):
                continue
            image = image_dict[torchio.DATA][0].numpy()
            image = rescale_pad_img(image, size=self.target_dim)
            tensor = torch.from_numpy(image).unsqueeze(0)
            image_dict[torchio.DATA] = tensor
        return sample
    
    
class BoundsCrop(Transform):

    def apply_transform(self, sample: dict) -> dict:
        for image_dict in sample.values():
            if not torchio.utils.is_image_dict(image_dict):
                continue
            image = image_dict[torchio.DATA][0].numpy()
            image = cut_img(image)
            tensor = torch.from_numpy(image).unsqueeze(0)
            image_dict[torchio.DATA] = tensor
        return sample