#the code was taken from https://github.com/jacobgil/pytorch-grad-cam
#it was corrected in order to be suitable to considered models and 3D images
#the original article https://arxiv.org/pdf/1610.02391v1.pdf

import torch
import numpy as np
import cv2
from skimage.transform import resize

class FeatureExtractor():
    """ Class for extracting activations and 
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []
        for name, module in self.model._modules.items():
            x = module(x)
            if name in self.target_layers:
                x.register_hook(self.save_gradient)
                outputs += [x]
        return outputs, x


class ModelOutputs():
    """ Class for making a forward pass, and getting:
    1. The network output.
    2. Activations from intermeddiate targetted layers.
    3. Gradients from intermeddiate targetted layers. """

    def __init__(self, model, target_layers):
        self.model = model
        self.feature_extractor = FeatureExtractor(self.model.features, target_layers)

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x, g):
        target_activations, output = self.feature_extractor(x)
        output = self.model.clf(output, g)
        return target_activations, output
    
    
class ModelOutputs_VAE():
    """ Class for making a forward pass, and getting:
    1. The network output.
    2. Activations from intermeddiate targetted layers.
    3. Gradients from intermeddiate targetted layers. """

    def __init__(self, model, target_layers):
        self.model = model
        self.feature_extractor = FeatureExtractor(
            self.model.encoder.down_sample, target_layers)

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x, g):
        target_activations, output = self.feature_extractor(x)
        output = self.model.encoder.flatten(output)
        output = torch.cat([output, g.view(-1, 1)], dim=1)
        output = self.model.encoder.age_mean(output)
        return target_activations, output
    
    
class GradCam:
    def __init__(self, model, target_layer_names, 
                 model_ouputs_class, device='cuda'):
        self.model = model
        self.model.eval()
        self.device = device
        self.model = model.to(device)

        self.extractor = model_ouputs_class(self.model, target_layer_names)

    def __call__(self, x, g, index=None):
        x = x.to(self.device)
        g = g.to(self.device)
        features, output = self.extractor(x, g)
        if index == None:
            index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        one_hot = torch.sum(one_hot.to(self.device) * output)

        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()

        target = features[-1]
        target = target.cpu().data.numpy()[0, :]
        weights = np.mean(grads_val, axis=(2, 3))[0, :]
        cam = np.zeros(target.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * target[i, :, :]

        cam = np.maximum(cam, 0)
        cam = resize(cam, (128, 128, 128), order=3)
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        return cam
    
    
def show_cam_on_image(img, mask):
    """
    combines original 3D image and mask obtained from grad-cam
    """
    heatmap = np.zeros((*mask.shape, 3))
    for i in range(mask.shape[0]):
        heatmap[i] = cv2.applyColorMap(np.uint8(255 * mask[i]), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.repeat(np.float32(img)[:,:,:,np.newaxis], 3, axis=-1)
    cam = cam / np.max(cam)
    return cam
    

    