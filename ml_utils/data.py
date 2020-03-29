import torchio
import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
from .preprocessing import process_ages, map_gender


class S500MRI_Dataset(torchio.ImagesDataset):
    def __init__(
              self,
              images,
              genders,
              ages=None,
              transform = None,
              check_nans = True,
              ):
        self._parse_subjects_list(images)
        self.subjects = images
        self._transform = transform
        self.check_nans = check_nans
        if ages is None:
            self.ages = None
        else:
            self.ages = torch.tensor(process_ages(ages), dtype=torch.long)
        self.genders = torch.tensor(map_gender(genders), dtype=torch.float32)

    def __getitem__(self, index: int) -> dict:
        if not isinstance(index, int):
            raise TypeError(f'Index "{index}" must be int, not {type(index)}')
        subject = self.subjects[index]
        sample = {}
        for image in subject:
            tensor, affine = image.load(check_nans=self.check_nans)
            image_dict = {
                torchio.DATA: tensor,
                torchio.AFFINE: affine,
                torchio.TYPE: image.type,
                torchio.PATH: str(image.path),
                torchio.STEM: torchio.utils.get_stem(image.path),
            }
            sample[image.name] = image_dict

        if self._transform is not None:
            sample = self._transform(sample)
        if self.ages is None:
            return sample, self.genders

        return sample, self.genders[index], self.ages[index]
    
    
class S500MRI_Dataset_simple(Dataset):
    def __init__(
              self,
              images,
              genders,
              ages=None,
              transform=None
              ):
        self.X = images.squeeze(1).transpose(0, 2, 1, 3)
        self.genders = torch.tensor(genders, dtype=torch.float32)
        if ages is None:
            self.ages = None
        else:
            self.ages = torch.tensor(ages, dtype=torch.long)

        self.transform = transform

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        sample = torch.tensor(self.X[idx], dtype=torch.float32)

        if self.transform is not None:
            sample = self.transform(sample)

        if self.ages is None:
            return sample, self.genders

        return sample.unsqueeze(0), self.genders[idx], self.ages[idx]
    

class ImbalancedDatasetSampler(Sampler):
    #function is taken from https://github.com/ufoym/imbalanced-dataset-sampler
    """
    function is taken from https://github.com/ufoym/imbalanced-dataset-sampler
    Samples elements randomly from a given list of indices for imbalanced dataset
    
    :param indices (list, optional): a list of indices
    :param num_samples (int, optional): number of samples to draw
    :param callback_get_label func: a callback-like function which takes two arguments - dataset and index
    """

    def __init__(self, dataset, indices=None, num_samples=None, callback_get_label=None):
                
        # if indices is not provided, 
        # all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) \
            if indices is None else indices

        # define custom callback
        self.callback_get_label = callback_get_label

        # if num_samples is not provided, 
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) \
            if num_samples is None else num_samples
            
        # distribution of classes in the dataset 
        label_to_count = {}
        for idx in self.indices:
            label = self._get_label(dataset, idx)
            if label in label_to_count:
                label_to_count[label] += 1
            else:
                label_to_count[label] = 1
                
        # weight for each sample
        weights = [1.0 / label_to_count[self._get_label(dataset, idx)]
                   for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)

    def _get_label(self, dataset, idx):
        return dataset.ages[idx].item()
                
    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(
            self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples
    
    
