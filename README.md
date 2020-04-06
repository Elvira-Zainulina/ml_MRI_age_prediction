# ml_MRI_age_prediction

This repository is made by [Zainulina Elvira](https://github.com/Elvira-Zainulina), [Kiseleva Elizaveta](https://github.com/KiselevaElizavetaA), [Maksim Brazhnikov](https://github.com/alhkad) and Kodirjon Akhmedov and contains the source code to reproduce the experiments in our Machine Learning course project. Skoltech 2020.

![ A 3-D view of the entire human brain, taken with a powerful 7 Tesla MRI and shown here from two angles, could reveal new details on structures in the mysterious organ. https://www.sciencenews.org/article/mri-scan-most-detailed-look-yet-whole-human-brain](https://www.sciencenews.org/wp-content/uploads/2019/07/070119_ls_brainscan_feat.jpg)

The connection of brain structure with aging attracts a lot of attention especially in the context of diagnosing aging-related diseases.
In this work we explore various ML approaches of age estimation from brain structural MRI data. We used MRI data from ConnectomeDB https://db.humanconnectome.org/ for our research. We found that the correlation between brain structure and age exists. The most promising method is conditional variation auto-encoders.

## Libraries
In order to work with the image data auxiliary functions were created. They are in the ```ml_utils```. Used model architectures are described in the ```model.py```. Grad-CAM model is suited in the ```grad_cam.py```.

During implementation of this project the following repositories were used:
* [torchio](https://github.com/fepegar/torchio)
* with [pytorch implementation of the Grad-CAM](https://github.com/jacobgil/pytorch-grad-cam)
* with [implentation of the ImbalancedDatasetSampler](from https://github.com/ufoym/imbalanced-dataset-sampler)

## Dataset
We used **image dataset:** 'WU-Minn HCP Data - 1200 Subjects', 'S900 new subjects' https://db.humanconnectome.org/app/action/DownloadPackagesAction, 
and **tabular datasets:** 'Behavioral Data' and 'Expanded FreeSurfer Data' https://db.humanconnectome.org/data/projects/HCP_1200

In order to download this data the Data Agreement should be signed.

To replicate the results of our project the Subjects with defined ids should be used (its split is in the ```data``` folder, each ```.npy``` file containes corresponding Subject ids). 

To make it easier to obtain image datasets right after uploading you can use ```notebooks/images/Data_preparation.ipynb``` to obtain ```train_dataset```, ```validation_dataset```, ```test_dataset``` that were used in the experiments.

## Experiments on the image data

During the project two model architectures were considered: 3D CNN and VAE-classifier.

The examples of the training of the models are in the ```notebooks/images```. As some models take long time to be trained, trained models are presented in the ```image_models``` (2 in the name of the model means that it is used for 2-class classification).

```notebooks/Visualization.ipynb``` presents results of the visualization of the latent vetors obtained by encoder of VAE-C.

## Experiments on the tabular data

The experiments carried out on the tabular data are in the ```notebooks/tables```.

## Feature importance for image data
![](/pictures/GRAD-CAM.png)
 Visualization of the feature maps of the trained model using Grad-CAM. The usage example is shown in the ```notebooks/images/Grad_CAM.ipynb```.
 
## Feature importance for tabular data
![](/pictures/fi.png)
