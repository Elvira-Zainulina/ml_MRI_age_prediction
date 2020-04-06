# ml_MRI_age_prediction

This repository is made by [Zainulina Elvira](https://github.com/Elvira-Zainulina), [Kiseleva Elizaveta](https://github.com/KiselevaElizavetaA), Maksim Brazhnikov and [Kodirjon Akhmedov](https://github.com/alhkad) and contains the source code to reproduce the experiments in our Machine Learning course project. Skoltech 2020.

![ A 3-D view of the entire human brain, taken with a powerful 7 Tesla MRI and shown here from two angles, could reveal new details on structures in the mysterious organ. https://www.sciencenews.org/article/mri-scan-most-detailed-look-yet-whole-human-brain](https://www.sciencenews.org/wp-content/uploads/2019/07/070119_ls_brainscan_feat.jpg)

The connection of brain structure with aging attracts a lot of attention especially in the context of diagnosing aging-related diseases.
In this work we explore various ML approaches of age estimation from brain structural MRI data. We used MRI data from ConnectomeDB https://db.humanconnectome.org/ for our research. We found that the correlation between brain structure and age exists. The most promising method is conditional variation auto-encoders.

## Setup and Dependencies
* Python/numpy
* TensorFlow (we used r10)

## Libraries

## Dataset
We used **image dataset:** 'WU-Minn HCP Data - 1200 Subjects', 'S900 new subjects' https://db.humanconnectome.org/app/action/DownloadPackagesAction, 
and **tabular datasets:** 'Behavioral Data' and 'Expanded FreeSurfer Data' https://db.humanconnectome.org/data/projects/HCP_1200
## Feature importance for image data

## Feature importance for tabular data

## Related projects
[torchio](https://github.com/fepegar/torchio)
