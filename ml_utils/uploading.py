import numpy as np
import pandas as pd
import os
from zipfile import ZipFile
import torchio


def unzip_imgs(in_path, out_path, verbose=True):
    """
    unzips folders from uploaded dataset S500
    (https://db.humanconnectome.org/data/projects/HCP_1200)
    from in_path to out_path
    
    :param in_path: path to folder containing downloaded dataset
    :param out_path: path to folder where extract folders
    """
    for file in os.listdir(in_path):
        if 'Structural_preproc' in file and file.endswith('.zip'):
            zip_file = os.path.join(in_path, file)
            with ZipFile(zip_file, 'r') as zipObj:
                listOfFileNames = zipObj.namelist()
                for name in listOfFileNames:
                    if 'T1w_acpc_dc_restore_brain' in name:
                        zipObj.extract(name, path=out_path)
            if verbose:
                print('Unzipped', file)
                
                
def upload_raw_data(img_path, table_path, names):
    subjects = []
    ages = []
    genders = []
    df = pd.read_csv(table_path)
    for name in names:
        file_ = os.path.join(img_path, str(name), 'T1w', 
                             'T1w_acpc_dc_restore_brain.nii.gz')
        subject = torchio.Subject(
            torchio.Image('MRI', file_, torchio.INTENSITY)
        )
        subjects.append(subject)
        ages.append(df.Age.values[df.Subject == name][0])
        genders.append(df.Gender.values[df.Subject == name][0])
        
    data = {
        'images' : subjects,
        'genders' : genders,
        'ages' : ages
    }
    return data


def load_data(path):
    data = np.load(path)
    X = data['images']
    y = data['ages']
    g = data['genders']
    return X, y, g


         