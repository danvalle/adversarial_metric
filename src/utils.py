import os
import pickle as pkl

import numpy as np
from torchvision.utils import save_image


def load_data():
    with open('../resources/data.pkl', 'rb') as ff:
        data = pkl.load(ff)
    return data


def visualization(tensor, file_path, file_name, image):
    tensor.abs_()
    
    percentile = np.percentile(tensor, 99)
    tensor.clamp_(0, percentile)

    # Pure relevnce in 3 channels
    complete_path = file_path + 'clamp/'
    if not os.path.exists(complete_path):
        os.makedirs(complete_path)
    save_image(tensor, complete_path+file_name, normalize=True)

    # Relevance map
    complete_path = file_path + 'map/'
    if not os.path.exists(complete_path):
        os.makedirs(complete_path)
    save_image(tensor.sum(dim=0), complete_path+file_name, normalize=True)

    # Relevance x image / 3 channels and map
    tensor.mul_(image)

    complete_path = file_path + 'final/'
    if not os.path.exists(complete_path):
        os.makedirs(complete_path)
    save_image(tensor, complete_path+file_name, normalize=True)

    complete_path = file_path + 'final_map/'
    if not os.path.exists(complete_path):
        os.makedirs(complete_path)
    save_image(tensor.sum(dim=0), complete_path+file_name, normalize=True)
