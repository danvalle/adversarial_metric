import pickle as pkl

import numpy as np
from torchvision.utils import save_image


def load_data():
    with open('../resources/data.pkl', 'rb') as ff:
        data = pkl.load(ff)
    return data


def visualization(tensor, file_name, image):
    tensor.abs_()
    
    percentile = np.percentile(tensor, 99)
    tensor.clamp_(0, percentile)

    tensor.mul_(image)
    save_image(tensor.sum(dim=0), file_name, normalize=True)