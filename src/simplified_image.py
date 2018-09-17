from PIL import Image
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

from utils import load_data, visualization
from torchvision.utils import save_image


def get_least(rel):
    rel[rel.le(0)] = 0
    rel[rel.gt(0)] = 1
    return rel


def main(num_images, device_name, seed):
    torch.manual_seed(seed)
    device = torch.device(device_name)
    transform = transforms.Compose([
        transforms.ToTensor()])

    data = load_data()
    model = models.vgg16(pretrained=True).to(device=device)
    model.eval()

    it = 0
    for name, (img, label) in data.items():
        image = Image.open('img/original/'+name+'.png')

        X = transform(image).to(device=device)
        target = torch.LongTensor([label]).to(device=device)

        out = model(X.unsqueeze(0))
        print('First:', name,  out.argmax().item(), label)
        

        rel = Image.open('img/lrp/'+name+'.png')
        rel = transform(rel).to(device=device)
        rel = get_least(rel)
        save_image(rel, 'img/simplified/rel_'+name+'.png')

        simple_X = X.detach().clone() * rel
        save_image(simple_X, 'img/simplified/'+name+'.png')
        
        out = model(simple_X.unsqueeze(0))
        print('Simplified:', name,  out.argmax().item(), label)

        it += 1
        if it == num_images:
            break


if __name__ == '__main__':
    num_images = 3
    device_name = 'cuda'
    seed = 0

    main(num_images, device_name, seed)