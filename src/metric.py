import sys
import os

from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt

from utils import load_data, visualization


def adv_test(model, img, label, sigma, rel, adv=True):
    sign = -1 if adv else 1
    new_image = img + adv*sigma*rel
    # save_image(new_image, 'test_image.png')

    out = model(new_image.unsqueeze(0))
    return out.argmax().item() == label


def adv_tests(model, img, label, sigma, rel, adv=True):
    sign = -1 if adv else 1
    new_image = img.unsqueeze(0).repeat(rel.size(0), 1, 1, 1) + \
                adv*sigma*rel
    out = model(new_image)
    return (out.argmax(dim=-1) == label).sum().item()


def main(img_path, rel_path):
    torch.manual_seed(0)
    device = torch.device('cuda')

    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()])
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])
    to_tensor = transforms.ToTensor()

    model = models.vgg16(pretrained=True).to(device=device)
    model.eval()
    criterion = nn.CrossEntropyLoss()

    data = load_data()
    print('Name,Rel,RelShuf,IrelShuf,Irel')
    for rel_file in os.listdir(rel_path):
        name = rel_file.split('.')[0]
        
        rel_img = Image.open(rel_path+rel_file)
        rel = to_tensor(rel_img).to(device=device)
        
        image = Image.open(img_path+name+'.JPEG').convert('RGB')
        image = transform(image).to(device)
        X = normalize(image.clone())
        X.requires_grad_()
        
        # label = data[name][1]
        out = model(X.unsqueeze(0))
        label = out.argmax().item()

        loss = criterion(out, out.argmax().view(-1))
        loss.backward()
        grad = X.grad.data.clone()
        
        rel_fell = -1
        irel_fell = -1
        rels_fell = -1
        irels_fell = -1
        num_rand = 10

        with torch.no_grad():
            # Find starting point
            start_point = 0
            for sigma in range(100, 10001, 100):
                rel_used = rel.clone()
                rel_used /= rel_used.norm(p=1)
                rel_used2 = rel_used*grad.sign()
                if adv_test(model, X, label, sigma, rel_used2):
                    start_point = sigma

            for _sigma in range(start_point, 10000, 5):
                sigma = _sigma
                imp = 0
                imp_s = 0
                nimp = 0
                nimp_s = 0


                rel_used = rel.clone()
                rel_used /= rel_used.norm(p=1)
                rel_used2 = rel_used*grad.sign()
                if adv_test(model, X, label, sigma, rel_used2):
                    imp += num_rand

                irel = 1 - rel.clone()
                irel /= irel.norm(p=1)
                irel2 = irel*grad.sign()
                if adv_test(model, X, label, sigma, irel2):
                    nimp += num_rand

                rel_shuffles = torch.zeros_like(rel).unsqueeze(0).repeat(num_rand, 1, 1, 1)
                irel_shuffles = torch.zeros_like(rel_shuffles)

                for j in range(num_rand):
                    rel_shuffle = (rel_used).view(-1)
                    rel_shuffle = rel_shuffle[torch.randperm(len(rel_shuffle))].view(rel.shape)
                    rel_shuffle = rel_shuffle*grad.sign()
                    rel_shuffles[j] = rel_shuffle

                    irel_shuffle = irel.view(-1)
                    irel_shuffle = irel_shuffle[torch.randperm(len(irel_shuffle))].view(rel.shape)
                    irel_shuffle = irel_shuffle*grad.sign()
                    irel_shuffles[j] = irel_shuffle

                imp_s = adv_tests(model, X, label, sigma, rel_shuffles)
                nimp_s = adv_tests(model, X, label, sigma, irel_shuffles)

                # print('Sigma:', sigma)
                # print('Importa:', imp)
                # print('Importa shuffle:', imp_s)
                # print('Nao importa:', nimp)
                # print('Nao importa shuffle:', nimp_s)

                if imp < 10 and rel_fell < 0:
                    rel_fell = sigma
                if imp_s < 10 and rels_fell < 0:
                    rels_fell = sigma
                if nimp < 10 and irel_fell < 0:
                    irel_fell = sigma
                if nimp_s < 10 and irels_fell < 0:
                    irels_fell = sigma

                if imp == 0 and imp_s == 0 and nimp == 0 and nimp_s == 0:
                    break

        print('{},{},{},{},{}'.format(
            name, rel_fell, rels_fell, irels_fell, irel_fell))


if __name__ == '__main__':
    img_path = '/media/dan/CAGE/ILSVRC/Data/CLS-LOC/val/'
    rel_path = 'img/smooth_grad/final_map/'
    main(img_path, rel_path)
