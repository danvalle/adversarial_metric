import sys

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
    save_image(new_image, 'test_image.png')

    out = model(new_image.unsqueeze(0))
    return out.argmax().item() == label


def adv_tests(model, img, label, sigma, rel, adv=True):
    # import ipdb; ipdb.set_trace()
    sign = -1 if adv else 1
    new_image = img.unsqueeze(0).repeat(rel.size(0), 1, 1, 1) + \
                adv*sigma*rel
    out = model(new_image)
    return (out.argmax(dim=-1) == label).sum().item()


def z_value(tensor):
    mean = tensor.mean()
    std = tensor.std()
    tensor_norm = tensor.sub(mean)
    tensor_norm = tensor_norm.div(std)
    return tensor_norm


def main():

    torch.manual_seed(0)
    device = torch.device('cuda')
    to_tensor = transforms.ToTensor()

    model = models.vgg16(pretrained=True).to(device=device)
    model.eval()
    criterion = nn.CrossEntropyLoss()

    data = load_data()
    data_path = sys.argv[1]
    name = data_path.split('.')[0].split('/')[-1]

    img = Image.open(sys.argv[2])
    rel_img = Image.open(data_path)

    X = to_tensor(img).to(device=device)
    X.requires_grad_()
    rel = to_tensor(rel_img).to(device=device)
    label = data[name][1]

    out = model(X.unsqueeze(0))
    print(name, ':', out.argmax().item(), label)
    loss = criterion(out, out.argmax().view(-1))
    loss.backward()
    grad = X.grad.data.clone()


    # Relevance using grandient
    rel = grad.data.clone()
    rel.abs_()
    percentile = np.percentile(rel, 99)
    rel.clamp_(0, percentile)
    
    rel.mul_(X)
    rel = rel.sum(dim=0)
    rel = rel.sub(rel.min()).div(rel.max() - rel.min())
    rel = rel.unsqueeze(0).repeat(3,1,1)
    save_image(rel, 'test_grad.jpg', normalize=True)
    

    rel_fell = -1
    irel_fell = -1
    rels_fell = -1
    irels_fell = -1

    sigmas = []
    results = [[], [], [], []]

    with torch.no_grad():
        for _sigma in range(5, 2000, 3):
            sigma = _sigma
            imp = 0
            imp_s = 0
            nimp = 0
            nimp_s = 0

            num_rand = 20

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


            print('Sigma:', sigma)
            print('Importa:', imp)
            print('Importa shuffle:', imp_s)
            print('Nao importa:', nimp)
            print('Nao importa shuffle:', nimp_s)

            sigmas += [sigma]
            results[0] += [imp]
            results[1] += [imp_s]
            results[2] += [nimp]
            results[3] += [nimp_s]

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

    print('Importa:', rel_fell)
    print('Importa shuffle:', rels_fell)
    print('Nao importa:', irel_fell)
    print('Nao importa shuffle:', irels_fell)
    print('{} - {} / {} - {}'.format(rel_fell, rels_fell, irels_fell, irel_fell))

    plt.plot(sigmas, results[0])
    plt.plot(sigmas, results[1])
    plt.plot(sigmas, results[2])
    plt.plot(sigmas, results[3])
    plt.legend(['Relevance', 'Shuffled Relevance', 'Irrelevance', 'Shuffled Irrelevance'])
    plt.xlabel('Epsilon')
    plt.ylabel('Correct classifications')
    plt.savefig('plot.jpg')



if __name__ == '__main__':
    main()