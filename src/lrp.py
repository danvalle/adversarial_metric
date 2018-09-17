from PIL import Image
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

from utils import load_data, visualization
from extractor_model import Extractor


def lrp(model, neurons, features_layers, classifier_layers, input):
    neurons_out = len(neurons)
    neurons_out -= 1

    R = neurons[neurons_out][0].data.clone()

    for idx in classifier_layers[::-1]:
        neurons_out -= 1
        i_layer = neurons[neurons_out].data

        print(model.classifier[idx])
        weight = model.classifier[idx].weight.data
    
        R = model.linear_back(
            model.classifier[idx], i_layer, weight, R.data)
    R = R.view(i_layer.size())

    for idx in features_layers[::-1]:
        j_layer = i_layer

        neurons_out -= 1
        if neurons_out >= 0:
            i_layer = neurons[neurons_out].data
        else:
            i_layer = input

        print(model.features[idx])
        if type(model.features[idx]) == nn.MaxPool2d:
            R = model.maxpool_back(
                model.features[idx], i_layer, j_layer.data, R.data)

        elif type(model.features[idx]) == nn.Conv2d:
            weight = model.features[idx].weight.data
            R = model.conv_back(
                model.features[idx], i_layer, weight, R.data)

    return R[0]


def main(num_images, device_name, seed):
    torch.manual_seed(seed)
    device = torch.device(device_name)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()])

    data = load_data()
    model = Extractor(
        models.vgg16(pretrained=True), 1, device)

    it = 0
    for name, (img, label) in data.items():
        image = Image.open(img)

        X = transform(image).to(device=device)
        target = torch.LongTensor([label]).to(device=device)

        neurons, features_layers, classifier_layers = model(X.unsqueeze(0))
        if neurons[-1].argmax().item() == label:
            print(name)

        R = lrp(
            model, neurons,
            features_layers, classifier_layers,
            X.data.clone().unsqueeze(0))

        visualization(R, 'img/lrp/'+name+'.png', X)

        it += 1
        if it == num_images:
            break


if __name__ == '__main__':
    num_images = 10
    device_name = 'cuda'
    seed = 0

    with torch.no_grad():
        main(num_images, device_name, seed)