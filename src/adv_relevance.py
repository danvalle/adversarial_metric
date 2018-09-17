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

        if type(model.features[idx]) == nn.MaxPool2d:
            R = model.maxpool_back(
                model.features[idx], i_layer, j_layer.data, R.data)

        elif type(model.features[idx]) == nn.Conv2d:
            weight = model.features[idx].weight.data
            R = model.conv_back(
                model.features[idx], i_layer, weight, R.data)

    return R[0]


def smooth_grad(x, n, sigma, model, criterion, device):
    x_clone = x.data.clone()
    target = torch.LongTensor(1).to(device=device)

    mean = torch.zeros_like(x)
    std = torch.zeros_like(x)
    std.fill_(sigma)

    total_grad = torch.zeros_like(x)
    for i in range(n):
        sample_x = x_clone + torch.normal(mean, std)
        sample_x.requires_grad_()

        out = model(sample_x.unsqueeze(0))
        target.fill_(out.argmax().item())
        
        loss = criterion(out, target)
        loss.backward()
        grad = sample_x.grad.data.clone()

        total_grad.add_(grad)

    total_grad.div_(n)
    return total_grad


def main(num_images, device_name, seed):
    torch.manual_seed(seed)
    device = torch.device(device_name)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()])

    data = load_data()
    model = models.vgg16(pretrained=True).to(device=device)
    lrp_model = Extractor(
        models.vgg16(pretrained=True), 1, device)
    model.eval()
    criterion = nn.CrossEntropyLoss()

    sigma = 0.2

    it = 0
    for name, (img, label) in data.items():
        image = Image.open(img)

        X = transform(image).to(device=device)
        X.requires_grad_()
        target = torch.LongTensor([label]).to(device=device)

        out = model(X.unsqueeze(0))
        print(name, 'normal:', out.argmax().item(), label)

        target.fill_(out.sort()[1][0][-5])
        loss = criterion(out, target)
        loss.backward()

        simple_grad = X.grad.data.clone()
        with torch.no_grad():
            visualization(simple_grad, 'img/adv/g_'+name+'.png', X)

        sm_grad = smooth_grad(X, 100, 0.2, model, criterion, device)
        with torch.no_grad():
            visualization(sm_grad, 'img/adv/s_'+name+'.png', X)


        # ADV
        adv_X = X.data.clone() + sigma * simple_grad
        adv_X.requires_grad_()

        target = torch.LongTensor([label]).to(device=device)

        out = model(adv_X.unsqueeze(0))
        print(name, 'adv:', out.argmax().item(), label)

        target.fill_(out.argmax().item())
        loss = criterion(out, target)
        loss.backward()

        grad = adv_X.grad.data.clone()
        with torch.no_grad():
            visualization(grad, 'img/adv/ag_'+name+'.png', adv_X)

        sm_grad = smooth_grad(adv_X, 100, 0.2, model, criterion, device)
        with torch.no_grad():
            visualization(sm_grad, 'img/adv/as_'+name+'.png', adv_X)


        neurons, features_layers, classifier_layers = lrp_model(adv_X.unsqueeze(0))
        R = lrp(
            lrp_model, neurons,
            features_layers, classifier_layers,
            adv_X.data.clone().unsqueeze(0))

        visualization(R, 'img/adv/alrp'+name+'.png', X)



        it += 1
        if it == num_images:
            break


if __name__ == '__main__':
    num_images = 3
    device_name = 'cuda'
    seed = 0

    main(num_images, device_name, seed)