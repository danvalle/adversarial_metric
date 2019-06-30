from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import interpolate
from torch.autograd import Function
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.utils import save_image

from utils import load_data, visualization


class FeatureExtractor():
    """ Class for extracting activations and 
    registering gradients from targetted intermediate layers """
    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []
        for name, module in self.model._modules.items():
            x = module(x)
            if name in self.target_layers:
                x.register_hook(self.save_gradient)
                outputs += [x]
        return outputs, x


class ModelOutputs():
    """ Class for making a forward pass, and getting:
    1. The network output.
    2. Activations from intermeddiate targetted layers.
    3. Gradients from intermeddiate targetted layers. """
    def __init__(self, model, target_layers):
        self.model = model
        self.feature_extractor = FeatureExtractor(self.model.features, target_layers)

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):
        target_activations, output  = self.feature_extractor(x)
        output = output.view(output.size(0), -1)
        output = self.model.classifier(output)
        return target_activations, output


class GradCam:

    def __init__(self, model, target_layer_names):
        self.model = model
        self.extractor = ModelOutputs(self.model, target_layer_names)

    def forward(self, input):
        return self.model(input) 

    def __call__(self, input, index):
        input = input.requires_grad_()
        features, output = self.extractor(input)

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_()
        one_hot = torch.sum(one_hot.cuda() * output)
        
        self.model.features.zero_grad()
        self.model.classifier.zero_grad()
        one_hot.backward(retain_graph=True)

        grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()

        target = features[-1]
        target = target[0, :]

        weights = torch.Tensor(np.mean(grads_val, axis=(2, 3))[0, :]).cuda()
        cam = torch.zeros(target.size()[1:]).cuda()
        
        for i, w in enumerate(weights):
            cam += w * target[i, :, :]

        cam = cam.unsqueeze(0).repeat(3, 1, 1).unsqueeze(0)
        cam = interpolate(cam, size=(224,224), mode='bilinear', align_corners=False)
        return cam[0]


class GuidedBackpropReLU(Function):

    def forward(self, input):
        positive_mask = (input > 0).type_as(input)
        output = torch.addcmul(torch.zeros(input.size()).type_as(input), input, positive_mask)
        self.save_for_backward(input, output)
        return output

    def backward(self, grad_output):
        input, output = self.saved_tensors
        grad_input = None

        positive_mask_1 = (input > 0).type_as(grad_output)
        positive_mask_2 = (grad_output > 0).type_as(grad_output)
        grad_input = torch.addcmul(
            torch.zeros(input.size()).type_as(input),
            torch.addcmul(torch.zeros(input.size()).type_as(input), grad_output, positive_mask_1),
            positive_mask_2)

        return grad_input


class GuidedBackpropReLUModel:

    def __init__(self, model):
        self.model = model

        # replace ReLU with GuidedBackpropReLU
        for idx, module in self.model.features._modules.items():
            if module.__class__.__name__ == 'ReLU':
                self.model.features._modules[idx] = GuidedBackpropReLU()

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, index):
        input = input.requires_grad_()
        output = self.forward(input)

        one_hot = torch.zeros(1, output.size()[-1]).cuda()
        one_hot[0][index] = 1

        one_hot = torch.sum(one_hot * output)
        one_hot.backward(retain_graph=True)

        return input.grad[0]


def main(num_images, device_name, seed):
    torch.manual_seed(seed)
    device = torch.device(device_name)
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()])
 
    data = load_data()
    model = models.vgg16(pretrained=True).to(device=device)
    model2 = models.vgg16(pretrained=True).to(device=device)
    model.eval()
    model2.eval()
    criterion = nn.CrossEntropyLoss()

    gcam_model = GradCam(model, ["29"])
    gb_model = GuidedBackpropReLUModel(model2)

    sotfmax = nn.Softmax(dim=1)
    print('Name,Label,Pred,Conf,Loss', flush=True)
    it = 0
    for name, (img, label) in data.items():
        image = Image.open(img).convert('RGB')
        X = transform(image).to(device)
        X.requires_grad_()
        out = model(X.unsqueeze(0))
        
        if out.argmax().item() != label:
            continue

        loss = criterion(out, out.argmax().view(-1))
        loss.backward()

        # Save log
        print('{},{},{},{},{}'.format(
            name, label, out.argmax().item(),
            sotfmax(out.detach().clone()).max(), loss.item()), flush=True)

        gcam = gcam_model(X.data.clone().unsqueeze(0), label)
        with torch.no_grad():
            visualization(gcam, 'img/gradcam/', name+'.png', X)

        gb = gb_model(X.data.clone().unsqueeze(0), label)
        with torch.no_grad():
            visualization(gb, 'img/guided/', name+'.png', X)

        cam_gb = torch.mul(gcam, gb)
        with torch.no_grad():
            visualization(cam_gb, 'img/guidedcam/', name+'.png', X)

        it += 1
        if it == num_images:
            break


if __name__ == '__main__':
    num_images = 5000
    device_name = 'cuda'
    seed = 0

    main(num_images, device_name, seed)
