import torch
import torch.nn as nn


class Extractor(nn.Module):
    def __init__(self, original_model, epsilon, device):
        super(Extractor, self).__init__()
        original_model = original_model.to(device=device)
        original_model.eval()
        self.features = list(original_model.features)
        self.classifier = list(original_model.classifier)

        self.epsilon = epsilon
        self.device = device

    def forward(self, x):
        neurons = []
        features_layers = []
        classifier_layers = []

        for layer, module in enumerate(self.features):
            x = module(x)
            if type(module) == nn.Conv2d or type(module) == nn.MaxPool2d:
                features_layers.append(layer)
                neurons.append(x)

        x = x.view(len(x), -1)
        
        for layer, module in enumerate(self.classifier):
            x = module(x)
            if type(module) == nn.Linear:
                classifier_layers.append(layer)
                neurons.append(x)

        return neurons, features_layers, classifier_layers

    def linear_back(self, layer, i_layer, weight, R):
        if i_layer.dim() > 2:
            i_layer = i_layer.view(len(i_layer), -1)

        Z = weight.transpose(0,1).mul(i_layer.transpose(0,1))
        
        Zs = Z.sum(0) + layer.bias.data
        Zs += self.epsilon * Zs.ge(0).float().mul(2).sub(1)

        Rx = Z.div(Zs).mul(R).sum(1)
        return Rx

    def maxpool_back(self, layer, i_layer, j_layer, R):
        N, D, H, W = i_layer.shape
        hpool = wpool = layer.kernel_size
        hstride = wstride = layer.stride
        Hout = int((H - hpool) / hstride + 1)
        Wout = int((W - wpool) / wstride + 1)

        Rx = torch.zeros_like(i_layer)

        for i in range(Hout):
            h_start = i*hstride
            h_end = h_start + hpool
            for j in range(Wout):
                w_start = j*wstride
                w_end = w_start + wpool

                Z = j_layer[:, :, i:i+1, j:j+1].eq(
                    i_layer[:, :, h_start:h_end , w_start:w_end]).float()

                Zs = Z.sum(2, keepdim=True).sum(3, keepdim=True)
        
                Rx[:, :, h_start:h_end, w_start:w_end] += (
                    Z.div(Zs).mul(R[:,:,i:i+1,j:j+1]))
        return Rx

    def conv_back(self, layer, i_layer, weight, R):
        N, C, H, W = i_layer.size()
        Hout, Wout = R.size()[2:]
        hf, wf = weight.size()[2:]
        hstride, wstride = layer.stride
        hpadding, wpadding = layer.padding

        pad_layer = torch.zeros(N, C, H+2*hpadding, W+2*wpadding)
        pad_layer = pad_layer.to(self.device)
        pad_layer[:, :, hpadding:H+hpadding, wpadding:W+wpadding] = i_layer.data
        Rx = torch.zeros(N, C, H+2*hpadding, W+2*wpadding).to(self.device)

        for i in range(Hout):
            h_start = i*hstride
            h_end = h_start + hf
            for j in range(Wout):
                w_start = j*wstride
                w_end = w_start + wf

                Z = weight.transpose(0,1).mul(
                    pad_layer.transpose(0,1)[:,:,h_start:h_end, w_start:w_end])

                Zs = Z.sum(0,keepdim=True).sum(2,keepdim=True).sum(3,keepdim=True)
                Zs = Zs + layer.bias.data.view(Zs.size())
                
                if self.epsilon == 0:
                    Zs += 1e-12 * Zs.ge(0).float().mul(2).sub(1)
                else:
                    Zs += self.epsilon * Zs.ge(0).float().mul(2).sub(1)
                
                Rx[:, :, h_start:h_end, w_start:w_end] += (
                    (Z.div(Zs).mul(R[:,:,i:i+1,j:j+1])).sum(1))

        return Rx[:, :, hpadding:-hpadding, wpadding:-wpadding]

