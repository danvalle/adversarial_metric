from PIL import Image
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.utils import save_image

from utils import load_data, visualization



def main(num_images, device_name, seed):
    torch.manual_seed(seed)
    device = torch.device(device_name)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()])

    data = load_data()
    model = models.vgg16(pretrained=True).to(device=device)
    model.eval()
    criterion = nn.CrossEntropyLoss()

    splits = 3
    split_range = 512/splits
    smooth = 30

    it = 0
    for name, (img, label) in data.items():
        image = Image.open(img)
        X = transform(image).to(device=device)
        # X.requires_grad_()      
        final = torch.zeros(X.size()).to(device=device)
        N = 0

        mean = torch.zeros_like(X)
        std = torch.zeros_like(X)
        std.fill_(0.01)

        for _ in range(smooth):
            sample_x = X.data.clone() + torch.normal(mean, std)
            sample_x.requires_grad_()
       
            for i in range(splits):
                feat = model.features(sample_x.unsqueeze(0))
                # feat[:, int(i*split_range):int((i+1)*split_range), :, :].fill_(0)
                feat[:, :int(i*split_range), :, :].fill_(0)
                feat[:, int((i+1)*split_range):, :, :].fill_(0)
                
                out = model.classifier(feat.view(1,-1))
                print(name, ':', out.argmax().item(), label)

                if out.argmax().item() == label:
                    loss = criterion(out, out.argmax().view(-1))
                    loss.backward()

                    final.add_(sample_x.grad.data.clone())
                    N += 1

                else:
                    loss = criterion(out, out.argmax().view(-1))
                    loss.backward()

                    final.sub_(sample_x.grad.data.clone())
                    N += 1

        if N > 0:
            final.div_(N)
            print(N)
            visualization(final, 'img/noise/'+name+'.png', X)

        it += 1
        if it == num_images:
            break


if __name__ == '__main__':
    num_images = 3
    device_name = 'cuda'
    seed = 0

    main(num_images, device_name, seed)