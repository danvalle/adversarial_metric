from PIL import Image
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.utils import save_image

from utils import load_data, visualization


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
        transforms.Resize((224,224)),
        transforms.ToTensor()])

    data = load_data()
    model = models.vgg16(pretrained=True).to(device=device)
    model.eval()
    criterion = nn.CrossEntropyLoss()

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
        
        save_image(X.clone(), 'img/original/'+name+'.png')

        simple_grad = X.grad.data.clone()
        with torch.no_grad():
            visualization(simple_grad, 'img/simple/', name+'.png', X)

        sm_grad = smooth_grad(X, 100, 0.2, model, criterion, device)
        with torch.no_grad():
            visualization(sm_grad, 'img/smooth/', name+'.png', X)

        it += 1
        if it == num_images:
            break


if __name__ == '__main__':
    num_images = 5000
    device_name = 'cuda'
    seed = 0

    main(num_images, device_name, seed)
