
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torch.nn import functional as F

__all__ = ['vgg19']
model_urls = {
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
}

class VGG(nn.Module):
    def __init__(self, features):
        super(VGG, self).__init__()
        self.features = features
        self.reg_layer1 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.density_layer1 = nn.Sequential(nn.Conv2d(128, 1, 1), nn.ReLU())
        
        self.reg_layer2 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.density_layer2 = nn.Sequential(nn.Conv2d(128, 1, 1), nn.ReLU())

    def forward(self, x):
        x = self.features(x)

        # x = F.upsample_bilinear(x, scale_factor=2)
        x = F.interpolate(x, scale_factor=2)

        x1 = self.reg_layer1(x)
        mu1 = self.density_layer1(x1)
        B, C, H, W = mu1.size()
        mu1_sum = mu1.view([B, -1]).sum(1).unsqueeze(1).unsqueeze(2).unsqueeze(3)
        mu1_normed = mu1 / (mu1_sum + 1e-6)
        
        x2 = self.reg_layer2(x)
        mu2 = self.density_layer2(x2)
        mu2_sum = mu2.view([B, -1]).sum(1).unsqueeze(1).unsqueeze(2).unsqueeze(3)
        mu2_normed = mu2 / (mu2_sum + 1e-6)
        
        return mu1,mu1_normed, mu2,mu2_normed
    

def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

cfg = {
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512]
}

def vgg19():
    """VGG 19-layer model (configuration "E")
        model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['E']))
    model.load_state_dict(model_zoo.load_url(model_urls['vgg19']), strict=False)
    return model


def print_model(model, file):
    with open(file, 'w') as f:
        for idx, m in enumerate(model.modules()):
            name = m.__class__.__name__
            f.write('Layer {} ({})\n'.format(idx, name))
            f.write('-' * 50 + '\n')
            f.write(str(m) + '\n')
            f.write('-' * 50 + '\n')

    print('Model structure written to file: ', file)

if __name__ == '__main__':
    
    # Usage example
    model = vgg19()
    # Downloading: "https://download.pytorch.org/models/vgg19-dcbb9e9d.pth" to 
    # C:\Users\10965/.cache\torch\hub\checkpoints\vgg19-dcbb9e9d.pth
    print_model(model, 'model_structure.txt')

