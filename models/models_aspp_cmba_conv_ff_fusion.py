
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torch.nn import functional as F
import torch
import torch.nn.init as init

__all__ = ['vgg19']
model_urls = {
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
}
# add
class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):  # 需修改 reduction=16
        super(CBAM, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0)
        self.sigmoid_channel = nn.Sigmoid()
        self.conv_after_concat = nn.Conv2d(channels * 2, channels, kernel_size=1, padding=0)
        self.sigmoid_spatial = nn.Sigmoid()

    def forward(self, x):

        # 对输入张量x在通道维度上分别进行最大池化和平均池化
        # 得到两个形状为(B, C, 1, 1)的张量max_out和avg_out
        # 将这两个张量分别送入两个全连接层，输出形状均为(B, C//reduction_factor)
        # 对这两个输出进行ReLU激活，
        # 对channel_out张量在通道维度上进行Sigmoid函数激活，得到形状为(B, C, 1, 1)的注意力权重张量。
        # 将x和channel_out张量相乘，得到通道注意力机制作用后的张量spatial_out。
        # 将spatial_out张量与原始输入x在通道维度上拼接起来，得到形状为(B, 2C, H, W)的张量concat_out。
        # 将concat_out张量送入一个卷积层，并在空间维度上进行Sigmoid激活，得到形状与输入x相同的CBAM注意力权重张量output。
        # 将输入张量x和output张量相乘，得到通过CBAM注意力机制调整后的张量作为最终输出
        
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        channel_out = self.sigmoid_channel(avg_out + max_out)
        spatial_out = torch.mul(x, channel_out)
        concat_out = torch.cat((spatial_out, x), dim=1)
        output = self.conv_after_concat(concat_out)
        output = self.sigmoid_spatial(output)

        return torch.mul(x, output)
# 空洞卷积
class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        super(ASPPConv, self).__init__(*modules)

# 池化 -> 1*1 卷积 -> 上采样
class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),  # 自适应均值池化
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())

    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        # 上采样
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)  

# 整个 ASPP 架构
class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates, out_channels=512):
        super(ASPP, self).__init__()
        modules = []
        # 1*1 卷积
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()))

        # 多尺度空洞卷积
        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))

        # 池化
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)
        
        # 拼接后的卷积
        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5))
        

                # 使用xavier_uniform_初始化所有卷积层的权重和偏置
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)
    



class VGG(nn.Module):
    def __init__(self, features):
        super(VGG, self).__init__()
        self.features = features

        self.cbam = CBAM(channels=512)  # 添加 CBAM 模块
        self.aspp = ASPP(512,[1,2,5],512)
        
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
        self.convs_bn = nn.Sequential(
            nn.Conv2d(512, 512, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU())
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

    def forward(self, x):

        x1 = self.features[:4](x)   # 64
        x2 = self.features[4:9](x1)  #  128
        x3 = self.features[9:18](x2)  # 256
        x4 = self.features[18:27](x3)  #512
        x5_00 = self.features[27:35](x4)  # 512

        x5_0 = self.aspp(x5_00)
        # x = F.upsample_bilinear(x, scale_factor=2)
        x5 = F.interpolate(x5_00, scale_factor=2)

        x5 = x5 + x4

        x5_3 = F.interpolate(x5_0, scale_factor=2)
        x5_1 = self.cbam(x5_3)



        x6 = self.reg_layer1(x5)
        mu1 = self.density_layer1(x6)
        B, C, H, W = mu1.size()
        mu1_sum = mu1.view([B, -1]).sum(1).unsqueeze(1).unsqueeze(2).unsqueeze(3)
        mu1_normed = mu1 / (mu1_sum + 1e-6)
        
        x7 = self.reg_layer2(x5_1)
        mu2 = self.density_layer2(x7)
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

