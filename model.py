import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=0.125):
        super(ChannelAttention, self).__init__()
        reduced_channels = int(in_channels * reduction_ratio)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, reduced_channels, kernel_size=1, bias=True)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(reduced_channels, in_channels, kernel_size=1, bias=True)

    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return torch.sigmoid(out) * x

class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        x = torch.cat([max_out, avg_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x) * x

class CSA(nn.Module):
    def __init__(self, in_channels, reduction_ratio=0.5):
        super(CSA, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        channel_refined = self.channel_attention(x)
        spatial_refined = self.spatial_attention(channel_refined)
        refined = channel_refined * spatial_refined
        return refined + x

class SelfAttention(nn.Module):
    def __init__(self, ch):
        super(SelfAttention, self).__init__()
        self.channels = ch
        self.filters_f_g = ch // 8
        self.filters_h = ch

        self.gamma = nn.Parameter(torch.zeros(1))
        self.conv_f = nn.Conv2d(ch, self.filters_f_g, kernel_size=1)
        self.conv_g = nn.Conv2d(ch, self.filters_f_g, kernel_size=1)
        self.conv_h = nn.Conv2d(ch, self.filters_h, kernel_size=1)

    def forward(self, x):
        batch_size, C, width, height = x.size()

        f = self.conv_f(x)  # [bs, c', w, h]
        g = self.conv_g(x)  # [bs, c', w, h]
        h = self.conv_h(x)  # [bs, c, w, h]

        f_flat = f.view(batch_size, -1, width * height)  # [bs, c', N]
        g_flat = g.view(batch_size, -1, width * height)  # [bs, c', N]
        h_flat = h.view(batch_size, -1, width * height)  # [bs, c, N]

        s = torch.bmm(g_flat.permute(0, 2, 1), f_flat)  # [bs, N, N]
        beta = F.softmax(s, dim=-1)  # attention map

        o = torch.bmm(h_flat, beta.permute(0, 2, 1))  # [bs, c, N]
        o = o.view(batch_size, C, width, height)  # [bs, C, w, h]

        x = self.gamma * o + x
        return x

class BreastCancerModel(nn.Module):
    def __init__(self, num_classes):
        super(BreastCancerModel, self).__init__()
        # Load pre-trained ResNet models for CC and MLO views
        self.resnet_cc = models.resnet50(pretrained=True).to(device)  # Explicitly move to device
        self.resnet_mlo = models.resnet50(pretrained=True).to(device)  # Explicitly move to device

        # Freeze all layers initially
        for param in self.resnet_cc.parameters():
            param.requires_grad = False
        for param in self.resnet_mlo.parameters():
            param.requires_grad = False

        # Unfreeze the last two layers of the ResNet for both CC and MLO models, including Layer 4
        for layer in [self.resnet_cc.layer4, self.resnet_cc.avgpool]:
            for param in layer.parameters():
                param.requires_grad = True
        for layer in [self.resnet_mlo.layer4, self.resnet_mlo.avgpool]:
            for param in layer.parameters():
                param.requires_grad = True

        # Modify ResNet models to extract features up to 'layer4' (conv4_block6_out equivalent)
        self.resnet_cc = nn.Sequential(*list(self.resnet_cc.children())[:-2]).to(device)  # Remove fully connected layers
        self.resnet_mlo = nn.Sequential(*list(self.resnet_mlo.children())[:-2]).to(device)  # Remove fully connected layers

        # Add BatchNormalization to the outputs of ResNet models
        self.bn_mlo = nn.BatchNorm2d(2048).to(device)  # BatchNormalization for MLO output
        self.bn_cc = nn.BatchNorm2d(2048).to(device)  # BatchNormalization for CC output

        # Channel Spatial Attention module
        self.csa = CSA(8192).to(device)  # Updated to reflect concatenation of four tensors
        self.bn_csa = nn.BatchNorm2d(8192).to(device)  # BatchNormalization after CSA

        # Final layers
        self.dropout = nn.Dropout(0.5)  # Dropout for regularization
        self.bn_fc = nn.BatchNorm1d(8192).to(device)  # BatchNormalization before fully connected layer
        self.fc = nn.Linear(8192, num_classes).to(device)  # Final classifier layer

    def forward(self, x_cc1, x_cc2, x_mlo1, x_mlo2):

        # Forward pass through ResNet models for all four inputs
        x_cc1 = self.resnet_cc(x_cc1)
        x_cc2 = self.resnet_cc(x_cc2)
        x_mlo1 = self.resnet_mlo(x_mlo1)
        x_mlo2 = self.resnet_mlo(x_mlo2)

        # Apply BatchNormalization to ResNet outputs
        x_cc1 = self.bn_cc(x_cc1)
        x_cc2 = self.bn_cc(x_cc2)
        x_mlo1 = self.bn_mlo(x_mlo1)
        x_mlo2 = self.bn_mlo(x_mlo2)

        # Concatenate all four feature maps along the channel axis
        x = torch.cat((x_cc1, x_cc2, x_mlo1, x_mlo2), dim=1)
        
        x = self.csa(x)  # Apply Channel Spatial Attention
        #x = self.bn_fc(x.float())
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.dropout(x)  # Apply Dropout
        x = self.fc(x)
        return x
