# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

from collections import OrderedDict

import torch
import torch.nn as nn

import torchvision.models as models
import torch.nn.functional as F


class MobileNetV2Encoder(nn.Module):
    def __init__(self, pretrained=True):
        super(MobileNetV2Encoder, self).__init__()
        # Load a pretrained MobileNetV2 model
        mobilenet = models.mobilenet_v2(pretrained=pretrained)
        self.features = mobilenet.features

        # We need to extract features at different scales.
        # MobileNetV2's features are a sequence of layers:
        # features[0]: Conv (output: 32ch, stride=2, ~1/2 resolution)
        # features[1]: InvertedResidual (output:16ch)
        # features[2-3]: InvertedResidual (output:24ch)
        # features[4-6]: InvertedResidual (output:32ch)
        # features[7-10]: InvertedResidual (output:64ch)
        # features[11-13]: InvertedResidual (output:96ch)
        # features[14-16]: InvertedResidual (output:160ch)
        # features[17]: InvertedResidual (output:320ch)
        # features[18]: Conv (output:1280ch)
        #
        # We want four scales of features at progressively smaller resolutions.
        # For example, we can pick:
        #  - After features[1]: ~1/2 res, 16 channels
        #  - After features[3]: ~1/4 res, 24 channels
        #  - After features[6]: ~1/8 res, 32 channels
        #  - After features[13]: ~1/16 res, 96 channels
        #
        # These four scales roughly match the original code's multiscale approach.
        self.scale_indices = [1, 3, 6, 13]

        # num_ch_enc corresponds to the channel counts of extracted feature maps
        self.num_ch_enc = [16, 24, 32, 96]

    def forward(self, x):
        # Normalize input similar to the original DepthEncoder
        # The original code normalizes as (x - 0.45)/0.225
        x = (x - 0.45) / 0.225

        features = []
        out = x
        for i, layer in enumerate(self.features):
            out = layer(out)
            if i in self.scale_indices:
                features.append(out)

        # 'features' will be a list of 4 tensors [f_half, f_quarter, f_eighth, f_sixteenth]
        return features



class DepthEncoder(nn.Module):
    def __init__(self):
        super(DepthEncoder, self).__init__()
        self.num_ch_enc = [64, 64, 128, 256]
        self.num_ch_enc_build = [64, 64, 128, 256]
        self.convs = nn.ModuleList()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)

        self.relu = nn.ReLU(inplace=True)

        for l, (ch_in, ch_out) in enumerate(zip(self.num_ch_enc_build[:-1], self.num_ch_enc_build[1:])):
            layer = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=2, padding=1, bias=True),
                nn.LeakyReLU(0.1, inplace=True),

                nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
                nn.LeakyReLU(0.1, inplace=True),

                nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
                nn.LeakyReLU(0.1, inplace=True)
            )
            self.convs.append(layer)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = []
        x = (x - 0.45) / 0.225
        x = self.conv1(x)
        x = self.relu(x)
        features.append(x)

        for conv in self.convs:
            x = conv(x)
            features.append(x)

        return features

class ConvBlock(nn.Module):
    """Layer to perform a convolution followed by ELU"""
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        self.pad = nn.ReflectionPad2d(1)
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3)
        self.nonlin = nn.ELU(inplace=True)

    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        out = self.nonlin(out)
        return out

class Decoder(nn.Module):
    def __init__(self, in_channels):
        super(Decoder, self).__init__()
        self.in_channels = in_channels

        self.conv1 = nn.Conv2d(in_channels, 64, 3, 1, 1)
        self.conv2 = nn.Conv2d(64, 32, 3, 1, 1)
        self.conv3 = nn.Conv2d(32, 1, 3, 1, 1)
        self.relu = nn.LeakyReLU(0.1, inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.conv3(self.relu(self.conv2(self.relu(self.conv1(x)))))
        return out

# class DepthDecoder(nn.Module):
#     def __init__(self, num_ch_enc, scales=range(4), use_skips=True):
#         super(DepthDecoder, self).__init__()

#         self.use_skips = use_skips
#         self.scales = scales

#         self.num_ch_enc = num_ch_enc
#         self.num_ch_dec = [16, 32, 64, 96]  

#         # decoder
#         self.convs = OrderedDict()
#         for i in range(3, -1, -1):
#             # upconv_0
#             num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
#             num_ch_out = self.num_ch_dec[i]
#             self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)

#             # upconv_1
#             num_ch_in = self.num_ch_dec[i]
#             if self.use_skips and i == 1:
#                 num_ch_in += self.num_ch_enc[i - 1] #//2
#             num_ch_out = self.num_ch_dec[i]

#             self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)

#         for s in self.scales:
#             self.convs[("dispconv", s)] = Decoder(self.num_ch_dec[s])

#         self.decoder = nn.ModuleList(list(self.convs.values()))
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, input_features):
#         self.outputs = {}

#         x = input_features[-1]  # 1/16
#         for i in range(3, -1, -1):

#             x = self.convs[("upconv", i, 0)](x)
#             x = nn.functional.interpolate(x, scale_factor=2, mode="nearest")

#             if self.use_skips and i > 1:
#                 x += input_features[i - 1]
#             elif self.use_skips and i == 1:
#                 x = [x,input_features[i - 1]]
#                 x = torch.cat(x, 1)

#             x = self.convs[("upconv", i, 1)](x)

#             if i in self.scales:
#                 depth = self.sigmoid(self.convs[("dispconv", i)](x))
#                 self.outputs[("disp", i)] = depth

#         return self.outputs

class DepthDecoder(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), use_skips=True):
        super(DepthDecoder, self).__init__()
        
        self.use_skips = use_skips
        self.scales = scales
        self.num_ch_enc = num_ch_enc
        
        # Adjust decoder channels to match what your encoder provides
        self.num_ch_dec = [16, 32, 64, 96]  
        self.convs = OrderedDict()

        for i in range(3, -1, -1):
            if i == 3:
                # top scale, start decoding from encoder's top-level features (96)
                num_ch_in = self.num_ch_enc[-1]  
            else:
                # for other scales, input channels come from the previous decoded layer
                num_ch_in = self.num_ch_dec[i + 1]

            num_ch_out = self.num_ch_dec[i]
            
            # upconv_0
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)

            # upconv_1
            if self.use_skips and i > 0:
                num_ch_in = num_ch_out + self.num_ch_enc[i - 1]
            else:
                num_ch_in = num_ch_out

            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)

        for s in self.scales:
            self.convs[("dispconv", s)] = Decoder(self.num_ch_dec[s])

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_features):
        self.outputs = {}
        x = input_features[-1]  # start from top-level features (96 ch)

        # decode
        for i in range(3, -1, -1):
            x = self.convs[("upconv", i, 0)](x)
            x = F.interpolate(x, scale_factor=2, mode="nearest")

            if self.use_skips and i > 0:
                x = torch.cat([x, input_features[i - 1]], 1)

            x = self.convs[("upconv", i, 1)](x)

            if i in self.scales:
                disp = self.sigmoid(self.convs[("dispconv", i)](x))
                self.outputs[("disp", i)] = disp

        return self.outputs



class RTMonoDepth(nn.Module):
    def __init__(self):
        super(RTMonoDepth, self).__init__()
        self.encoder = DepthEncoder().to('cuda')
        self.decoder = DepthDecoder(self.encoder.num_ch_enc).to('cuda')

    def forward(self, x):
        return self.decoder(self.encoder(x))[("disp",0)]
