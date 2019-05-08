import numpy as np
import math
import torch.nn as nn
from .utils import unetConv2, unetUp, conv2DBatchNormRelu, conv2DBatchNorm, UnetConv3
import torch
import torch.nn.functional as F
from layers.grid_attention_layer import _GridAttentionBlockND_TORR as AttentionBlock3D
from networks_other import init_weights


class huo_net_conv1(nn.Module):

    def __init__(self, feature_scale=4, n_classes=3, in_channel=2, is_batchnorm=True, n_convs=None,
                 nonlocal_mode='concatenation', aggregation_mode='concat'):
        super(huo_net_conv1, self).__init__()
        self.in_channel = in_channel
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale
        self.n_classes = n_classes
        self.aggregation_mode = aggregation_mode
        self.deep_supervised = True

        self.ec0 = self.encoder(self.in_channel, 16, bias=True, batchnorm=True)
        self.ec1 = self.encoder(16, 32, bias=True, batchnorm=True)
        self.ec2 = self.encoder(32, 32, bias=True, batchnorm=True)
        self.ec3 = self.encoder(32, 64, bias=True, batchnorm=True)
        self.ec4 = self.encoder(64, 64, bias=True, batchnorm=True)
        self.ec5 = self.encoder(64, 128, bias=True, batchnorm=True)
        self.ec6 = self.encoder(128, 128, bias=True, batchnorm=True)
        # self.ec7 = self.encoder(128, 256, bias=True, batchnorm=True)

        self.pool0 = nn.MaxPool3d(2)
        self.pool1 = nn.MaxPool3d(2)
        self.pool2 = nn.MaxPool3d(2)

        filters = [32, 64, 128, 128]
        # filters = [int(x / self.feature_scale) for x in filters]

        ################
        # Attention Maps
        self.compatibility_score1 = AttentionBlock3D(in_channels=filters[2], gating_channels=filters[3],
                                                     inter_channels=filters[3], sub_sample_factor=(1, 1, 1),
                                                     mode=nonlocal_mode)

        self.compatibility_score2 = AttentionBlock3D(in_channels=filters[1], gating_channels=filters[3],
                                                     inter_channels=filters[3], sub_sample_factor=(1, 1, 1),
                                                     mode=nonlocal_mode)

        self.compatibility_score3 = AttentionBlock3D(in_channels=filters[0], gating_channels=filters[3],
                                                     inter_channels=32, sub_sample_factor=(1, 1, 1),
                                                     mode=nonlocal_mode)
        self.bn3D_3 = nn.BatchNorm3d(filters[3])

        # self.compatibility_score4 = AttentionBlock3D(in_channels=filters[0], gating_channels=filters[3],
        #                                              inter_channels=filters[3], sub_sample_factor=(1,1,1),
        #                                              mode=nonlocal_mode)
        # #########################
        # Aggreagation Strategies
        self.attention_filter_sizes = [filters[3], filters[2], filters[1], filters[0]]

        if aggregation_mode == 'concat':
            self.classifier = nn.Linear(filters[3] + 128, n_classes)
            # self.classifier = nn.Linear(filters[2]+filters[3]+filters[1]+256, n_classes)
            # self.classifier = nn.Linear(256, n_classes)
            self.aggregate = self.aggregation_concat

        else:
            self.classifier1 = nn.Linear(filters[2], n_classes)
            self.classifier2 = nn.Linear(filters[3], n_classes)
            self.classifier3 = nn.Linear(filters[3], n_classes)
            self.classifiers = [self.classifier1, self.classifier2, self.classifier3]

            if aggregation_mode == 'mean':
                self.aggregate = self.aggregation_sep

            elif aggregation_mode == 'deep_sup':
                self.classifier = nn.Linear(filters[2] + filters[3] + filters[3], n_classes)
                self.aggregate = self.aggregation_ds

            elif aggregation_mode == 'ft':
                self.classifier = nn.Linear(n_classes * 3, n_classes)
                self.aggregate = self.aggregation_ft
            else:
                raise NotImplementedError

        ####################
        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')

    def encoder(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                bias=True, batchnorm=False):
        if batchnorm:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm3d(out_channels),
                nn.ReLU())
        else:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.ReLU())
        return layer

    def aggregation_sep(self, *attended_maps):
        return [clf(att) for clf, att in zip(self.classifiers, attended_maps)]

    def aggregation_ft(self, *attended_maps):
        preds = self.aggregation_sep(*attended_maps)
        return self.classifier(torch.cat(preds, dim=1))

    def aggregation_ds(self, *attended_maps):
        preds_sep = self.aggregation_sep(*attended_maps)
        pred = self.aggregation_concat(*attended_maps)
        return [pred] + preds_sep

    def aggregation_concat(self, *attended_maps):
        return self.classifier(torch.cat(attended_maps, dim=1))

    def forward(self, inputs):

        if_print = False
        # Feature Extraction
        e0 = self.ec0(inputs)
        syn0 = self.ec1(e0)
        e1 = self.pool0(syn0)
        e2 = self.ec2(e1)
        syn1 = self.ec3(e2)
        if if_print:
            print("e0 = %s" % str(e0.size()))
            print("syn0 = %s" % str(syn0.size()))
            print("e1 = %s" % str(e1.size()))
            print("e2 = %s" % str(e2.size()))
            print("syn1 = %s" % str(syn1.size()))

        e3 = self.pool1(syn1)
        e4 = self.ec4(e3)
        syn2 = self.ec5(e4)
        if if_print:
            print("e3 = %s" % str(e3.size()))
            print("e4 = %s" % str(e4.size()))
            print("syn2 = %s" % str(syn2.size()))

        e5 = self.pool2(syn2)
        e6 = self.ec6(e5)
        # e7 = self.ec7(e6)
        if if_print:
            print("e5 = %s" % str(e5.size()))
            print("e6 = %s" % str(e6.size()))
            # print("e7 = %s" % str(e7.size()))

        batch_size = inputs.size(0)  # inputs.shape[0]
        pooled = F.avg_pool3d(e6, (8, 24, 16)).view(batch_size, -1)
        if if_print:
            print("pooled = %s" % str(pooled.size()))


        # # Attention Mechanism
        g_conv1, att1 = self.compatibility_score1(syn2, e6)
        g_conv1 = self.bn3D_3(g_conv1)
        # print("g_conv1 = %s" % str(g_conv1.size()))
        g1 = F.adaptive_avg_pool3d(g_conv1, 1)
        g1 = g1.view(batch_size,-1)
        # print("g1 = %s" % str(g1.size()))
        # print("pooled = %s" % str(pooled.size()))
        # g3 = F.avg_pool3d(g_conv3, (64, 96, 128)).view(batch_size, -1)
        return self.aggregate(g1, pooled)

    @staticmethod
    def apply_argmax_softmax(pred):
        log_p = F.softmax(pred, dim=1)

        return log_p