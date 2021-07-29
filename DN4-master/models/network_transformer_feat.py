import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import functools
import pdb
import math
import sys
import numpy as np

sys.dont_write_bytecode = True

''' 

	This Network is designed for Few-Shot Learning Problem. 

'''

###############################################################################
# Functions
###############################################################################

from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM


# X: (N,3,H,W) a batch of non-negative RGB images (0~255)
# Y: (N,3,H,W)

# calculate ssim & ms-ssim for each image
# ssim_val = ssim( X, Y, data_range=255, size_average=False) # return (N,)
# ms_ssim_val = ms_ssim( X, Y, data_range=255, size_average=False ) #(N,)
#
# # set 'size_average=True' to get a scalar value as loss. see tests/tests_loss.py for more details
# ssim_loss = 1 - ssim( X, Y, data_range=255, size_average=True) # return a scalar
# ms_ssim_loss = 1 - ms_ssim( X, Y, data_range=255, size_average=True )
#
# # reuse the gaussian kernel with SSIM & MS_SSIM.
# ssim_module = SSIM(data_range=255, size_average=True, channel=3)
# ms_ssim_module = MS_SSIM(data_range=255, size_average=True, channel=3)
#
# ssim_loss = 1 - ssim_module(X, Y)
# ms_ssim_loss = 1 - ms_ssim_module(X, Y)


def weights_init_normal(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_xavier(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.xavier_normal_(m.weight.data, gain=0.02)
    elif classname.find('Linear') != -1:
        init.xavier_normal_(m.weight.data, gain=0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    print(classname)
    if classname.find('Conv') != -1:
        init.orthogonal_(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.orthogonal_(m.weight.data, gain=1)
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def init_weights(net, init_type='normal'):
    print('initialization method [%s]' % init_type)
    if init_type == 'normal':
        net.apply(weights_init_normal)
    elif init_type == 'xavier':
        net.apply(weights_init_xavier)
    elif init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def define_DN4Net(pretrained=False, model_root=None, which_model='Conv64', norm='batch', init_type='normal',
                  use_gpu=True, **kwargs):
    DN4Net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if use_gpu:
        assert (torch.cuda.is_available())

    if which_model == 'Conv64F':
        DN4Net = FourLayer_64F(norm_layer=norm_layer, **kwargs)
    elif which_model == 'ResNet256F':
        net_opt = {'userelu': False, 'in_planes': 3, 'dropout': 0.5, 'norm_layer': norm_layer}
        DN4Net = ResNetLike(net_opt)
    else:
        raise NotImplementedError('Model name [%s] is not recognized' % which_model)
    init_weights(DN4Net, init_type=init_type)

    if use_gpu:
        DN4Net.cuda()

    if pretrained:
        DN4Net.load_state_dict(model_root)

    return DN4Net


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v):
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature
        log_attn = F.log_softmax(attn, 2)
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)
        return output, attn, log_attn


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)  # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)  # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)  # (n*b) x lv x dv

        output, attn, log_attn = self.attention(q, k, v)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)  # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output


##############################################################################
# Classes: FourLayer_64F
##############################################################################

# Model: FourLayer_64F
# Input: One query image and a support set
# Base_model: 4 Convolutional layers --> Image-to-Class layer
# Dataset: 3 x 84 x 84, for miniImageNet
# Filters: 64->64->64->64
# Mapping Sizes: 84->42->21->21->21


class FourLayer_64F(nn.Module):
    def __init__(self, norm_layer=nn.BatchNorm2d, num_classes=5, neighbor_k=3):
        super(FourLayer_64F, self).__init__()

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.features = nn.Sequential(  # 3*84*84
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(64),
            nn.LeakyReLU(0.2, True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 64*42*42

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(64),
            nn.LeakyReLU(0.2, True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 64*21*21

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(64),
            nn.LeakyReLU(0.2, True),  # 64*21*21
            nn.MaxPool2d(kernel_size=2, stride=2),  # 64*10*10

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(64),
            nn.LeakyReLU(0.2, True),  # 64*7*7
            nn.MaxPool2d(kernel_size=2, stride=2),  # 64*5*5
        )

        self.slf_attn = MultiHeadAttention(1, 64, 64, 64, dropout=0.5)
        self.imgtoclass = ImgtoClass_Metric(neighbor_k=neighbor_k)  # 1*num_classes

    def forward(self, input1, input2):
        # extract features of input1--query image
        q_feature_map = self.features(input1)
        q = nn.MaxPool2d(5)(q_feature_map)
        q = q.view(q.size(0), -1)

        # extract features of input2--support set
        S = torch.zeros((3, 64)).cuda()
        for i in range(len(input2)):
            support_set_sam_feature_map = self.features(input2[i])
            support_set_sam = nn.MaxPool2d(5)(support_set_sam_feature_map)
            support_set_sam = support_set_sam.view(support_set_sam.size(0), -1)

            support_set_sam = support_set_sam.mean(dim=0, keepdim=True)

            S[i, :] = support_set_sam
        S = S.unsqueeze(0)
        S = self.slf_attn(S, S, S)
        # print("-----------------"+str(len(S)))
        x = self.imgtoclass(q, S)  # get Batch*num_classes
        return x


# ========================== Define an image-to-class layer ==========================#


class ImgtoClass_Metric(nn.Module):
    def __init__(self, neighbor_k=3):
        super(ImgtoClass_Metric, self).__init__()
        self.neighbor_k = neighbor_k

    # Calculate the k-Nearest Neighbor of each local descriptor
    def cal_cosinesimilarity(self, input1, input2):
        B, C = input1.size()  # 15，64
        Similarity_list = []

        for i in range(B):
            query_sam = input1[i].unsqueeze(0)  # 1*64
            query_sam_norm = torch.norm(query_sam, 2, 1,
                                        True)  # calculate 2-norm for each row (ie. for each descriptor)
            query_sam = query_sam / query_sam_norm  # normalization

            if torch.cuda.is_available():
                inner_sim = torch.zeros(1, 3).cuda()  # 1*3

            input2 = input2.squeeze() # 3*64
            for j in range(3):  # j = 0,1,2
                support_set_sam = input2[j].unsqueeze(-1)  # 64*1
                support_set_sam_norm = torch.norm(support_set_sam, 2, 0,
                                                  True)  # calculate 2-norm for each column(ie. for each descriptor)
                support_set_sam = support_set_sam / support_set_sam_norm  # normalization

                # cosine similarity between a query sample and a support category
                innerproduct_matrix = query_sam @ support_set_sam  # 1

                inner_sim[0, j] = innerproduct_matrix

            Similarity_list.append(inner_sim)

        Similarity_list = torch.cat(Similarity_list, 0)  # 15*3 (3 classes)

        return Similarity_list

    # def cal_euclideandistance(self, input1, input2):
    #     B, C, h, w = input1.size()
    #     Similarity_list = []
    #
    #     for i in range(B):
    #         query_sam = input1[i]
    #         query_sam = query_sam.view(C, -1)
    #         query_sam = torch.transpose(query_sam, 0, 1)
    #         # print(query_sam.shape)
    #         # query_sam_norm = torch.norm(query_sam, 2, 1, True)
    #         # query_sam = query_sam/query_sam_norm
    #
    #         if torch.cuda.is_available():
    #             inner_sim = torch.zeros(1, len(input2)).cuda()
    #             # print("inner_sim"+str(inner_sim.shape))
    #
    #         for j in range(len(input2)):
    #             support_set_sam = input2[j]
    #             # print(support_set_sam.shape)
    #             support_set_sam = torch.transpose(support_set_sam, 0, 1)
    #             # support_set_sam_norm = torch.norm(support_set_sam, 2, 0, True)
    #             # support_set_sam = support_set_sam/support_set_sam_norm
    #
    #             # euclidean distance between a query sample and a support category
    #             innerproduct_matrix = torch.cdist(query_sam, support_set_sam, p=2)
    #             # print("innerproduct"+str(innerproduct_matrix.shape))
    #
    #             # choose the top-k nearest neighbors
    #             topk_value, topk_index = torch.topk(innerproduct_matrix, self.neighbor_k, 1, largest=False)
    #             # print(topk_value, topk_index)
    #             inner_sim[0, j] = torch.sum(topk_value)
    #
    #         Similarity_list.append(inner_sim)
    #
    #     Similarity_list = torch.cat(Similarity_list, 0)
    #     # print(Similarity_list.shape)
    #
    #     return Similarity_list
    #
    # def cal_SSIM(self, input1, input2):
    #     # print(input1.shape) # 15*49*64
    #
    #     input1 = input1.view(15, 7, 7, 64).permute(0, 3, 1, 2)
    #
    #     B, C, h, w = input1.size()  # 15，64，7, 7
    #     Similarity_list = []
    #
    #     for i in range(B):
    #         query_sam = input1[i]  # 64*7*7
    #         query_sam_norm = torch.norm(query_sam, 2, 0, True)
    #         query_sam = (query_sam / query_sam_norm).unsqueeze(0)
    #         query_sam = query_sam.repeat(5, 1, 1, 1)
    #
    #         if torch.cuda.is_available():
    #             inner_sim = torch.zeros(1, len(input2)).cuda()  # 1*3
    #
    #         for j in range(len(input2)):  # j = 0,1,2
    #             support_set_sam = input2[j].permute(0, 2, 1).view(5, 64, 7, 7)  # 5*64*7*7
    #             # print(support_set_sam.shape)
    #             support_set_sam_norm = torch.norm(support_set_sam, 2, 1, True)
    #             support_set_sam = support_set_sam / support_set_sam_norm  # normalization
    #
    #             # SSIM between a query sample and a support category
    #             innerproduct_matrix = ssim(query_sam, support_set_sam, data_range=1, size_average=False)
    #
    #             # choose the top-k nearest neighbors
    #             # topk_value, topk_index = torch.topk(innerproduct_matrix, self.neighbor_k, 1)  # 441*3
    #             inner_sim[0, j] = torch.sum(innerproduct_matrix)  # sum mk value
    #
    #         Similarity_list.append(inner_sim)
    #
    #     Similarity_list = torch.cat(Similarity_list, 0)  # 15*3 (3 classes)
    #
    #     return Similarity_list
    #
    def forward(self, x1, x2):

        Similarity_list = self.cal_cosinesimilarity(x1, x2)

        return Similarity_list


##############################################################################
# Classes: ResNetLike
##############################################################################

# Model: ResNetLike
# Refer to: https://github.com/gidariss/FewShotWithoutForgetting
# Input: One query image and a support set
# Base_model: 4 ResBlock layers --> Image-to-Class layer
# Dataset: 3 x 84 x 84, for miniImageNet
# Filters: 64->96->128->256


class ResBlock(nn.Module):
    def __init__(self, nFin, nFout):
        super(ResBlock, self).__init__()

        self.conv_block = nn.Sequential()
        self.conv_block.add_module('BNorm1', nn.BatchNorm2d(nFin))
        self.conv_block.add_module('LRelu1', nn.LeakyReLU(0.2))
        self.conv_block.add_module('ConvL1', nn.Conv2d(nFin, nFout, kernel_size=3, padding=1, bias=False))
        self.conv_block.add_module('BNorm2', nn.BatchNorm2d(nFout))
        self.conv_block.add_module('LRelu2', nn.LeakyReLU(0.2))
        self.conv_block.add_module('ConvL2', nn.Conv2d(nFout, nFout, kernel_size=3, padding=1, bias=False))
        self.conv_block.add_module('BNorm3', nn.BatchNorm2d(nFout))
        self.conv_block.add_module('LRelu3', nn.LeakyReLU(0.2))
        self.conv_block.add_module('ConvL3', nn.Conv2d(nFout, nFout, kernel_size=3, padding=1, bias=False))

        self.skip_layer = nn.Conv2d(nFin, nFout, kernel_size=1, stride=1)

    def forward(self, x):
        return self.skip_layer(x) + self.conv_block(x)


class ResNetLike(nn.Module):
    def __init__(self, opt, neighbor_k=3):
        super(ResNetLike, self).__init__()

        self.in_planes = opt['in_planes']
        self.out_planes = [64, 96, 128, 256]
        self.num_stages = 4

        if type(opt['norm_layer']) == functools.partial:
            use_bias = opt['norm_layer'].func == nn.InstanceNorm2d
        else:
            use_bias = opt['norm_layer'] == nn.InstanceNorm2d

        if type(self.out_planes) == int:
            self.out_planes = [self.out_planes for i in range(self.num_stages)]

        assert (type(self.out_planes) == list)
        assert (len(self.out_planes) == self.num_stages)
        num_planes = [self.out_planes[0], ] + self.out_planes
        userelu = opt['userelu'] if ('userelu' in opt) else False
        dropout = opt['dropout'] if ('dropout' in opt) else 0

        self.feat_extractor = nn.Sequential()
        self.feat_extractor.add_module('ConvL0', nn.Conv2d(self.in_planes, num_planes[0], kernel_size=3, padding=1))

        for i in range(self.num_stages):
            self.feat_extractor.add_module('ResBlock' + str(i), ResBlock(num_planes[i], num_planes[i + 1]))
            if i < self.num_stages - 2:
                self.feat_extractor.add_module('MaxPool' + str(i), nn.MaxPool2d(kernel_size=2, stride=2, padding=0))

        self.feat_extractor.add_module('ReluF1', nn.LeakyReLU(0.2, True))  # get Batch*256*21*21

        self.imgtoclass = ImgtoClass_Metric(neighbor_k=neighbor_k)  # Batch*num_classes

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, input1, input2):

        # extract features of input1--query image
        q = self.feat_extractor(input1)

        # extract features of input2--support set
        S = []
        for i in range(len(input2)):
            support_set_sam = self.feat_extractor(input2[i])
            B, C, h, w = support_set_sam.size()
            support_set_sam = support_set_sam.permute(1, 0, 2, 3)
            support_set_sam = support_set_sam.contiguous().view(C, -1)
            S.append(support_set_sam)

        x = self.imgtoclass(q, S)  # get Batch*num_classes

        return x


