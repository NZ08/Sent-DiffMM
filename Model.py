import torch
from torch import nn
import torch.nn.functional as F
from Params import args
import numpy as np
import random
import math
from Utils.Utils import *

# 使用Xavier初始化方法
init = nn.init.xavier_uniform_
# 使用均匀分布初始化方法
uniformInit = nn.init.uniform


class Model(nn.Module):
    def __init__(self, image_embedding, text_embedding):
        """
        DiffMM模型的主要类
        Args:
            image_embedding: 图像特征嵌入
            text_embedding: 文本特征嵌入
        """
        super(Model, self).__init__()

        # 初始化用户嵌入矩阵
        self.uEmbeds = nn.Parameter(init(torch.empty(args.user, args.latdim)))
        # 初始化物品嵌入矩阵
        self.iEmbeds = nn.Parameter(init(torch.empty(args.item, args.latdim)))
        # 创建多层GCN网络
        self.gcnLayers = nn.Sequential(*[GCNLayer() for i in range(args.gnn_layer)])

        # 边缘随机丢弃层，用于防止过拟合
        self.edgeDropper = SpAdjDropEdge(args.keepRate)

        # 根据不同的转换模式选择特征转换方法
        if args.trans == 1:  # 使用线性层进行转换
            self.image_trans = nn.Linear(args.image_feat_dim, args.latdim)
            self.text_trans = nn.Linear(args.text_feat_dim, args.latdim)
        elif args.trans == 0:  # 使用可学习的转换矩阵
            self.image_trans = nn.Parameter(init(torch.empty(size=(args.image_feat_dim, args.latdim))))
            self.text_trans = nn.Parameter(init(torch.empty(size=(args.text_feat_dim, args.latdim))))
        else:  # 混合模式：图像使用矩阵，文本使用线性层
            self.image_trans = nn.Parameter(init(torch.empty(size=(args.image_feat_dim, args.latdim))))
            self.text_trans = nn.Linear(args.text_feat_dim, args.latdim)

        # 存储图像和文本的特征嵌入
        self.image_embedding = image_embedding
        self.text_embedding = text_embedding

        # 初始化模态权重，用于平衡图像和文本特征的重要性
        self.modal_weight = nn.Parameter(torch.Tensor([0.5, 0.5]))
        self.softmax = nn.Softmax(dim=0)

        # Dropout层，用于防止过拟合
        self.dropout = nn.Dropout(p=0.1)

        # LeakyReLU激活函数，用于非线性变换
        self.leakyrelu = nn.LeakyReLU(0.2)

    def getItemEmbeds(self):
        """获取物品嵌入向量"""
        return self.iEmbeds

    def getUserEmbeds(self):
        """获取用户嵌入向量"""
        return self.uEmbeds

    def getImageFeats(self):
        """
        获取转换后的图像特征
        根据不同的转换模式选择不同的特征转换方法
        """
        if args.trans == 0 or args.trans == 2:
            image_feats = self.leakyrelu(torch.mm(self.image_embedding, self.image_trans))
            return image_feats
        else:
            return self.image_trans(self.image_embedding)

    def getTextFeats(self):
        """
        获取转换后的文本特征
        根据不同的转换模式选择不同的特征转换方法
        """
        if args.trans == 0:
            text_feats = self.leakyrelu(torch.mm(self.text_embedding, self.text_trans))
            return text_feats
        else:
            return self.text_trans(self.text_embedding)

    def forward_MM(self, adj, image_adj, text_adj, att_image_list, att_text_list):
        """
        多模态前向传播函数
        Args:
            adj: 用户-物品交互图的邻接矩阵
            image_adj: 图像相似度邻接矩阵
            text_adj: 文本相似度邻接矩阵
            att_image_list: 图像权重列表，维度为(26495,)
            att_text_list: 文本权重列表，维度为(26495,)
        Returns:
            用户和物品的最终嵌入表示
        """
        # 根据不同的转换模式获取图像和文本特征
        if args.trans == 0:
            # 使用矩阵变换和LeakyReLU激活
            image_feats = self.leakyrelu(torch.mm(self.image_embedding, self.image_trans))
            text_feats = self.leakyrelu(torch.mm(self.text_embedding, self.text_trans))
        elif args.trans == 1:
            # 使用线性层转换
            image_feats = self.image_trans(self.image_embedding)
            text_feats = self.text_trans(self.text_embedding)
        else:
            # 混合模式
            image_feats = self.leakyrelu(torch.mm(self.image_embedding, self.image_trans))
            text_feats = self.text_trans(self.text_embedding)

        # 处理图像模态
        # 1. 基于图像相似度的消息传递
        embedsImageAdj = torch.concat([self.uEmbeds, self.iEmbeds])
        embedsImageAdj = torch.spmm(image_adj, embedsImageAdj)

        # 2. 基于用户-物品交互的图像特征传播
        embedsImage = torch.concat([self.uEmbeds, F.normalize(image_feats)])
        embedsImage = torch.spmm(adj, embedsImage)

        # 3. 二阶消息传播
        embedsImage_ = torch.concat([embedsImage[:args.user], self.iEmbeds])
        embedsImage_ = torch.spmm(adj, embedsImage_)
        embedsImage += embedsImage_

        # 处理文本模态（过程类似图像模态）
        # 1. 基于文本相似度的消息传递
        embedsTextAdj = torch.concat([self.uEmbeds, self.iEmbeds])
        embedsTextAdj = torch.spmm(text_adj, embedsTextAdj)

        # 2. 基于用户-物品交互的文本特征传播
        embedsText = torch.concat([self.uEmbeds, F.normalize(text_feats)])
        embedsText = torch.spmm(adj, embedsText)

        # 3. 二阶消息传播
        embedsText_ = torch.concat([embedsText[:args.user], self.iEmbeds])
        embedsText_ = torch.spmm(adj, embedsText_)
        embedsText += embedsText_

        # 融合相似度信息
        embedsImage += args.ris_adj_lambda * embedsImageAdj
        embedsText += args.ris_adj_lambda * embedsTextAdj

        # 将列表转换为张量
        att_image_tensor = torch.tensor(att_image_list, device=embedsImage.device)
        att_text_tensor = torch.tensor(att_text_list, device=embedsImage.device)
        
        # 计算归一化的权重
        weight_sum = att_image_tensor + att_text_tensor
        # 避免除以零
        weight_sum = torch.where(weight_sum == 0, torch.ones_like(weight_sum), weight_sum)
        
        # 归一化权重
        norm_att_image = att_image_tensor / weight_sum
        norm_att_text = att_text_tensor / weight_sum
        
        # 权重扩展为与嵌入相同的维度
        norm_att_image = norm_att_image.unsqueeze(1).expand_as(embedsImage)
        norm_att_text = norm_att_text.unsqueeze(1).expand_as(embedsText)
        
        # 使用归一化权重融合两个模态的特征
        embedsModal = norm_att_image * embedsImage + norm_att_text * embedsText

        # GCN层的消息传播
        embeds = embedsModal
        embedsLst = [embeds]
        for gcn in self.gcnLayers:
            embeds = gcn(adj, embedsLst[-1])
            embedsLst.append(embeds)
        embeds = sum(embedsLst)  # 残差连接

        # 加入模态信息的残差连接
        embeds = embeds + args.ris_lambda * F.normalize(embedsModal)

        # 返回用户和物品的最终嵌入
        return embeds[:args.user], embeds[args.user:]

    def forward_cl_MM(self, adj, image_adj, text_adj):
        """
        对比学习的多模态前向传播函数
        Args:
            adj: 用户-物品交互图的邻接矩阵
            image_adj: 图像相似度邻接矩阵
            text_adj: 文本相似度邻接矩阵
        Returns:
            两个模态下的用户和物品嵌入表示
        """
        # 根据不同的转换模式获取图像和文本特征
        if args.trans == 0:
            # 使用矩阵变换和LeakyReLU激活
            image_feats = self.leakyrelu(torch.mm(self.image_embedding, self.image_trans))
            text_feats = self.leakyrelu(torch.mm(self.text_embedding, self.text_trans))
        elif args.trans == 1:
            # 使用线性层转换
            image_feats = self.image_trans(self.image_embedding)
            text_feats = self.text_trans(self.text_embedding)
        else:
            # 混合模式
            image_feats = self.leakyrelu(torch.mm(self.image_embedding, self.image_trans))
            text_feats = self.text_trans(self.text_embedding)

        # 处理图像模态的特征传播
        embedsImage = torch.concat([self.uEmbeds, F.normalize(image_feats)])
        embedsImage = torch.spmm(image_adj, embedsImage)

        # 处理文本模态的特征传播
        embedsText = torch.concat([self.uEmbeds, F.normalize(text_feats)])
        embedsText = torch.spmm(text_adj, embedsText)

        # 图像模态的GCN传播
        embeds1 = embedsImage
        embedsLst1 = [embeds1]
        for gcn in self.gcnLayers:
            embeds1 = gcn(adj, embedsLst1[-1])
            embedsLst1.append(embeds1)
        embeds1 = sum(embedsLst1)  # 残差连接

        # 文本模态的GCN传播
        embeds2 = embedsText
        embedsLst2 = [embeds2]
        for gcn in self.gcnLayers:
            embeds2 = gcn(adj, embedsLst2[-1])
            embedsLst2.append(embeds2)
        embeds2 = sum(embedsLst2)  # 残差连接

        # 返回两个模态下的用户和物品嵌入
        return embeds1[:args.user], embeds1[args.user:], embeds2[:args.user], embeds2[args.user:]

    def reg_loss(self):
        """
        计算正则化损失，用于防止过拟合
        Returns:
            用户和物品嵌入的L2正则化损失
        """
        ret = 0
        ret += self.uEmbeds.norm(2).square()  # 用户嵌入的L2正则化
        ret += self.iEmbeds.norm(2).square()  # 物品嵌入的L2正则化
        return ret


class GCNLayer(nn.Module):
    """
    图卷积网络层
    实现了简单的图卷积操作，通过邻接矩阵进行消息传播
    """

    def __init__(self):
        super(GCNLayer, self).__init__()

    def forward(self, adj, embeds):
        """
        前向传播函数
        Args:
            adj: 邻接矩阵
            embeds: 节点特征矩阵
        Returns:
            更新后的节点特征
        """
        return torch.spmm(adj, embeds)


class SpAdjDropEdge(nn.Module):
    """
    稀疏邻接矩阵的边随机丢弃层
    用于防止过拟合，通过随机丢弃一些边来实现数据增强
    """

    def __init__(self, keepRate):
        """
        初始化函数
        Args:
            keepRate: 边的保留概率
        """
        super(SpAdjDropEdge, self).__init__()
        self.keepRate = keepRate

    def forward(self, adj):
        """
        前向传播函数
        Args:
            adj: 稀疏邻接矩阵
        Returns:
            处理后的稀疏邻接矩阵，其中一些边被随机丢弃
        """
        # 获取邻接矩阵的值和索引
        vals = adj._values()
        idxs = adj._indices()
        edgeNum = vals.size()

        # 生成随机掩码，决定哪些边被保留
        mask = ((torch.rand(edgeNum) + self.keepRate).floor()).type(torch.bool)

        # 更新边的权重值并保留选中的边
        newVals = vals[mask] / self.keepRate
        newIdxs = idxs[:, mask]

        # 返回新的稀疏邻接矩阵
        return torch.sparse.FloatTensor(newIdxs, newVals, adj.shape)


class Denoise(nn.Module):
    """
    去噪模块
    用于在扩散过程中对噪声数据进行处理和还原
    实现了一个多层感知机结构来预测噪声
    """
    def __init__(self, in_dims, out_dims, emb_size, norm=False, dropout=0.5):
        """
        初始化函数
        Args:
            in_dims: 输入维度列表，定义了多层感知机的输入维度
            out_dims: 输出维度列表，定义了多层感知机的输出维度
            emb_size: 时间嵌入的维度大小
            norm: 是否使用归一化，默认False
            dropout: dropout比率，默认0.5
        """
        super(Denoise, self).__init__()
        self.in_dims = in_dims
        self.out_dims = out_dims
        self.time_emb_dim = emb_size
        self.norm = norm

        # 时间嵌入的线性层
        self.emb_layer = nn.Linear(self.time_emb_dim, self.time_emb_dim)

        # 构建输入层维度列表（加入时间嵌入维度）
        in_dims_temp = [self.in_dims[0] + self.time_emb_dim] + self.in_dims[1:]
        out_dims_temp = self.out_dims

        # 构建输入和输出层的线性变换
        self.in_layers = nn.ModuleList([nn.Linear(d_in, d_out) for d_in, d_out in zip(in_dims_temp[:-1], in_dims_temp[1:])])
        self.out_layers = nn.ModuleList([nn.Linear(d_in, d_out) for d_in, d_out in zip(out_dims_temp[:-1], out_dims_temp[1:])])

        # Dropout层
        self.drop = nn.Dropout(dropout)
        # 初始化网络参数
        self.init_weights()

    def init_weights(self):
        """
        初始化网络权重
        使用正态分布初始化权重和偏置
        """
        # 初始化输入层权重
        for layer in self.in_layers:
            size = layer.weight.size()
            std = np.sqrt(2.0 / (size[0] + size[1]))
            layer.weight.data.normal_(0.0, std)
            layer.bias.data.normal_(0.0, 0.001)

        # 初始化输出层权重
        for layer in self.out_layers:
            size = layer.weight.size()
            std = np.sqrt(2.0 / (size[0] + size[1]))
            layer.weight.data.normal_(0.0, std)
            layer.bias.data.normal_(0.0, 0.001)

        # 初始化时间嵌入层权重
        size = self.emb_layer.weight.size()
        std = np.sqrt(2.0 / (size[0] + size[1]))
        self.emb_layer.weight.data.normal_(0.0, std)
        self.emb_layer.bias.data.normal_(0.0, 0.001)

    def forward(self, x, timesteps, mess_dropout=True):
        """
        前向传播函数
        Args:
            x: 输入特征
            timesteps: 时间步
            mess_dropout: 是否使用消息dropout，默认True
        Returns:
            处理后的特征
        """
        # 计算时间步的位置编码
        freqs = torch.exp(-math.log(10000) * torch.arange(start=0, end=self.time_emb_dim//2, dtype=torch.float32) / (self.time_emb_dim//2)).cuda()
        temp = timesteps[:, None].float() * freqs[None]
        time_emb = torch.cat([torch.cos(temp), torch.sin(temp)], dim=-1)
        if self.time_emb_dim % 2:
            time_emb = torch.cat([time_emb, torch.zeros_like(time_emb[:, :1])], dim=-1)

        # 时间嵌入的线性变换
        emb = self.emb_layer(time_emb)

        # 特征归一化（如果启用）
        if self.norm:
            x = F.normalize(x)
        # 应用dropout（如果启用）
        if mess_dropout:
            x = self.drop(x)

        # 连接特征和时间嵌入
        h = torch.cat([x, emb], dim=-1)

        # 通过输入层
        for i, layer in enumerate(self.in_layers):
            h = layer(h)
            h = torch.tanh(h)

        # 通过输出层
        for i, layer in enumerate(self.out_layers):
            h = layer(h)
            if i != len(self.out_layers) - 1:
                h = torch.tanh(h)

        return h


# class Denoise(nn.Module):
#     """
#     增强版去噪模块
#     通过加深网络结构、改进时间嵌入处理和优化激活函数提升去噪能力
#     """
#
#     def __init__(self, in_dims, out_dims, emb_size, norm=False, dropout=0.5):
#         super(Denoise, self).__init__()
#         self.in_dims = in_dims
#         self.out_dims = out_dims
#         self.time_emb_dim = emb_size
#         self.norm = norm
#
#         # 增强时间嵌入处理：两层的MLP
#         self.emb_layer = nn.Sequential(
#             nn.Linear(self.time_emb_dim, self.time_emb_dim * 2),
#             nn.SiLU(),
#             nn.Linear(self.time_emb_dim * 2, self.time_emb_dim)
#         )
#
#         # 构建带残差连接的深度网络结构
#         in_dims_temp = [self.in_dims[0] + self.time_emb_dim] + self.in_dims[1:]
#         out_dims_temp = [self.out_dims[0]] + self.out_dims[1:]  # 保持输出维度一致
#
#         self.linear_layer = nn.Linear(in_features=7060, out_features=7050)
#
#         # 输入处理模块（带残差连接）
#         self.in_layers = nn.ModuleList()
#         for d_in, d_out in zip(in_dims_temp[:-1], in_dims_temp[1:]):
#             block = nn.Sequential(
#                 nn.Linear(d_in, d_out),
#                 nn.LayerNorm(d_out),
#                 nn.SiLU(),
#                 nn.Dropout(dropout)
#             )
#             self.in_layers.append(block)
#
#         # 输出处理模块（最后一层不加激活）
#         self.out_layers = nn.ModuleList()
#         for d_in, d_out in zip(out_dims_temp[:-1], out_dims_temp[1:]):
#             if d_out == out_dims_temp[-1]:  # 最后一层
#                 block = nn.Linear(d_in, d_out)
#             else:
#                 block = nn.Sequential(
#                     nn.Linear(d_in, d_out),
#                     nn.LayerNorm(d_out),
#                     nn.SiLU(),
#                     nn.Dropout(dropout)
#                 )
#             self.out_layers.append(block)
#
#         self.drop = nn.Dropout(dropout)
#         self.init_weights()
#
#     def init_weights(self):
#         """改进的权重初始化方法"""
#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('relu'))
#                 if m.bias is not None:
#                     nn.init.zeros_(m.bias)
#
#     def forward(self, x, timesteps, mess_dropout=True):
#         # 时间步的傅里叶特征编码
#         freqs = torch.exp(-math.log(10000) * torch.arange(
#             start=0, end=self.time_emb_dim // 2, dtype=torch.float32
#         )).to(x.device)
#         temp = timesteps[:, None].float() * freqs[None]
#         time_emb = torch.cat([torch.cos(temp), torch.sin(temp)], dim=-1)
#         if self.time_emb_dim % 2:
#             time_emb = torch.cat([time_emb, torch.zeros_like(time_emb[:, :1])], dim=-1)
#
#         # 增强的时间嵌入处理
#         emb = self.emb_layer(time_emb)
#
#         # 输入预处理
#         if self.norm:
#             x = F.layer_norm(x, x.shape[1:])
#         if mess_dropout:
#             x = self.drop(x)
#
#         # 特征融合
#         h = torch.cat([x, emb], dim=-1)
#
#         # 深度特征提取
#         residual = h  # 保存用于残差连接
#         residual = self.linear_layer(residual)
#         for block in self.in_layers:
#             h = block(h)
#
#         # 输出解码
#         for block in self.out_layers:
#             h = block(h)
#
#         return h + residual  # 添加残差连接

class GaussianDiffusion(nn.Module): # 需要修改成适合有情感得分的扩散模型
    """
    高斯扩散模型
    实现了扩散过程中的前向和反向过程，包括噪声添加和去噪
    """
    def __init__(self, noise_scale, noise_min, noise_max, steps, beta_fixed=True):
        """
        初始化函数
        Args:
            noise_scale: 噪声尺度
            noise_min: 最小噪声水平
            noise_max: 最大噪声水平
            steps: 扩散步数
            beta_fixed: 是否固定beta值，默认True
        """
        super(GaussianDiffusion, self).__init__()

        self.noise_scale = noise_scale
        self.noise_min = noise_min
        self.noise_max = noise_max
        self.steps = steps

        if noise_scale != 0:
            # 计算beta值序列
            self.betas = torch.tensor(self.get_betas(), dtype=torch.float64).cuda()
            if beta_fixed:
                self.betas[0] = 0.0001

            # 计算扩散过程需要的各种系数
            self.calculate_for_diffusion()

    def get_betas(self):
        """
        计算beta值序列
        Returns:
            numpy数组，包含每个时间步的beta值
        """
        # 计算方差范围
        start = self.noise_scale * self.noise_min
        end = self.noise_scale * self.noise_max
        variance = np.linspace(start, end, self.steps, dtype=np.float64)
        alpha_bar = 1 - variance
        
        # 计算beta序列
        betas = []
        betas.append(1 - alpha_bar[0])
        for i in range(1, self.steps):
            betas.append(min(1 - alpha_bar[i] / alpha_bar[i-1], 0.999))
        return np.array(betas) 

    def calculate_for_diffusion(self):
        """
        计算扩散过程中需要的各种系数
        包括累积alpha值、方差等
        """
        # 计算alpha值和累积乘积
        alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(alphas, axis=0).cuda()
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]).cuda(), self.alphas_cumprod[:-1]]).cuda()
        self.alphas_cumprod_next = torch.cat([self.alphas_cumprod[1:], torch.tensor([0.0]).cuda()]).cuda()

        # 计算各种系数
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)

        # 计算后验分布的参数
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_log_variance_clipped = torch.log(torch.cat([self.posterior_variance[1].unsqueeze(0), self.posterior_variance[1:]]))
        self.posterior_mean_coef1 = (self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod))
        self.posterior_mean_coef2 = ((1.0 - self.alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - self.alphas_cumprod))

    def p_sample(self, model, x_start, steps, sampling_noise=False):
        """
        采样过程（去噪过程）
        Args:
            model: 去噪模型
            x_start: 初始数据
            steps: 当前步数
            sampling_noise: 是否添加采样噪声，默认False
        Returns:
            去噪后的数据
        """
        # 初始化采样起点
        if steps == 0:
            x_t = x_start
        else:
            t = torch.tensor([steps-1] * x_start.shape[0]).cuda()
            x_t = self.q_sample(x_start, t)
        
        # 逐步去噪
        indices = list(range(self.steps))[::-1]
        for i in indices:
            t = torch.tensor([i] * x_t.shape[0]).cuda()
            model_mean, model_log_variance = self.p_mean_variance(model, x_t, t)
            if sampling_noise:
                # 添加采样噪声
                noise = torch.randn_like(x_t)
                nonzero_mask = ((t!=0).float().view(-1, *([1]*(len(x_t.shape)-1))))
                x_t = model_mean + nonzero_mask * torch.exp(0.5 * model_log_variance) * noise
            else:
                x_t = model_mean
        return x_t

    def q_sample(self, x_start, t, noise=None):
        """
        前向扩散过程采样
        Args:
            x_start: 初始数据
            t: 时间步
            noise: 噪声，如果为None则随机生成
        Returns:
            添加噪声后的数据
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        return self._extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start + self._extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise

    def _extract_into_tensor(self, arr, timesteps, broadcast_shape):
        """
        从数组中提取时间步对应的值并广播到指定形状
        Args:
            arr: 源数组
            timesteps: 时间步
            broadcast_shape: 目标广播形状
        Returns:
            广播后的张量
        """
        arr = arr.cuda()
        res = arr[timesteps].float()
        while len(res.shape) < len(broadcast_shape):
            res = res[..., None]
        return res.expand(broadcast_shape)

    def p_mean_variance(self, model, x, t):
        """
        计算去噪过程中的均值和方差
        Args:
            model: 去噪模型
            x: 输入数据
            t: 时间步
        Returns:
            预测的均值和方差的对数
        """
        model_output = model(x, t, False)

        model_variance = self.posterior_variance
        model_log_variance = self.posterior_log_variance_clipped

        model_variance = self._extract_into_tensor(model_variance, t, x.shape)
        model_log_variance = self._extract_into_tensor(model_log_variance, t, x.shape)

        model_mean = (self._extract_into_tensor(self.posterior_mean_coef1, t, x.shape) * model_output + self._extract_into_tensor(self.posterior_mean_coef2, t, x.shape) * x)
        
        return model_mean, model_log_variance

    def training_losses(self, model, x_start, itmEmbeds, batch_index, model_feats):
        """
        计算扩散模型训练的双重损失（扩散损失 + 图对比损失）

        参数说明：
        x_start: 原始交互数据矩阵 [batch_size, item_num]
                 包含用户真实交互（如评分/点击），可能已融合图像和文本特征
        model_feats: 多模态特征矩阵 [item_num, latent_dim]
                    来自Model类的image_trans/text_trans转换后的图像或文本特征

        数据变化流程：
        原始交互矩阵 → 加噪 → 去噪预测 → 对比特征空间一致性
        """
        batch_size = x_start.size(0)

        # 随机采样时间步（决定加噪强度） 代码行：564-565
        ts = torch.randint(0, self.steps, (batch_size,)).long().cuda()# 如果 self.steps 为5， batch_size 为3，那么可能生成的 ts 张量为 [3, 4, 1] ，表示每个样本在不同的时间步上进行噪声添加。
        noise = torch.randn_like(x_start)  # 生成高斯噪声

        # 前向扩散过程：x_start → x_t 代码行：568-572
        if self.noise_scale != 0:
            # 公式：x_t = sqrt(α_bar_t)x_start + sqrt(1-α_bar_t)ε
            x_t = self.q_sample(x_start, ts, noise)  # 线性混合原始数据和噪声
        else:  # 当noise_scale=0时跳过加噪，用于调试
            x_t = x_start

        # 去噪模型预测 代码行：575
        model_output = model(x_t, ts)  # 预测原始数据（非噪声）

        # 扩散损失计算（带SNR权重）代码行：578-585
        mse = self.mean_flat((x_start - model_output) ** 2)  # 原始数据与预测的MSE
        # SNR权重：早期时间步权重更大（去噪难度更高）
        weight = self.SNR(ts - 1) - self.SNR(ts)  # 信噪比差值作为权重
        weight = torch.where((ts == 0), 1.0, weight)  # 处理初始时间步
        diff_loss = weight * mse  # 时间步自适应的加权损失

        # 图对比损失计算（多模态一致性）代码行：588-590
        # 模型预测结果在特征空间的投影（与图像/文本特征对齐）
        usr_model_embeds = torch.mm(model_output, model_feats)  # [batch, latent]
        # 原始交互数据在ID嵌入空间的投影
        usr_id_embeds = torch.mm(x_start, itmEmbeds)  # [batch, latent]
        # 强制两个空间的一致性（使去噪结果符合多模态特征）
        gc_loss = self.mean_flat((usr_model_embeds - usr_id_embeds) ** 2)

        return diff_loss, gc_loss # 最小化预测交互与真实交互的均方误差，对应论文的ELBO损失（式13）。 # 对齐用户嵌入与模态特征（如image_feats），实现模态感知信号注入（MSI，式14）。
        
    def mean_flat(self, tensor):
        """
        计算张量在除第一维外所有维度上的平均值
        Args:
            tensor: 输入张量
        Returns:
            平均值
        """
        return tensor.mean(dim=list(range(1, len(tensor.shape))))
    
    def SNR(self, t):
        """
        计算信噪比
        Args:
            t: 时间步
        Returns:
            对应时间步的信噪比
        """
        self.alphas_cumprod = self.alphas_cumprod.cuda()
        return self.alphas_cumprod[t] / (1 - self.alphas_cumprod[t])