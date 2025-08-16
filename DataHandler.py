import pickle
import numpy as np
from scipy.sparse import coo_matrix
from Params import args
import scipy.sparse as sp
import torch
import torch.utils.data as data
import torch.utils.data as dataloader
from collections import defaultdict
from tqdm import tqdm
import random

class DataHandler:
	"""数据处理类，负责加载和预处理数据集"""
	def __init__(self):
		# 根据数据集类型设置数据目录
		if args.data == 'baby':
			predir = './Datasets/baby/'
		elif args.data == 'sports':
			predir = './Datasets/sports/'
		elif args.data == 'Toys':
			predir = './Datasets/Toys/'
		elif args.data == 'Clothing':
			predir = './Datasets/Clothing/'
		self.predir = predir
		# 设置训练集和测试集文件路径
		self.trnfile = predir + 'trnMat.pkl'  # 训练矩阵文件
		self.tstfile = predir + 'tstMat.pkl'  # 测试矩阵文件

		# 设置多模态特征文件路径
		self.imagefile = predir + 'image_feat.npy'  # 图像特征文件
		self.textfile = predir + 'text_feat.npy'  # 文本特征文件

	def loadOneFile(self, filename):
		"""加载并保留原始评分"""
		with open(filename, 'rb') as fs:
			ret = pickle.load(fs).astype(np.float32)  # 移除二值化转换
		if type(ret) != coo_matrix:
			ret = sp.coo_matrix(ret)
		return ret

	def normalizeAdj(self, mat):
		"""对邻接矩阵进行归一化
		Args:
			mat: 输入矩阵
		Returns:
			归一化后的矩阵
		"""
		degree = np.array(mat.sum(axis=-1))  # 计算节点的度
		dInvSqrt = np.reshape(np.power(degree, -0.5), [-1])  # 计算度的-1/2次方
		dInvSqrt[np.isinf(dInvSqrt)] = 0.0  # 处理无穷大的情况
		dInvSqrtMat = sp.diags(dInvSqrt)  # 创建对角矩阵
		# 执行归一化: D^(-1/2) * A * D^(-1/2)
		# ret = mat.dot(dInvSqrtMat).transpose().dot(dInvSqrtMat).multiply(mat)

		return mat.dot(dInvSqrtMat).transpose().dot(dInvSqrtMat).tocoo()

	def makeTorchAdj(self, mat):
		# 保留原始评分值
		a = sp.csr_matrix((args.user, args.user))
		b = sp.csr_matrix((args.item, args.item))
		mat = sp.vstack([sp.hstack([a, mat]), sp.hstack([mat.transpose(), b])])
		mat = mat.tocsr()  # 不再二值化
		mat = mat + sp.eye(mat.shape[0])*args.self_loop_val # 自环
		mat = self.normalizeAdj(mat)  # 需要调整归一化逻辑

		idxs = torch.from_numpy(np.vstack([mat.row, mat.col]).astype(np.int64))
		vals = torch.from_numpy(mat.data.astype(np.float32))  # 这里现在包含实际评分
		shape = torch.Size(mat.shape)
		return torch.sparse.FloatTensor(idxs, vals, shape).cuda()

	def loadFeatures(self, filename):
		"""加载特征文件
		Args:
			filename: 特征文件路径
		Returns:
			特征张量和特征维度
		"""
		feats = np.load(filename)
		return torch.tensor(feats).float().cuda(), np.shape(feats)[1]

	def LoadData(self):
		"""加载所有必要的数据"""
		# 加载训练集和测试集
		trnMat = self.loadOneFile(self.trnfile)
		tstMat = self.loadOneFile(self.tstfile)
		self.trnMat = trnMat
		args.user, args.item = trnMat.shape  # 设置用户数和物品数
		self.torchBiAdj = self.makeTorchAdj(trnMat)  # 创建二部图邻接矩阵 # 可不可以将二部图邻接矩阵中的1换成用户对物品的情感得分

		# 创建数据加载器
		trnData = TrnData(trnMat)
		self.trnLoader = dataloader.DataLoader(trnData, batch_size=args.batch, shuffle=True, num_workers=0)
		tstData = TstData(tstMat, trnMat)
		self.tstLoader = dataloader.DataLoader(tstData, batch_size=args.tstBat, shuffle=False, num_workers=0)

		# 加载多模态特征
		self.image_feats, args.image_feat_dim = self.loadFeatures(self.imagefile)
		self.text_feats, args.text_feat_dim = self.loadFeatures(self.textfile)

		# 创建扩散模型数据加载器
		self.diffusionData = DiffusionData(torch.FloatTensor(self.trnMat.A))
		self.diffusionLoader = dataloader.DataLoader(self.diffusionData, batch_size=args.batch, shuffle=True, num_workers=0)

class TrnData(data.Dataset):
	"""训练数据集类"""
	def __init__(self, coomat):
		"""
		Args:
			coomat: COO格式的训练矩阵
		"""
		self.rows = coomat.row  # 用户索引
		self.cols = coomat.col  # 物品索引
		self.dokmat = coomat.todok()  # 转换为字典格式，用于快速查找
		self.negs = np.zeros(len(self.rows)).astype(np.int32)  # 负样本数组

	def negSampling(self):
		"""为每个正样本采样一个负样本"""
		for i in range(len(self.rows)):
			u = self.rows[i]
			while True:
				iNeg = np.random.randint(args.item)  # 随机选择一个物品
				if (u, iNeg) not in self.dokmat:  # 确保是真正的负样本
					break
			self.negs[i] = iNeg

	def __len__(self):
		"""返回数据集大小"""
		return len(self.rows)

	def __getitem__(self, idx):
		"""返回一个训练样本
		Returns:
			(用户索引, 正样本物品索引, 负样本物品索引)
		"""
		return self.rows[idx], self.cols[idx], self.negs[idx]

class TstData(data.Dataset):
	"""测试数据集类"""
	def __init__(self, coomat, trnMat):
		"""
		Args:
			coomat: 测试集矩阵
			trnMat: 训练集矩阵
		"""
		self.csrmat = (trnMat.tocsr() != 0) * 1.0  # 转换为CSR格式的二值矩阵

		# 构建测试集的用户-物品对
		tstLocs = [None] * coomat.shape[0]
		tstUsrs = set()
		for i in range(len(coomat.data)):
			row = coomat.row[i]
			col = coomat.col[i]
			if tstLocs[row] is None:
				tstLocs[row] = list()
			tstLocs[row].append(col)
			tstUsrs.add(row)
		tstUsrs = np.array(list(tstUsrs))
		self.tstUsrs = tstUsrs  # 测试集中的用户列表
		self.tstLocs = tstLocs  # 每个用户的测试物品列表

	def __len__(self):
		"""返回测试集用户数量"""
		return len(self.tstUsrs)

	def __getitem__(self, idx):
		"""返回一个测试样本
		Returns:
			(用户索引, 用户的训练集交互向量)
		"""
		return self.tstUsrs[idx], np.reshape(self.csrmat[self.tstUsrs[idx]].toarray(), [-1])

class DiffusionData(data.Dataset):
	"""扩散模型数据集类"""
	def __init__(self, data):
		"""
		Args:
			data: 输入数据
		"""
		self.data = data

	def __getitem__(self, index):
		"""返回一个数据样本和其索引"""
		item = self.data[index]
		return item, index
	
	def __len__(self):
		"""返回数据集大小"""
		return len(self.data)