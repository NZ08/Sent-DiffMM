import time
import torch
import Utils.TimeLogger as logger
from Utils.TimeLogger import log, init_log_file, write_params_to_log
from Params import args
from Model import Model, GaussianDiffusion, Denoise
from DataHandler import DataHandler
import numpy as np
from Utils.Utils import *
import os
import scipy.sparse as sp
import random
# import setproctitle
from scipy.sparse import coo_matrix

class Coach:
	def __init__(self, handler):
		"""
		初始化教练类，负责训练和测试过程
		Args:
			handler: 数据处理器实例
		"""
		self.handler = handler

		log(f'USER, {args.user}, ITEM, {args.item}')  # 修改
		log(f'NUM OF INTERACTIONS, {self.handler.trnLoader.dataset.__len__()}')  # 修改
		self.metrics = dict()
		mets = ['Loss', 'preLoss', 'Recall', 'NDCG']
		for met in mets:
			self.metrics['Train' + met] = list()
			self.metrics['Test' + met] = list()
		
		self.att_image_list = None
		self.att_text_list = None

		
		# 添加用户情感得分字典，用于存储所有用户的情感得分
		self.all_user_sentiments = {}
		self.current_epoch = 0

	def makePrint(self, name, ep, reses, save):
		"""
		生成训练/测试结果的打印字符串
		Args:
			name: 阶段名称（训练/测试）
			ep: 当前轮次
			reses: 结果字典
			save: 是否保存指标
		"""
		ret = 'Epoch %d/%d, %s: ' % (ep, args.epoch, name)
		for metric in reses:
			val = reses[metric]
			ret += '%s = %.4f, ' % (metric, val)
			tem = name + metric
			if save and tem in self.metrics:
				self.metrics[tem].append(val)
		ret = ret[:-2] + '  '
		return ret

	def run(self):
		"""
		运行训练和测试的主循环
		- 准备模型
		- 循环训练指定的轮次
		- 定期进行测试评估
		- 记录最佳性能
		"""
		self.prepareModel()  # 准备模型
		log('Model Prepared')

		# 初始化最佳性能指标
		recallMax = 0
		ndcgMax = 0
		precisionMax = 0
		bestEpoch = 0

		log('Model Initialized')
		start_time = time.time()  # 修改

		# 主训练循环
		for ep in range(0, args.epoch):
			ep_start_time = time.time()  # 修改
			self.current_epoch = ep  # 记录当前的epoch
			tstFlag = (ep % args.tstEpoch == 0)  # 判断是否需要进行测试
			reses = self.trainEpoch()  # 训练一个轮次
			log(self.makePrint('Train', ep, reses, tstFlag))
			if tstFlag:  # 如果需要测试
				reses = self.testEpoch()  # 进行测试
				# 更新最佳性能
				if (reses['Recall'] > recallMax):
					recallMax = reses['Recall']
					ndcgMax = reses['NDCG']
					precisionMax = reses['Precision']
					bestEpoch = ep
				log(self.makePrint('Test', ep, reses, tstFlag))
			ep_end_time = time.time()  # 修改
			log('Epoch %d time: %.2f seconds' % (ep, ep_end_time - ep_start_time))  # 修改
			print()
		log(f'Best epoch : {bestEpoch} , Recall : {recallMax} , NDCG : {ndcgMax} , Precision {precisionMax}')  # 修改
		end_time = time.time()  # 修改
		log(f'Total time: {end_time - start_time} seconds')  # 修改

	def prepareModel(self):
		"""
		准备模型和优化器
		- 初始化主模型和优化器
		- 初始化扩散模型
		- 初始化图像和文本的去噪模型及其优化器
		"""
		# 初始化主模型，使用图像和文本特征
		self.model = Model(self.handler.image_feats.detach(), self.handler.text_feats.detach()).cuda()
		# 使用Adam优化器
		self.opt = torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=0)

		# 初始化高斯扩散模型
		self.diffusion_model = GaussianDiffusion(args.noise_scale, args.noise_min, args.noise_max, args.steps).cuda()
		
		# 设置去噪模型的维度
		out_dims = eval(args.dims) + [args.item]
		in_dims = out_dims[::-1]
		
		# 初始化图像去噪模型和优化器
		self.denoise_model_image = Denoise(in_dims, out_dims, args.d_emb_size, norm=args.norm).cuda()
		self.denoise_opt_image = torch.optim.Adam(self.denoise_model_image.parameters(), lr=args.lr, weight_decay=0)

		# 初始化文本去噪模型和优化器
		self.denoise_model_text = Denoise(in_dims, out_dims, args.d_emb_size, norm=args.norm).cuda()
		self.denoise_opt_text = torch.optim.Adam(self.denoise_model_text.parameters(), lr=args.lr, weight_decay=0)

	def normalizeAdj(self, mat):
		"""
		对邻接矩阵进行归一化处理
		Args:
			mat: 输入的邻接矩阵
		Returns:
			归一化后的邻接矩阵（COO格式）
		"""
		# 计算每个节点的度
		degree = np.array(mat.sum(axis=-1))
		# 计算度的-0.5次方
		dInvSqrt = np.reshape(np.power(degree, -0.5), [-1])
		# 处理无穷大的情况
		dInvSqrt[np.isinf(dInvSqrt)] = 0.0
		# 创建对角矩阵
		dInvSqrtMat = sp.diags(dInvSqrt)
		# 进行归一化：D^(-0.5) * A * D^(-0.5)

		# ret = mat.dot(dInvSqrtMat).transpose().dot(dInvSqrtMat).multiply(mat+mat)

		return mat.dot(dInvSqrtMat).transpose().dot(dInvSqrtMat).tocoo()

	def buildUIMatrix(self, u_list, i_list, edge_list): # 做了适合情感得分的修改
		"""
		构建用户-物品交互矩阵
		Args:
			u_list: 用户ID列表
			i_list: 物品ID列表
			edge_list: 边的权重列表
		Returns:
			构建好的稀疏张量形式的交互矩阵
		"""
		# 创建用户-物品交互矩阵
		mat = coo_matrix((edge_list, (u_list, i_list)), shape=(args.user, args.item), dtype=np.float32)

		# 创建用户-用户和物品-物品的空矩阵
		a = sp.csr_matrix((args.user, args.user))
		b = sp.csr_matrix((args.item, args.item))
		
		# 构建完整的交互矩阵 [[U-U, U-I], [I-U, I-I]]
		mat = sp.vstack([sp.hstack([a, mat]), sp.hstack([mat.transpose(), b])])
		# 将非零元素设为1
		# mat = (mat != 0) * 1.0 # 不再二值化
		att = mat.copy()
		# 添加自环
		mat = mat + sp.eye(mat.shape[0])*args.self_loop_val
		# 对矩阵进行归一化
		mat = self.normalizeAdj(mat)

		# 转换为PyTorch稀疏张量格式
		idxs = torch.from_numpy(np.vstack([mat.row, mat.col]).astype(np.int64))
		vals = torch.from_numpy(mat.data.astype(np.float32))
		shape = torch.Size(mat.shape)

		return torch.sparse.FloatTensor(idxs, vals, shape).cuda(),att

	def trainEpoch(self):
		"""
		训练一个轮次
		包含两个主要步骤：
		1. 扩散模型训练
		2. 推荐模型训练
		Returns:
			包含各种损失值的字典
		"""
		# 获取训练数据加载器并进行负采样
		trnLoader = self.handler.trnLoader
		trnLoader.dataset.negSampling()
		
		# 初始化各种损失值
		epLoss, epRecLoss, epClLoss = 0, 0, 0
		epDiLoss = 0
		epDiLoss_image, epDiLoss_text = 0, 0
		steps = trnLoader.dataset.__len__() // args.batch

		# 获取扩散模型的数据加载器
		diffusionLoader = self.handler.diffusionLoader

		# 第一阶段：扩散模型训练
		for i, batch in enumerate(diffusionLoader):
			batch_item, batch_index = batch
			batch_item, batch_index = batch_item.cuda(), batch_index.cuda()

			# 获取当前的物品和用户嵌入
			iEmbeds = self.model.getItemEmbeds().detach()
			uEmbeds = self.model.getUserEmbeds().detach()

			# 获取图像和文本特征
			image_feats = self.model.getImageFeats().detach()
			text_feats = self.model.getTextFeats().detach()

			# 清空优化器梯度
			self.denoise_opt_image.zero_grad()
			self.denoise_opt_text.zero_grad()

			# 计算图像和文本的扩散损失 # 去噪模型（denoise_model_image/text）预测原始交互，计算扩散损失
			diff_loss_image, gc_loss_image = self.diffusion_model.training_losses(self.denoise_model_image, batch_item, iEmbeds, batch_index, image_feats) # 对batch_item添加噪声，模拟论文中的前向过程（逐步噪声注入）。
			diff_loss_text, gc_loss_text = self.diffusion_model.training_losses(self.denoise_model_text, batch_item, iEmbeds, batch_index, text_feats)

			# 计算总损失
			loss_image = diff_loss_image.mean() + gc_loss_image.mean() * args.e_loss
			loss_text = diff_loss_text.mean() + gc_loss_text.mean() * args.e_loss

			# 累积损失
			epDiLoss_image += loss_image.item()
			epDiLoss_text += loss_text.item()

			loss = loss_image + loss_text

			# 反向传播
			loss.backward() # 通过反向传播更新图像和文本模态的去噪模型参数，优化目标是生成更符合多模态上下文的交互结构。

			# 更新模型参数
			self.denoise_opt_image.step()
			self.denoise_opt_text.step()

			if i == diffusionLoader.dataset.__len__() // args.batch:  # 修改
				log('Diffusion Step %d/%d' % (i, diffusionLoader.dataset.__len__() // args.batch))  # 修改
				print('Diffusion Step %d/%d' % (i, diffusionLoader.dataset.__len__() // args.batch), end='\r')  # 修改

		log('')
		log('Start to re-build UI matrix')

		def calculate_user_sentiment_topk(user_id, k):
			"""
			获取用户前k个最高的情感得分
			Args:
				user_id: 用户ID
				k: 返回的最高情感得分数量，默认为3
			Returns:
				list: 用户前k个最高的情感得分，如果得分数量不足，就取最小得分并减去0.1，如果没有得分则返回[0.5]
			"""
			# 获取当前用户的情感得分
			if isinstance(self.handler.trnMat, coo_matrix):
				self.trnMat_score = self.handler.trnMat.tocsr()
			user_scores = self.trnMat_score[int(user_id)].toarray().flatten()  # 转换为数组
			# 只考虑非零得分
			nonzero_scores = user_scores[user_scores != 0]

			if len(nonzero_scores) == 0:
				return [0.5]  # 如果没有得分则返回默认值

			# 对得分进行排序（降序）
			sorted_scores = np.sort(nonzero_scores)[::-1]
			# 获取前k个最高的得分
			top_scores = sorted_scores[:min(k, len(sorted_scores))].tolist()
			
			# 如果得分数量不足k个，取最小得分并减去0.1进行填充
			if len(top_scores) < k and len(top_scores) > 0:
				min_score = top_scores[-1] - args.sentiment_offset
				# 确保得分不小于0
				min_score = max(0.0, min_score)
				# 填充到k个
				top_scores.extend([min_score] * (k - len(top_scores)))
			# 如果没有得分，用0.5填充
			elif len(top_scores) == 0:
				top_scores = [0.5] * k
				
			return top_scores


		def get_cached_user_sentiment_topk(user_ids, k):
			"""
			批量获取用户的情感得分，使用缓存减少重复计算
			Args:
				user_ids: 用户ID列表
				k: 返回的最高情感得分数量，默认为3
			Returns:
				dict: 用户ID到情感得分的映射
			"""
			# 如果不是第一个epoch，且all_user_sentiments已经有数据，直接返回已存储的结果
			if self.current_epoch > 0 and self.all_user_sentiments:
				results = {}
				for user_id in user_ids:
					key = f"{user_id}_{k}"
					if key in self.all_user_sentiments:
						results[user_id] = self.all_user_sentiments[key]
					else:
						# 如果找不到，可能是新用户或k值不同，需要计算
						top_scores = calculate_user_sentiment_topk(user_id, k)
						self.all_user_sentiments[key] = top_scores
						results[user_id] = top_scores
				return results
			
			# 第一个epoch或all_user_sentiments为空，需要计算并存储结果
			results = {}
			for user_id in user_ids:
				key = f"{user_id}_{k}"
				if key in self.all_user_sentiments:
					results[user_id] = self.all_user_sentiments[key]
				else:
					top_scores = calculate_user_sentiment_topk(user_id, k)
					self.all_user_sentiments[key] = top_scores
					results[user_id] = top_scores
			
			return results

		# 重建用户-物品交互矩阵
		with torch.no_grad():
			# 初始化图像和文本的交互列表
			u_list_image = []
			i_list_image = []
			edge_list_image = []

			u_list_text = []
			i_list_text = []
			edge_list_text = []

			# 使用扩散模型生成新的交互
			for _, batch in enumerate(diffusionLoader): # 做了修改
				batch_item, batch_index = batch
				batch_item, batch_index = batch_item.cuda(), batch_index.cuda()

				# 将batch_index转换为CPU上的整数列表
				user_ids = [int(b_i.cpu().numpy()) for b_i in batch_index]

				# 批量获取所有用户的情感得分（使用缓存）
				user_sentiments = get_cached_user_sentiment_topk(user_ids, k=args.rebuild_k)

				# 处理图像模态
				denoised_batch = self.diffusion_model.p_sample(self.denoise_model_image, batch_item, args.sampling_steps, args.sampling_noise)
				top_item, indices_ = torch.topk(denoised_batch, k=args.rebuild_k)
				# indices_是什么意思？ indices_是扩散模型去噪后预测的用户潜在兴趣物品索引
				# 收集图像模态的交互
				for i in range(batch_index.shape[0]):
					user_id = int(batch_index[i].cpu().numpy())
					for j in range(indices_[i].shape[0]):
						u_list_image.append(user_id)
						i_list_image.append(int(indices_[i][j].cpu().numpy()))
					edge_list_image.append(user_sentiments[user_id])

				# 处理文本模态
				denoised_batch = self.diffusion_model.p_sample(self.denoise_model_text, batch_item, args.sampling_steps, args.sampling_noise)
				top_item, indices_ = torch.topk(denoised_batch, k=args.rebuild_k)

				# 收集文本模态的交互
				for i in range(batch_index.shape[0]):
					user_id = int(batch_index[i].cpu().numpy())
					for j in range(indices_[i].shape[0]):
						u_list_text.append(user_id)
						i_list_text.append(int(indices_[i][j].cpu().numpy()))
					edge_list_text.append(user_sentiments[user_id])


			def row_means(att):
				# 提取矩阵属性
				rows = att.row  # 行索引数组
				data = att.data  # 数据数组
				max_row = att.shape[0]  # 总行数

				# 使用bincount高效计算（自动处理稀疏性）
				row_sums = np.bincount(rows, weights=data, minlength=max_row)
				row_counts = np.bincount(rows, minlength=max_row)

				# 计算平均值（保证无零除）
				means = row_sums / row_counts
				means[np.isnan(means)] = args.self_loop_val
				return means.tolist()

			# 将得分列表展平
			flat_score_list_image = []
			for scores in edge_list_image:
				flat_score_list_image.extend(scores)

			flat_score_list_text = []
			for scores in edge_list_text:
				flat_score_list_text.extend(scores)

			# 构建图像模态的UI矩阵
			u_list_image = np.array(u_list_image)
			i_list_image = np.array(i_list_image)
			flat_score_list_image = np.array(flat_score_list_image)
			self.image_UI_matrix,att_image = self.buildUIMatrix(u_list_image, i_list_image, flat_score_list_image)
			self.image_UI_matrix = self.model.edgeDropper(self.image_UI_matrix)
			att_image_list = row_means(att_image)

			# 构建文本模态的UI矩阵
			u_list_text = np.array(u_list_text)
			i_list_text = np.array(i_list_text)
			flat_score_list_text = np.array(flat_score_list_text)
			self.text_UI_matrix,att_text = self.buildUIMatrix(u_list_text, i_list_text, flat_score_list_text)
			self.text_UI_matrix = self.model.edgeDropper(self.text_UI_matrix) ###???为什么归一化后的数值那么小
			att_text_list = row_means(att_text)

			self.att_image_list = att_image_list
			self.att_text_list = att_text_list

		log('UI matrix built!')

		# 第二阶段：推荐模型训练
		for i, tem in enumerate(trnLoader):
			ancs, poss, negs = tem
			ancs = ancs.long().cuda() # 用户索引
			poss = poss.long().cuda() # 正样本索引
			negs = negs.long().cuda() # 负样本索引

			# 清空优化器梯度
			self.opt.zero_grad()

			# 前向传播，获取用户和物品嵌入 
			usrEmbeds, itmEmbeds = self.model.forward_MM(self.handler.torchBiAdj, self.image_UI_matrix, self.text_UI_matrix, att_image_list, att_text_list)
			ancEmbeds = usrEmbeds[ancs]
			posEmbeds = itmEmbeds[poss]
			negEmbeds = itmEmbeds[negs]
			
			# 计算BPR损失
			scoreDiff = pairPredict(ancEmbeds, posEmbeds, negEmbeds)
			bprLoss = - (scoreDiff).sigmoid().log().sum() / args.batch
			regLoss = self.model.reg_loss() * args.reg
			loss = bprLoss + regLoss
			
			# 累积损失
			epRecLoss += bprLoss.item()
			epLoss += loss.item()
			# 计算对比学习损失
			usrEmbeds1, itmEmbeds1, usrEmbeds2, itmEmbeds2 = self.model.forward_cl_MM(self.handler.torchBiAdj, self.image_UI_matrix, self.text_UI_matrix)
			clLoss = (contrastLoss(usrEmbeds1, usrEmbeds2, ancs, args.temp) + contrastLoss(itmEmbeds1, itmEmbeds2, poss, args.temp)) * args.ssl_reg

			clLoss1 = (contrastLoss(usrEmbeds, usrEmbeds1, ancs, args.temp) + contrastLoss(itmEmbeds, itmEmbeds1, poss, args.temp)) * args.ssl_reg
			clLoss2 = (contrastLoss(usrEmbeds, usrEmbeds2, ancs, args.temp) + contrastLoss(itmEmbeds, itmEmbeds2, poss, args.temp)) * args.ssl_reg
			clLoss_ = clLoss1 + clLoss2

			if args.cl_method == 1:
				clLoss = clLoss_

			loss += clLoss

			epClLoss += clLoss.item()

			# 反向传播和参数更新
			loss.backward()
			self.opt.step()

			if i == steps:  # 修改
				log('Step %d/%d: bpr : %.3f ; reg : %.3f ; cl : %.3f ' % (
					i,
					steps,
					bprLoss.item(),
					regLoss.item(),
					clLoss.item()
				))  # 修改
				print('Step %d/%d: bpr : %.3f ; reg : %.3f ; cl : %.3f ' % (
					i,
					steps,
					bprLoss.item(),
					regLoss.item(),
					clLoss.item()
				), end='\r')  # 修改

		# 返回训练结果
		ret = dict()
		ret['Loss'] = epLoss / steps
		ret['BPR Loss'] = epRecLoss / steps
		ret['CL loss'] = epClLoss / steps
		ret['Di image loss'] = epDiLoss_image / (diffusionLoader.dataset.__len__() // args.batch)
		ret['Di text loss'] = epDiLoss_text / (diffusionLoader.dataset.__len__() // args.batch)
		return ret

	def testEpoch(self):
		"""
		测试一个轮次的模型性能
		Returns:
			包含评估指标的字典（Recall, NDCG, Precision）
		"""
		# 获取测试数据加载器
		tstLoader = self.handler.tstLoader
		epRecall, epNdcg, epPrecision = [0] * 3
		i = 0
		num = tstLoader.dataset.__len__()
		steps = num // args.tstBat

		# 获取用户和物品的嵌入
		usrEmbeds, itmEmbeds = self.model.forward_MM(self.handler.torchBiAdj, self.image_UI_matrix, self.text_UI_matrix, self.att_image_list, self.att_text_list)

		# 对每个测试批次进行评估
		for usr, trnMask in tstLoader:
			i += 1
			usr = usr.long().cuda()
			trnMask = trnMask.cuda()
			
			# 计算所有物品的预测分数
			allPreds = torch.mm(usrEmbeds[usr], torch.transpose(itmEmbeds, 1, 0)) * (1 - trnMask) - trnMask * 1e8
			# 获取topk的预测结果
			_, topLocs = torch.topk(allPreds, args.topk)
			
			# 计算评估指标
			recall, ndcg, precision = self.calcRes(topLocs.cpu().numpy(), self.handler.tstLoader.dataset.tstLocs, usr)
			epRecall += recall
			epNdcg += ndcg
			epPrecision += precision
			if i == steps:  # 修改
				log('Steps %d/%d: recall = %.2f, ndcg = %.2f , precision = %.2f   ' % (
				i, steps, recall, ndcg, precision))  # 修改
				print('Steps %d/%d: recall = %.2f, ndcg = %.2f , precision = %.2f   ' % (
				i, steps, recall, ndcg, precision), end='\r')  # 修改
		
		# 返回平均评估结果
		ret = dict()
		ret['Recall'] = epRecall / num
		ret['NDCG'] = epNdcg / num
		ret['Precision'] = epPrecision / num
		return ret

	def calcRes(self, topLocs, tstLocs, batIds):
		"""
		计算评估指标
		Args:
			topLocs: 预测的top-k物品位置
			tstLocs: 真实的测试集位置
			batIds: 批次中的用户ID
		Returns:
			recall: 召回率
			ndcg: NDCG值
			precision: 精确率
		"""
		assert topLocs.shape[0] == len(batIds)
		allRecall = allNdcg = allPrecision = 0
		
		# 对每个用户计算指标
		for i in range(len(batIds)):
			temTopLocs = list(topLocs[i])  # 预测的top-k物品
			temTstLocs = tstLocs[batIds[i]]  # 真实的测试集物品
			tstNum = len(temTstLocs)
			
			# 计算理想DCG值
			maxDcg = np.sum([np.reciprocal(np.log2(loc + 2)) for loc in range(min(tstNum, args.topk))])
			
			# 初始化指标
			recall = dcg = precision = 0
			
			# 计算各项指标
			for val in temTstLocs:
				if val in temTopLocs:  # 如果预测正确
					recall += 1  # 增加召回计数
					dcg += np.reciprocal(np.log2(temTopLocs.index(val) + 2))  # 计算DCG
					precision += 1  # 增加精确度计数
			
			# 归一化指标
			recall = recall / tstNum  # 计算召回率
			ndcg = dcg / maxDcg  # 计算NDCG
			precision = precision / args.topk  # 计算精确率
			
			# 累积批次结果
			allRecall += recall
			allNdcg += ndcg
			allPrecision += precision
			
		return allRecall, allNdcg, allPrecision

def seed_it(seed):
	"""
	设置随机种子，确保实验可重复性
	Args:
		seed: 随机种子值
	"""
	random.seed(seed)  # 设置Python的随机种子
	os.environ["PYTHONSEED"] = str(seed)  # 设置环境变量
	np.random.seed(seed)  # 设置NumPy的随机种子
	torch.cuda.manual_seed(seed)  # 设置CUDA的随机种子
	torch.cuda.manual_seed_all(seed)  # 设置所有CUDA设备的随机种子
	torch.backends.cudnn.deterministic = True  # 确保CUDNN是确定性的
	torch.backends.cudnn.benchmark = True  # 启用CUDNN的自动优化
	torch.backends.cudnn.enabled = True  # 启用CUDNN
	torch.manual_seed(seed)  # 设置PyTorch的随机种子

if __name__ == '__main__':
	# 设置随机种子
	seed_it(args.seed)

	# 设置GPU设备
	os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
	
	# 初始化日志文件并写入参数
	init_log_file()
	write_params_to_log(args)
	
	# 启用日志保存
	logger.saveDefault = True
	
	log('Start')
	# 初始化数据处理器
	handler = DataHandler()
	# 加载数据
	handler.LoadData()
	log('Load Data')

	# 创建并运行训练器
	coach = Coach(handler)
	coach.run()

	# 输出最终的实验结果
	# baby:		Best epoch :  78  , Recall :  0.09861820274522767  , NDCG :  0.04171448504471465  , Precision 0.0052044227307791215			 k=3
	# sports:	Best epoch :  134  , Recall :  0.10290266531345071  , NDCG :  0.04592183751300481  , Precision 0.005438507781336032