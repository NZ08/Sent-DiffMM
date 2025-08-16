import datetime  # 导入日期时间模块，用于获取当前时间
import os  # 导入操作系统模块，用于文件操作

# 全局变量用于存储日志消息
logmsg = ''  # 初始化空字符串，用于累积存储日志信息
# 用于存储时间标记的字典
timemark = dict()  # 初始化空字典，用于记录不同标记点的时间
# 是否默认保存日志的标志
saveDefault = False  # 设置默认不保存日志到logmsg变量
# 日志文件路径
log_file_path = None  # 初始化日志文件路径变量

def init_log_file():
	"""
	初始化日志文件，创建以当前时间命名的日志文件
	返回:
		日志文件路径
	"""
	global log_file_path
	# 确保logs目录存在
	if not os.path.exists('logs'):
		os.makedirs('logs')
	
	# 以当前时间创建日志文件名
	timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
	log_file_path = f'logs/log_{timestamp}.txt'
	
	# 创建空日志文件
	with open(log_file_path, 'w', encoding='utf-8') as f:
		f.write(f"日志创建时间: {datetime.datetime.now()}\n")
		f.write("="*50 + "\n\n")
	
	return log_file_path

def write_params_to_log(args):
	"""
	将参数写入日志文件，并进行对齐
	参数:
		args: 参数对象
	"""
	if log_file_path is None:
		init_log_file()
	
	with open(log_file_path, 'a', encoding='utf-8') as f:
		f.write("运行参数:\n")
		f.write("-"*50 + "\n")
		
		# 获取所有参数
		params = vars(args)
		# 计算最长参数名称，用于对齐
		max_param_len = max(len(param) for param in params.keys())
		
		# 写入每个参数
		for param, value in params.items():
			f.write(f"{param:{max_param_len}} = {value}\n")
		
		f.write("-"*50 + "\n\n")

def log(msg, save=None, oneline=False):
	"""
	记录带有时间戳的消息
	参数:
		msg: 要记录的消息
		save: 是否保存消息到全局logmsg变量（None表示使用saveDefault值）
		oneline: 是否在同一行打印（用于进度更新）
	"""
	global logmsg  # 声明使用全局变量logmsg
	global saveDefault  # 声明使用全局变量saveDefault
	global log_file_path  # 声明使用全局变量log_file_path
	
	time = datetime.datetime.now()  # 获取当前时间
	tem = '%s: %s' % (time, msg)  # 格式化时间和消息，生成日志条目
	
	# 保存到全局变量
	if save != None:  # 如果明确指定了是否保存
		if save:  # 如果save为True
			logmsg += tem + '\n'  # 将日志条目添加到全局logmsg变量中
	elif saveDefault:  # 如果未指定save且saveDefault为True
		logmsg += tem + '\n'  # 将日志条目添加到全局logmsg变量中
	
	# 保存到日志文件
	if log_file_path is not None:
		with open(log_file_path, 'a', encoding='utf-8') as f:
			f.write(tem + '\n')
	
	# 打印到控制台
	if oneline:  # 如果需要在同一行打印
		print(tem, end='\r')  # 打印日志条目并将光标返回到行首
	else:  # 如果需要正常打印（换行）
		print(tem)  # 打印日志条目并自动换行

def marktime(marker):
	"""
	记录当前时间点并与标记关联
	参数:
		marker: 时间标记的名称
	"""
	global timemark  # 声明使用全局变量timemark
	timemark[marker] = datetime.datetime.now()  # 记录当前时间并与标记关联存储在字典中


if __name__ == '__main__':  # 如果直接运行此脚本
	init_log_file()  # 初始化日志文件
	log('')  # 打印一个空日志信息（仅包含时间戳）