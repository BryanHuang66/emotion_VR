'''
全局参数设置
'''
EPOCH = 300  # 训练Epoch数
BATCHSIZE = 256  # 训练batch数, 3090中不会出现内存溢出
PATIENCE = 30  # Earlystopping 耐心
NPYSAVEPATH = 'npy'  # 数据预处理后npy文件存放位置
MODELSAVEPATH = 'results'   # 模型训练完毕后放置位置
PROCESSWITHNOISE = True   # 是否需要通过添加白噪声倍多数据集大小
N_MFCC = 24   # MFCC提取维度


'''
模型参数, 需要重新训练的时候修改, 如果需要运行我训练完毕的模型，请勿修改
'''
## FNN 输出维度
FNNUNIT = 16
## CNN 特征数
CNNUNIT = 16
## LSTM 输出维度
LSTMUNIT = 34
## BiLSTM 输出维度
BILSTMUNIT = 20
## Transformer 输出维度
TRANSUNIT = 24