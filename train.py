from model import BiLSTM,CNN,FNN,LSTM,Transformer
from settings import *
import matplotlib.pyplot as plt
def train_model(modelList:list,data,batchsize,patience,epoch):
    model_list = {}
    for cur_pos in range(len(modelList)):
        if modelList[cur_pos] == 'CNN':
            history,cnn_model = CNN.CNNModel(data,CNNUNIT,batchsize,patience,epoch)
            model_list['CNN'] = cnn_model
        elif modelList[cur_pos] == 'FNN':
            history,FNN_model = FNN.FNNModel(data,FNNUNIT,batchsize,patience,epoch)
            model_list['FNN'] = FNN_model
        elif modelList[cur_pos] == 'LSTM':
            history,lstm_model = LSTM.LstmModel(data,LSTMUNIT,batchsize,patience,epoch)
            model_list['LSTM'] = lstm_model
        elif modelList[cur_pos] == 'BiLSTM':
            history,bilstm_model = BiLSTM.BilstmModel(data,BILSTMUNIT,batchsize,patience,epoch)
            model_list['BiLSTM'] = bilstm_model
        elif modelList[cur_pos] == 'Transformer':
            history,transformer_model = Transformer.TransModel(data,batchsize,patience,epoch)
            model_list['Transformer'] = transformer_model    

        para_list = ['loss', 'accuracy', 'val_loss', 'val_accuracy']
        for i in range(4):
            plt.subplot(2,2,i+1)
            plt.title(para_list[i])
            para = history.history[para_list[i]]
            epochs = range(1, len(para) + 1)
            plt.plot(epochs, para, label=modelList[cur_pos])
            plt.legend()
            plt.tight_layout() 
    plt.savefig('fig/model_info.png')
    return model_list