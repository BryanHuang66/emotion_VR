from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
def BilstmModel(data:dict,unit:int,batch_size:int,patience:int,epochs:int):
    '''
    data: 输入词典数据，需要包含训练集和验证集
    unit: LSTM输出维数，由于使用了BiLSTM因此输出维数为2*unit
    batch_size: 批量输入大小
    patience: earlystopping的宽容度
    epochs: 迭代轮数
    '''
    
    early_stopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=patience)
    BiLSTMmodel = Sequential(name='BiLSTM')
    BiLSTMmodel.add(Bidirectional(LSTM(unit)))
    BiLSTMmodel.add(Dropout(0.2))
    BiLSTMmodel.add(Dense(1, activation='sigmoid'))

    BiLSTMmodel.compile('adam', 'binary_crossentropy', metrics=['accuracy'])

    print('Train BiLSTM Model...')
    history = BiLSTMmodel.fit(np.array(data["X_train"]), np.array(data["y_train"]),
            batch_size=batch_size,
            epochs=epochs,
            validation_data=[np.array(data["X_valid"]), np.array(data["y_valid"])],
            callbacks=[early_stopping])

    output = BiLSTMmodel.predict(np.array(data["X_test"]))
    output = np.around(output)
    output = np.reshape(output,(-1))
    a = np.where(output!=data["y_test"])
    test_acc = 1 - len(a[0])/len(data["y_test"])
    print(f"测试集精度为{test_acc}")
    
    return history,BiLSTMmodel