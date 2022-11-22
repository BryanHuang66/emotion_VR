from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
def LstmModel(data:dict,unit:int,batch_size:int,patience:int,epochs:int):
    '''
    data: 输入词典数据，需要包含训练集和验证集
    unit: LSTM输出维数
    batch_size: 批量输入大小
    patience: earlystopping的宽容度
    epochs: 迭代轮数
    '''
    
    early_stopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=patience)
    LSTMmodel = Sequential(name='LSTM')
    LSTMmodel = Sequential()
    LSTMmodel.add(LSTM(unit))
    LSTMmodel.add(Dropout(0.2))
    LSTMmodel.add(Dense(1, activation='sigmoid'))
    # model.summary()

    LSTMmodel.compile('adam', 'binary_crossentropy', metrics=['accuracy'])

    print('Train LSTM Model...')
    
    history = LSTMmodel.fit(np.array(data["X_train"]), np.array(data["y_train"]),
            batch_size=batch_size,
            epochs=epochs,
            validation_data=[np.array(data["X_valid"]), np.array(data["y_valid"])],
            callbacks=[early_stopping])
    
    output = LSTMmodel.predict(np.array(data["X_test"]))
    output = np.around(output)
    output = np.reshape(output,(-1))
    a = np.where(output!=data["y_test"])
    test_acc = 1 - len(a[0])/len(data["y_test"])
    print(f"测试集精度为{test_acc}")
    return history,LSTMmodel