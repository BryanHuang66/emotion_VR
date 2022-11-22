from keras.models import Sequential
from keras.layers import Dense, Dropout,Input,Flatten
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
def FNNModel(data:dict,units:int,batch_size:int,patience:int,epochs:int):
    '''
    data: 输入词典数据，需要包含训练集和验证集
    units: 输出维数
    batch_size: 批量输入大小
    patience: earlystopping的宽容度
    epochs: 迭代轮数
    '''
    
    early_stopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=patience)

    FNNmodel = Sequential(name='FNN')
    
    FNNmodel.add(Input(shape=data['X_train'][0].shape, name='input'))
    FNNmodel.add(Dense(units,activation='relu'))
    FNNmodel.add(Flatten())
    FNNmodel.add(Dropout(0.2))
    FNNmodel.add(Dense(1, activation='sigmoid'))
    # model.summary()

    FNNmodel.compile('adam', 'binary_crossentropy', metrics=['accuracy'])

    print('Train FNN Model...')
    history = FNNmodel.fit(np.array(data["X_train"]), np.array(data["y_train"]),
            batch_size = batch_size,
            epochs = epochs,
            validation_data = [np.array(data["X_valid"]), np.array(data["y_valid"])],
            callbacks = [early_stopping])
    output = FNNmodel.predict(np.array(data["X_test"]))
    output = np.around(output)
    output = np.reshape(output,(-1))
    a = np.where(output!=data["y_test"])
    test_acc = 1 - len(a[0])/len(data["y_test"])
    print(f"测试集精度为{test_acc}")
    
    return history,FNNmodel
