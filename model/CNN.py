from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D,Input,MaxPool2D,Flatten
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
def CNNModel(data:dict,filters:int,batch_size:int,patience:int,epochs:int):
    '''
    data: 输入词典数据，需要包含训练集和验证集
    filters: 特征卷基层数
    batch_size: 批量输入大小
    patience: earlystopping的宽容度
    epochs: 迭代轮数
    '''
    
    early_stopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=patience)

    CNNmodel = Sequential(name='CNN')
    CNNmodel.add(Input(shape=(*data['X_train'][0].shape,1), name='input'))
    CNNmodel.add(Conv2D(filters,(7,7),activation='relu'))
    CNNmodel.add(MaxPool2D((4,4)))
    CNNmodel.add(Flatten())
    CNNmodel.add(Dropout(0.2))
    CNNmodel.add(Dense(1, activation='sigmoid'))
    # model.summary()

    CNNmodel.compile('adam', 'binary_crossentropy', metrics=['accuracy'])

    print('Train CNN Model...')
    history = CNNmodel.fit(np.array(data["X_train"]), np.array(data["y_train"]),
            batch_size=batch_size,
            epochs=epochs,
            validation_data=[np.array(data["X_valid"]), np.array(data["y_valid"])],
            callbacks=[early_stopping])
    output = CNNmodel.predict(np.array(data["X_test"]))
    output = np.around(output)
    output = np.reshape(output,(-1))
    a = np.where(output!=data["y_test"])
    test_acc = 1 - len(a[0])/len(data["y_test"])
    print(f"测试集精度为{test_acc}")
    
    return history,CNNmodel
