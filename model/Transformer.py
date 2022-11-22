import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from settings import *
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Input,GlobalAveragePooling1D
from tensorflow.keras.callbacks import EarlyStopping

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim=N_MFCC, num_heads=3, ff_dim=TRANSUNIT, rate=0.3,**kwargs):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def get_config(self):
        config = super().get_config()
        config.update({
            "att": self.att,
            "ffn": self.ffn,
            "layernorm1": self.layernorm1,
            "layernorm2": self.layernorm2,
            "dropout1": self.dropout1,
            "dropout2": self.dropout2,

        })
        return config
    
    
    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)
    
def TransModel(data:dict,batch_size:int,patience:int,epochs:int):
    '''
    data: 输入词典数据，需要包含训练集和验证集
    unit: 输出维数
    batch_size: 批量输入大小
    patience: earlystopping的宽容度
    epochs: 迭代轮数
    '''
    
    early_stopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=patience)
    Transmodel = Sequential()
    Transmodel.add(Input(shape=data['X_train'][0].shape, name='input'))
    Transmodel.add(TransformerBlock())
    Transmodel.add(GlobalAveragePooling1D())
    Transmodel.add(Dropout(0.1))
    Transmodel.add(Dense(1, activation='sigmoid'))

    Transmodel.compile('adam', 'binary_crossentropy', metrics=['accuracy'])

    print('Train Transformer Model...')
    history = Transmodel.fit(np.array(data["X_train"]), np.array(data["y_train"]),
            batch_size=batch_size,
            epochs=epochs,
            validation_data=[np.array(data["X_valid"]), np.array(data["y_valid"])],
            callbacks=[early_stopping])

    output = Transmodel.predict(np.array(data["X_test"]))
    output = np.around(output)
    output = np.reshape(output,(-1))
    a = np.where(output!=data["y_test"])
    test_acc = 1 - len(a[0])/len(data["y_test"])
    print(f"测试集精度为{test_acc}")
    
    return history,Transmodel
    