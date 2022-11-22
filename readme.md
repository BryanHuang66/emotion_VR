# 多模型联合投票声音情感分析模型

> 通过FNN、CNN、LSTM、BiLSTM和Transformer模型进行声音情感分析。并在bagging算法启发下，联合打分增强模型泛化能力

## 环境配置
### conda环境配置
推荐重新安装一个conda环境，方便三方库和项目的管理。
请先进入该文档路径，之后运行下方代码

```shell
conda create -n emotion_vr python=3.7
conda activate emotion_vr
pip install -r requirements.txt
```


### base环境配置
如果有本地base环境，可以直接安装需求

```shell
pip install -r requirements.txt
```


==注意: 1) 本实验没有尝试windows环境使用NVIDIA显卡加速，但是在云服务器linux系统+3090下训练没有问题。由于本人使用Mac，因此使用的是tensorflow+metal的环境，具体配置metal加速的方式可见[APPLE官网](https://developer.apple.com/metal/tensorflow-plugin/)，即使不适用gpu加速，使用cpu运行的速度也不是很慢。2) 在linux系统中，有可能会报错cannot load library 'libsndfile.so'，请在终端输入apt install libsndfile1==


## 根目录结构解释
报告在根目录下main.pdf，预测结果在test_main.xlsx
```
    EMOTION_VR
    ├─dataset
    │  ├─test.xlsx  ## 用于测试的数据集excel表
    │  ├─test
    │  │  └─...
    │  ├─train.xlsx ## 需要验证的数据集或者要用于训练的数据集excel表
    │  ├─train
    │     └─...
    ├─fig  ## 存储运行过程中生成的图像
    │  └─...
    ├─model  ## 各类模型的.py文件
    │  └─...
    ├─npy  ## 生成预处理后的数组，用于训练和预测
    │  └─...
    ├─results  ## 生成训练完毕的模型
    │  └─...
    ├─utils  ## 工具类.py文件
    │  └─process.py
    ├─results  ## 生成训练完毕的模型
    │  └─...
    ├─train.py  ## 训练相关代码
    ├─settings.py  ## 配置文件
    ├─main.py  ## 入口文件
    ├─requirements.txt  ##环境需求
    ├─test_main.xlsx ## 测试结果
    ├─main.pdf  ## 报告
    └─readme.md
```


## 复现excel表格信息
dataset下的test.xlsx表格已经填充了我的预测结果。如果仅需要进行该结果的复现，请如下操作
1. **拷贝数据文件**
   首先将原本的数据文件都拷贝到dataset文件夹下，当然也可以只替换test.xlsx文件和test文件夹
2. **数据预处理**
   在终端键入
   ```shell
    python main.py --process=predict
    ```
3. **预测**
    我训练好的模型均在model文件夹下，可以直接进行预测
    ```shell
    python main.py --predict=True --model_list=CNN,FNN,LSTM,BiLSTM,Transformer
    ```
    复现结果在dataset/test.xlsx表格中



## 进一步运行
### 运行老师提供的数据集
1. **拷贝数据集到对应文件夹下**
    由于音频文件较大，因此不上传。助教老师如果需要复现模型训练全过程中音频处理步骤，可以首先用音频文件夹替换dataset下的两个空文件夹，在根目录下settings.py文件夹中修改你想要修改的参数，每个参数都有备注。
2. **数据的预处理**
   本次实验采用课程提供总共1784份音频文件进行深度学习的模型训练，在我们普通人的经验中，轻微的白噪声不会对人耳的识别产生影响，因此我通过添加白噪声，将所有的数据复制了一份添加少量的白噪声。而后，为了能够进行batch化的训练，我将所有音频长度补长，word中有说时间为2-10秒，所以我们设置补长到11秒。最后通过提取mfcc这一重要特征，作为模型的输入。预处理后的音频数据格式为[np.array,...]的列表，列表长度为总音频个数，其中np.array的shape为(474,24)，其中24为mfcc维数（mfcc可以在settings.py中修改提取维度, 人声一般提取维度不需要太高）
    ```shell
    python main.py --process=train
    python main.py --process=predict
    ```

3.  **模型的训练**
    当处理完数据之后，程序会将处理后的数组存储到npy文件夹中（存储位置也可以在settings.py中修改），如果你认为预处理比较浪费时间，可以直接下载我处理完的npy文件（添加了白噪声），将解压后的三个.npy文件放置到根目录下的npy文件夹中。
    [下载链接](https://pan.baidu.com/s/1Unci0lRU7zdAZI5xipWhZQ) → 提取码: 9hdn 
    我目前总共使用了五种模型[CNN,FNN,LSTM,BiLSTM,Transformer]，--model_list中输入上述需要训练的模型，注意不同模型之间用英文逗号隔开
    ```shell
    python main.py --train=True --model_list=CNN,FNN,LSTM,BiLSTM,Transformer
    ```
4. **模型的预测**
    由于我已经事先处理完了所有文件，因此如果需要预测文件，可以直接运行此步，但是需要确保有第一步处理完成的npy文件，比如我的位于"npy/test_set.npy"
    其中model_list可以选择其中的几个模型或者全部模型进行投票，如果有模型在某个音频上有分歧，采用少数服从多数的方法
    ```shell
    python main.py --predict=True --model_list=CNN,FNN,LSTM,BiLSTM,Transformer
    ```

### 运行你的数据集
考虑到有可能你需要用你的数据集进行验证我的模型、训练你的模型或者预测其他的音频，这里给出方法。
1. **替换数据文件**
    要求用你的dataset文件夹替换我的dataset文件夹，dataset目录为
    ```
    EMOTION_VR
    ├─dataset
    │  ├─test.xlsx  ## 用于测试的数据集excel表
    │  ├─test
    │  │  └─...
    │  ├─train.xlsx ## 需要验证的数据集或者要用于训练的数据集excel表
    │  ├─train
    │     └─...
    ...
    ```
    其中，如果你只要进行预测，那么只需要替换test.xlsx和test文件夹；如果你只要进行训练或者验证，那么只需要替换train的相关文件文件夹，但是train.xlsx必须有label的信息，excel表格形式详见我的数据集。
2. **验证你的数据**
    通过上面一步处理完成后，需要先生成valid文件.
    ```shell
    python main.py --process=valid 
    ```
    而后进行验证，model_list同样可选五个模型中的几个，最终的输出结果将会是你的数据集在这些模型下分别验证精度
    ```shell
    python main.py --valid=True --model_list=CNN,FNN,LSTM,BiLSTM,Transformer
    ```
3. **训练你的数据**
    类似于“运行老师提供的数据集”板块描述的样子，先对你的数据集进行预处理
    ```shell
    python main.py --process=train 
    ```
    而后进行训练，model_list同样可选五个模型中的几个
    ```shell
    python main.py --train=True --model_list=CNN,FNN,LSTM,BiLSTM,Transformer
    ```
4. **预测你的数据**
    先生成数据集
    ```shell
    python main.py --process=predict
    ```
    而后进行训练，model_list同样可选五个模型中的几个
    ```shell
    python main.py --predict=True --model_list=CNN,FNN,LSTM,BiLSTM,Transformer
    ```