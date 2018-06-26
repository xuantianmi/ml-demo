LMTCNN--Lightweight Multi-task CNN

## Joint Estimation of Age and Gender from Unconstrained Face Images using Lightweight Multi-task CNN for Mobile Applications
ref: https://arxiv.org/pdf/1806.02023.pdf 
- Raw images: 227 red line?, 256 yellow line?
- Convolution: 96 filters
- Depthwise Separable Convolution: 256 filters
- Depthwise Separable Convolution: 384 filters
- Max-Pooling
- FCI: 512
- ReLU
- Dropout
- FCI 512
- ReLU
- Dropout
- Softmax for Age Prob / Softmax for Gender Prob

## Install python3 on MacOS
brew install python3
sudo mkdir /usr/local/Frameworks
sudo chown $(whoami):admin /usr/local/Frameworks
brew link python3

## Install pip3 on MacOS
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
pyton3 get-pip.py
ln -s /usr/local/Cellar/python/3.6.5_1/Frameworks/Python.framework/Versions/3.6/bin/pip /usr/local/bin/pip3 
sudo pip3 install --index https://pypi.mirrors.ustc.edu.cn/simple/ keras
sudo pip3 install --index https://pypi.mirrors.ustc.edu.cn/simple/ tensorflow
sudo pip3 install --index https://pypi.mirrors.ustc.edu.cn/simple/ matplotlib

## Glossary
|Name|Content|
- | :- | 
|dw Conv|DWConv表示depthwise separable convolution|
|pw Conv|？？？|
|K-fold cross-validation|K层交叉检验就是把原始的数据随机分成K个部分。在这K个部分中，选择一个作为测试数据，剩下的K-1个作为训练数据。<br/>交叉检验的过程实际上是把实验重复做K次，每次实验都从K个部分中选择一个不同的部分作为测试数据（保证K个部分的数据都分别做过测试数据），剩下的K-1个当作训练数据进行实验，最后把得到的K个实验结果平均。| 
|LMTCNN|Lightweight Multi-task CNN|

