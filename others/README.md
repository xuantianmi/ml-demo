# Joint Estimation of Age and Gender from Unconstrained Face Images using Lightweight Multi-task CNN for Mobile Applications

## LMTCNN study
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
