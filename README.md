# Mache Learning Demos(Mac OS X, Python2.7)

## Prepare
```
更新pip，以便支持TLSv1.0和TLSv1.1
$ curl https://bootstrap.pypa.io/get-pip.py | sudo python
Install Numpy, nose and tornado for matplotlib.
$ python -m pip install --user numpy nose tornado
Install TensorFlow, 当前版本只支持 CPU
$ sudo pip install -i https://pypi.tuna.tsinghua.edu.cn/simple tensorflow  --ignore-installed numpy
用以解决import tensorflow错误
$ pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --force-reinstall --upgrade protobuf
```
注意：如下方式可以强制安装最新版本，避免如下错误“Cannot uninstall ‘***’. It is a distutils installed project and thus we cannot accurately determine which files belong to it which would lead to only a partial uninstall.”
$ sudo pip install six --ignore-installed six

## Tensor's mnist demo
ref: https://tensorflow.googlesource.com/tensorflow/+/master/tensorflow/examples/tutorials/mnist

### Softmax Regression(mnist_softmax.py)
阅读代码并增加注释

### 

## (BTW)Python3.3 up内置了pip包管理器
```
python3 -m pip install --upgrade pip
sudo pip3 install numpy
sudo pip3 install scipy
sudo pip3 install matplotlib
sudo pip3 install tensorflow-gpu #安装gpu版tensorflow
```
注：目前用Python2.7，所以这部分暂时用不到～

## Ref
- http://www.tensorfly.cn/
- https://tensorflow.google.cn/
- https://tensorflow.google.cn/tutorials/
- TF基本用法：http://www.tensorfly.cn/tfdoc/get_started/basic_usage.html
