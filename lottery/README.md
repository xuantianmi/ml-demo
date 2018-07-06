# 机器学习应用于彩票（乐透/竞猜）

## 创建首个.ipynb文件
- 安装jupter notebook for python2
```
sudo pip install --index https://pypi.mirrors.ustc.edu.cn/simple/ jupyter notebook
# maybe, python -m pip install --user jupyter and add "/Users/dora/Library/Python/2.7/bin" to PATH
```
- 安装jupter notebook for python3
```
python3 -m pip install --index https://pypi.mirrors.ustc.edu.cn/simple/ --upgrade pip
python3 -m pip install --index https://pypi.mirrors.ustc.edu.cn/simple/ jupyter
# To add "/usr/local/Cellar/python/3.6.5_1/Frameworks/Python.framework/Versions/3.6/bin" to PATH
```

启动，浏览器就会弹出一个页面http://localhost:6006/
```
cd ./ml_demo/lottery
jupyter notebook
```

## 排三开奖结果预测
- digital3_lstm.py
结合两个例子（参考REF）编写一个排三开奖号码预测程序。不同点在于，参考中的排三预测是基于奖期整体开奖号码作为原子输入，本例子是将每个开奖球作为原子输入，并结合英语单词预测的模型逻辑进行培训和预测。
- digital3_lstm.ipynb
首个jupyter尝试～

## REF
- [*(译)理解LSTM网络](https://www.jianshu.com/p/9dc9f41f0b29)
- [排三LSTM预测](https://github.com/chengstone/LotteryPredict)
- [带你了解LSTM](https://www.cnblogs.com/DjangoBlog/p/6888812.html)
- [各种Deep Learning参考](https://github.com/roatienza/Deep-Learning-Experiments)
- [*TensorFlow中RNN实现的正确打开方式](https://yq.aliyun.com/articles/229291)