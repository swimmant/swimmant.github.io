# 24_深度卷积神经网络AlexNet_QA

## 问题1:老师， Imagenet?数据集是怎么构建的，现在看是不是要成为历史了？

答：现在还是常见数据集

## 问题2:为什么2000年的时候，神经网络被核方法代替？是因为神经网络运算量太大，数据多，硬件跟不上吗？

答：深的网络算不动，但理论性更好，学术界喜欢

## 问题4: alexnet让机器自己寻找特征，这些找到的特征都符合人类的逻辑马？如果不复合的话，要怎么解释？

答：深度学习寻找的特征都不是符合人类逻辑的，符合人类逻辑的特征都是碰巧，网络拟合过程不知道人类的存在。深度学习的可解释性很差。如果要模拟人能解释的特征，需要设计特定的网络。

## 问题9:为什么 Alexnete最后要有两个相同的全连接层 Dense(4096)?一个行吗？

答：一个效果会差，复杂度不够。

