# 34_多GPU训练实现QA

## 问题1:最近 keras从分离，书籍会不会需要重新整理呀？

答：keras可以看为python的库。

## 问题3:请问 resets中的卷积层能否全部替換成mlp来实现一个很深的的深层mlp网络？

答：1*1的卷积层=共享参数的全连接层

## 问题4:既然 XX norm是一种正则，那么原则上它应该能像 dropout那样加强模型的泛化能力，那就应该能提高模型的测试精度。那为什么说 batch norm只能加快训练而对精度没影响呢？

答：不知道

## 问题6:李老师您好，我想做建筑图纸的处理，请问有没有相关的深度学习网络可以实现输入一张图纸，进行一定的处理后再输出一张图纸的操作

答：有，gan，AI辅助作图

## 问题7:老师，all reduce, all gather主要起什么作用啊？实际用的时候发现， pytorch的类似分布式op不能传导梯度，会破坏计算图，不能自动求导,怎么解决呢？

答：取决于实现，all_reduce 就是将所有东西加在一起，然后再复制回去。scatter ：将一个东西切成n分发给设备，all gather：将分布在不同地方的东西合并起来。分布式计算放进去会破坏一些自动求导。

## 问题8:两个GPU训练时，最后的梯度是把两个GPU上的梯度相加吗？

答：是的，minibatch的梯度就是每一个样本梯度求和。各自gpu求和后两个再求和。

## 问题9:老师，为什么参数量大的模型速度不一定慢？还有一种说法是，flops越多的模型性能越好，又是什么原理呢？

答：不是那么唯一，性能取决于每次计算访问多少参数量也就是多少byte。 计算量/内存访问，越高越好。

## 问题10:老师，想问一下为什么分布到多个GPU上测试精度会比单GPU抖动厉害呢？

答：其实不是的，抖动是因为学习率变大了。学习率不变，批量大小不变，多gpu和单gpu是没有变化的，为了更好的速度就要调大batchsize；batchsize变大后导致参数量，收敛发生变化。

## 问题11:老师，Ir太大会导致训练不收敛嘛？另外， batch size,太大会导致loss nan嘛？

答：lr太大会导致不收敛，batch size大数值稳定性会更好。

## 问题12:GPU的显存如何优化呢？最近跑实验经常显存OOM,我的显存用到了14

答：手动优化很难，靠的是框架。batchsize调小点，或者模型做简单点

## 问题13:老师，对于精度来说， batch size=1是一种最好的情况吗？

答;是的，可能是最好的。

## 问题14: parameter server p可以和 pytorch结合吗，具体要如何实现呢？

答：pytorch没有parameter server ；由第三方实现，头条

## 问题15:老师，网络模型使用了nn. Dataparallel（）,是不是数据集也被自动的分配到了多个GPU니

答：是的，

## 问题17:老师，验证集准确率震荡较大是哪个参数影响最大呢

答：lr

## 问题19:请问老师，在用orch的高级数据并行中，还是将 inputs和 labels放到第0块gpu,这会不会导致性能问题，因为这些数据最终会被挪一次到其他gpu上

答：挪动数据不是问题，看上去挺额外的，但是不挪动会出错。对于小数据不算啥。

## 问题20:为什么 batch size调的比较小，比如8,精度会一直在0.1左右，直不怎么变化

答:lr太大了。batch size变小 lr不能太大

## 问题21:请问老师，如果训练集和验证集不同分布，该怎么训练

答：现在都是假设独立同分布，但现实不是的。

## 问题22:多叉树结构的输入(t如 dependency parse tree)如何 encode?有相应的成熟的方案吗？

答：用bert，GCN,TREENet相关看看

## 问题25:老师，个人dⅳy主机上装两块不同型号的gu,影响深度学习的性能吗

答：要注意两块GPU的性能强度，越强的就分的样本多一点。

## 问题26:这次课内赛有尝试直接用教材上的Vgg11,但不会有收敛，但同样的 dataloader, resets可以正常收敛。如果想解决相关问题的话，可以从什么角度入手呢？（除去学习率意外）

答：这个模型还是很不错的，在做finetune效果还是不错的