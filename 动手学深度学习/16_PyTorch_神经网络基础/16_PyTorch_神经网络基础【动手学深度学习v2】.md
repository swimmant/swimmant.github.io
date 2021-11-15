# 1、层和块


```python
import torch 
from torch import nn
from torch.nn import functional as F      #定义了一些有用的函数

net = nn.Sequential(nn.Linear(20,256),
                    nn.ReLU(),
                    nn.Linear(256,10)
                   )
X = torch.rand(2 ,20)         #2为批处理大小
net(X)
```




    tensor([[ 0.0575,  0.0646, -0.1531, -0.2502,  0.2134, -0.2487,  0.1119, -0.2793,
             -0.0771,  0.2374],
            [ 0.0821, -0.1458, -0.1899, -0.3212,  0.2784, -0.1832,  0.1434, -0.3080,
             -0.2397,  0.1752]], grad_fn=<AddmmBackward>)



nn.Sequential 定义一种特殊的Module

## 1.1、自定义MLP


```python

class MLP(nn.Module):                               #继承nn.Module
    def __init__(self):                             #定义组件
        super().__init__()
        self.hidden = nn.Linear(20,256)             #定义输入为20输出为256的线性层
        self.out = nn.Linear(256,10)
    def forward(self ,X):                           #定义前向传播
        return self.out(F.relu(self.hidden(X)))
        
```


```python
net = MLP()
net(X)
```




    tensor([[ 0.1075,  0.1499,  0.0019, -0.1983,  0.1055,  0.0090,  0.0848,  0.0717,
              0.2854, -0.1110],
            [ 0.1165,  0.0994, -0.0794, -0.2184,  0.1116, -0.0013,  0.0785, -0.0218,
              0.3297, -0.1320]], grad_fn=<AddmmBackward>)



## 1.2、顺序块


```python
class MySequentail(nn.Module):
    def __init__(self,*args):
        super().__init__()
        for block in args:                            #
            self._modules[block] = block               # 特殊的容器，按序的字典
    def forward(self,X):
        for block in self._modules.values():
            X = block(X)
        return X
net = MySequentail(nn.Linear(20,256),nn.ReLU(),nn.Linear(256,10))
net(X)
```




    tensor([[ 0.0464, -0.1645,  0.0924, -0.0242,  0.1961,  0.2435,  0.0517, -0.0803,
             -0.0844, -0.1829],
            [ 0.1450, -0.2823,  0.0479, -0.0597,  0.1657,  0.1154,  0.0692, -0.0372,
             -0.1532, -0.2183]], grad_fn=<AddmmBackward>)



## 1.3、灵活构造函数，在正向传播中执行代码


```python
class FixedHiddenMLP(nn.Module):           #在init和forward中可以做大量的自定义计算
    def __init__(self):
        super().__init__()
        self.rand_weight = torch.rand((20,20),requires_grad=False)    #随机weight不参与训练
        self.linear = nn.Linear(20,20)
    def forward(self ,X):
        X =  self.linear(X)                           #先做完linear
        X = F.relu(torch.mm(X,self.rand_weight)+1)    #和随机weights做乘法
        X = self.linear(X)                            #在调用linear
        while X.abs().sum()>1:                       #对X绝对值求和如果大于一，除二
            X/=2
        return X.sum()
net = FixedHiddenMLP()
net(X)
```




    tensor(0.1398, grad_fn=<SumBackward0>)



## 1.4、嵌套使用Sequential或自定义方法自由组合搭配


```python
class NestMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(20,64),nn.ReLU(),
                                 nn.Linear(64,32),nn.ReLU()
                                )
        self.linear = nn.Linear(32,16)
        
    def forward(self,X):
        return self.linear(self.net(X))

chimra = nn.Sequential(NestMLP(),nn.Linear(16,20),FixedHiddenMLP())
chimra(X)
```




    tensor(-0.0030, grad_fn=<SumBackward0>)



## 2、参数管理
    首先关注单隐藏层的多层感知机


```python
import torch
from torch import nn

net = nn.Sequential(nn.Linear(4,8),nn.ReLU(),nn.Linear(8,1))
X = torch.rand(size=(2,4))
net(X)
```




    tensor([[-0.3017],
            [-0.3544]], grad_fn=<AddmmBackward>)



### 2.1 参数访问
    将每一层中的权重拿出来

sequential可以看做为python 的list ； net[2]相当于拿到nn.Linear(8,1)


```python
print(net[2].state_dict())
```

    OrderedDict([('weight', tensor([[-0.3533,  0.0262, -0.1229, -0.0872,  0.3234, -0.3202, -0.2343,  0.2017]])), ('bias', tensor([-0.1021]))])


### 2.2、访问具体的参数


```python
print(type(net[2].bias))
print(net[2].bias)
print(net[2].bias.data)     #访问数据，其中还有梯度
```

    <class 'torch.nn.parameter.Parameter'>
    Parameter containing:
    tensor([-0.1021], requires_grad=True)
    tensor([-0.1021])



```python
net[2].weight.grad == None
```




    True




```python
### 2.3、一次性访问所有参数
print(*[(name,param.shape) for name,param in net[0].named_parameters()])
print(*[(name,param.shape) for name,param in net.named_parameters()])                #* 是对list进行拆解，访问liet中每一组数据， 解包
```

    ('weight', torch.Size([8, 4])) ('bias', torch.Size([8]))
    ('0.weight', torch.Size([8, 4])) ('0.bias', torch.Size([8])) ('2.weight', torch.Size([1, 8])) ('2.bias', torch.Size([1]))



```python
net.state_dict()['2.bias'].data          #通过某一层的名字直接访问其中数值   
```




    tensor([-0.1021])



### 2.3、嵌套网络中收集参数


```python
def block1():
    return nn.Sequential(nn.Linear(4,8),nn.ReLU(), nn.Linear(8,4),nn.ReLU())

def block2():
    net = nn.Sequential()
    for i in range(4):
        net.add_module(f'block{i}',block1())
    return net

rgnet = nn.Sequential(block2(),nn.Linear(4,1))
rgnet(X)
```




    tensor([[-0.2302],
            [-0.2303]], grad_fn=<AddmmBackward>)




```python
print(rgnet)
```

    Sequential(
      (0): Sequential(
        (block0): Sequential(
          (0): Linear(in_features=4, out_features=8, bias=True)
          (1): ReLU()
          (2): Linear(in_features=8, out_features=4, bias=True)
          (3): ReLU()
        )
        (block1): Sequential(
          (0): Linear(in_features=4, out_features=8, bias=True)
          (1): ReLU()
          (2): Linear(in_features=8, out_features=4, bias=True)
          (3): ReLU()
        )
        (block2): Sequential(
          (0): Linear(in_features=4, out_features=8, bias=True)
          (1): ReLU()
          (2): Linear(in_features=8, out_features=4, bias=True)
          (3): ReLU()
        )
        (block3): Sequential(
          (0): Linear(in_features=4, out_features=8, bias=True)
          (1): ReLU()
          (2): Linear(in_features=8, out_features=4, bias=True)
          (3): ReLU()
        )
      )
      (1): Linear(in_features=4, out_features=1, bias=True)
    )


## 3、内置初始化


```python
def init_normal(m):                                  #定义正太分布
    if type(m)==nn.Linear:                           #如果是线性层做均值为0，方差为0.01的初始化
        nn.init.normal_(m.weight,mean=0,std=0.01)    #有_ 的函数表示是替换函数，原地操作 ，将其中的weights替换掉
        nn.init.zeros_(m.bias)
net.apply(init_normal)
net[0].weight.data[0],net[0].bias.data[0]
```




    (tensor([0.0060, 0.0060, 0.0031, 0.0024]), tensor(0.))




```python
def init_constant(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight,1)
        nn.init.zeros_(m.bias)
net.apply(init_constant)
net[0].weight.data[0],net[0].bias.data[0]
```




    (tensor([1., 1., 1., 1.]), tensor(0.))



### 3.1、对某些块使用不同的初始化方法


```python

def xavier(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)

def init_42(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight,42)

net[0].apply(xavier)                    #对第一个层使用Xavier初始化
net[2].apply(init_42)                   #对最后一个层使用init_42的初始化
print(net[0].weight.data[0])
print(net[2].weight.data)
```

    tensor([ 0.0405, -0.6049, -0.5710,  0.0473])
    tensor([[42., 42., 42., 42., 42., 42., 42., 42.]])


### 3.2 自定义初始化


```python
def my_init(m):
    if type(m) == nn.Linear:
        print(
            "Init",
            *[(name,param.shape) for name , param in m.named_parameters()][0]
        )
        nn.init.uniform_(m.weight,-10,10)
        m.weight.data *= m.weight.data.abs() >=5         # 当前权重数据乘以(自己绝对值中，保留大于等于5的值，其余为0）
net.apply(my_init)
net[0].weight[:2]
```

    Init weight torch.Size([8, 4])
    Init weight torch.Size([1, 8])





    tensor([[ 8.5630, -0.0000, -0.0000,  0.0000],
            [ 0.0000,  0.0000, -8.5134, -6.0404]], grad_fn=<SliceBackward>)




```python
# 暴力直接修改
net[0].weight.data[:] +=1
net[0].weight.data[0,0] = 42
net[0].weight.data[0]
```




    tensor([42.,  1.,  1.,  1.])



### 3.3、参数绑定
    小应用：在一些层之间share一些param，连个输入数据流进来如何共享权重
    先构造shared层，构造出来时参数就生成


```python
shared = nn.Linear(8,8)
net = nn.Sequential(nn.Linear(4,8),nn.ReLU(),shared,nn.ReLU(),shared,nn.ReLU(),nn.Linear(8,1))

net(X)
print(net[2].weight.data[0] == net[4].weight.data[0])
net[2].weight.data[0,0] == 100
print(net[2].weight.data[0] == net[4].weight.data[0])
```

    tensor([True, True, True, True, True, True, True, True])
    tensor([True, True, True, True, True, True, True, True])


## 4、自定义层
    构造一个没有任何参数的自定义层


```python
import torch 
import torch.nn.functional as F
from torch import nn

class CenteredLayer(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self ,X):
        return X - X.mean()
layer = CenteredLayer()
layer(torch.FloatTensor([1,2,3,4,5]))
```




    tensor([-2., -1.,  0.,  1.,  2.])



### 4.1 将自己构造的层放入sequential


```python
net = nn.Sequential(nn.Linear(8,128),CenteredLayer())

Y = net(torch.rand(4,8))
Y.mean()
```




    tensor(1.8626e-09, grad_fn=<MeanBackward0>)



### 4.2 自定义带有参数的层


```python
class MyLinear(nn.Module):
    def __init__(self,in_units,units):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_units,units))  #两参数， 随机初始化(输入大小，输出大小)矩阵
        self.bias = nn.Parameter(torch.randn(units,))               #randn 正态分布  ；加逗号是创建列向量
        
    def forward(self, X):
        linear = torch.matmul(X,self.weight.data) + self.bias.data
        return F.relu(linear)

dense = MyLinear(5,3)
dense.weight
```




    Parameter containing:
    tensor([[ 0.0804, -0.5038, -1.3182],
            [-0.2190,  0.1457,  1.5468],
            [ 0.0219, -0.7152, -0.6811],
            [-0.8235,  2.0442,  1.8378],
            [-1.4888, -0.1597,  0.9653]], requires_grad=True)



###  4.3 自定层直接执行正向传播计算


```python
dense(torch.rand(2,5))
```




    tensor([[1.1066, 1.1407, 0.4707],
            [0.1582, 2.8575, 1.7879]])



###  4.4 自定层构建模型


```python
net = nn.Sequential(MyLinear(64,8),MyLinear(8,1))
net(torch.rand(2,64))
```




    tensor([[4.2691],
            [0.0000]])



## 5、读写文件
    加载和保存张量


```python
%ls
```

    04 数据操作 + 数据预处理【动手学深度学习v2】.ipynb  [0m[01;34md2l-zh[0m/
    16 PyTorch 神经网络基础【动手学深度学习v2】.ipynb   V10多层感知机.ipynb



```python
x = torch.arange(4)
torch.save(x,'x-file')
x2 = torch.load("x-file")
x2
```




    tensor([0, 1, 2, 3])




```python
%ls
```

    04 数据操作 + 数据预处理【动手学深度学习v2】.ipynb  [0m[01;34md2l-zh[0m/              x-file
    16 PyTorch 神经网络基础【动手学深度学习v2】.ipynb   V10多层感知机.ipynb


### 5.1、存一个张量列表，然后将它们读回内存


```python
y = torch.zeros(4)
torch.save([x,y],'x-file')
x2,y2 = torch.load("x-file")
(x2,y2)
```




    (tensor([0, 1, 2, 3]), tensor([0., 0., 0., 0.]))



### 5.2、写入并读取字符串映射的张量字典


```python
mydict = {'x':x , 'y':y}
torch.save(mydict,'mydict')
mydict2 = torch.load('mydict')
mydict2
```




    {'x': tensor([0, 1, 2, 3]), 'y': tensor([0., 0., 0., 0.])}



### 5.3、加载和保存模型参数  
    pytorch不存储计算图，只存储权重参数，torchScript存储计算图


```python


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20,256)
        self.output = nn.Linear(256,10)
    def forward(self,x):
        return self.output(F.relu(self.hidden(x)))
    
net = MLP()
X = torch.randn(size=(2,20))
Y = net(X)
```


```python
# 将模型参数存储一个mlp.params
torch.save(net.state_dict(),'mlp.params')
```

实例化原始多层感知机模型，直接读取文件中存储的参数


```python
clone = MLP()
clone.load_state_dict(torch.load("mlp.params"))
clone.eval()
```




    MLP(
      (hidden): Linear(in_features=20, out_features=256, bias=True)
      (output): Linear(in_features=256, out_features=10, bias=True)
    )




```python
Y_clone = clone(X)                   #clone一个网络，输入和同样之前的随机X，拿到Y 进行对比
Y_clone == Y
```




    tensor([[True, True, True, True, True, True, True, True, True, True],
            [True, True, True, True, True, True, True, True, True, True]])


