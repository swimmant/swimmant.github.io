# 第一天

## 第一部分为环境配置（过于简单略）后续参考mmdection框架或者triton的环境配置


```python
import torch
import numpy as np
```

##  二、张量的简介与创建

### 2.1 Tensor概念 
    张量是一个多维数组，标量，向量，矩阵都为张量。

#### Tensor与Variable  #从pytorch0.4版本并入张量
   ##### Variable 是 torch.autograd的数据类型。
    主要用于封装Tensor，方便进行自动求导
    主要参数：
        data : 存放被封装的Tensor
        grad : data的梯度
        grad_fn : 创建Tensor的函数Function，是计算导数的关键
        requires_grad : 标识是否计算梯度
        is_leaf : 标识是否为叶子节点张量  
<font color="red">注意：</font>一般情况只对叶子节点更新梯度，不保留非叶子节点梯度，减低显存消耗。若要实现非叶子节点的梯度，可以使用Tensor.register_hook函数和Tensor.retain_grad()，两者区别在于，前者直接打印占用显存很少，后者使用方便但耗显存。
##### pytorch 0.4版本后，Variable并入Tensor
   ###### torch.Tensor中的参数：
        data : 存放被封装的Tensor
        grad : data的梯度
        grad_fn : 创建Tensor的函数Function，是计算导数的关键
        requires_grad : 标识是否计算梯度
        is_leaf : 标识是否为叶子节点张量
        dtype : 张量的数据类型，如 torch.FloatTensor, torch.cuda.FloatTensor
        shape ：张量的形状,(3,244,244)
        device ：标识张量所在的设备 CPU or GPU ，用于加速环节
    


### 2.2 张量的创建
   #### 2.2.1、直接创建
   ##### 2.2.1.1、torch.tensor()  功能：从data创建tensor
        参数：
            data : 数据 可以是list，numpy
            dtype : 数据类型，默认与data输入类型一致
            device :所在设备, gpu(cuda)/cpu
            requires_grad : 是否需要梯度
            pin_memory : 是否存入锁页内存
<font color="red">注意：</font> ①主机中的内存有两种方式，一是**锁页**，二是***不锁页***,锁页内存中存放的内容在任何情况不与虚拟内存（硬盘）进行交换，而不锁页内存，在主机内存不足时，数据会存放在虚拟内存中。
②显存均为锁页内存
③创建DataLoader时，若设置pin_memory=True，则意味着从数据加载时属于内存中的锁页内存，这将加快Tensor转义到GPU的显存上。
            


```python
#------example----------
flag = True
# flag = False
if flag:
    list1 = [[5., 5., 5.],
             [1., 1. ,1.],
             [1., 1., 1.]]
    arr = np.ones((3,3))
    print("ndarray的数据类型为",type(arr))
    print("ndarray：",arr)
    print("list的数据类型为",type(list1))
    print("list1：",list1)
    tens = torch.tensor(arr, device='cuda')
    tens1 = torch.tensor(arr)
    tens2 = torch.tensor(list1)
    print(tens)
    
    print(tens1)
    print(tens2)
```

    ndarray的数据类型为 <class 'numpy.ndarray'>
    ndarray： [[1. 1. 1.]
     [1. 1. 1.]
     [1. 1. 1.]]
    list的数据类型为 <class 'list'>
    list1： [[5.0, 5.0, 5.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
    tensor([[1., 1., 1.],
            [1., 1., 1.],
            [1., 1., 1.]], device='cuda:0', dtype=torch.float64)
    tensor([[1., 1., 1.],
            [1., 1., 1.],
            [1., 1., 1.]], dtype=torch.float64)
    tensor([[5., 5., 5.],
            [1., 1., 1.],
            [1., 1., 1.]])


#####  2.2.1.2 torch.from_numpy(ndarray)
    功能：从numpy创建tensor
<font color="red">注意：</font> 使用该方法创建tensor与原ndarray共享内存，当修改其中的数据，另一个变量也会改变


```python
#------example---------
arr = np.array([[1,2,3],[4,5,6]])
print("原始arr",arr)
tens = torch.from_numpy(arr)
print(tens)
print("修改tensor")
tens[0,0]= -2
print("修改后arr",arr)
```

    原始arr [[1 2 3]
     [4 5 6]]
    tensor([[1, 2, 3],
            [4, 5, 6]])
    修改tensor
    修改后arr [[-2  2  3]
     [ 4  5  6]]


#### 2.2.2 依据数值创建
   ##### 2.2.2.1 torch.zeros()
   **功能**： 依据size创建全0张量</br>
   参数：
   + size : 张量的形状，如 （3,3）
   + out : 输出的张量； out参数是把张量复制给另一个变量
   + layout : 内存中布局形式，有strided, sparse_coo 等 当是稀疏矩阵时，设置为 sparse_coo 可以减少内存占用
   + device : 所在设备
   + require_grad : 是否需要梯度
   


```python
#-----example————————————
out_t = torch.tensor([1])
print(out_t)
tens = torch.zeros((3,3),out=out_t)
print(tens,'\n',out_t)
print(id(tens),id(out_t),id(tens) == id(out_t))   #内存地址相同，说明生成全0张量同时也输出给out_t
```

    tensor([1])
    tensor([[0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]]) 
     tensor([[0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]])
    139670394641472 139670394641472 True



```python

```


```python

```


```python

```


```python

```


```python

```
