# GPU的使用和选购

## 1、查看设备


```python
!nvidia-smi
```

    Mon Nov 15 18:56:56 2021       
    +-----------------------------------------------------------------------------+
    | NVIDIA-SMI 440.33.01    Driver Version: 440.33.01    CUDA Version: 10.2     |
    |-------------------------------+----------------------+----------------------+
    | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
    | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
    |===============================+======================+======================|
    |   0  TITAN Xp            Off  | 00000000:01:00.0  On |                  N/A |
    | 23%   34C    P8    10W / 250W |    163MiB / 12190MiB |      0%      Default |
    +-------------------------------+----------------------+----------------------+
    |   1  GeForce RTX 208...  Off  | 00000000:02:00.0 Off |                  N/A |
    | 27%   34C    P8    13W / 250W |   1725MiB / 11019MiB |      0%      Default |
    +-------------------------------+----------------------+----------------------+
                                                                                   
    +-----------------------------------------------------------------------------+
    | Processes:                                                       GPU Memory |
    |  GPU       PID   Type   Process name                             Usage      |
    |=============================================================================|
    |    0      1098      G   /usr/lib/xorg/Xorg                           112MiB |
    |    0      2704      G   compiz                                        37MiB |
    |    1      4977      C   python                                      1713MiB |
    +-----------------------------------------------------------------------------+


### 1.1、计算设备


```python
import torch
from torch import nn
torch.device('cpu'),torch.cuda.device('cuda'),torch.cuda.device('cuda:1')
```




    (device(type='cpu'),
     <torch.cuda.device at 0x7ff893603130>,
     <torch.cuda.device at 0x7ff893603310>)



### 1.2、查看可用gpu数量


```python
torch.cuda.device_count()
```




    2



### 1.3、定义当gpu不存在时运行代码


```python
def try_gpu(i=0):
    """如果存在，则返回gpu(i),否则返回cpu()"""
    if torch.cuda.device_count()>=i+1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

def try_all_gpus():
    """返回所有可用的GPU，如果没有GPU，则返回[cpu,]"""
    devices = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]

try_gpu(), try_gpu(10),try_all_gpus()
```




    (device(type='cuda', index=0),
     device(type='cpu'),
     [device(type='cuda', index=0), device(type='cuda', index=1)])



## 2、gpu上数据计算

### 2.1、查询张量所在设备


```python
x = torch.tensor([1,2,3])           #默认在cpu内存上
x.device
```




    device(type='cpu')



## 2.2、创建在gpu


```python

X = torch.ones(2,3,device=try_gpu())
X
```




    tensor([[1., 1., 1.],
            [1., 1., 1.]], device='cuda:0')



## 2.3、在第二个GPU上创建随机张量


```python
Y = torch.rand(2,3,device=try_gpu(1))
Y
```




    tensor([[0.4411, 0.3717, 0.7920],
            [0.5166, 0.7394, 0.6777]], device='cuda:1')



## 2.4、要计算X+Y，需要决定在哪执行操作


```python

Z = X.cuda(1)
print(X)
print(Z)
```

    tensor([[1., 1., 1.],
            [1., 1., 1.]], device='cuda:0')
    tensor([[1., 1., 1.],
            [1., 1., 1.]], device='cuda:1')


现在数据在同一GPU上，可以见它们相加


```python
Y+Z
```




    tensor([[1.4411, 1.3717, 1.7920],
            [1.5166, 1.7394, 1.6777]], device='cuda:1')




```python
Z.cuda(1) is Z             #如果Z在cuda：1 ，就不会将Z进行copy
```




    True



## 3、神经网络与GPU


```python
net = nn.Sequential(nn.Linear(3,1))
net = net.to(device=try_gpu())                #将所有参数copy到0号GPU

net(X)
```




    tensor([[-1.1968],
            [-1.1968]], device='cuda:0', grad_fn=<AddmmBackward>)



确认模型参数存储在同一GPU上


```python
net[0].weight.data.device                  #模型参数在0号GPU
```




    device(type='cuda', index=0)




```python

```

## 购买GPU

两件事比较重要：①显存 ②计算能力：每秒钟计算浮点数次数 ③价格


```python
①买新的不买旧的 ②买承受最贵的
```


```python

```


```python

```


```python

```
