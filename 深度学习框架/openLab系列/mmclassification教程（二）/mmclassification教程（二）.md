# MMclassification  教程（二）
   1、主要重要内容
   - 修改配置文件，对预训练模型进行微调
   - 自带的工具包shell命令使用

## 1、安装
参考教程一中，pytorch，mmcv，mmcls 安装

## 2、下载数据集
这里使用kaggle上的猫狗数据集


```python
# 国内网访问不了：直接进kaggle下载 https://www.kaggle.com/dhirensk/cats-vs-dogs-training8000test2000
!mkdir data
!unzip -q archive.zip -d ./data/
```


```python
%ls data/dataset/
%mv data/dataset data/cats_dogs
%rm -r data/cats_dogs/dataset
```

    [0m[01;34mdataset[0m/  [01;34msingle_prediction[0m/  [01;34mtest_set[0m/  [01;34mtraining_set[0m/



```python
%ls data/cats_dogs/
```

    [0m[01;34msingle_prediction[0m/  [01;34mtest_set[0m/  [01;34mtraining_set[0m/



```python
#可视化一张图片
from PIL import Image
Image.open('data/cats_dogs/training_set/cats/cat.1.jpg')
```




​    
![png](output_6_0.png)
​    



## 3、MMClassification 数据集介绍
1、要求将数据集中图像和标签放在统同级目录下， 有两种方法支持多层自定义数据集。
- 最简单方式是将数据集转换成现有数据集（ImageNet，CoCo） ，GitHub中有数据转化换脚本，[链接](https://github.com/swimmant/DataProcessing) 如果有帮助，麻烦大大们给个Star,谢谢啦！
- 第二种方法： 新建数据集的类。详细查看 [文档](https://mmclassification.readthedocs.io/zh_CN/latest/tutorials/new_dataset.html)

### 3.1、ImageNet标准文件格式：
       1、根据图片的类别，存放至不同子目录下。训练数据文件夹结构如下所示：      （官网教程有点小错误）
            imagenet
            ├── ...
            ├── train
            │   ├── n01440764
            │   │   ├── n01440764_10026.JPEG
            │   │   ├── n01440764_10027.JPEG
            │   │   ├── ...
            │   ├── ...
            │   ├── n15075141
            │   │   ├── n15075141_999.JPEG
            │   │   ├── n15075141_9993.JPEG
            │   │   ├── ...
       2、提供了一个注释列表。列表的每一行都包含一个文件名及其相应的真实标签。格式如下：
            ILSVRC2012_val_00000001.JPEG 65
            ILSVRC2012_val_00000002.JPEG 970
            ILSVRC2012_val_00000003.JPEG 230
            ILSVRC2012_val_00000004.JPEG 809
            ILSVRC2012_val_00000005.JPEG 516
       注：真实标签的值应该位于 [0, 类别数目 - 1] 之间

### 3.2、将现用数据集转换为ImageNet数据集格式   


```python
# 当前文件夹格式
!tree -d data/cats_dogs/        #命令详细内容： !tree --help
```

    [01;34mdata/cats_dogs/[00m
    ├── [01;34msingle_prediction[00m
    ├── [01;34mtest_set[00m
    │   ├── [01;34mcats[00m
    │   └── [01;34mdogs[00m
    └── [01;34mtraining_set[00m
        ├── [01;34mcats[00m
        └── [01;34mdogs[00m
    
    7 directories



```python
!mkdir data/cats_dogs_ImageNet
```


```python
### Author: KequanChen
### Data: 2021年11月11日


# 创建ImageNet数据集格式，并创建标签。
from glob import  glob
import os
import shutil
from PIL import Image
import random

#函数功能：传入层级 图片路径字典，根目录，目标文件夹
def Image2File(imageNet,target_dir,cls_labelList):

    for d in imageNet:
        #对train，val，test写标签
        lines = []
        for cls in imageNet[f'{d}']:
            #os.makedirs(target_dir+d+'/'+cls,exist_ok=True)
            os.makedirs(target_dir+d,exist_ok=True)
            for img in imageNet[f'{d}'][f'{cls}']:
                #将图片copy到类别文件夹
                img_name = img.split('/')[-1]
                #target_img_path = target_dir+d+'/'+cls+'/'+ img_name
                target_img_path = target_dir+d+'/'+ img_name
                shutil.copy(img,target_img_path)

                #写入标签
                cls_index = cls_labelList.index(f'{cls}')
                label_line = f'{img_name} {cls_index}'
                lines.append(label_line+'\n')
                
        os.makedirs(f'{target_dir}meta',exist_ok=True)
        txt_path = target_dir+'meta'+'/'+d+'.txt'
        with open(txt_path, "w") as f:
            f.writelines(lines)
        cls_path = target_dir+'meta'+'/classes.txt'
    with open(cls_path,'w') as f1:
            #方法一：
        for line in cls_labelList:
            f1.write(line+'\n')
            #方法二：
#             lists=[line+"\n" for line in lists]
#            f1.writelines(lists)
            #方法三：
#             f1.write('\n'.join(lists))

def data2ImageNet():
    # 读取当前数据，整理成统一列表， 训练集，验证集，测试集 列表
    cls2label = ['cat', 'dog']
    cls_dir_list = ['cats', 'dogs']
    #root = 'F:/ChromeDown/archive/dataset/'
    root = 'data/cats_dogs/'
    data_dir_list = ['training_set','test_set']

    # imageNet 划分文件列表
    imageNet = {'train': {}, 'val': {}, 'test': {}}

    # 按类别读取所有的图片路径，后续还需要shuffle
    All_data = {}
    for clsLbael in cls2label:
        imageNet['train'].update({f'{clsLbael}': []})
        imageNet['val'].update({f'{clsLbael}': []})
        imageNet['test'].update({f'{clsLbael}': []})
        All_data.update({f'{clsLbael}': []})


    for dataDir in data_dir_list:
        for cls in cls_dir_list:
            #读取文件夹下某个类别
            img_dir =f'{root}{dataDir}/{cls}/'
            img_list = glob(img_dir+'*.jpg')
            #判断是某一类的图片
            for clsLbael in cls2label:
                    if clsLbael in cls:
                        #和全部数据路径取并集,并打乱
                        All_data[f'{clsLbael}'] =list(set(All_data[f'{clsLbael}']).union(set(img_list)))
                        random.shuffle(All_data[f'{clsLbael}'])

    print(len(All_data['dog']))
    print(len(All_data['cat']))

    #按类别比例划分数据集， 分为 train:0.6，val:0.2 ，test:0.2
    train = 0.6
    val = 0.2
    test = 0.2
    #获得截取各类数据下标并分配给标准格式
    for cls in All_data:
        clsList = All_data[f'{cls}']
        imageNet['train'][f'{cls}'] = clsList[0:int(train*len(clsList))]
        imageNet['val'][f'{cls}'] = clsList[int(train*len(clsList)):int((train+val)*len(clsList))]
        imageNet['test'][f'{cls}'] = clsList[int((train+val)*len(clsList)):]
        print('1',cls,len(imageNet['train'][f'{cls}']))
        print('2',cls,len(imageNet['val'][f'{cls}']))
        print('3',cls,len(imageNet['test'][f'{cls}']))
        print('4',cls,len(imageNet['test'][f'{cls}'])+len(imageNet['val'][f'{cls}'])+len(imageNet['train'][f'{cls}']))
        print('5',cls,int(train*len(clsList)))
        print('6',cls,int((train+val)*len(clsList)))
    # 创建ImageNet数据集 文件夹格式
    target_dir = 'data/cats_dogs_ImageNet/imagenet/'
    Image2File(imageNet,target_dir,cls2label)

```


```python
 data2ImageNet()
```

    5000
    5000
    1 cat 3000
    2 cat 1000
    3 cat 1000
    4 cat 5000
    5 cat 3000
    6 cat 4000
    1 dog 3000
    2 dog 1000
    3 dog 1000
    4 dog 5000
    5 dog 3000
    6 dog 4000



```python
# !cat data/cats_dogs_ImageNet/imagenet/meta/train.txt
```


```python
# !tree  data/cats_dogs_ImageNet/
```


```python
# !tree -d data/cats_dogs/
```

## 4、行工具使用
    MMCls 提供：
      1、模型训练
      2、模型微调
      3、模型测试
      4、对立计算
### 4.1、模型微调
    步骤：
        1、准备自定义数据集
        2、数据集适配MMCls要求
        3、在py脚本中修改配置文件
        4、使用命令工具进行训练，推理
    第1、2 步参考前面教程，下面介绍3，4
    在配置文件中修改配置文件
    特点：
        支持多配置文件继承（下面用configs中mobilenet_v2_b32x8_imagenet.py举例）


```python
_base_ = [
    '../_base_/models/mobilenet_v2_1x.py',                        #继承该文件创建模型的基本结构     
    '../_base_/datasets/imagenet_bs32_pil_resize.py',             #继承该文件定义数据集
    '../_base_/schedules/imagenet_bs256_epochstep.py',            #继承改文件定义学习策略
    '../_base_/default_runtime.py'                                #继承该文件配置运行策略
]

```


```python
#写好自己微调的config文件
!cat mmclassification/ownConfigs/mobilenet_v2_1x_cats_dogs.py
```

    #复制并修改configs/_base_/models/mobilenet_v2_1x.py
    # model settings
    model = dict(
        type='ImageClassifier',
        backbone=dict(type='MobileNetV2', widen_factor=1.0),
        neck=dict(type='GlobalAveragePooling'),
        head=dict(
            type='LinearClsHead',
            num_classes=2,
            in_channels=1280,
            loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
            topk=(1,),
        ))
    
    #复制并修改 configs/_base_/datasets/imagenet_bs32.py
    
    # dataset settings
    dataset_type = 'ImageNet'
    img_norm_cfg = dict(
        mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
    train_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(type='RandomResizedCrop', size=224),
        dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
        dict(type='Normalize', **img_norm_cfg),
        dict(type='ImageToTensor', keys=['img']),
        dict(type='ToTensor', keys=['gt_label']),
        dict(type='Collect', keys=['img', 'gt_label'])
    ]
    test_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(type='Resize', size=(256, -1)),
        dict(type='CenterCrop', crop_size=224),
        dict(type='Normalize', **img_norm_cfg),
        dict(type='ImageToTensor', keys=['img']),
        dict(type='Collect', keys=['img'])
    ]
    
    data = dict(
        samples_per_gpu=32,
        workers_per_gpu=1,
        train=dict(
            type=dataset_type,
            data_prefix='../../data/cats_dogs_ImageNet/imagenet/train',
            ann_file='../../data/cats_dogs_ImageNet/imagenet/meta/train.txt',
            classes = '../../data/cats_dogs_ImageNet/imagenet/meta/classes.txt',
            pipeline=train_pipeline),
        val=dict(
            type=dataset_type,
            data_prefix='../../data/cats_dogs_ImageNet/imagenet/val',
            ann_file='../../data/cats_dogs_ImageNet/imagenet/meta/val.txt',
            classes = '../../data/cats_dogs_ImageNet/imagenet/meta/classes.txt',
            pipeline=test_pipeline),
        test=dict(
            # replace `data/val` with `data/test` for standard test
            type=dataset_type,
            data_prefix='../../data/cats_dogs_ImageNet/imagenet/test',
            ann_file='../../data/cats_dogs_ImageNet/imagenet/meta/test.txt',
            classes = '../../data/cats_dogs_ImageNet/imagenet/meta/classes.txt',
            pipeline=test_pipeline))
    evaluation = dict(interval=1, metric='accuracy')


​    
    #复制并修改 configs/_base_/schedules/imagenet_bs256_epochstep.py
    # optimizer
    optimizer = dict(type='SGD', lr=0.045, momentum=0.9, weight_decay=0.00004)
    optimizer_config = dict(grad_clip=None)
    # learning policy
    lr_config = dict(policy='step', gamma=0.98, step=1)
    runner = dict(type='EpochBasedRunner', max_epochs=2)


​    
    #复制并修改 configs/_base_/default_runtime.py
    
    # checkpoint saving
    checkpoint_config = dict(interval=1)
    # yapf:disable
    log_config = dict(
        interval=100,
        hooks=[
            dict(type='TextLoggerHook'),
            dict(type='TensorboardLoggerHook')         #Tensorboard 看板
        ])
    # yapf:enable
    
    dist_params = dict(backend='nccl')
    log_level = 'INFO'
    load_from = 'mmclassification/checkpoints/mobilenet_v2_batch256_imagenet_20200708-3b2dc3af.pth'
    resume_from = None                         #断点训练
    workflow = [('train', 1)]




```python
%pwd

```




    '/home/snnu/chenkequan/notebooke/mmcls'




```python
%ls
```



```python
#安装tensorboard
# !pip install tensorboard
```


```python
### 使用脚本训练
# !python mmclassification/tools/train.py mmclassification/ownConfigs/mobilenet_v2_1x_cats_dogs.py --work-dir work_dirs/mobilenet_v2_1x_cats_dogs
```


```python
%ls ./work_dirs/mobilenet_v2_1x_cats_dogs/
```

    20211111_184110.log       20211111_192356.log.json
    20211111_184148.log       20211111_192409.log
    20211111_184215.log       20211111_192409.log.json
    20211111_184333.log       20211111_192439.log
    20211111_184512.log       20211111_192439.log.json
    20211111_191653.log       20211111_192514.log
    20211111_191848.log       20211111_192514.log.json
    20211111_191906.log       20211111_192740.log
    20211111_191932.log       20211111_192740.log.json
    20211111_192005.log       20211111_203648.log
    20211111_192005.log.json  20211111_203752.log
    20211111_192121.log       20211111_203752.log.json
    20211111_192121.log.json  epoch_1.pth
    20211111_192255.log       epoch_2.pth
    20211111_192255.log.json  [0m[01;36mlatest.pth[0m@
    20211111_192340.log       mobilenet_v2_1x_cats_dogs.py
    20211111_192340.log.json  [01;34mtf_logs[0m/
    20211111_192356.log


## 5、测试模型
        使用tools/test.py 对模型进行推理
   `python tools/test.py 配置文件 权重文件 其他参数 `
   1、参数说明：
   - --metrics: 评价方式，依赖于数据集 如 准确率
   - --metric-options: 对评估操的自定义操作 如 topk=1



```python
!python mmclassification/tools/test.py mmclassification/ownConfigs/mobilenet_v2_1x_cats_dogs.py work_dirs/mobilenet_v2_1x_cats_dogs/latest.pth --metrics=accuracy --metric-options=topk=1
```

    /home/snnu/miniconda3/envs/mmcls/lib/python3.8/site-packages/mmcv/cnn/bricks/transformer.py:28: UserWarning: Fail to import ``MultiScaleDeformableAttention`` from ``mmcv.ops.multi_scale_deform_attn``, You should install ``mmcv-full`` if you need this module. 
      warnings.warn('Fail to import ``MultiScaleDeformableAttention`` from '
    Use load_from_local loader
    [>>>>>>>>>>>>>>>>>>>>>>>>>>>] 2000/2000, 418.0 task/s, elapsed: 5s, ETA:     0s
    accuracy : 53.95



```python
# 推理计算并保存结果
!python mmclassification/tools/test.py mmclassification/ownConfigs/mobilenet_v2_1x_cats_dogs.py work_dirs/mobilenet_v2_1x_cats_dogs/latest.pth --out=result.json
```

    /home/snnu/miniconda3/envs/mmcls/lib/python3.8/site-packages/mmcv/cnn/bricks/transformer.py:28: UserWarning: Fail to import ``MultiScaleDeformableAttention`` from ``mmcv.ops.multi_scale_deform_attn``, You should install ``mmcv-full`` if you need this module. 
      warnings.warn('Fail to import ``MultiScaleDeformableAttention`` from '
    Use load_from_local loader
    [>>>>>>>>>>>>>>>>>>>>>>>>>>>] 2000/2000, 418.9 task/s, elapsed: 5s, ETA:     0s
    dumping results to result.json



```python
#文件中保存每张图片预测的结果
# !cat result.json
```

