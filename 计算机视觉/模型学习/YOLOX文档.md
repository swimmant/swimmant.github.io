# YOLOX 文档

## 环境配置

### 第一步 安装YoloX

	当前教程环境为 cuda11 torch1.7.1 python3.8
	git clone git@github.com:Megvii-BaseDetection/YOLOX.git
	cd YOLOX
	pip3 install -U pip && pip3 install -r requirements.txt
	pip3 install -v -e .  # or  python3 setup.py develop

### 第2步。安装pycocotools （使用coco数据集）

```python
pip3 install cython; pip3 install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
```

## Demo 

### 第1步。从基准表下载预训练模型。

现在权重文件地址：https://github.com/monsterchan/YOLOX  的Benchmark的权重地址文

### 第2步。使用 -n 或 -f 来指定检测器的配置。例如：

```python
python tools/demo.py image -n yolox-s -c /path/to/your/yolox_s.pth --path assets/dog.jpg --conf 0.25 --nms 0.45 --tsize 640 --save_result --device [cpu/gpu]
```

或者

```python
python tools/demo.py image -f exps/default/yolox_s.py -c /path/to/your/yolox_s.pth --path assets/dog.jpg --conf 0.25 --nms 0.45 --tsize 640 --save_result --device [cpu/gpu]
```

视频推理

```python
python tools/demo.py video -n yolox-s -c /path/to/your/yolox_s.pth --path /path/to/your/video --conf 0.25 --nms 0.45 --tsize 640 --save_result --device [cpu/gpu]
```

参数说明：

**video 或image ：**指定推理类型

**-n 或 -f** ：指定实验配置文件和参数 ；使用-n时一般只有官方推荐的几种模型配置，只需要指定模型的类型如 yolox-s或者yolox-m 程序自动生成exp文件。使用-f时，是自定义exp文件，控制训练模型的大小和迭代次数等参数

**--path：**存放要推理的资源路径

**--conf：** 设置自信度

 **--nms**：设置非极大抑制筛选阈值

**--tsize：**输入图片的尺寸大小

**--save_result**：存放输出结果

**--device**：选择推理设备

## 模型自定义训练

### step1：准备数据集

数据集：目前支持coco和voc数据集

准备coco数据集：cd <YOLOX_HOME>  

​								ln -s /path/to/your/COCO ./datasets/COCO

​	python tools/train.py -n yolox-s -d 8 -b 64 --fp16 -o [--cache]

或者准备voc数据集

### step2：编写对应Dataset类

通过__getitem__方法加载图片和标签

voc：

```python
   @Dataset.resize_getitem
    def __getitem__(self, index):
        img, target, img_info, img_id = self.pull_item(index)

        if self.preproc is not None:
            img, target = self.preproc(img, target, self.input_dim)

        return img, target, img_info, img_id
```

**注意：**在pull_item和load_anno方法中实现 Mosiac MixUp 数据增强

### step3：准备评估器

目前支持 COCO 格式和 VOC 格式

### step4：将数据集放入datasets文件夹下

例如VOC：**ln -s /path/to/your/VOCdevkit ./datasets/VOCdevkit**    linux中建立软连接

### **step5：创建EXP文件来控制整个实验**

**exp文件：**包括模型设置、训练设置和测试设置

 **yolox_base.py**： 完整的exp文件，继承此类，覆盖方法即可

**voc例子：** ①选择yolox_模型②应该改变网络的深度和宽度，以及voc的种类数量③还需要在训练模型前 覆盖dataset和evalutor

### step6：训练自己模型

```python
python tools/train.py -f exps/example/yolox_voc/yolox_voc_s.py -d 8 -b 64 --fp16 -o -c /path/to/yolox_s.pth [--cache]
```

**–cache：**我们现在支持 RAM 缓存以加快训练速度！采用它时，请确保您有足够的系统 RAM。

```python
python tools/train.py -f exps/chan/yolox_voc_m.py -d 0 -b8 --fp16 -o -c /checkpoint/yolox_m.pth
```

## YOLOX-TensorRT

### 工具包

官方文档：

https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html

项目地址：

https://github.com/NVIDIA-AI-IOT/torch2trt

yolox项目中自带 torch2trt 轻松转换为 TensorRT 模型

```
python tools/trt.py -n <YOLOX_MODEL_NAME> -c <YOLOX_CHECKPOINT>

python tools/trt.py -n yolox-s -c your_ckpt.pth
```

自定义模型使用标志 -f 指定您的 exp 文件

```python
python tools/trt.py -f <YOLOX_EXP_FILE> -c <YOLOX_CHECKPOINT>

python tools/trt.py -f /path/to/your/yolox/exps/yolox_s.py -c your_ckpt.pth
```

转换后的模型和序列化引擎文件（用于 C++ 演示）将保存在您的实验输出目录中。

示例：

```python
python tools/demo.py image -n yolox-s --trt --save_result

python tools/demo.py image -f exps/default/yolox_s.py --trt --save_result
```

### 转ONNX

**进入项目目录**

**通过 -n 转换标准 YOLOX 模型**：

```python
python3 tools/export_onnx.py --output-name yolox_s.onnx -n yolox-s -c yolox_s.pth
```

-n：指定型号名称

-c：训练过的模型

-o：opset 版本，默认 11。但是，如果您将 onnx 模型进一步转换为OpenVINO，请将 opset 版本指定为 10。

–no-onnxsim: 禁用 onnxsim

要 onnx 模型自定义输入形状，请在 tools/export.py 中修改以下代码：

```
dummy_input = torch.randn(1, 3, exp.test_size[0], exp.test_size[1])
```

**通过 -f 转换标准 YOLOX 模型：**

```
python3 tools/export_onnx.py --output-name yolox_s.onnx -f exps/default/yolox_s.py -c yolox_s.pth

python3 tools/export_onnx.py --output-name your_yolox.onnx -f exps/your_dir/your_yolox.py -c your_yolox.pth

python tools/export_onnx.py --output-name helmet.onnx -f exps/chan/yolox_voc_m.py -c YOLOX_outputs/yolox_voc_m/last_epoch_ckpt.pth
```

在ONNXRuntime环境下推理：

```
cd <YOLOX_HOME>/demo/ONNXRuntime

python3 onnx_inference.py -m <ONNX_MODEL_PATH> -i <IMAGE_PATH> -o <OUTPUT_DIR> -s 0.3 --input_shape 640,640
```

-m：您转换后的 onnx 模型

-i: 输入图像

-s：可视化的分数阈值

--input_shape：应该和你用于onnx转换的shape一致



