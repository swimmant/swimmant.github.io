# Deep learning

作者：Yann Lecun ；Yoshua Bengio ；Geoffrey Hinton 

## 一、引言

> Deep learning allows computational models that are composed of multiple processing layers to learn representations of data with multiple levels of abstraction. These methods have dramatically improved the state-of-the-art in speech recognition, visual object recognition, object detection and many other domains such as drug discovery and genomes. Deep learning discovers intricate structure in large data sets by using the back-propagation algorithm to indicate how a machine should change its internal parameters that are used to compute the representation in each layer from the representation in the previous layer. Deep convolutional nets have brought about breakthroughs in processing images, video, speech and audio, whereas recurrent nets have shone light on sequential data such as text and speech

深度学习由多个处理层组成的计算模型用来学习具有多抽象水平的数据表征。 这些方法极大地提高了语音识别、视觉目标识别（分类）、检测和许多其他领域（如药物发现和基因序列）的最新技术水平。 深度学习通过使用反向传播算法来发现大型数据集中的复杂结构，以指导机器（电脑）应如何修改其内部参数，这些参数用于从每一层的前一层中的表征信息来计算该层的表征信息。 深度卷积网络在处理图像、视频、语音和音频方面取得了突破，而循环网络则在处理文本和语音等序列数据方面取得了突破。

**感想**：深度学习从宏观角度理解就是让机器自己在数据中找到一个映射关系（在数学上我们叫作函数），打个比方就像是我们眼镜看到了一只动物的画像。大脑就相当于一个黑盒子或函数，给我们返回这个动物叫猫。这就是一种映射关系。而深度学习就是利用数据教机器找到一种映射关系后，给机器一张图片，它便能够识别其中的类别。（这里仅仅是使用分类打的比方，它可以用在各种场景）

## 二、背景

> Machine-learning technology powers many aspects of modern society: from web searches to content filtering on social net- works to recommendations on e-commerce websites, and it is increasingly present in consumer products such as cameras and smart phones. Machine-learning systems are used to identify objects in images, transcribe speech into text, match news items, posts or products with users’ interests, and select relevant results of search. Increasingly, these applications make use of a class of techniques called deep learning.

机器学习技术为现代社会的许多方面提供动力：从网络搜索到社交网络上的内容过滤，再到电子商务网站上的推荐，它越来越多地出现在相机和智能手机等消费产品中。 机器学习系统用于识别图像中的对象，将语音转录成文本，将新闻项目、帖子或产品与用户的兴趣相匹配，并选择相关的搜索结果。 这些应用程序越来越多地使用一类称为深度学习的技术。

> Conventional machine-learning techniques were limited in their ability to process natural data in their raw form. For decades, constructing a pattern-recognition or machine-learning system required careful engineering and considerable domain expertise to design a feature extractor that transformed the raw data (such as the pixel values of an image) into a suitable internal representation or feature vector from which the learning subsystem, often a classifier, could detect or classify patterns in the input.

传统的机器学习技术以原始形式处理自然数据的能力有限。 几十年来，构建模式识别或机器学习系统需要仔细的工程和大量的领域专业知识来设计特征提取器，将原始数据（例如图像的像素值）转换为合适的内部表示或特征向量，从中学习子系统，通常是一个分类器，可以检测或分类输入中的模式。

> Representation learning is a set of methods that allows a machine to be fed with raw data and to automatically discover the representations needed for detection or classification.

特征学习是一种允许机器输入原始数据并自动发现检测或分类所需的特征的方法。

> **Deep-learning methods are representation-learning methods with multiple levels of representation, obtained by composing simple but non-linear modules that each transform the representation at one level (starting with the raw input) into a representation at a higher, slightly more abstract level. With the composition of enough such transformations, very complex functions can be learned.** 

**深度学习方法是具有多级特征的特征学习方法，它是通过组合简单但非线性的模块获得的，每个模块将一个级别的特征（从原始输入开始）转换为更高、稍多一点抽象层次的特征。 通过组合足够多的转换器可以学习非常复杂的函数。**

**感想：**特征层数越高，感受野越大，得到的图像信息越抽象，可能已经不能被人能理解，但是这些信息之间存在这某些能被神经元组成的映射关系所理解。

> For classification tasks, higher layers of representation amplify aspects of the input that are important for discrimination and suppress irrelevant variations. An image, for example, comes in the form of an array of pixel values, and the learned features in the first layer of representation typically represent the presence or absence of edges at particular orientations and locations in the image.

对于分类任务，更高层的特征会放大输入中不同点并抑制不相关的变化是很重要的方面。 例如，图像以像素值数组的形式出现，第一层特征中的学习特征通常表示图像中特定方向和位置处边缘的存在或不存在。

> The second layer typically detects motifs by spotting particular arrangements of edges, regardless of small variations in the edge positions.

第二层通常通过发现边缘的特定排列来检测图案，而不管边缘位置的微小变化。

> The third layer may assemble motifs into larger combinations that correspond to parts of familiar objects, and subsequent layers would detect objects as combinations of these parts. The key aspect of deep learning is that these layers of features are not designed by human engineers: they are learned from data using a general-purpose learning procedure.

第三层可以将图案组合成更大的组合，对应于熟悉的对象的部分，随后的层将检测对象作为这些部分的组合。 深度学习的关键方面是这些特征层不是由人类工程师设计的：它们是使用通用学习程序从数据中学习的。

**感想：**就像开始说讲，神经元组成的一个具有映射关系的黑匣子，想起前两天想拍视频通俗化讲解深度学习的例子。它就好比是一个带着很多旋钮开关的小电视（有点像老式那种天线电视）我们暂时叫他小深，电视从天线接收信号经过处理然后呈现画面。小深和普通的老电视不太一样，他的旋钮很多很多，大部分在内部（这就是权重文件的参数），一小部分在外部（这就是超参数）。我们使用时先喂小深吃一堆带标签数据，还是用猫狗分类的例子打比方，从天线口喂了一堆猫狗的数据。小深就一块一块吃，边吃变看这个是啥（数据标签教小深），吃完一部分，就用特定一部分考小深，这是猫还是狗呀，回答错了，就敲一棍（损失函数），小深就根据敲的力度来扭他内部的旋钮。这个过程就这样重复着。逐渐着他就样一点一点提升自己喂进来图片的差异。这时有人问外部旋钮干嘛用？外部旋钮是我们根据小深的学习情况，来帮助她让他变得更强。 小深学了好久，也吃了几千张猫狗的图像了。这时我们喂他一张图片时，他就能帮我们分辨出这是猫还是狗。（当然这个例子是通俗化讲解，并没有涉及底层知识。也存在不合理地方，仅供参考）

> Deep learning is making major advances in solving problems that have resisted the best attempts of the artificial intelligence community for many years. **It has turned out to be very good at discovering intricate structures in high-dimensional data** and is therefore applicable to many domains of science, business and government. 

深度学习在解决多年来一直抵制人工智能社区最佳尝试的问题方面取得了重大进展。 事实证明，**它非常擅长发现高维数据中的复杂结构**，因此适用于科学、商业和政府的许多领域。

> In addition to beating records in image recognition and speech recognition , it has beaten other machine-learning techniques at predicting the activity of potential drug molecules ,analysing particle accelerator data, reconstructing brain circuits, and predicting the effects of mutations in non-coding DNA on gene expression and disease

除了在图像识别和语音识别方面打破记录之外，它还在预测潜在药物分子的活动、分析粒子加速器数据、重建大脑回路以及预测非编码 DNA 突变对基因表达和疾病的影响方面击败了其他机器学习技术。 

> Perhaps more surprisingly, deep learning has produced extremely promising results for various tasks in natural language understanding, particularly topic classification, sentiment analysis, question answering and language translation

也许更令人惊讶的是，深度学习在自然语言理解的各种任务中产生了非常有希望的结果，特别是主题分类、情感分析、问答和语言翻译

> We think that deep learning will have many more successes in the near future because it requires very little engineering by hand, so it can easily take advantage of increases in the amount of available computation and data. New learning algorithms and architectures that are currently being developed for deep neural networks will only accelerate this progress.

我们认为深度学习在不久的将来会取得更多成功，因为它只需要很少的手工工程，因此可以轻松利用可用计算量和数据量的增加。 目前正在为深度神经网络开发的新学习算法和架构只会加速这一进程。

## 三、监督学习

> The most common form of machine learning, deep or not, is supervised learning. Imagine that we want to build a system that can classify images as containing, say, a house, a car, a person or a pet. We first collect a large data set of images of houses, cars, people and pets, each labelled with its category. During training, the machine is shown an image and produces an output in the form of a vector of scores, one for each category. We want the desired category to have the highest score of all categories, but this is unlikely to happen before training.

最常见的机器学习形式，无论是否深入，都是监督学习。 想象一下，我们想要构建一个系统，可以将图像分类为包含房屋、汽车、人或宠物。 我们首先收集大量房屋、汽车、人和宠物的图像数据集，每个图像都标有其类别。 在训练期间，机器会看到一张图像，并以分数向量的形式产生输出，每个类别一个。 我们希望期望的类别在所有类别中得分最高，但这在训练之前不太可能发生。

> **We compute an objective function that measures the error (or distance) between the output scores and the desired pattern of scores. The machine then modifies its internal adjustable parameters to reduce this error. These adjustable parameters, often called weights, are real numbers that can be seen as ‘knobs’ that define the input–output function of the machine. In a typical deep-learning system, there may be hundreds of millions of these adjustable weights, and hundreds of millions of labelled examples with which to train the machine.**

**我们计算一个目标函数，用于测量输出分数与所需分数模式之间的误差（或距离）。 然后机器修改其内部可调参数以减少该误差。 这些可调参数，通常称为权重，是实数，可以看作是定义机器输入输出函数的“旋钮”。 在典型的深度学习系统中，可能有数亿个这样的可调整权重，以及数亿个标记示例用于训练机器。**

> To properly adjust the weight vector, the learning algorithm computes a gradient vector that, for each weight, indicates by what amount the error would increase or decrease if the weight were increased by a tiny amount. The weight vector is then adjusted in the opposite direction to the gradient vector.

为了正确调整权重向量，学习算法计算一个梯度向量，对于每个权重，该向量指示如果权重增加很小的量，误差会增加或减少多少。 然后在与梯度向量相反的方向上调整权重向量。

> **The objective function, averaged over all the training examples, can be seen as a kind of hilly landscape in the high-dimensional space of weight values. The negative gradient vector indicates the direction of steepest descent in this landscape, taking it closer to a minimum, where the output error is low on average.**

**对所有训练示例求平均值的目标函数可以看作是权重值的高维空间中的一种丘陵景观。 负梯度向量表示该景观中最速下降的方向，使其更接近最小值，其中输出误差平均较低。**

**感想：**借用梯度下降算法示意图，训练目的就是找到当前模型的权重参数为何值，使损失函数最小。

![image-20211030172019092](\img\image-20211030172019092.png)

> In practice, most practitioners use a procedure called stochastic gradient descent (SGD). This consists of showing the input vector for a few examples, computing the outputs and the errors, computing the average gradient for those examples, and adjusting the weights accordingly. The process is repeated for many small sets of examples from the training set until the average of the objective function stops decreasing. It is called stochastic because each small set of examples gives a noisy estimate of the average gradient over all examples. This simple procedure usually finds a good set of weights surprisingly quickly when compared with far more elaborate optimization techniques.

在实践中，大多数从业者使用称为随机梯度下降 (SGD) 的方法。 主要包括显示一些样本的输入向量，计算输出和误差，计算这些示例的平均梯度，并相应地调整权重。 对训练集中的许多小样本集重复该过程，直到目标函数的平均值停止下降。 之所以称为随机，是因为每一小组样本都替代了所有样本的平均梯度的噪声估计。 与更复杂的优化技术相比，这个简单的过程通常会以惊人的速度找到一组好的权重。

> **After training, the performance of the system is measured on a different set of examples called a test set. This serves to test the generalization ability of the machine .its ability to produce sensible answers on new inputs that it has never seen during training.**

训练后，系统的性能在称为测试集的不同示例集上进行测量。 这用于测试机器的泛化能力。这种能力是在训练期间学习的并能够从未见过的新输入上产生合理答案的能力。

> Many of the current practical applications of machine learning use linear classifiers on top of hand-engineered features. A two-class linear classifier computes a weighted sum of the feature vector components. If the weighted sum is above a threshold, the input is classified as belonging to a particular category.

机器学习的许多当前实际应用使用手工设计特征的线性分类器。 二分类线性分类器计算特征向量分量的加权和。 如果加权和高于阈值，则输入被归类为属于特定类别。

> Since the 1960s we have known that linear classifiers can only carve their input space into very simple regions, namely half-spaces separated by a hyperplane. But problems such as image and speech recognition require the input–output function to be insensitive to irrelevant variations of the input, such as variations in position, orientation or illumination of an object, or variations in the pitch or accent of speech, while being very sensitive to particular minute variations (for example, the difference between a white wolf and a breed of wolf-like white dog called a Samoyed). 

自 1960 年代以来，我们就知道线性分类器只能将其输入空间划分为非常简单的区域，即由超平面分隔的半空间。 **但是图像和语音识别等问题要求输入-输出函数对输入的无关变化不敏感，例如物体的位置、方向或照明的变化，或者语音音调或口音的变化，同时非常对特定的微小变化敏感**（例如，白狼和一种叫做萨摩耶的类似狼的白狗之间的区别）。

> At the pixel level, images of two Samoyeds in different poses and in different environments may be very different from each other, whereas two images of a Samoyed and a wolf in the same position and on similar backgrounds may be very similar to each other. A linear classifier, or any other ‘shallow’ classifier operating on Figure 1 | Multilayer neural networks and backpropagation.

在像素层面，两个萨摩耶犬在不同姿势和不同环境中的图像可能彼此非常不同，而在相同位置和相似背景下的两个萨摩耶犬和狼的图像可能彼此非常相似。 线性分类器，或任何其他“浅”分类器操作（如图 1 ）多层神经网络和反向传播。

![tmp61D6](\img\tmp61D6.png)

> a, A multi- layer neural network (shown by the connected dots) can distort the input space to make the classes of data (examples of which are on the red and blue lines) linearly separable. Note how a regular grid (shown on the left) in input space is also transformed (shown in the middle panel) by hidden units. This is an illustrative example with only two input units, two hidden units and one output unit, but the networks used for object recognition or natural language processing contain tens or hundreds of thousands of units. Reproduced with permission from C. Olah (http://colah.github.io/).

图a，多层神经网络（由连接的点表示）可以扭曲输入空间，使数据类别（红色和蓝色线上的示例）线性可分。 请注意输入空间中的常规网格（显示在左侧）也如何被隐藏单元转换（显示在中间面板中）。 这是一个只有两个输入单元、两个隐藏单元和一个输出单元的说明性示例，但用于对象识别或自然语言处理的网络包含数万或数十万个单元。 经 C. Olah (http://colah.github.io/) 许可转载。

> b, The chain rule of derivatives tells us how two small effects (that of a small change of x on y, and that of y on z) are composed. A small change Δx in x gets transformed first into a small change Δy in y by getting multiplied by ∂y/∂x (that is, the definition of partial derivative). Similarly, the change Δy creates a change Δz in z. Substituting one equation into the other gives the chain rule of derivatives — how Δx gets turned into Δz through multiplication by the product of ∂y/∂x and ∂z/∂x. It also works when x, y and z are vectors (and the derivatives are Jacobian matrices).

图b，导数的链式法则告诉我们两个小的影响（x 对 y 的微小变化和 y 对 z 的微小变化）是如何组成的。 通过乘以∂y/∂x（即偏导数的定义），x 的微小变化Δx 首先转化为y 的微小变化Δy。 类似地，这个变化 Δy导致 在 z 中Δz变化 。 将一个方程代入另一个方程给出了导数的链式法则——Δx 如何通过乘以 ∂y/∂x 和 ∂z/∂x 的乘积变成 Δz。 它也适用于当 x、y 和 z 是向量（并且导数是雅可比矩阵）。

> c, The equations used for computing the forward pass in a neural net with two hidden which one can back-propagate gradients. At each layer, we first compute the total input z to each unit, which is a weighted sum of the outputs of the units in the layer below. Then a non-linear function f(.) is applied to z to get the output of the unit. For simplicity, we have omitted bias terms. The non-linear functions used in neural networks include the rectified linear unit (ReLU) f(z) = max(0,z), commonly used in recent years, as well as the more conventional sigmoids, such as the hyberbolic tangent, f(z) = (exp(z)− exp(−z))/(exp(z) + exp(−z)) and logistic function logistic, f(z) = 1/(1 + exp(−z)). 

图c，用于计算具有两个隐藏层的神经网络中的前向传递的方程，其中一个可以反向传播梯度。 在每一层，我们首先计算每个单元的总输入 z，它是下层单元输出的加权和。 然后将非线性函数 f(.) 应用于 z 以获得单元的输出。 为简单起见，我们省略了偏差项。 神经网络中使用的非线性函数包括整流线性单元 (ReLU) f(z) = max(0,z)，近年来常用，以及更传统的 sigmoid，如双曲正切，f (z) = (exp(z)− exp(−z))/(exp(z) + exp(−z)) 和logistic函数，f(z) = 1/(1 + exp(−z)) .

> d, The equations used for computing the backward pass. At each hidden layer we compute the error derivative with respect to the output of each unit, which is a weighted sum of the error derivatives with respect to the total inputs to the units in the layer above. We then convert the error derivative with respect to the output into the error derivative with respect to the input by multiplying it by the gradient of f(z). At the output layer, the error derivative with respect to the output of a unit is computed by differentiating the cost function. This gives yl− tl if the cost function for unit l is 0.5(yl − tl)2 , where tlis the target value. Once the ∂E/∂zk is known, the error-derivative for the weight wjk on the connection from unit j in the layer below is just yj ∂E/∂zk .

d，用于计算反向传播的方程。 在每个隐藏层，我们计算每个单元输出的误差导数，它是误差导数相对于上层单元所有输入的加权和。 然后，我们通过乘以 f(z) 的梯度，将关于输出的误差导数转换为关于输入的误差导数。 在输出层，通过对成本函数进行微分来计算单元输出的误差导数。 如果单位 l 的成本函数为 0.5(yl − tl)平方，则这给出 yl− tl ，其中 tli 是目标值。 一旦 ∂E/∂zk 已知，来自下层单元 j 的连接上的权重 wjk 的误差导数就是 yj乘∂E/∂zk 。

![tmp1AEB](\img\tmp1AEB.png)

> Figure 2 | Inside a convolutional network. The outputs (not the filters) of each layer (horizontally) of a typical convolutional network architecture applied to the image of a Samoyed dog (bottom left; and RGB (red, green, blue) inputs, bottom right). Each rectangular image is a feature map corresponding to the output for one of the learned features, detected at each of the image positions. Information flows bottom up, with lower-level features acting as oriented edge detectors, and a score is computed for each image class in output. ReLU, rectified linear unit.

图2 | 在卷积网络内部。 应用于萨摩耶犬图像的典型卷积网络架构的每一层（水平）的输出（不是滤波器）（左下；RGB（红色、绿色、蓝色）输入，右下）。 每个矩形图像都是一个特征图，对应于在每个图像位置检测到的学习特征之一的输出。 信息自下而上流动，低级特征充当定向边缘检测器，并为输出中的每个图像类计算分数。 ReLU，整流线性单元。

> raw pixels could not possibly distinguish the latter two, while putting the former two in the same category. This is why shallow classifiers require a good feature extractor that solves the selectivity–invariance dilemma — one that produces representations that are selective to the aspects of the image that are important for discrimination, but that are invariant to irrelevant aspects such as the pose of the animal.

原始像素无法区分后两者，而将前两者归入同一类别。 这就是为什么浅层分类器需要一个好的特征提取器来解决选择性-不变性困境——一个能够产生对图像中对区分很重要的方面有选择性的表示，但对不相关的方面（例如动物的姿势）是不变的。

> To make classifiers more powerful, one can use generic non-linear features, as with kernel methods. but generic features such as those arising with the Gaussian kernel do not allow the learner to generalize well far from the training examples. The conventional option is to hand design good feature extractors, which requires a consider- able amount of engineering skill and domain expertise. But this can all be avoided if good features can be learned automatically using a general-purpose learning procedure. This is the key advantage of deep learning.

**为了使分类器更强大，可以使用通用非线性特征**，就像核方法一样。 但是一般特征（例如由高斯核产生的特征）不允许学习者在远离训练示例的情况下很好地泛化。 **传统的选择是手工设计好的特征提取器，这需要大量的工程技能和领域专业知识**。 但是，**如果可以使用通用学习程序自动学习好的特征，这一切都可以避免。** 这是深度学习的主要优势。

> A deep-learning architecture is a multilayer stack of simple modules, all (or most) of which are subject to learning, and many of which compute non-linear input–output mappings. Each module in the stack transforms its input to increase both the selectivity and the invariance of the representation. With multiple non-linear layers, say a depth of 5 to 20, **a system can implement extremely intricate functions of its inputs that are simultaneously sensitive to minute details** — distinguishing Samoyeds from white wolves — and **insensitive to large irrelevant variations** such as the background, pose, lighting and surrounding objects.

深度学习架构是简单模块的多层堆栈，所有（或大部分）模块都需要学习，其中许多计算非线性输入-输出映射。 堆栈中的每个模块都转换其输入以增加表示的选择性和不变性。 **通过多个非线性层，比如 5 到 20 的深度，系统可以实现其输入的极其复杂的功能，这些功能同时对微小细节敏感——区分萨摩耶犬和白狼——并且对大的不相关变化不敏感，例如背景， 姿势、灯光和周围物体。**
