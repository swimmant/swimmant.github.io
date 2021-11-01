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

## 四、反向传播算法

> From the earliest days of pattern recognition, the aim of researchers has been to replace hand-engineered features with trainable multilayer networks, but despite its simplicity, the solution was not widely understood until the mid 1980s. As it turns out, multilayer architectures can be trained by simple stochastic gradient descent. As long as the modules are relatively smooth functions of their inputs and of their internal weights, one can compute gradients using the backpropagation procedure. The idea that this could be done, and that it worked, was discovered independently by several different groups during the 1970s and 1980s。

从模式识别的早期开始，研究人员的目标就是用可训练的多层网络代替手工设计的特征，但尽管它很简单，但直到 1980 年代中期，该解决方案才被广泛理解。 事实证明，**多层架构可以通过简单的随机梯度下降进行训练**。 只要模块是其输入及其内部权重的相对平滑的函数，就可以使用反向传播程序计算梯度。 在 1970 年代和 1980 年代，几个不同的团队独立地发现了这种方法可以做到并且有效的想法。

> The backpropagation procedure to compute the gradient of an objective function with respect to the weights of a multilayer stack of modules is nothing more than a practical application of the chain rule for derivatives.

计算目标函数相对于多层模块权重的梯度的反向传播过程只不过是导数链式法则的实际应用。

> **The key insight is that the derivative (or gradient) of the objective with respect to the input of a module can be computed by working backwards from the gradient with respect to the output of that module (or the input of the subsequent module) (Fig. 1). The backpropagation equation can be applied repeatedly to propagate gradients through all modules, starting from the output at the top (where the network produces its prediction) all the way to the bottom (where the external input is fed). Once these gradients have been computed, it is straightforward to compute the gradients with respect to the weights of each module.**

**关键的见解是，目标相对于模块输入的导数（或梯度）可以通过从相对于该模块的输出（或后续模块的输入）的梯度向后计算（图 . 1). 反向传播方程可以重复应用以通过所有模块传播梯度，从顶部的输出（网络产生预测的地方）一直到底部（外部输入的地方）。 一旦计算了这些梯度，就可以直接计算每个模块权重的梯度。**

> Many applications of deep learning use feed forward neural network architectures (Fig. 1), which learn to map a fixed-size input (for example, an image) to a fixed-size output (for example, a probability for each of several categories). To go from one layer to the next, a set of units compute a weighted sum of their inputs from the previous layer and pass the result through a non-linear function.

深度学习的许多应用使用前馈神经网络架构（图 1），它学习将固定大小的输入（例如，图像）映射到固定大小的输出（例如，每个 几个类别）。 为了从一层到下一层，一组单元计算它入来自前一层的输入的加权总和，并将结果传递给非线性函数。

> At present, the most popular non-linear function is the rectified linear unit (ReLU), which is simply the half-wave rectifier f(z) = max(z, 0). In past decades, neural nets used smoother nonlinearities, such as tan h(z) or 1/(1 + exp(−z)), but the ReLU typically learns much faster in networks with many layers, allowing training of a deep supervised network without unsupervised pre-training. Units that are not in the input or output layer are conventionally called hidden units. The hidden layers can be seen as distorting the input in a non-linear way so that categories become linearly separable by the last layer (Fig. 1)

目前最流行的非线性函数是整流线性单元（ReLU），简单来说就是半波整流器 f(z) = max(z, 0)。 在过去的几十年中，神经网络使用更平滑的非线性，例如 tan h(z) 或 1/(1 + exp(−z))，但 ReLU 通常在具有多层的网络中学习得更快，允许训练深度监督网络而无需 无监督预训练。 不在输入或输出层中的单元通常称为隐藏单元。 隐藏层可以被视为以非线性方式扭曲输入，以便类别变得线性可分到最后一层（图 1）

> In the late 1990s, neural nets and backpropagation were largely forsaken by the machine-learning community and ignored by the computer-vision and speech-recognition communities. It was widely thought that learning useful, multistage, feature extractors with little prior knowledge was infeasible. In particular, it was commonly thought that simple gradient descent would get trapped in poor local minima — weight configurations for which no small change would reduce the average error

在 1990 年代后期，神经网络和反向传播在很大程度上被机器学习社区所抛弃，而被计算机视觉和语音识别圈子所忽视。 人们普遍认为，在先验知识很少的情况下学习有用的、多阶段的特征提取器是不可行的。 特别是，人们普遍认为简单的梯度下降会陷入糟糕的局部最小值——权重配置，不小的变化会减少平均误差

> **In practice, poor local minima are rarely a problem with large networks. Regardless of the initial conditions, the system nearly always reaches solutions of very similar quality.**

**在实践中，较差的局部最小值很少会成为大型网络的问题。 无论初始条件如何，系统几乎总能达到效果非常相似的解。**

> Recent theoretical and empirical results strongly suggest that local minima are not a serious issue in general. Instead, the landscape is packed with a combinatorially large number of saddle points where the gradient is zero, and the surface curves up in most dimensions and curves down in the remainder。

最近的理论和实证结果强烈表明，局部最小值通常不是一个严重的问题。 取而代之的是，三维权重全景图中包含大量梯度为零的鞍点，并且表面在大多数维度上向上弯曲，在其余部分向下弯曲。

**注**：借用其他笔记中图，方便大家理解：其中红点停下的地方就是鞍点，就像骑马那个马鞍。单方向梯度下降，很有可能到红点处，模型就停止了。

<img src="\img\tmp2302.png" alt="tmpCF79" style="zoom:50%;" />

> **The analysis seems to show that saddle points with only a few downward curving directions are present in very large numbers, but almost all of them have very similar values of the objective function. Hence, it does not much matter which of these saddle points the algorithm gets stuck at.**

**分析似乎表明，只有少数向下弯曲方向的鞍点数量非常多，但几乎所有鞍点的目标函数值都非常相似。 因此，算法卡在这些鞍点中的哪个点并不重要。**

> Interest in deep feed forward networks was revived around 2006 by a group of researchers brought together by the Canadian Institute for Advanced Research (CIFAR). The researchers introduced unsupervised learning procedures that could create layers of feature detectors without requiring labelled data. The objective in learning each layer of feature detectors was to be able to reconstruct or model the activities of feature detectors (or raw inputs) in the layer below.

2006 年左右，由加拿大高级研究所 (CIFAR) 召集的一组研究人员重新引起了对深度前馈网络的兴趣。 研究人员引入了无监督学习程序，可以在不需要标记数据的情况下创建特征检测器层。 学习每一层特征检测器的目标是能够重建或建模下层特征检测器（或原始输入）的活动。

> By ‘pre-training’ several layers of progressively more complex feature detectors using this reconstruction objective, the weights of a deep network could be initialized to sensible values. A final layer of output units could then be added to the top of the network and the whole deep system could be fine-tuned using standard backpropagation. This worked remarkably well for recognizing handwritten digits or for detecting pedestrians, especially when the amount of labelled data was very limited.

通过使用这个重建目标“预训练”多层逐渐复杂的特征检测器，深度网络的权重可以被初始化为合理的值。 然后可以将最后一层输出单元添加到网络的顶部，并且可以使用标准反向传播对整个深层系统进行微调。 这对于识别手写数字或检测行人非常有效，尤其是在标记数据量非常有限的情况下。

The first major application of this pre-training approach was in speech recognition, and it was made possible by the advent of fast graphics processing units (GPUs) that were convenient to program and allowed researchers to train networks 10 or 20 times faster. In 2009, the approach was used to map short temporal windows of coefficients extracted from a sound wave to a set of probabilities for the various fragments of speech that might be represented by the frame in the centre of the window

这种预训练方法的第一个主要应用是语音识别，它的出现得益于快速图形处理单元 (GPU) 的出现，这些单元便于编程，并使研究人员能够以 10 或 20 倍的速度训练网络。 2009 年，该方法用于将从声波中提取的系数的短时间窗口映射到可能由窗口中心的帧表示的各种语音片段的一组概率

> **It achieved record-breaking results on a standard speech recognition benchmark that used a small vocabulary and was quickly developed to give record-breaking results on a large vocabulary task**.

**它在使用小词汇量的标准语音识别基准测试中取得了破纪录的结果，并迅速开发以在大词汇量任务上获得破纪录的结果**。

By 2012, versions of the deep net from 2009 were being developed by many of the major speech groups and were already being deployed in Android phones. For smaller data sets, unsupervised pre-training helps to prevent overfitting, leading to significantly better generalization when the number of labelled exam- ples is small, or in a transfer setting where we have lots of examples for some ‘source’ tasks but very few for some ‘target’ tasks. Once deep learning had been rehabilitated, it turned out that the pre-training stage was only needed for small data sets.

到 2012 年，许多主要语音团队都在开发 2009 年以来的深度网络版本，并且已经在 Android 手机中部署。 对于较小的数据集，无监督的预训练有助于防止过度拟合，当标记样本的数量很少时，或者在我们有很多用于某些“源”任务的示例但很少的传输设置中，可以显着改善泛化 对于一些“目标”任务。 一旦深度学习得到修复，结果证明只有小数据集才需要预训练阶段。

> There was, however, one particular type of deep, feedforward network that was much easier to train and generalized much better than networks with full connectivity between adjacent layers. This was the convolutional neural network (ConvNet). It achieved many practical successes during the period when neural networks were out of favour and it has recently been widely adopted by the computer- vision community.

然而，有一种特殊类型的深度前馈网络比相邻层之间具有完全连接性的网络更容易训练和推广。 这就是卷积神经网络 (ConvNet)。 在神经网络不受欢迎的时期，它取得了许多实际的成功，最近它被计算机视觉社区广泛采用。

## 五、卷积神经网络

> ConvNets are designed to process data that come in the form of multiple arrays, for example a colour image composed of three 2D arrays containing pixel intensities in the three colour channels. Many data modalities are in the form of multiple arrays: 1D for signals and sequences, including language; 2D for images or audio spectrograms; and 3D for video or volumetric images. There are four key ideas behind ConvNets that take advantage of the properties of natural signals: local connections, shared weights, pooling and the use of many layers.

卷积神经网络旨在处理以多个阵列形式出现的数据，例如由三个二维阵列组成的彩色图像，其中包含三个颜色通道中的像素强度。 许多数据模态采用多个数组的形式：信号和序列的一维，包括语言； 2D 图像或音频频谱图和 3D 视频或立体图像。 卷积神经网络背后有四个利用自然信号特性的关键思想：本地连接、共享权重、池化和多层的使用。

> The architecture of a typical ConvNet (Fig. 2) is structured as a series of stages. The first few stages are composed of two types of layers: convolutional layers and pooling layers. Units in a convolutional layer are organized in feature maps, within which each unit is connected to local patches in the feature maps of the previous layer through a set of weights called a filter bank. 

典型的 卷积神经网络（图 2）的架构由一系列阶段构成。 前几个阶段由两种类型的层组成：卷积层和池化层。 卷积层中的单元被组织在特征图中，其中每个单元通过一组称为滤波器组的权重连接到前一层特征图中的局部补丁。

> The result of this local weighted sum is then passed through a non-linearity such as a ReLU. All units in a feature map share the same filter bank. Different feature maps in a layer use different filter banks. 

然后将此局部加权和的结果传递给非线性，例如 ReLU。 特征图中的所有单元共享相同的过滤器组。 层中的不同特征图使用不同的滤波器组。

> The reason for this architecture is twofold. First, in array data such as images, local groups of values are often highly correlated, forming distinctive local motifs that are easily detected. Second, the local statistics of images and other signals are invariant to location.

这种架构的原因是双重的。 首先，在图像等数组数据中，局部值组通常高度相关，形成易于检测的独特局部图案。 其次，图像和其他信号的局部统计对于位置是不变的。

>  In other words, if a motif can appear in one part of the image, it could appear anywhere, hence the idea of units at different locations sharing the same weights and detecting the same pattern in different parts of the array. Mathematically, the filtering operation performed by a feature map is a discrete convolution, hence the name.

换句话说，如果一个主题可以出现在图像的一个部分，它就可以出现在任何地方，因此不同位置的单元共享相同的权重，并在阵列的不同部分检测相同的模式。 从数学上讲，特征图执行的过滤操作是离散卷积，因此得名。

> Although the role of the convolutional layer is to detect local conjunctions of features from the previous layer, the role of the pooling layer is to merge semantically similar features into one. Because the relative positions of the features forming a motif can vary somewhat, reliably detecting the motif can be done by coarse-graining the position of each feature. 

虽然卷积层的作用是检测前一层特征的局部连接，但池化层的作用是将语义相似的特征合并为一个。 因为形成一个主题的特征的相对位置可能会有所不同，所以可以通过粗粒度化每个特征的位置来可靠地检测主题。

> A typical pooling unit computes the maximum of a local patch of units in one feature map (or in a few feature maps). Neighbouring pooling units take input from patches that are shifted by more than one row or column, thereby reducing the dimension of the representation and creating an invariance to small shifts and distortions. 

典型的池化单元计算一个特征映射（或几个特征映射）中单元局部补丁的最大值。 相邻的池化单元从移动了不止一行或一列的补丁中获取输入，从而减少表示的维度并创建对小移动和扭曲的不变性。

> Two or three stages of convolution, non-linearity and pool- ing are stacked, followed by more convolutional and fully-connected layers. Backpropagating gradients through a ConvNet is as simple as through a regular deep network, allowing all the weights in all the filter banks to be trained.

两个或三个阶段的卷积、非线性和池化被堆叠起来，然后是更多的卷积和全连接层。 通过 ConvNet 反向传播梯度就像通过常规深度网络一样简单，允许训练所有滤波器组中的所有权重。

Deep neural networks exploit the property that many natural signals are compositional hierarchies, in which higher-level features are obtained by composing lower-level ones. In images, local combinations of edges form motifs, motifs assemble into parts, and parts form objects. Similar hierarchies exist in speech and text from sounds to phones, phonemes, syllables, words and sentences. The pooling allows representations to vary very little when elements in the previous layer vary in position and appearance.

深度神经网络利用许多自然信号是组合层次结构的特性，其中通过组合较低级别的特征来获得更高级别的特征。 在图像中，边缘的局部组合形成图案，图案组合成部分，部分形成物体。 从声音到音素、音素、音节、单词和句子，语音和文本中也存在类似的层次结构。 当前一层中的元素在位置和外观上发生变化时，池化允许表示变化很小。

> The convolutional and pooling layers in ConvNets are directly inspired by the classic notions of simple cells and complex cells in visual neuroscience, and the overall architecture is reminiscent of the LGN–V1–V2–V4–IT hierarchy in the visual cortex ventral pathway. When ConvNet models and monkeys are shown the same picture, the activations of high-level units in the ConvNet explains half of the variance of random sets of 160 neurons in the monkey’s inferotemporal cortex. ConvNets have their roots in the neocognitronthe architecture of which was somewhat similar, but did not have an end-to-end supervised-learning algorithm such as backpropagation. A primitive 1D ConvNet called a time-delay neural net was used for the recognition of phonemes and simple words.

卷积神经网络中的卷积层和池化层直接受到视觉神经科学中简单细胞和复杂细胞的经典概念的启发，整体架构让人联想到视觉皮层腹侧通路中的 LGN-V1-V2-V4-IT 层次结构。 当卷积神经网络模型和猴子看到相同的图片时，卷积神经网络中高级单元的激活解释了猴子下颞叶皮层中 160 个神经元的随机集方差的一半。 卷积神经网络起源于新认知器，其架构有些相似，但没有端到端的监督学习算法，例如反向传播。 一种称为延时神经网络的原始 1D 卷积神经网络用于识别音素和简单单词。

> There have been numerous applications of convolutional networks going back to the early 1990s, starting with time-delay neural networks for speech recognition and document reading. The document reading system used a ConvNet trained jointly with a probabilistic model that implemented language constraints. By the late 1990s this system was reading over 10% of all the cheques in the United States. A number of ConvNet-based optical character recognition and handwriting recognition systems were later deployed by Microsoft. ConvNets were also experimented with in the early 1990s for object detection in natural images, including faces and hands,and for face recognition.

回溯到 1990 年代初期，卷积网络有很多应用，首先是用于语音识别和文档阅读的延时神经网络。 文档阅读系统使用了一个 卷积网络和一个实现语言约束的概率模型联合训练。 到 1990 年代后期，该系统读取了美国所有支票的 10% 以上。 微软后来部署了许多基于卷积网络的光学字符识别和手写识别系统。 卷积网络在 1990 年代初期也进行了实验，用于自然图像中的对象检测，包括面部和手部，以及面部识别。

## 六、深度卷积网络的图像理解

> Since the early 2000s, ConvNets have been applied with great success to the detection, segmentation and recognition of objects and regions in images. These were all tasks in which labelled data was relatively abundant, such as traffic sign recognition , the segmentation of biological images particularly for connectomics, and the detection of faces,text, pedestrians and human bodies in natural images. A major recent practical success of ConvNets is face recognition.

自 2000 年代初以来，卷积网络已成功应用于图像中对象和区域的检测、分割和识别。 这些都是标记数据相对丰富的任务，例如交通标志识别，生物图像的分割，特别是连接组学，以及自然图像中人脸、文本、行人和人体的检测。 卷积网络最近的一个主要实际成功是人脸识别。

Importantly, images can be labelled at the pixel level, which will have applications in technology, including autonomous mobile robots and self-driving cars.

重要的是，图像可以在像素级别进行标记，这将在技术中得到应用，包括自主移动机器人和自动驾驶汽车。

<img src="\img\tmpCF79.png" alt="tmpCF79" style="zoom:75%;" />

<img src="img\tmp1638.png" alt="tmpCF79" style="zoom:60%;" />

> Figure 3 | From image to text. Captions generated by a recurrent neural network (RNN) taking, as extra input, the representation extracted by a deep convolution neural network (CNN) from a test image, with the RNN trained to ‘translate’ high-level representations of images into captions (top).

图 3 | 从图片到文字。 由循环神经网络 (RNN) 生成的字幕将深度卷积神经网络 (CNN) 从测试图像中提取的表征作为额外输入，并训练 RNN 将图像的高级表征“翻译”为字幕（ 最佳）。

> Reproduced with permission from ref.  When the RNN is given the ability to focus its attention on a different location in the input image (middle and bottom; the lighter patches were given more attention) as it generates each word (bold), we found that it exploits this to achieve better ‘translation’ of images into captions

经参考许可转载。 当 RNN 能够在生成每个单词（粗体）时将注意力集中在输入图像中的不同位置（中间和底部；较轻的补丁受到更多关注）时，我们发现它利用这一点来实现更好的效果 将图像“翻译”为标题

> Companies such as Mobileye and NVIDIA are using such ConvNet-based methods in their upcoming vision systems for cars. Other applications gaining importance involve natural language understanding and speech recognition

Mobileye 和 NVIDIA 等公司正在他们即将推出的汽车视觉系统中使用这种基于 卷积网络的方法。 其他越来越重要的应用包括自然语言理解和语音识别

> Despite these successes, ConvNets were largely forsaken by the mainstream computer-vision and machine-learning communities until the ImageNet competition in 2012.

尽管取得了这些成功，但在 2012 年的 ImageNet 竞赛之前，卷积网络在很大程度上被主流计算机视觉和机器学习社区所抛弃。

> When deep convolutional networks were applied to a data set of about a million images from the web that contained 1,000 different classes, they achieved spectacular results, almost halving the error rates of the best competing approaches. This success came from the efficient use of GPUs,ReLUs, a new regularization technique called dropout, and techniques to generate more training examples by deforming the existing ones. This success has brought about a revolution in computer vision; ConvNets are now the dominant approach for almost all recognition and detection tasks and approach human performance on some tasks. A recent stunning demonstration combines ConvNets and recurrent net modules for the generation of image captions (Fig. 3)

当深度卷积网络应用于包含 1,000 个不同类别的来自网络的大约一百万张图像的数据集时，它们取得了惊人的结果，几乎将最佳竞争方法的错误率减半。 这一成功来自于高效使用 GPU、ReLU、一种称为 **dropout 的新正则化**技术，以及通过对现有示例进行变形来生成更多训练示例的技术。 这一成功带来了计算机视觉的革命； 卷积网络现在是几乎所有识别和检测任务的主要方法，并在某些任务上接近人类的表现。 最近一个令人惊叹的演示结合了 卷积网络和循环网络模块来生成图像标题（图 3）

> Recent ConvNet architectures have 10 to 20 layers of ReLUs, hundreds of millions of weights, and billions of connections between units. Whereas training such large networks could have taken weeks only two years ago, progress in hardware, software and algorithm parallelization have reduced training times to a few hours.

最近的卷积网络架构有 10 到 20 层 ReLU，数亿个权重，以及单元之间的数十亿个连接。 两年前，训练如此大型的网络可能需要数周时间，但硬件、软件和算法并行化的进步已将训练时间缩短到几个小时。

> The performance of ConvNet-based vision systems has caused most major technology companies, including Google, Facebook, Microsoft, IBM, Yahoo!, Twitter and Adobe, as well as a quickly growing number of start-ups to initiate research and development projects and to deploy ConvNet-based image understanding products and services.

基于卷积网络 的视觉系统的性能已经促使包括谷歌、Facebook、微软、IBM、雅虎、Twitter 和 Adobe 在内的大多数主要技术公司，以及越来越多的初创企业启动研发项目并 部署基于卷积网络的图像理解产品和服务。

> ConvNets are easily amenable to efficient hardware implementations in chips or field-programmable gate arrays. A number of companies such as NVIDIA, Mobileye, Intel, Qualcomm and Samsung are developing ConvNet chips to enable real-time vision applications in smartphones, cameras, robots and self-driving cars.

ConvNets 很容易适应芯片或现场可编程门阵列中的高效硬件实现。 英伟达、Mobileye、英特尔、高通和三星等多家公司正在开发 ConvNet 芯片，以实现智能手机、相机、机器人和自动驾驶汽车中的实时视觉应用。

## 七、分布式表示和语言处理

> Deep-learning theory shows that deep nets have two different exponential advantages over classic learning algorithms that do not use distributed representations. Both of these advantages arise from the power of composition and depend on the underlying data-generating distribution having an appropriate componential structure. First,learning distributed representations enable generalization to new combinations of the values of learned features beyond those seen during training (for example,  2^{n} combinations are possible with n binary features). Second, composing layers of representation in a deep net brings the potential for another exponential advantage (exponential in the depth).

深度学习理论表明，与不使用分布式表示的经典学习算法相比，深度网络具有两个不同的指数优势。 这两个优势都源于组合的力量，并取决于具有适当组件结构的底层数据生成分布。 首先，学习分布式表示可以泛化到训练期间所见之外的学习特征值的新组合（例如，2 的n方 组合可能具有 n 个二元特征）。 其次，在深度网络中组合表示层带来了另一个指数优势（深度指数）的潜力。

> The hidden layers of a multilayer neural network learn to represent the network’s inputs in a way that makes it easy to predict the target outputs. This is nicely demonstrated by training a multilayer neural network to predict the next word in a sequence from a local context of earlier words . Each word in the context is presented to the network as a one-of-N vector, that is, one component has a value of 1 and the rest are 0. In the first layer, each word creates a different pattern of activations, or word vectors (Fig. 4). 

多层神经网络的隐藏层学习以一种易于预测目标输出的方式来表示网络的输入。 通过训练多层神经网络从较早单词的局部上下文中预测序列中的下一个单词，可以很好地证明这一点。 上下文中的每个单词都作为 N 中的一个向量呈现给网络，即一个组件的值为 1，其余为 0。在第一层，每个单词创建不同的激活模式，或 词向量（图4）。

> In a language model, the other layers of the network learn to convert the input word vectors into an output word vector for the predicted next word, which can be used to predict the probability for any word in the vocabulary to appear as the next word. The network learns word vectors that contain many active components each of which can be interpreted as a separate feature of the word, as was first demonstrated in the context of learning distributed representations for symbols.

在语言模型中，网络的其他层学习将输入词向量转换为预测下一个词的输出词向量，该词向量可用于预测词汇表中任何词出现作为下一个单词的概率。 网络学习包含许多活动成分的词向量，每个成分都可以解释为词的一个单独特征，正如在学习符号的分布式表示的上下文中首次展示的那样。

> These semantic features were not explicitly present in the input. They were discovered by the learning procedure as a good way of factorizing the structured relationships between the input and output symbols into multiple ‘micro-rules’. Learning word vectors turned out to also work very well when the word sequences come from a large corpus of real text and the individual micro-rules are unreliable.

这些语义特征未明确存在于输入中。 学习过程发现它们是将输入和输出符号之间的结构化关系分解为多个“微规则”的好方法。 当词序列来自大量真实文本并且单个微规则不可靠时，学习词向量也很有效。

> When trained to predict the next word in a news story, for example, the learned word vectors for Tuesday and Wednesday are very similar, as are the word vectors for Sweden and Norway. Such representations are called distributed representations because their elements (the features) are not mutually exclusive and their many configurations correspond to the variations seen in the observed data. These word vectors are composed of learned features that were not determined ahead of time by experts, but automatically discovered by the neural network.

例如，当训练预测新闻故事中的下一个词时，星期二和星期三的学习词向量非常相似，瑞典和挪威的词向量也是如此。 这种表示被称为分布式表示，因为它们的元素（特征）不是相互排斥的，并且它们的许多配置对应于在观察到的数据中看到的变化。 这些词向量由学习的特征组成，这些特征不是由专家提前确定，而是由神经网络自动发现的。

> Vector representations of words learned from text are now very widely used in natural language applications。

从文本中学习的单词的向量表示现在在自然语言应用中得到了非常广泛的应用

> The issue of representation lies at the heart of the debate between the logic-inspired and the neural-network-inspired paradigms for cognition. In the logic-inspired paradigm, an instance of a symbol is something for which the only property is that it is either identical or non-identical to other symbol instances. It has no internal structure that is relevant to its use; and to reason with symbols, they must be bound to the variables in judiciously chosen rules of inference. By contrast, neural networks just use big activity vectors, big weight matrices and scalar non-linearities to perform the type of fast ‘intuitive’ inference that underpins effortless commonsense reasoning.

表征问题是逻辑启发和神经网络启发的认知范式之间争论的核心。 在逻辑启发范式中，符号的实例是唯一属性是它与其他符号实例相同或不相同的东西。 它没有与其使用相关的内部结构； 为了用符号进行推理，它们必须与明智选择的推理规则中的变量绑定。 相比之下，神经网络仅使用大活动向量、大权重矩阵和标量非线性来执行支持轻松常识推理的快速“直观”推理类型。

> Before the introduction of neural language models, the standard approach to statistical modelling of language did not exploit distributed representations: it was based on counting frequencies of occurrences of short symbol sequences of length up to N (called N-grams). The number of possible N-grams is on the order of VN, where V is the vocabulary size, so taking into account a context of more than a handful of words would require very large training corpora. N-grams treat each word as an atomic unit, so they cannot generalize across semantically related sequences of words, whereas neural language models can because they associate each word with a vector of real valued features, and semantically related words end up close to each other in that vector space (Fig. 4).

在引入神经语言模型之前，语言统计建模的标准方法没有利用分布式表示：它基于对长度高达 N（称为 N-gram）的短符号序列的出现频率进行计数。 可能的 N-gram 的数量在 VN 的数量级上，其中 V 是词汇量的大小，因此考虑多个单词的上下文将需要非常大的训练语料库。 N-gram 将每个单词视为一个原子单元，因此它们不能泛化语义相关的单词序列，而神经语言模型可以，因为它们将每个单词与一个实值特征向量相关联，并且语义相关的单词最终彼此接近 在那个向量空间中（图 4）。

## 八、循环神经网络（RNN）

> When backpropagation was first introduced, its most exciting use was for training **recurrent neural networks** (RNNs). For tasks that involve sequential inputs, such as speech and language, it is often better to use RNNs (Fig. 5). RNNs process an input sequence one element at a time, maintaining in their hidden units a ‘state vector’ that implicitly contains information about the history of all the past elements of the sequence. When we consider the outputs of the hidden units at different discrete time steps as if they were the outputs of different neurons in a deep multilayer network (Fig. 5, right), it becomes clear how we can apply backpropagation to train RNNs.

首次引入反向传播时，其最令人兴奋的用途是训练循环神经网络 (RNN)。 对于涉及顺序输入的任务，例如语音和语言，通常最好使用 RNN（图 5）。 RNN 一次处理一个输入序列一个元素，在它们的隐藏单元中维护一个“状态向量”，该向量隐式包含有关该序列所有过去元素的历史信息。 当我们考虑隐藏单元在不同离散时间步长的输出时，就好像它们是深层多层网络中不同神经元的输出一样（图 5，右），我们如何应用反向传播来训练 RNN 就变得很清楚了。

> RNNs are very powerful dynamic systems, but training them has proved to be problematic because the backpropagated gradients either grow or shrink at each time step, so over many time steps they typically explode or vanish.

RNN 是非常强大的动态系统，但训练它们已被证明是有问题的，因为反向传播的梯度在每次迭代中要么增长要么缩小，所以在许多次迭代中它们通常会爆炸或消失。

> Thanks to advances in their architecture and ways of training them, RNNs have been found to be very good at predicting the next character in the text or the next word in a sequence , but they can also be used for more complex tasks. For example, after reading an English sentence one word at a time, an English ‘encoder’ network can be trained so that the final state vector of its hidden units is a good representation of the thought expressed by the sentence。

由于其结构和训练方法的进步，人们发现 RNN 非常擅长预测文本中的下一个字符或序列中的下一个单词，但它们也可用于更复杂的任务。 例如，在一次读一个英语句子后，可以训练英语“编码器”网络，使其隐藏单元的最终状态向量很好地表示句子所表达的思想。

> This thought vector can then be used as the initial hidden state of (or as extra input to) a jointly trained French ‘decoder’ network, which outputs a prob- ability distribution for the first word of the French translation. If a particular first word is chosen from this distribution and provided as input to the decoder network it will then output a probability distribution for the second word of the translation and so on until a full stop is chosen.

然后，该思想向量可以用作联合训练的法语“解码器”网络的初始隐藏状态（或作为额外输入），该网络输出法语翻译第一个单词的概率分布。 如果从这个分布中选择一个特定的第一个单词并作为输入提供给解码器网络，它将输出翻译的第二个单词的概率分布，依此类推，直到选择句号。

> Overall, this process generates sequences of French words according to a probability distribution that depends on the English sentence. This rather naive way of performing machine translation has quickly become competitive with the state-of-the-art, and this raises serious doubts about whether understanding a sentence requires anything like the internal symbolic expressions that are manipulated by using inference rules. It is more compatible with the view that everyday reasoning involves many simultaneous analogies that each contribute plausibility to a conclusion.

总体而言，此过程根据取决于英语句子的概率分布生成法语单词序列。 这种相当幼稚的机器翻译方式很快就与最先进的技术竞争，这引发了人们对理解句子是否需要诸如使用推理规则操纵的内部符号表达式之类的东西的严重怀疑。 它更符合以下观点，即日常推理涉及许多同时进行的类比，每个类比都有助于得出结论。

<img src="\img\tmpFBE9.png" alt="tmpCF79" style="zoom:75%;" />

> Figure 4 | Visualizing the learned word vectors. On the left is an illustration of word representations learned for modelling language, non-linearly projected to 2D for visualization using the t-SNE algorithm. On the right is a 2D representation of phrases learned by an English-to-French encoder–decoder recurrent neural network. One can observe that semantically similar words or sequences of words are mapped to nearby representations. The distributed representations of words are obtained by using backpropagation to jointly learn a representation for each word and a function that predicts a target quantity such as the next word in a sequence (for language modelling) or a whole sequence of translated words (for machine translation)

图 4 | 可视化学习到的词向量。 左侧是为建模语言学习的单词表示的图示，使用 t-SNE 算法非线性投影到 2D 以进行可视化。 右侧是由英法编码器-解码器循环神经网络学习的短语的二维表示。 可以观察到语义相似的单词或单词序列被映射到附近的表示。 单词的分布式表示是通过使用反向传播联合学习每个单词的表示和预测目标数量的函数获得的，例如序列中的下一个单词（用于语言建模）或整个翻译单词序列（用于机器翻译） )

<img src="\img\tmpE8DC.png" alt="tmpCF79" style="zoom:75%;" />

> Figure 5 | A recurrent neural network and the unfolding in time of the computation involved in its forward computation. 

图 5 | 循环神经网络及其前向计算所涉及的计算时间展开。

> The artificial neurons (for example, hidden units grouped under node s with values st at time t) get inputs from other neurons at previous time steps (this is represented with the black square, representing a delay of one time step, on the left). 

人工神经元（例如，分组在节点 s 下的隐藏单元在时间 t 的值为 st）在前一时间步从其他神经元获取输入（这用黑色方块表示，代表一个时间步的延迟，在左侧）。

> In this way, a recurrent neural network can map an input sequence with elements xt into an output sequence with elements ot , with each ot depending on all the previous xtʹ (for tʹ ≤ t). 

通过这种方式，循环神经网络可以将具有元素 xt 的输入序列映射到具有元素 ot 的输出序列，每个 ot 取决于所有先前的 xtʹ（对于 tʹ ≤ t）。

> The same parameters (matrices U,V,W ) are used at each time step. Many other architectures are possible, including a variant in which the network can generate a sequence of outputs (for example, words), each of which is used as inputs for the next time step. 

每个时间步都使用相同的参数（矩阵 U、V、W）。 许多其他架构也是可能的，包括一种变体，其中网络可以生成一系列输出（例如，单词），每个输出都用作下一个时间步的输入。

> The backpropagation algorithm (Fig. 1) can be directly applied to the computational graph of the unfolded network on the right, to compute the derivative of a total error (for example, the log-probability of generating the right sequence of outputs) with respect to all the states st and all the parameters.

反向传播算法（图 1）可以直接应用于右侧展开网络的计算图，计算总误差的导数（例如，生成正确输出序列的对数概率）关于到所有状态 st 和所有参数。

> Instead of translating the meaning of a French sentence into an English sentence, one can learn to ‘translate’ the meaning of an image into an English sentence (Fig. 3). The encoder here is a deep ConvNet that converts the pixels into an activity vector in its last hidden layer. The decoder is an RNN similar to the ones used for machine translation and neural language modelling. There has been a surge of interest in such systems recently (see examples mentioned in ref.)

与其将法语句子的含义翻译成英语句子，不如学习将图像的含义“翻译”成英语句子（图 3）。 这里的编码器是一个深度卷积网络，它将像素转换为最后一个隐藏层中的活动向量。 解码器是一个类似于用于机器翻译和神经语言建模的 RNN。 最近对此类系统的兴趣激增（参见参考文献中提到的示例）

> RNNs, once unfolded in time (Fig. 5), can be seen as very deep feedforward networks in which all the layers share the same weights. Although their main purpose is to learn long-term dependencies, theoretical and empirical evidence shows that it is difficult to learn to store information for very long.

RNN 一旦及时展开（图 5），就可以被视为非常深的前馈网络，其中所有层共享相同的权重。 尽管它们的主要目的是学习长期依赖，但理论和经验证据表明，很难学会长期存储信息。

> To correct for that, one idea is to augment the network with an explicit memory. The first proposal of this kind is the **long short-term memory** (LSTM) networks that use special hidden units, the natural behaviour of which is to remember inputs for a long time . A special unit called the memory cell acts like an accumulator or a gated leaky neuron: it has a connection to itself at the next time step that has a weight of one, so it copies its own real-valued state and accumulates the external signal, but this self-connection is multiplicatively gated by another unit that learns to decide when to clear the content of the memory.

为了纠正这一点，一个想法是用**显式记忆来增强网络**。 这种类型的第一个提议是**使用特殊隐藏单元的长短期记忆 (LSTM) 网络**，其自然行为是长时间记住输入。 一个称为记忆单元的特殊单元就像一个累加器或门控泄漏神经元：它在下一个时间步与自身建立连接，权重为 1，因此它复制自己的实值状态并累积外部信号， 但是这种自连接被另一个学习决定何时清除记忆内容的单元乘法门控。

> LSTM networks have subsequently proved to be more effective than conventional RNNs, especially when they have several layers for each time step, enabling an entire speech recognition system that goes all the way from acoustics to the sequence of characters in the transcription. LSTM networks or related forms of gated units are also currently used for the encoder and decoder networks that perform so well at machine translation.

LSTM 网络随后被证明比传统 RNN 更有效，尤其是当它们每个时间步长有多个层时，可以实现从声学到转录中的字符序列的整个语音识别系统。 LSTM 网络或相关形式的门控单元目前也用于在机器翻译中表现如此出色的编码器和解码器网络。

> Over the past year, several authors have made different proposals to augment RNNs with a memory module. Proposals include the Neural Turing Machine in which the network is augmented by a ‘tape-like’ memory that the RNN can choose to read from or write to, and memory networks, in which a regular network is augmented by a kind of associative memory . Memory networks have yielded excellent performance on standard question-answering benchmarks. The memory is used to remember the story about which the network is later asked to answer questions.

在过去的一年里，几位作者提出了不同的建议，用记忆模块来增强 RNN。 提议包括神经图灵机，其中网络通过 RNN 可以选择读取或写入的“磁带状”内存进行增强，以及内存网络，其中常规网络通过一种关联内存进行增强。 记忆网络在标准问答基准测试中表现出色。 记忆用于记住网络后来被要求回答问题的故事。

> Beyond simple memorization, neural Turing machines and memory networks are being used for tasks that would normally require reasoning and symbol manipulation. Neural Turing machines can be taught ‘algorithms’. Among other things, they can learn to output a sorted list of symbols when their input consists of an unsorted sequence in which each symbol is accompanied by a real value that indicates its priority in the list. Memory networks can be trained to keep track of the state of the world in a setting similar to a text adventure game and after reading a story, they can answer questions that require complex inference. In one test example, the network is shown a 15-sentence version of the The Lord of the Rings and correctly answers questions such as “where is Frodo now?”

除了简单的记忆之外，神经图灵机和记忆网络正被用于通常需要推理和符号操作的任务。 可以教授神经图灵机“算法”。 除此之外，当他们的输入由一个未排序的序列组成时，他们可以学习输出一个排序的符号列表，其中每个符号都伴随着一个实际值，表明其在列表中的优先级。 可以训练记忆网络在类似于文本冒险游戏的设置中跟踪世界状态，并且在阅读故事后，它们可以回答需要复杂推理的问题。 在一个测试示例中，网络展示了《指环王》的 15 句版本，并正确回答了诸如“佛罗多现在在哪里？”之类的问题。

## 九、深度学习的未来

> Unsupervised learning had a catalytic effect in reviving interest in deep learning, but has since been overshadowed by the successes of purely supervised learning.

无监督学习在重振对深度学习的兴趣方面具有催化作用，但此后被纯监督学习的成功所掩盖。

> Although we have not focused on it in this Review, we expect unsupervised learning to become far more important in the longer term. Human and animal learning is largely unsupervised: we discover the structure of the world by observing it, not by being told the name of every object.

虽然我们在本回顾中没有关注它，但我们预计无监督学习在长期内会变得更加重要。 人类和动物的学习在很大程度上是无监督的：我们通过观察来发现世界的结构，而不是通过被告知每个物体的名称。

> Human vision is an active process that sequentially samples the optic array in an intelligent, task-specific way using a small, high-resolution fovea with a large, low-resolution surround.

人类视觉是一个主动过程，它使用具有大型低分辨率环绕的小型高分辨率中央凹以智能、特定于任务的方式对光学阵列进行顺序采样。

> We expect much of the future progress in vision to come from systems that are trained end-to- end and combine ConvNets with RNNs that use reinforcement learning to decide where to look. Systems combining deep learning and reinforcement learning are in their infancy, but they already outperform passive vision systems at classification tasks and produce impressive results in learning to play many different video games

我们预计未来视觉方面的大部分进展将来自经过端到端训练并将 ConvNet 与 RNN 相结合的系统，这些系统使用强化学习来决定看哪里应用。 结合深度学习和强化学习的系统还处于起步阶段，但它们在分类任务上的表现已经超过被动视觉系统，并且在学习玩许多不同的视频游戏方面产生了令人印象深刻的结果

> Natural language understanding is another area in which deep learning is poised to make a large impact over the next few years. We expect systems that use RNNs to understand sentences or whole documents will become much better when they learn strategies for selectively attending to one part at a time

自然语言理解是深度学习有望在未来几年产生重大影响的另一个领域。 我们希望使用 RNN 来理解句子或整个文档的系统在学习一次有选择地关注一个部分的策略时会变得更好

> Ultimately, major progress in artificial intelligence will come about through systems that combine representation learning with complex reasoning. Although deep learning and simple reasoning have been used for speech and handwriting recognition for a long time, new paradigms are needed to replace rule-based manipulation of symbolic expressions by operations on large vectors.

最终，人工智能的重大进展将通过将表征学习与复杂推理相结合的系统实现。 尽管深度学习和简单推理已被用于语音和笔迹识别已有很长时间，但仍需要新的范式来通过对大向量的操作来取代基于规则的符号表达式操作。

