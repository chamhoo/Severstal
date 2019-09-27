# U-Net: Convolutional Networks for Biomedical Image Segmentation
# U-Net: 用于生物医学图像分割的卷积网络

***
**Abstract.** There is large consent that successful training of deep networks requires many thousand annotated training samples. In this paper, we present a network and training strategy that relies on the strong use of data augmentation to use the available annotated samples more efficiently. The architecture consists of a contracting path to capture context and a symmetric expanding path that enables precise localization. We show that such a network can be trained end-to-end from very few images and outperforms the prior best method (a sliding-window convolutional network) on the ISBI challenge for segmentation of neuronal structures in electron microscopic stacks. Using the same network trained on transmitted light microscopy images (phase contrast and DIC) we won the ISBI cell tracking challenge 2015 in these categories by a large margin. Moreover, the network is fast. Segmentation of a 512x512 image takes less than a second on a recent GPU. The full implementation (based on Caffe) and the trained networks are available at http://lmb.informatik.uni-freiburg.de/people/ronneber/u-net.

**摘要.**我们一般认为一次成功的深度学习训练需要数千个带注释的训练样本。在本文中，我们提出了一个网络和训练策略，它依赖于强大的数据扩充来更有效的利用注释样本。该体系包含了用于捕捉上下文的收缩路径和一个用于实现精确定位的扩张路径。我们展示的这个网络可以使用非常少的图像进行end-to-end训练，并且优于ISBI挑战赛（使用电子显微镜堆叠中的神经元结构的分割）的最佳方法（滑动窗口卷积网络）。使用相同的网络，我们以很大的优势赢得了2015年ISBI细胞追踪挑战赛的冠军。并且网络速度很快，最近在GPU上，512x512图像的分割不到一秒。完整的搭建以及训练过的网络可以在 http://lmb.informatik.uni-freiburg.de/people/ronneber/u-net 获得。

## 1. Introduction

In the last two years, deep convolutional networks have outperformed the state of the art in many visual recognition tasks, e.g. [7,3]. While convolutional networks have already existed for a long time [8], their success was limited due to the size of the available training sets and the size of the considered networks. The breakthrough by Krizhevsky et al. [7] was due to supervised training of a large network with 8 layers and millions of parameters on the ImageNet dataset with 1 million training images. Since then, even larger and deeper networks have been trained [12].

在最近的两年里,深度卷积网络在许多视觉识别任务中表现优于现有技术, e.g. [7,3].尽管卷积神经网络已经出现很长时间 [8], 但是因为训练集大小以及网络的规模，它仍没有取得成功. Krizhevsky 等人的突破[7]。是由于使用了一个包含八层以及上百万个参数的巨大的神经网络，在ImageNet的包含上百万张已标注图像进行监督训练。从那时起，更大更深的网络都被训练出来 [12].

The typical use of convolutional networks is on classification tasks, where the output to an image is a single class label. However, in many visual tasks,especially in biomedical image processing, the desired output should include localization, i.e., a class label is supposed to be assigned to each pixel. Moreover, thousands of training images are usually beyond reach in biomedical tasks.Hence, Ciresan et al. [1] trained a network in a sliding-window setup to predict the class label of each pixel by providing a local region (patch) around that pixel as input. First, this network can localize. Secondly, the training data in terms of patches is much larger than the number of training images. The resulting network won the EM segmentation challenge at ISBI 2012 by a large margin.

卷积网络的典型用途是分类任务，一个图像的输出是一个单个类别的标签。然而，在许多视觉任务中，尤其在医学图像处理中，我们期望最终的输出应该包含位置信息，即，一个分类标签应该被分配给每个像素。此外，医学任务往往无法获取数以千计的训练数据。Hence, Ciresan et al. [1]通过提供该像素周围的局部区域（补丁）作为输入，使用滑动窗口训练网络以预测每个像素的类别标签。首先，这个网络可以定位。并且，补丁的训练数据远远的多于训练图像。由此产生的网络以大优势赢得了2012年的ISBI的分段挑战。

![](fig1.png)

<font size=2> Fig. 1. U-net architecture (example for 32x32 pixels in the lowest resolution). Each blue box corresponds to a multi-channel feature map. The number of channels is denoted on top of the box. The x-y-size is provided at the lower left edge of the box. White boxes represent copied feature maps. The arrows denote the different operations. </font>

<font size=2>  图片1U-net架构（最低分辨率32x32的示例）每个蓝色框对应多通道特征图。在箱型顶部表示通道的数量。在箱型的左下边缘提供x-y尺寸。白色框表示复制的特征图。箭头表示不同的操作。 </font>

Obviously, the strategy in Ciresan et al. [1] has two drawbacks. First, it is quite slow because the network must be run separately for each patch, and there is a lot of redundancy due to overlapping patches. Secondly, there is a trade-off between localization accuracy and the use of context. Larger patches require more max-pooling layers that reduce the localization accuracy, while small patches allow the network to see only little context. More recent approaches[11,4] proposed a classifier output that takes into account the features from multiple layers. Good localization and the use of context are possible at the same time.

很显然的，Ciresan等人的策略有两个缺点。首先，他非常的慢，因为它需要为每个像素周围的补丁单独运行网络，并且会因为补丁之间重叠产生许多的冗余。第二点，在定位精度与使用的视野之间需要权衡。一个大的补丁需要更多的max-pooling层来降低定位精度。而一个小的补丁则允许网络只拥有很小的视野。最近的方法[11,4] 则提出了一种分类输出，其考虑了来自多个层的特征。使得兼顾定位精度与视野成为可能。

In this paper, we build upon a more elegant architecture, the so-called “fully convolutional network” [9]. We modify and extend this architecture such that it works with very few training images and yields more precise segmentations; see Figure 1. The main idea in [9] is to supplement a usual contracting network by successive layers, where pooling operators are replaced by upsampling operators. Hence, these layers increase the resolution of the output. In order to localize, high resolution features from the contracting path are combined with the upsampled output. A successive convolution layer can then learn to assemble a more precise output based on this information.

在这篇论文中，我们建立了一个更加优雅的架构，即所谓的“完全卷积网络”，我们对这个架构进行了修改和拓展，使得它只需要很少的训练图像就可以进行更加精确的分割；参见图片1.主要的思想是将 successive 层添加进通常构建的网络，其中的池化运算被上采样运算代替。因此，这些层增加了输出的分辨率。为了进行定位，收缩路径上的高分辨率特征与上采样输出相结合。然后，连续卷积层基于此可以学习更精准的输出。

One important modification in our architecture is that in the upsampling part we have also a large number of feature channels, which allow the network to propagate context information to higher resolution layers. As a consequence, the expansive path is more or less symmetric to the contracting path, and yields a u-shaped architecture. The network does not have any fully connected layer and only uses the valid part of each convolution, i.e., the segmentation map only contains the pixels, for which the full context is available in the input image. This strategy allows the seamless segmentation of arbitrarily large images by an overlap-tile strategy (see Figure 2). To predict the pixels in the border region of the image, the missing context is extrapolated by mirroring the input image. This tiling strategy is important to apply the network to large images, since otherwise the resolution would be limited by the GPU memory.

我们的一个重要的修改是在上采样部分使用大量的特征通道，以允许网络将信息传播给更高分辨率的层，由此，扩张路径或多或少的与收缩路径对称，并形成一个“U”型结构。这个网络没有任何全连接层并且只使用每个卷积的有效部分。即，分割图仅包含在输入图像中可获得完整视野的像素！！！。该策略允许通过 重叠块策略无缝分割任意大的图像。（参见图片2）为了预测边缘区域的像素，将通过镜像输入的图像来推断缺失的视野。这种平铺策略对于大图像是十分重要的，否则分辨率会受到GPU显存的限制。

As for our tasks there is very little training data available, we use excessive data augmentation by applying elastic deformations to the available training images. This allows the network to learn invariance to such deformations, without the need to see these transformations in the annotated image corpus. This is particularly important in biomedical segmentation, since deformation used to be the most common variation in tissue and realistic deformations can be simulated efficiently. The value of data augmentation for learning invariance has been shown in Dosovitskiy et al. [2] in the scope of unsupervised feature learning.

由于我们的任务可用的训练数据非常少，我们通过对可用的训练数据进行弹性形变来进行过度的数据增强。这允许网络学习这种形变中的不变性，而不需要从带注释的图像库中见到这种形变。这在医学分割中尤为重要，因为形变是组织中最常见的变化，并且可以有效的模拟真实的变形。在无监督学习领域，Dosovitskiy等人证明了数据增强对于学习不变性的价值。[2]

Another challenge in many cell segmentation tasks is the separation of touching objects of the same class; see Figure 3. To this end, we propose the use of a weighted loss, where the separating background labels between touching cells obtain a large weight in the loss function.

细胞分割任务中的另一个挑战是分离同一类的相邻的细胞；参见图3。为此，我们建议使用加权损失，其中分离相邻细胞之间的背景标签在损失函数中赋予较大的权重。

The resulting network is applicable to various biomedical segmentation problems. In this paper, we show results on the segmentation of neuronal structures in EM stacks (an ongoing competition started at ISBI 2012), where we outperformed the network of Ciresan et al. [1]. Furthermore, we show results for cell segmentation in light microscopy images from the ISBI cell tracking challenge 2015. Here we won with a large margin on the two most challenging 2D transmitted light datasets.

由此产生的网络适用于各种医学分割问题。在本文中，我们展示了EM stacks中神经元分割的结果（ISBI 2012 开始的持续的竞争），我们的结果优于Ciresan等人的结果。此外，我们展示在2015年ISBI细胞追踪挑战赛（对光学显微镜下的图像进行细胞分割）显示的结果。在这里，我们在两个最具挑战性的2D透视光数据集上以巨大的优势获胜。

## 2 Network Architecture

The network architecture is illustrated in Figure 1. It consists of a contracting path (left side) and an expansive path (right side). The contracting path follows the typical architecture of a convolutional network. It consists of the repeated application of two 3x3 convolutions (unpadded convolutions), each followed by a rectified linear unit (ReLU) and a 2x2 max pooling operation with stride 2 for downsampling. At each downsampling step we double the number of feature channels. Every step in the expansive path consists of an upsampling of the feature map followed by a 2x2 convolution (“up-convolution”) that halves the number of feature channels, a concatenation with the correspondingly cropped feature map from the contracting path, and two 3x3 convolutions, each followed by a ReLU. The cropping is necessary due to the loss of border pixels in every convolution. At the final layer a 1x1 convolution is used to map each 64-component feature vector to the desired number of classes. In total the network has 23 convolutional layers.

网络架构由图1所示，它包含了一个收缩路径（左侧）以及一个扩张路径（右侧）。收缩路径是一个典型的卷积网络架构。它包括了重复应用两个 3x3 卷积（无padding），每个跟随一个整流线型单元（ReLU）和一个 2x2 的最大池化操作，步长为2的下采样。在每个下采样步骤中我们将加倍特征通道数量。在扩展路径中的每一步包含一个对特征图的上采样，使用一个 2x2 卷积（“向上卷积”），将特征通道数量减半，与来自收缩路径的相应裁剪特征映射串联，以及两个 3x3 的卷积，每个都紧跟一个 ReLU。由于每个卷积中边界像素的丢失， 裁剪是必要的。在最后一层，一个 1x1 卷积被用来将每个64-分量的特征向量映射到所需数量的类中。总的来说，网络有23个卷积层。

To allow a seamless tiling of the output segmentation map (see Figure 2), it is important to select the input tile size such that all 2x2 max-pooling operations are applied to a layer with an even x- and y-size.

为了实现输出分割图的无缝平铺（见图2），选择合适的输入切片大小是很重要的，要使得所有  2x2 最大池化操作都应用于具有合适的x与y大小

## 3. Training

The input images and their corresponding segmentation maps are used to train the network with the stochastic gradient descent implementation of Caffe [6]. Due to the unpadded convolutions, the output image is smaller than the input by a constant border width. To minimize the overhead and make maximum use of the GPU memory, we favor large input tiles over a large batch size and hence reduce the batch to a single image. Accordingly we use a high momentum (0.99) such that a large number of the previously seen training samples determine the update in the current optimization step.

输入图像以及相应的分割图的网络训练使用caffe 的随机梯度下降实现。对于未填充的卷积，输出图像小于输入的恒定边界宽度。为了最大限度的减少开销以及利用 GPU 显存，我们更倾向于使用大的输入切片而不是大的批量，所以我们将批量减小到单个图像。因此，我们使用一个高动量（0.99）使得当前优化步骤中的更新由大量先前看到的训练样本决定。

The energy function is computed by a pixel-wise soft-max over the final feature map combined with the cross entropy loss function. The soft-max is defined as $p_{k}(x)=exp(a_{k}(x))/(\sum^{K}_{k^{'}} exp(a_{k^{'}}(x)))$ where $a_{k}(x)$ denotes the activation in feature channel $k$ at the pixel position $x\in\Omega$ with $\Omega\subset\mathbb{Z}^{2}$. $K$ is the number of classes and $p_{k}(x)$ is the approximated maximum-function. I.e. $p_{k}(x) \approx 1$ for the $k$ that has the maximum activation $a_{k}(x)$ and $p_{k}(x) \approx 0$ for all other $k$. The cross entropy then penalizes at each position the deviation of $p_{l_{(x)}}(x)$ from 1 using

能量函数由最后一个特征图进行逐像素的soft-max之后计算交叉熵损失函数（cross entropy ）得出。soft-max定义为 $p_{k}(x)=exp(a_{k}(x))/(\sum^{K}_{k^{'}} exp(a_{k^{'}}(x)))$ 其中 $a_{k}(x)$ 表示像素位置$x\in\Omega$（$\Omega\subset\mathbb{Z}^{2}$）在特征通道$k$处的激活值。  $K$ 是分类的数量以及 $p_{k}(x)$ 是近似最大函数. 换句话说.对于特征通道$k$激活 $a_{k}(x)$ 使得 $p_{k}(x) \approx 1$， 对于其他特征通道 $p_{k}(x) \approx 0$.交叉熵会惩罚每个位置的偏差，使用：

 $$E=\sum_{X\in\Omega}w(X)\log(p_{l(X)}(X))$$

  where $l: \Omega \to \{1, . . . , K\}$ is the true label of each pixel and $w : \Omega \to \mathbb{R}$ is a weight map that we introduced to give some pixels more importance in the training.

  其中 $l: \Omega \to \{1, . . . , K\}$ 是每一个像素的真实值，以及 $w : \Omega \to \mathbb{R}$ 是我们引入的权重图，以使某些像素在训练中更加重要。

We pre-compute the weight map for each ground truth segmentation to compensate the different frequency of pixels from a certain class in the training data set, and to force the network to learn the small separation borders that we introduce between touching cells (See Figure 3c and d).

我们预计算每个背景分割的权重图，以补训练数据集中的某个类别的不同像素频率，并且迫使网络学习在相邻细胞之间的小分离边界。（见图3c 与 d）

The separation border is computed using morphological operations. The weight map is then computed as

使用形态学来计算分离边界。权重图被计算为：

$$w(x) = w_{c}(x) + w_{0}(x).exp(-\frac{(d_{1}(x) + d_{2}(x))^{2}}{2\sigma^{2}})$$

where $w_{c} : \Omega \to \mathbb{R}$ is the weight map to balance the class frequencies, $d_{1} : \Omega \to \mathbb{R}$ denotes the distance to the border of the nearest cell and $d_{2} : \Omega \to \mathbb{R}$ the distance to the border of the second nearest cell. In our experiments we set $w_{0} = 10$ and $\sigma \approx 5$ pixels.

其中 $w_{c} : \Omega \to \mathbb{R}$ 是用于平衡类别频率的权重图，$d_{1} : \Omega \to \mathbb{R}$ 表示到最近的细胞边界的距离，$d_{2} : \Omega \to \mathbb{R}$ 表示到第二近的细胞边界的距离，在我们的实验中，我们设置$w_{0} = 10$ 以及 $\sigma \approx 5$ 像素。

In deep networks with many convolutional layers and different paths through the network, a good initialization of the weights is extremely important. Otherwise, parts of the network might give excessive activations, while other parts never contribute.Ideally the initial weights should be adapted such that each feature map in the network has approximately unit variance. For a network with our architecture (alternating convolution and ReLU layers) this can be achieved by drawing the initial weights from a Gaussian distribution with a standard deviation of $\sqrt{2/N}$, where N denotes the number of incoming nodes of one neuron [5]. E.g. for a $3 \times 3$ convolution and 64 feature channels in the previous layer $N = 9 \cdot 64 = 576$.

在拥有许多卷积层以及不同路径的深度神经网络中，一个良好的权重初始化是十分重要的。否则，网络的某些部分可能会过度激活，而另一部分完全没有贡献。理想情况之下，应该调整初始权重，使得每一个神经网络中的特征图都有相似的方差。对于具有我们架构的网络（交替卷积与ReLU层）这可以通过标准差 $\sqrt{2/N}$ 的高斯分布来初始化权重，其中$N$代表每一个神经元输入节点的数量。例如，对于前一层有 $3 \times 3$ 卷积和64的特征通道，$N = 9 \cdot 64 = 576$。

## 3.1 Data Augmentation

Data augmentation is essential to teach the network the desired invariance and robustness properties, when only few training samples are available. In case of microscopical images we primarily need shift and rotation invariance as well as robustness to deformations and gray value variations. Especially random elastic deformations of the training samples seem to be the key concept to train a segmentation network with very few annotated images. We generate smooth deformations using random displacement vectors on a coarse 3 by 3 grid. The displacements are sampled from a Gaussian distribution with 10 pixels standard deviation. Per-pixel displacements are then computed using bicubic interpolation. Drop-out layers at the end of the contracting path perform further implicit data augmentation.

当只有少数训练样本可用时，数据增强对于网络所需要的不变性与鲁棒性十分重要。在显微镜图像下，我们主要需要移位和旋转来应对不变性以及需要变形与灰度值变化来应对鲁棒性。特别的是引入弹性形变似乎是训练具有极少图像的分割网络的关键所在。我们使用一个粗糙的$3 \times 3$ 的网络上进行随机位移矢量生成平滑形变。位移是从10像素标准差的高斯分布中采样的。然后使用双三次插值计算每个像素的位移。收缩路径末端的 Drop-out 层负责执行进一步的隐式数据增强。

## 4. Experiments

We demonstrate the application of the u-net to three different segmentation tasks. The first task is the segmentation of neuronal structures in electron microscopic recordings. An example of the dataset and our obtained segmentation is displayed in Figure 2. We provide the full result as Supplementary Material. The data set is provided by the EM segmentation challenge [14] that was started at ISBI 2012 and is still open for new contributions. The training data is a set of 30 images (512x512 pixels) from serial section transmission electron microscopy of the Drosophila first instar larva ventral nerve cord (VNC). Each image comes with a corresponding fully annotated ground truth segmentation map for cells (white) and membranes (black). The test set is publicly available, but its segmentation maps are kept secret. An evaluation can be obtained by sending the predicted membrane probability map to the organizers. The evaluation is done by thresholding the map at 10 different levels and computation of the “warping error”, the “Rand error” and the “pixel error” [14].

我们演示了u-net在三种不同分割任务中的应用。第一项任务是电子显微镜记录中神经元结构的分割。图2展示了数据集和我们获得的分割的示例。我们提供完整的结果作为补充材料。该数据集是由ISBI 2012启动的EM细分挑战赛[14]提供的，并且仍然对新的贡献开放。训练数据是来自果蝇第一龄幼虫腹侧神经索（VNC）的连续切片透射电子显微镜的一组30个图像（512×512像素）。每个图像都带有相应的完全注释的真实分割图，用于细胞（白色）和膜（黑色）。该测试集是公开的，但其分段图保密。可以通过将预测的膜概率图发送给组织者来获得评估。通过在10个不同的级别对地图进行阈值处理并计算“扭曲误差”，“兰德误差”和“像素误差”[14]来完成评估。

The u-net (averaged over 7 rotated versions of the input data) achieves without any further pre- or postprocessing a warping error of 0.0003529 (the new best score, see Table 1) and a rand-error of 0.0382.

u-net（输入数据的7个旋转版本的平均值）在没有任何进一步预处理或后处理的情况下实现了0.0003529的翘曲误差（新的最佳分数，见表1）和兰特误差0.0382。

This is significantly better than the sliding-window convolutional network result by Ciresan et al. [1], whose best submission had a warping error of 0.000420 and a rand error of 0.0504. In terms of rand error the only better performing algorithms on this data set use highly data set specific post-processing methods1 applied to the probability map of Ciresan et al. [1].

这明显优于Ciresan等人的滑动窗口卷积网络结果。[1]，其最佳提交的翘曲误差为0.000420，兰特误差为0.0504。 就rand误差而言，该数据集上唯一性能更好的算法使用高度数据集特定的后处理方法1应用于Ciresan等人的概率图。[1]。

We also applied the u-net to a cell segmentation task in light microscopic images. This segmenation task is part of the ISBI cell tracking challenge 2014 and 2015 [10,13]. The first data set “PhC-U373”2 contains Glioblastoma-astrocytoma U373 cells on a polyacrylimide substrate recorded by phase contrast microscopy (see Figure 4a,b and Supp. Material). It contains 35 partially annotated training images. Here we achieve an average IOU (“intersection over union”) of 92%, which is significantly better than the second best algorithm with 83% (see Table 2). The second data set “DIC-HeLa”3 are HeLa cells on a flat glass recorded by differential interference contrast (DIC) microscopy (see Figure 3, Figure 4c,d and Supp. Material). It contains 20 partially annotated training images. Here we achieve an average IOU of 77.5% which is significantly better than the second best algorithm with 46%.

我们还将u-net应用于光学显微图像中的细胞分割任务。 这种分段任务是2014年和2015年ISBI小区跟踪挑战的一部分[10,13]。 第一组数据“PhC-U373”2包含通过相差显微镜记录的聚丙烯酰亚胺底物上的成胶质细胞瘤 - 星形细胞瘤U373细胞（参见图4a，b和Supp。材料）。 它包含35个部分注释的训练图像。 在这里，我们实现了92％的平均IOU（“交联结合”），这明显优于第二个最佳算法（83％）（见表2）。 第二组数据“DIC-HeLa”3是通过微分干涉对比（DIC）显微镜记录的平板玻璃上的HeLa细胞（参见图3，图4c，d和Supp。材料）。 它包含20个部分注释的训练图像。 在这里，我们的平均IOU为77.5％，明显优于第二好的算法，为46％。
