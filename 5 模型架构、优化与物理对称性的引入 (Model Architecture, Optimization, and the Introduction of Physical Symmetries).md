## 3.4 标准化流 (Normalizing Flows)

标准化流的核心思想如下。正如初等积分中常见的那样，一种常用的方法是换元积分法。使用以下积分变量的代换：

$$\phi=f(z), \quad z=f^{-1}(\phi)$$

其中 $f$ 是一个可逆函数，配分函数 $Z=\int\mathcal{D}\phi P(\phi)$ 可以转换为：

$$Z=\int\mathcal{D}z\left|\det\frac{\partial f(z)}{\partial z}\right|P(f(z))\equiv\int\mathcal{D}z~V(z)$$

当 $V(z)$ 变成一个简单的函数时，例如高斯函数 $V(z)\propto e^{-|z|^{2}/2}$，使用变换 $\phi=f(z)$ 从 $V(z)$ 的分布中生成场系综就会变得非常容易。不幸的是，在量子场论（QFT）的情况下，精确的变换 $f$ 并不容易获得。然而，借助最近的人工智能（AI）和神经网络技术，我们可以获得一种计算成本可接受的近似方法。我们将使用 $f_{\theta}$ 作为 $f$ 的近似器，定义为 $\phi=f_{\theta}(z)$，其中 $\theta$ 是函数的参数集。$f_{\theta}$ 与理想变换 $f$ 之间的偏差可以使用 MH 算法来补偿。效率的提升高度依赖于近似器 $f_{\theta}$。

构建这个参数化近似器 $f_{\theta}$ 正是标准化流旨在实现的目标。在机器学习的背景下，标准化流基于可逆变换。其基本机制与我们的数学动机完美契合：它利用一个简单、易于采样的先验分布 $r(z)$（扮演我们期望的简单函数 $V(z)$ 的角色），并通过一系列可逆且可微的映射系统地将其转换为复杂的目标分布。形式上，令

$$\phi=f_{\theta}(z), \quad z\sim r(z)$$

其中 $z$ 是先验变量，$\phi$ 代表生成的场配置，而 $f_{\theta}$ 是可逆变换。通常，$f_{\theta}$ 由一系列简单的可逆变换（层）组成：

$$f_{\theta}=g_{K}\circ g_{K-1}\circ\dots\circ g_{1}$$

每一个单独的层 $g_{k}$（在后续部分将被显式地实现为仿射耦合层）都被故意设计为一个简单的函数。这种约束确保了其雅可比行列式可以被轻松高效地计算。关键在于，虽然每一层都很简单，但组合大量这些简单的变换能够保证整个网络 $f_{\theta}$ 获得高度的表达能力。定义中间变量 $z_{k}$，且 $z_{0}=z$，

$$z_{k}=g_{k}(z_{k-1}), \quad k=1,2,...,K$$

我们最终得到 $\phi=z_{K}$。将逆变换 $z=f_{\theta}^{-1}(\phi)$ 代入公式(42)中，我们可以将积分展开成一个完整的链式结构：

$$Z=\int\mathcal{D}z~r(z)=\int\mathcal{D}\phi\left|\det\frac{\partial f_{\theta}^{-1}(\phi)}{\partial\phi}\right|r(f_{\theta}^{-1}(\phi))=\int\mathcal{D}\phi~q_{\theta}(\phi)$$

理想情况下，我们期望有一个精确的变换 $f$，使得先验分布 $r(z)$ 能够完美地映射到目标分布 $p(\phi)$。然而，由于寻找这样一个精确的解析映射通常是棘手的，我们将实际模型生成的分布记为 $q_{\theta}(\phi)$。因此，这些分布之间的关系由下式给出：

$$q_{\theta}(\phi)=r(z)\left|\det\frac{\partial f_{\theta}^{-1}(\phi)}{\partial\phi}\right|$$

等价地，对于正向变换 $\phi=f_{\theta}(z)$，利用关系式 $\left|\det\frac{\partial z}{\partial\phi}\right|=\left|\det\frac{\partial\phi}{\partial z}\right|^{-1}$，我们有：

$$\log q_{\theta}(\phi)=\log r(z)-\log\left|\det\frac{\partial f_{\theta}(z)}{\partial z}\right|$$

由于 $f_{\theta}$ 是多个变换的复合，其雅可比行列式可以分解为每一层雅可比行列式的乘积：

$$\log q_{\theta}(\phi)=\log r(z)-\sum_{k=1}^{K}\log\left|\det\frac{\partial g_{k}(z_{k-1})}{\partial z_{k-1}}\right|$$

因此，只要每个单独层 $g_{k}$ 的逆映射和雅可比行列式能够被高效计算，标准化流就可以同时实现快速的样本生成和精确的概率密度计算。然后，我们就能更有效地生成场配置。

## 3.5 传统采样方法的局限性

虽然传统的马尔可夫链蒙特卡洛 (MCMC) 方法是格点场论的基础，但它们内在地受到被称为样本自相关的关键局限性的影响。马尔可夫链生成的相邻样本在统计上并不是独立的。因此，有效样本量显著小于生成的配置总数。大部分效率低下都源于这种样本自相关：
* **热化时间 (Thermalization Time)**：在实际测量阶段之前，需要一个相当长的预热（热化）期，以消除人为初始配置引入的任何偏差。
* **临界慢化 (Critical Slowing Down)**：随着系统接近连续相变（临界点），物理相关长度发散。这导致局部更新方法的算法自相关时间急剧增加，严重阻碍了效率。
* **拓扑冻结 (Topological Freezing)**：在某些规范场论中，MCMC 更新难以克服高作用量势垒以在不同的拓扑扇区之间隧穿，导致模拟在单一拓扑电荷扇区中“冻结”。
* **高维度的效率下降 (Diminished Efficiency in Large Dimensions)**：随着物理格点体积的扩大，配置空间的维度呈指数级增长，由于临界慢化和拓扑冻结，进行有效的相空间探索变得极其困难。

这些艰巨的挑战迫使研究人员探索现代机器学习生成模型，以增强或完全取代传统的采样范式。配备神经网络技术的标准化流是解决这些困难的候选方案之一。

# 4 神经网络与标准化流算法

在本节中，我们简要介绍神经网络的基本结构和工作原理，包括前向传播和反向传播的过程。在前面介绍的标准化流基本概念的基础上，我们将详细说明特定可逆变换（即仿射耦合层）的神经网络实现，并定义基于 Kullback-Leibler 散度的网络训练目标。

## 4.1 神经网络作为参数化函数

从数学的角度来看，神经网络可以被视为一个具有高度表达能力的参数化函数 $F_{\theta}(x)$，其中 $x$ 表示输入变量。在我们的研究中，神经网络被定义为映射 $F_{\theta}(x)=h^{(L)}$，其中 $\theta=\{W^{(l)},b^{(l)}\}_{l=1}^{L}$ 表示跨所有层的所有可训练参数（权重和偏置）的集合。

神经网络通过堆叠 $L$ 个顺序层构建而成。令 $h^{(0)}=x$ 表示输入向量。第 $l$ 层的激活输出 $h^{(l)}$ 递归定义为：

$$h^{(l)}=\sigma^{(l)}(W^{(l)}h^{(l-1)}+b^{(l)}), \quad l=1,2,...,L$$

其中 $W^{(l)}$ 和 $b^{(l)}$ 分别是第 $l$ 层的权重矩阵和偏置向量，$\sigma^{(l)}$ 是应用于每个元素的非线性激活函数。“层”指的是神经元的集合，它将上一层的输出 $h^{(l-1)}$ 作为输入，并产生一个新的向量表示 $h^{(l)}$。通过堆叠这些层，深度神经网络获得了近似从输入空间到目标空间的高度复杂非线性映射的能力，最终网络输出由下式给出：

$$F_{\theta}(x)=h^{(L)}$$

## 4.2 前向与反向传播

神经网络代表一个参数化函数 $F_{\theta}(x)$，其中 $x$ 是输入，$\theta$ 表示网络中所有的可训练参数。引入这个函数是出于特定的目的。在一般的机器学习问题中，$F_{\theta}(x)$ 可能用于预测标签、近似未知函数或将数据转换为期望的输出形式。在目前的工作中，$F_{\theta}$ 被用作标准化流的一部分：它将从简单先验分布中抽取的样本转换为应该近似目标格点场分布的场配置。因此，训练的目的是调整参数，使 $F_{\theta}$ 的输出更接近期望的目标分布。

神经网络的训练过程通常包括两个阶段：前向传播和反向传播。在前向传播期间，输入数据依次通过每一层的变换，产生最终的模型输出 $F_{\theta}(x)$。基于该输出和期望的训练目标，计算损失函数 $\mathcal{L}(\theta)$。在我们的案例中，该损失衡量了生成分布与目标玻尔兹曼分布之间的距离。

在反向传播中，应用链式法则计算损失函数相对于每一层参数的梯度 $\nabla_{\theta}\mathcal{L}(\theta)$。这些梯度指示了为了减小损失应该如何改变参数。然后通过梯度下降或其变体来更新参数：

$$\theta\leftarrow\theta-\eta\nabla_{\theta}\mathcal{L}(\theta)$$

其中 $\eta$ 是学习率。通过重复前向传播、损失评估、反向传播和参数更新，网络逐渐学习到针对给定目标的更好函数 $F_{\theta}$。

## 4.3 前向与反向传播 (续)

神经网络的训练过程通常包括两个阶段：前向传播和反向传播。我们可以通过与物理系统进行类比来直观地理解这个过程：想象一个球在由参数空间 $\theta$ 定义的高维势能面上运动，如图 3 所示。

在前向传播期间，输入数据依次通过每一层的变换，产生最终的模型输出。在我们的物理类比中，这一步计算了球在势能面上的当前位置。基于此输出，计算损失函数 $\mathcal{L}(\theta)$。该损失衡量了生成分布与目标玻尔兹曼分布的距离，有效地表示了当前位置势能的大小。我们的最终目标是导航该曲面并找到此势能的最小值（目标位置）。

在反向传播中，应用链式法则计算损失函数相对于每一层参数的梯度 $\nabla_{\theta}\mathcal{L}(\theta)$。在数学上，梯度向量始终指向最陡峭上升的方向。因此，为了最小化损失，我们必须在完全相反的方向——负梯度 $(-\nabla_{\theta}\mathcal{L}(\theta))$——上移动参数，这在物理上对应于最陡峭下降的方向（最快的向下路径）。这些负梯度指示了应该如何改变参数以便最有效地降低损失。

在机器学习中，这个优化过程被正式称为训练。然后对参数空间进行操作，通过梯度下降或其变体使参数沿这个最陡峭下降方向移动：

$$\theta\leftarrow\theta-\eta\nabla_{\theta}\mathcal{L}(\theta)$$

其中 $\eta$ 是学习率。使用这些更新的参数，网络重新计算球在表面上的新位置。通过重复前向传播、损失评估、反向传播和参数更新，网络逐渐学习到更好的映射。由于参数空间 $\theta$ 极其庞大，人们开发了许多先进的优化方法来有效地在此高维表面上导航。在本文中，我们跳过反向传播复杂的数学细节；在标准的机器学习教科书中可以找到全面的讨论和推导。

## 4.4 带有神经网络的标准化流细节

为了构建前面介绍的可逆变换 $g$，我们利用了仿射耦合层。仿射耦合层是标准化流中广泛使用的可逆变换结构。它的核心优势在于它使得雅可比行列式的计算变得极其简单和高效。

其基本思想是将输入变量分成两部分，其中一部分保持不变，而另一部分则经历一个由从第一部分派生出的缩放和平移函数所参数化的仿射变换。设输入变量为 $\phi=(\phi_{a},\phi_{b})$。仿射耦合层的前向变换定义为：

$$z_{a}=\phi_{a}$$

$$z_{b}=\phi_{b}\odot\exp[s_{\theta}(\phi_{a})]+t_{\theta}(\phi_{a})$$

其中 $\odot$ 表示逐元素乘法，而 $s_{\theta}(\cdot)$ 和 $t_{\theta}(\cdot)$ 分别表示缩放和平移函数，它们通常由神经网络参数化。在实践中，分区 a 和 b 的功能角色可以在交替的层中进行切换，以确保所有场变量都被均匀更新。逆变换 $g^{-1}$ 可以明确写成：

$$\phi_{a}=z_{a}$$

$$\phi_{b}=[z_{b}-t_{\theta}(z_{a})]\odot\exp[-s_{\theta}(z_{a})]$$

因此，仿射耦合层自然满足可逆性要求，并且我们可以看到输入需要经过两次放射耦合层才可以得到一次完整的更新。

由于这种特定的分区设计，仿射耦合层的雅可比矩阵 $J=\frac{\partial z}{\partial\phi}$ 呈现出下分块三角形式：

$$J=\begin{pmatrix}I&0\\ A&D\end{pmatrix}$$

其中

$$A=\frac{\partial z_{b}}{\partial\phi_{a}}$$

$$D=\text{diag}(e^{s_{\theta,1}},e^{s_{\theta,2}},...)$$

因此，其行列式就是其对角元素的乘积：

$$\det J=\det(D)=\prod_{i}e^{s_{\theta,i}(\phi_{a})}$$

这产生了一个高度易于处理的对数行列式：

$$\log|\det J|=\sum_{i}s_{\theta,i}(\phi_{a})$$

相应地，逆变换的雅可比矩阵表现出类似的结构：

$$J^{-1}=\frac{\partial\phi}{\partial z}=\begin{pmatrix}I&0\\ B&D^{-1}\end{pmatrix}$$

其中 $D^{-1}=\text{diag}(e^{-s_{\theta,1}},e^{-s_{\theta,2}},...)$。因此，

$$\log|\det J^{-1}|=-\sum_{i}s_{\theta,i}(\phi_{a})$$

## 4.5 Kullback-Leibler 散度与训练目标

接下来，我们简要介绍 Kullback-Leibler (KL) 散度，概述其推导过程，并通过将 KL 散度与我们的目标分布相结合来设计损失函数。在统计分析中，通常需要量化两个分布之间的相似性，而 KL 散度正是用于此目的。对于两个概率分布 $q(\phi)$ 和 $p(\phi)$，KL 散度定义为：

$$D_{KL}(q||p)=\int d\phi~q(\phi)\log\frac{q(\phi)}{p(\phi)}$$

根据琴生不等式（Jensen's inequality） $\log(\mathbb{E}[x])\ge\mathbb{E}[\log(x)]$，我们得到：

$$\log\left(\int d\phi~q(\phi)\frac{p(\phi)}{q(\phi)}\right)\ge\int d\phi~q(\phi)\log\frac{p(\phi)}{q(\phi)}$$

注意到概率分布满足归一化条件 $\int d\phi~p(\phi)=1$，我们有：

$$\log\left(\int d\phi~q(\phi)\frac{p(\phi)}{q(\phi)}\right)=\log(1)=0$$

因此，

$$0\ge\int d\phi~q(\phi)\log\frac{p(\phi)}{q(\phi)}$$

两边同时乘以 -1 得到：

$$D_{KL}(q||p)=-\int d\phi~q(\phi)\log\frac{p(\phi)}{q(\phi)}\ge0$$

因此，KL 散度始终是非负的，当且仅当 $q(\phi)=p(\phi)$ 时等号成立。这表明 KL 散度有效地测量了两个概率分布之间的差异。当 KL 散度为 0 时，模型分布等于目标分布。

在格点场论中，目标分布通常是 $p(\phi)=\frac{1}{Z}e^{-S[\phi]}$，而由标准化流生成的分布记为 $q_{\theta}(\phi)$。为了使 $q_{\theta}(\phi)$ 逼近 $p(\phi)$，我们将最小化 KL 散度作为我们的训练目标。将目标分布和生成的分布代入方程中得到：

$$D_{KL}(q_{\theta}||p)=\int d\phi~q_{\theta}(\phi)[\log q_{\theta}(\phi)+S[\phi]+\log Z]$$

这可以表示为期望值：

$$D_{KL}(q_{\theta}||p)=\mathbb{E}_{\phi\sim q_{\theta}}[\log q_{\theta}(\phi)+S[\phi]]+\log Z$$

由于 $\log Z$ 独立于模型参数 $\theta$，这个常数项在训练期间可以忽略。训练损失随后被定义为：

$$\mathcal{L}(\theta)=\mathbb{E}_{\phi\sim q_{\theta}}[\log q_{\theta}(\phi)+S[\phi]]$$

使用 $\phi=f_{\theta}(z)$ 和 $z\sim r(z)$，这可以进一步改写为：

$$\mathcal{L}(\theta)=\mathbb{E}_{z\sim r(z)}[S(f_{\theta}(z))+\log q_{\theta}(f_{\theta}(z))]$$

根据变量替换公式，

$$\log q_{\theta}(f_{\theta}(z))=\log r(z)-\log\left|\det\frac{\partial f_{\theta}(z)}{\partial z}\right|$$

因此，最终的训练目标是：

$$\mathcal{L}(\theta)=\mathbb{E}_{z\sim r(z)}\left[S(f_{\theta}(z))+\log r(z)-\log\left|\det\frac{\partial f_{\theta}(z)}{\partial z}\right|\right]$$

在实践中，先验分布上的期望值通过使用蒙特卡洛采样对积分进行离散化来进行数值评估。对于给定的、从先验分布 $r(z)$ 中抽取的 $B$ 个独立样本批次 $\{z^{(i)}\}_{i=1}^{B}$，离散化的经验损失函数变为：

$$\mathcal{L}(\theta)\approx\frac{1}{B}\sum_{i=1}^{B}\left[S(f_{\theta}(z^{(i)}))+\log r(z^{(i)})-\log\left|\det\frac{\partial f_{\theta}(z^{(i)})}{\partial z^{(i)}}\right|\right]$$

## 4.6 Kullback-Leibler 散度与训练目标 (文中重复部分)

正如我们在反向传播部分提到的，我们需要定义一个损失函数来优化参数。接下来，我们简要介绍 Kullback-Leibler (KL) 散度，概述其推导，并通过结合 KL 散度与我们的目标分布来设计损失函数。

在统计分析中，经常需要量化两个分布之间的相似性，而 KL 散度正是起这个作用。对于两个概率分布 $q(\phi)$ 和 $p(\phi)$，KL 散度定义为：

$$D_{KL}(q||p)=\int d\phi~q(\phi)\log\frac{q(\phi)}{p(\phi)}$$

由琴生不等式 $\log(\mathbb{E}[x])\ge\mathbb{E}[\log(x)]$，我们得到：

$$\log\left(\int d\phi~q(\phi)\frac{p(\phi)}{q(\phi)}\right)\ge\int d\phi~q(\phi)\log\frac{p(\phi)}{q(\phi)}$$

注意到概率分布满足归一化条件 $\int d\phi~p(\phi)=1$，我们有：

$$\log\left(\int d\phi~q(\phi)\frac{p(\phi)}{q(\phi)}\right)=\log(1)=0$$

因此，

$$0\ge\int d\phi~q(\phi)\log\frac{p(\phi)}{q(\phi)}$$

两边乘以 -1 得到：

$$D_{KL}(q||p)=-\int d\phi~q(\phi)\log\frac{p(\phi)}{q(\phi)}\ge0$$

因此，KL 散度始终是非负的，并且仅当 $q(\phi)=p(\phi)$ 时等式成立。这证明了 KL 散度有效地测量了两个概率分布之间的差异。当 KL 散度为 0 时，模型分布等于目标分布。

在格点场论中，目标分布通常为 $p(\phi)=\frac{1}{Z}e^{-S[\phi]}$，而标准化流生成的分布表示为 $q_{\theta}(\phi)$。为了使 $q_{\theta}(\phi)$ 近似 $p(\phi)$，我们最小化 KL 散度作为训练目标。将目标分布和生成分布代入方程得到：

$$D_{KL}(q_{\theta}||p)=\int d\phi~q_{\theta}(\phi)[\log q_{\theta}(\phi)+S[\phi]+\log Z]$$

这可以表示为一个期望值：

$$D_{KL}(q_{\theta}||p)=\mathbb{E}_{\phi\sim q_{\theta}}[\log q_{\theta}(\phi)+S[\phi]]+\log Z$$

因为 $\log Z$ 独立于模型参数 $\theta$，这个常数项可以在训练期间被忽略。然后训练损失定义为：

$$\mathcal{L}(\theta)=\mathbb{E}_{\phi\sim q_{\theta}}[\log q_{\theta}(\phi)+S[\phi]]=\int d\phi~q_{\theta}(\phi)[\log q_{\theta}(\phi)+S[\phi]]$$

利用 $\phi=f_{\theta}(z)$ 且 $z\sim r(z)$，这可以进一步重写为：

$$\mathcal{L}(\theta)=\mathbb{E}_{z\sim r(z)}[S(f_{\theta}(z))+\log q_{\theta}(f_{\theta}(z))]$$

根据变量替换公式，

$$\log q_{\theta}(f_{\theta}(z))=\log r(z)-\log\left|\det\frac{\partial f_{\theta}(z)}{\partial z}\right|$$

因此，最终的训练目标为：

$$\mathcal{L}(\theta)=\mathbb{E}_{z\sim r(z)}\left[S(f_{\theta}(z))+\log r(z)-\log\left|\det\frac{\partial f_{\theta}(z)}{\partial z}\right|\right]$$

在实践中，我们可以通过使用蒙特卡洛 (M-C) 采样对积分进行离散化，在数值上近似这个先验分布上的期望值。对于给定的、从先验分布 $r(z)$ 抽取的 $B$ 个独立样本批次 $\{z^{(i)}\}_{i=1}^{B}$，离散化的经验损失函数变为：

$$\mathcal{L}(\theta)\approx\frac{1}{B}\sum_{i=1}^{B}\left[S(f_{\theta}(z^{(i)}))+\log r(z^{(i)})-\log\left|\det\frac{\partial f_{\theta}(z^{(i)})}{\partial z^{(i)}}\right|\right]$$

然后我们最小化这个损失，并使用许多优化方法来实现这一点，例如随机梯度下降 (SGD) 及其自适应变体，如 Adam 或 RMSprop。在随后的训练和优化中，我们使用了 Adam 方法。

## 4.7 结合标准化流的 Metropolis-Hastings 算法

如果流生成的分布 $q_{\theta}(\phi)$ 完美匹配目标分布 $p(\phi)$，那么流生成的样本可以直接用于计算物理观测量。然而，在实际训练中，$q_{\theta}(\phi)$ 通常只是 $p(\phi)$ 的一个近似。为了确保精确采样，流模型可以被用作独立的提议分布，并结合 Metropolis-Hastings 校正一起使用。

假设当前配置为 $\phi$，候选配置 $\phi^{\prime}$ 从流模型中独立生成，使得 $\phi^{\prime}\sim q_{\theta}(\phi^{\prime})$。因此，我们的提议分布为 $g(\phi^{\prime}|\phi)=q_{\theta}(\phi^{\prime})$，接受概率为：

$$A(\phi\rightarrow\phi^{\prime})=\min\left[1,\frac{p(\phi^{\prime})q_{\theta}(\phi)}{p(\phi)q_{\theta}(\phi^{\prime})}\right]$$

由于 $p(\phi)\propto e^{-S[\phi]}$，接受率可以写成：

$$A(\phi\rightarrow\phi^{\prime})=\min\left[1,e^{-S[\phi^{\prime}]+S[\phi]+\log q_{\theta}(\phi)-\log q_{\theta}(\phi^{\prime})}\right]$$

这种方法保证了目标分布的正确性，同时利用流模型生成全局提议，从而有效地绕过了临界慢化现象并减少了样本自相关。

# 5 模型架构、优化与物理对称性的引入 (Model Architecture, Optimization, and the Introduction of Physical Symmetries)



## 5.1 卷积神经网络与平移同变性 (Convolutional Neural Networks and Translational Equivariance)

在第四节，我们介绍了全连接网络（FCN）。将 FCN 直接应用于格点场论面临着实际挑战：参数数量会随着格点体积 $V=L^{2}$ 的平方呈爆炸式增长。对于具有 $V$ 个自由度的输入，稠密权重矩阵的尺度通常为 $O(V^{2})$。这带来了巨大的内存和计算开销，使得模型很难推广到更大的格点尺寸。

更重要的是，格点场论中的物理作用量具有空间平移对称性。换句话说，控制系统的物理规律并不取决于格点位点的绝对位置，而仅取决于相邻位点之间的相对关系。如果我们使用普通 FCN，网络从根本上无法感知这种结构，必须在训练期间从数据中重新学习这种平移不变性，这是极其低效的。

为了解决这个问题，我们将流模型中的上下文网络替换为卷积神经网络（CNN）。CNN 的核心计算组件是卷积核和通道。假设我们有一个形状为 $C_{in}\times L\times L$ 的二维输入场 $h^{(l-1)}$，其中 $C_{in}$ 是输入通道数，$L$ 是格点维度。卷积操作在输入数据上滑动一个小权重矩阵（例如 $3\times3$ 的核），计算每个局部邻域内的加权和。对于第 $l$ 层的第 $c_{out}$ 个输出通道，在位置 $(x, y)$ 处的输出由下式给出：

$$h_{cout}^{(l)}(x,y)=\sigma\left(\sum_{c_{in}=1}^{C_{in}}\sum_{i=-k}^{k}\sum_{j=-k}^{k}W_{cout,c_{in}}^{(l)}(i,j)h_{c_{in}}^{(l-1)}(x+i,y+j)+b_{cout}^{(l)}\right)$$

其中：
* $W^{(l)}$ 是第 $l$ 层的卷积核；
* 卷积核大小为 $(2k+1)\times(2k+1)$；
* $b_{cout}^{(l)}$ 是偏置项；
* $\sigma$ 是非线性激活函数；
* $(x+i,y+j)$ 表示以 $(x, y)$ 为中心的局部邻域。

如该公式所示，整个格点上的所有位置都共享同一组卷积权重 $W$。这种权重共享机制极大地压缩了参数空间。更准确地说，CNN 中可训练参数的数量主要由卷积核大小、通道数和网络深度决定，并不直接随格点体积 $V=L^{2}$ 缩放。自然地，前向和反向传播的计算成本仍然随着位点数 $V$ 的增加而增长，但与 FCN 的 $O(V^{2})$ 缩放相比，CNN 从根本上更适合格点系统。

对于具有周期性边界条件的物理系统，我们必须在卷积计算中正确处理边界效应。格点场论通常使用周期性边界条件，定义为：

$$\phi(x+L\hat{\mu})=\phi(x)$$

因此，我们在 CNN 中使用循环填充（circular padding）。当卷积核滑动到格点边缘时，它会自动绕回并读取对面的值，完美地尊重了周期性格点的物理拓扑结构。在这种设置下，CNN 自然表现出一种被称为平移同变性的平移对称性的充分条件（translational equivariance）。设 $T_{a}$ 为将场配置平移 $a$ 个格点位点的算子。一个理想的卷积网络满足：

$$f_{\theta}(T_{a}\phi)=T_{a}f_{\theta}(\phi)$$

这意味着将输入场移动一定距离并将其通过网络，得到的结果与将原始场通过网络然后再将输出移动相同距离完全相同。网络不再承担从数据中学习平移对称性的负担，因为这种对称性在结构上由卷积架构和循环填充所保证。

然而，正如我们在第四节介绍过的，我们的标准化流模型在更新过程中使用了棋盘格掩码（checkerboard masking）策略。棋盘格掩码将格点位点分为两个不同的子格点，通常称为“红”和“黑”位点。由于单个耦合层仅更新一半的位点，同时冻结另一半，因此固定的棋盘格掩码显式地分离了这两个子格点。单格子平移（$1a$）会交换黑红子格点。因为红黑位点进入网络的顺序不一致，后续位点更新所依赖的信息也不一致。因此，即使我们保持红黑网格的网络参数完全相同，严格的 $1a$ 平移同变性仍然被破坏了。

**如图，$r$代表红色掩码位置的输入,$b$代表黑色掩码位置的输入，$r_u$和$b_u$代表经过放射耦合层之后对应位置的更新，可以看出根据进入耦合层的顺序的不同，平移前后相同的值，更新后的值也是不同的。**

下面我们来证明卷积神经网络对于normalizing flow 具有$2\hat\mu$平移等变性：

定义掩码矩阵 $M$，其中 $M=1$ 表示黑色位点（冻结），$M=0$ 表示红色位点（更新）。定义平移 $2a$ 个位点的算子为 $T_{2a}$。根据棋盘格格点的几何特性，掩码在 $2a$ 平移下保持不变：

$$T_{2a}M=M$$

$$T_{2a}(1-M)=1-M$$

同时，因为网络 $s$ 和 $t$ 使用了带有循环填充的 CNN，它们内在地满足严格的平移同变性。对于任何输入场 $\psi$：

$$T_{2a}t(\psi)=t(T_{2a}\psi), \quad T_{2a}s(\psi)=s(T_{2a}\psi)$$

此外，平移算子与逐元素乘法可交换：$T_{2a}(A\odot B)=(T_{2a}A)\odot(T_{2a}B)$。

1. **红色位点更新层 $f_{r}$**：第一步，冻结黑色位点（$M$），更新红色位点（$1-M$）。网络映射为 $f_{r}$：

$$f_{r}(\phi)=M\odot\phi+(1-M)\odot[(\phi-t_{r}(M\odot\phi))\odot e^{-s_{r}(M\odot\phi)}]$$

我们首先证明单层 $f_{r}$ 具有 $2a$ 平移同变性。在两边应用 $T_{2a}$ 并将其分配到内部：

$$T_{2a}f_{r}(\phi)=(T_{2a}M)\odot(T_{2a}\phi)+T_{2a}(1-M)\odot[(T_{2a}\phi-T_{2a}t_{r}(M\odot\phi))\odot e^{-T_{2a}s_{r}(M\odot\phi)}]$$

代入掩码的平移不变性 $T_{2a}M=M$ 以及 CNN 的同变性 $T_{2a}t_{r}(M\odot\phi)=t_{r}(M\odot T_{2a}\phi)$：

$$f_{r}(\phi)=M\odot\phi+(1-M)\odot[(\phi-t_{r}(M\odot\phi))\odot e^{-s_{r}(M\odot\phi)}]$$

观察等式右边，这正是将 $T_{2a}\phi$ 作为初始输入代入 $f_{r}$ 的形式，因此：

$$T_{2a}f_{r}(\phi)=f_{r}(T_{2a}\phi)$$

2. **黑色位点更新层 $f_{b}$**：第二步，冻结红色位点（$1-M$），更新黑色位点（$M$）。网络映射为 $f_{b}$：

$$f_{b}(\psi)=(1-M)\odot\psi+M\odot[(\psi-t_{b}((1-M)\odot\psi))\odot e^{-s_{b}((1-M)\odot\psi)}]$$

类似地，在两边应用 $T_{2a}$ 并利用同变性和不变性性质：

$$T_{2a}f_{b}(\psi)=(1-M)\odot(T_{2a}\psi)+M\odot[(T_{2a}\psi-t_{b}((1-M)\odot T_{2a}\psi))\odot e^{-s_{b}((1-M)\odot T_{2a}\psi)}]$$

这给出了：

$$T_{2a}f_{b}(\psi)=f_{b}(T_{2a}\psi)$$

3. **完整流模型 $F$ 的 $2a$ 同变性**：一个完整的更新步骤是红黑层的复合映射：

$$F(\phi)=(f_{b}\circ f_{r})(\phi)=f_{b}(f_{r}(\phi))$$

我们在整个复合过程中应用 $T_{2a}$：

$$T_{2a}F(\phi)=T_{2a}[f_{b}(f_{r}(\phi))]$$

因为 $f_{b}$ 具有 $2a$ 同变性，平移算子 $T_{2a}$ 可以穿过外层的 $f_{b}$：

$$T_{2a}[f_{b}(f_{r}(\phi))]=f_{b}(T_{2a}f_{r}(\phi))$$

然后，因为 $f_{r}$ 也具有 $2a$ 同变性，$T_{2a}$ 继续穿过内层的 $f_{r}$：

$$f_{b}(T_{2a}f_{r}(\phi))=f_{b}(f_{r}(T_{2a}\phi))$$

将结果写回复合形式即可完成证明：

$$T_{2a}F(\phi)=F(T_{2a}\phi)$$

因此，尽管 CNN 本身具有平移同变性，但具有棋盘格耦合的整体流架构通常只能保持 $2a$ 平移同变性，而不是完全的 $1a$ 同变性，即使我们对红黑网格使用不同的网络参数也是如此。这是掩码结构不可避免的后果。

## normalizing flow的卷积神经网络训练架构

在正式训练中，我们的神经网络设置为12层耦合层，每层耦合层包含3层residual block，和两个各包含一层residual block的分支网络。这三层residual block的输出被当作输入进入分支网络进行训练，然后这两个分支网络分别输出$s^i_{\theta_a}$和$t^i_{\theta_a}$，其中$i$代表第$i$层耦合层。每个residual block又包含两层卷积层，通过残差连接。另外在初始化时，我们将每个耦合层的最后一层卷积层的参数全都设置为0，这样每层输出的$s^i_{\theta_a}$和$t^i_{\theta_a}$相应的也是0，这样在仿射变化下，能保证我们的网络在训练初始阶段接近恒等映射。其他隐藏层则初始化为均值为0，方差为0.01的高斯分布。

## 5.2 流模型更新策略与动量空间预编码（自由场先验） (Flow Model Update Strategy and Momentum-Space Pre-coding (Free-Field Prior))

在第四节中我们提到，在标准化流的仿射耦合层中，我们通常使用棋盘掩码将格点位点分成两个不相交的集合：

$$\phi=\phi_{a}+\phi_{b}$$

其中 $\phi_{a}$ 是冻结部分，$\phi_{b}$ 是更新部分。在棋盘格掩码下，一半的位点被冻结，另一半被更新。

Under a checkerboard mask, half of the sites are frozen, and the other half are updated. A typical affine coupling transformation looks like this:
\begin{equation}
\phi_b' = \left(\phi_b - t_\theta(\phi_a)\right) \odot \exp[-s_\theta(\phi_a)],
\end{equation}
while the frozen part stays the same:
\begin{equation}
\phi_a'=\phi_a.
\end{equation}

而冻结部分保持不变：

$$\phi_{a}^{\prime}=\phi_{a}.$$

然而，这种棋盘格耦合不可避免地制造了一个连通性瓶颈。当网络更新 $\phi_{b}$ 中的特定位点时，该点自身的值并没有被输入网络来进行训练，而只是在最后的仿射变换时与网络的输出值进行点积计算。这对于格点场论中的作用量却会带来严重的问题。以标量场的动能项为例，连续动能项为：

$$(\partial_{\mu}\phi)^{2}.$$

在离散格点上，这相当于：

$$\sum_y \Box(x,y)\phi(y) = \sum_\mu (2\phi(x) - \phi(x-\hat{\mu}) - \phi(x+\hat{\mu})).$$

网络想要获取到完整的动能项必须知道格点自身与周围四个格点的全部信息.掩码会阻止它知道该位点自身的原始值，因此它必须完全依赖 $\phi_{a}$ 中周围的冻结位点来进行预测。

更具体地说，在更新红色子格点时，网络只能将来自黑色子格点的信息注入到红色位点。在随后的层中，当更新黑色子格点时，网络终于可以使用红色位点（在前一步中被修改）来更新黑色位点。尽管这种交替更新方案保持了可逆性和易于处理的雅可比行列式，但它从根本上导致了信息传播的延迟。对于具有强长程关联的场配置，网络必须堆叠大量的耦合层才能迭代传播这些信息。

**为此我们在12层耦合层，隐藏层输出通道为16的小模型上对比了14x14格点上，初始分布为标准高斯分布，目标分布分别为$S_{\mathrm{kin}+m_{\mathrm{eff}}}(\phi) = \sum_x \left[ \sum_y \phi(x)\Box(x,y)\phi(y) + m_{\mathrm{eff}}^2 \phi(x)^2 \right]$ 和$S_{\mathrm{pot}}(\phi) = \sum_x \left[ m^2\phi(x)^2 + \lambda \phi(x)^4 \right]$的训练接受率，其中为了避免动能项发散，我们添加了有效质量：$m_{\mathrm{eff}}$是14x14$\phi^4$场的有效质量.$m$和$\lambda$是$\phi^4$场的裸参数**

可以看出，$\phi^4$作用量中的动能项相比质量项，确实更加难以学习。

为了缓解这个问题，我们引入了自由场先验（动量空间预编码）。核心思想很简单：我们不再从不相关的高斯白噪声开始，而是预先在动量空间中构建一个初始场，使其具有由自由场作用量控制的正确两点关联函数结构。换句话说，我们希望送入流模型的变量 $z$ 内在地包含自由场的空间关联。正因为如此，the neural network, build upon the Affine coupling layers, does not need to learn the finite difference structure of the kinetic term from scratch. 相反，它可以将所有的表达能力集中在学习由非线性相互作用项 $\lambda\phi^{4}$ 引起的非高斯形变上。

连续空间中的自由标量场作用量被给为：

$$S_{free}=\frac{1}{2}\int d^{2}x~\phi(x)(-\partial^{2}+m_{free}^{2})\phi(x).$$

在周期格点上，我们使用离散傅里叶变换将实空间场 $\phi(x)$ 映射到动量空间：

$$\phi(x)=\frac{1}{\sqrt{V}}\sum_{k}\tilde{\phi}(k)e^{ik\cdot x}$$

其中 $V=L^{2}$ 是格点体积，允许的动量为：

$$k_{\mu}=\frac{2\pi n_{\mu}}{L}, \quad n_{\mu}=0,1,...,L-1$$

由于 $\phi(x)$ 是实标量场，即 $\phi^{*}(x)=\phi(x)$，动量空间中的傅里叶模必须满足厄米条件：

$$\tilde{\phi}^{*}(-k)=\tilde{\phi}(k)$$

这意味着 $\tilde{\phi}(k)$ 和 $\tilde{\phi}(-k)$ 不是独立的自由度，而是互为复共轭。

在离散格点上，拉普拉斯算子在动量空间中是对角化的。二维周期格点拉普拉斯算子的特征值为：

$$\tilde{K}(k)=\sum_{\mu=1}^{2}(2-2\cos k_{\mu})=4\sin^{2}\frac{k_{1}}{2}+4\sin^{2}\frac{k_{2}}{2}.$$

因此，自由场的二次型可以在动量空间中表示为：

$$S_{free}=\sum_{k}[\hat{K}(k)+m_{free}^{2}]|\tilde{\phi}(k)|^{2}.$$

为了避免发散，$m_{free}^{2}$必须取正值，在这里我们先简单的$m_{free}^{2} = |m^2|$，关于$m_{free}^{2}$的具体取值我们在下一subsection会进一步讨论。

定义 $\tilde{M}(k)=\hat{K}(k)+m_{free}^{2}$，自由场分布变为：

$$r(\phi)=\frac{1}{Z_{free}}\exp\left[-\sum_{k}\tilde{M}(k)|\tilde{\phi}(k)|^{2}\right]$$

从该表达式中可以明显看出，自由场的不同动量模在动量空间中是完全解耦的。在实空间中表现出长程关联的自由场，在动量空间中分解为一组独立的高斯模。因此，对自由场采样可以通过为每个动量模独立采样一个高斯变量来实现。如果我们暂时忽略实数条件施加的共轭约束，我们可以形式上将复傅里叶模写为：

$$\tilde{\phi}(k)=\phi^{R}(k)+i\phi^{I}(k).$$

那么，

$$|\tilde{\phi}(k)|^{2}=(\phi^{R}(k))^{2}+(\phi^{I}(k))^{2}.$$

因此，每个模的实部和虚部都对应于独立的高斯变量，其方差由 $\tilde{M}(k)$ 控制。给定标准正态变量 $\gamma\sim\mathcal{N}(0,1)$，我们可以形式上像这样生成相应的自由场模：

$$\phi^{R}(k)=\frac{\gamma^{R}(k)}{\sqrt{\tilde{M}(k)}}, \quad \phi^{I}(k)=\frac{\gamma^{I}(k)}{\sqrt{\tilde{M}(k)}}.$$

利用快速傅里叶变换(FFT),我们可以高效地生成自由场采样。

### 5.2.1 如何找到一个“好”的初始分布 (How to Find a "Good" Initial Distribution)

自由场预采样不仅仅是一个经验性的工程技巧，它还可以从优化的角度来解释。我们的目标不仅是选择一个随机的先验分布，而是要找到一个在满足物理结构和训练启动稳定性的同时，尽可能接近目标分布的初始分布。在训练初期，因为流网络使用小参数初始化，所以模型变换大致是一个恒等映射：

$$f_{\theta}(z)\approx z.$$

因此初始生成的场满足：

$$\phi_{init}=f_{\theta}(z)\approx z.$$

换句话说，模型在训练初期的生成分布 $q_{0}(\phi)$ 近似等于我们选择的先验分布：

$$q_{0}(\phi_{init})\approx r(\phi_{init}).$$

因此，我们如何选择先验分布实际上决定了模型在训练开始时的起点。如果我们选择一个好的先验，流网络只需要学习剩余的非高斯相互作用修正；如果我们选择一个坏的先验，网络就必须从零开始学习动能关联和长程结构，这使得训练变得困难得多。

**三个初始化原则** 我们希望我们的初始分布满足以下三个原则：
1. 初始分布应尽可能紧密地匹配目标分布中最难学习的二次动能结构；
2. 初始分布应该保持Z_2对称性，这样网络在初始时也会保持Z_2对称性，后续的训练中也不会因为初始的不对称性干扰训练
3. 在满足前两个条件的候选先验中，初始分布与目标分布之间的 KL 散度应尽可能小。

第一个原则要求先验分布包含格点动能项的结构。对于二维周期格点，动量空间中离散拉普拉斯算子的特征值为：

$$\hat{K}(k)=\sum_{\mu=1}^{2}(2-2\cos k_{\mu})=4\sin^{2}\frac{k_{1}}{2}+4\sin^{2}\frac{k_{2}}{2}.$$

其中允许的动量为：

$$k_{\mu}=\frac{2\pi n_{\mu}}{L}, \quad n_{\mu}=0,1,...,L-1.$$

所以，$\hat{K}(k)$ 不是什么额外的裸参数；它是一个完全由格点尺寸 $L$ 和周期性边界条件决定的量。

因此我们将初始分布选定具有两个自由参数的动量空间的高斯分布：$e^-(K\hat(k)+m_{free}^2)|\phi(p)-\mu|^2$

第二个原则让我们选择$\mu = 0$

第三个原则是关于在这些允许的 $m_{free}^2$ 中挑选出使 KL 散度最小的值：

$$m_{free}^2{*}=\arg \min_{m_{free}^2\in\mathcal{A}}D_{KL}(q_{\mu}||p).$$

为了找到在动量空间高斯分布下使 KL 散度最小的$m_{free}^2$,我们进行以下推导：

目标分布是玻尔曼分布：

$$p(\phi)=\frac{1}{Z}e^{-S_{target}[\phi]}$$

根据公式（前面的公式引用，暂且占位），目标作用量为：

$$S_{target}[\phi]=\sum_{x}[\phi(x)(-\Box\phi)(x)+m^{2}\phi(x)^{2}+\lambda\phi(x)^{4}].$$

它在动量空间中的二次部分为：

$$S_{quad}[\phi]=\sum_{k}[\tilde{K}(k)+m^{2}]|\tilde{\phi}(k)|^{2}.$$

KL 散度定义为：

$$D_{KL}(q_{\mu}||p)=\int \mathcal{D}\phi~q_{\mu}(\phi)\log\frac{q_{\mu}(\phi)}{p(\phi)}$$

代入 $p(\phi)=\frac{1}{Z}e^{-S_{target}[\phi]}$，我们得到：

$$D_{KL}(q_{\mu}||p)=\mathbb{E}_{q_{\mu}}[S_{target}(\phi)]-H(q_{\mu})+\log Z$$

其中

$$H(q_{\mu})=-\mathbb{E}_{q_{\mu}}[\log q_{\mu}(\phi)]$$

是自由场先验的熵。

**目标作用量的期望值**：目标作用量分为二次项和四次相互作用项：

$$S_{target}=S_{quad}+S_{int}.$$

其中

$$S_{int}=\lambda\sum_{x}\phi(x)^{4}.$$

在恒等映射初始化下，对于二次项，我们有：

$$\mathbb{E}_{q_{\mu}}[S_{quad}]=\sum_{k}[\hat{K}(k)+m^{2}]\langle|\tilde{\phi}(k)|^{2}\rangle_{q_{\mu}}$$

代入 $\langle|\tilde{\phi}(k)|^{2}\rangle_{q_{\mu}}=\frac{1}{2[\tilde{K}(k)+\mu]}$，我们得到：

$$\mathbb{E}_{q_{\mu}}[S_{quad}]=\frac{1}{2}\sum_{k}\frac{\tilde{K}(k)+m^{2}}{\hat{K}(k)+\mu}.$$

对于四次项，由于 $q_{\mu}$ 是高斯分布，我们可以使用维克定理（Wick's theorem）：

$$\langle\phi(x)^{4}\rangle_{q_{\mu}}=3\langle\phi(x)^{2}\rangle_{q_{\mu}}^{2}=3G_{\mu}(0)^{2}.$$

因此

$$\mathbb{E}_{q_{\mu}}[S_{int}]=\lambda\sum_{x}\langle\phi(x)^{4}\rangle_{q_{\mu}}=3\lambda VG_{\mu}(0)^{2}.$$

综合起来：

$$\mathbb{E}_{q_{\mu}}[S_{target}]=\frac{1}{2}\sum_{k}\frac{\tilde{K}(k)+m^{2}}{\hat{K}(k)+\mu}+3\lambda VG_{\mu}(0)^{2}.$$

**自由场先验的熵项**：自由场先验是高斯分布，其在动量空间的协方差正比于：

$$C_{\mu}(k)\propto\frac{1}{\hat{K}(k)+\mu}$$

高斯分布的熵满足：

$$H(q_{\mu})=\frac{1}{2}\log \det C_{\mu}+\text{const.}$$

因此

$$H(q_{\mu})=-\frac{1}{2}\sum_{k}\log[\hat{K}(k)+\mu]+\text{const.}$$

因此，忽略与 $\mu$ 无关的常数，KL 散度可以写为：

$$D_{KL}(\mu)=\frac{1}{2}\sum_{k}\frac{\tilde{K}(k)+m^{2}}{\hat{K}(k)+\mu}+3\lambda VG_{\mu}(0)^{2}+\frac{1}{2}\sum_{k}\log[\hat{K}(k)+\mu].$$

其中

$$G_{\mu}(0)=\frac{1}{V}\sum_{k}\frac{1}{2[\hat{K}(k)+\mu]}$$

为了找到使 KL 散度最小的自由场质量参数，我们对 $\mu=m_{free}^{2}$ 求导。
定义

$$M_{\mu}(k)=\hat{K}(k)+\mu.$$

那么

$$G_{\mu}(0)=\frac{1}{V}\sum_{k}\frac{1}{2M_{\mu}(k)}.$$

因此

$$\frac{\partial G_{\mu}(0)}{\partial\mu}=-\frac{1}{2V}\sum_{k}\frac{1}{M_{\mu}(k)^{2}}.$$

对 KL 散度求导：

$$\frac{\partial D_{KL}}{\partial\mu}=-\frac{1}{2}\sum_{k}\frac{\tilde{K}(k)+m^{2}}{M_{\mu}(k)^{2}}+6\lambda VG_{\mu}(0)\frac{\partial G_{\mu}(0)}{\partial\mu}+\frac{1}{2}\sum_{k}\frac{1}{M_{\mu}(k)}.$$

合并第一项和第三项。因为

$$\frac{1}{M_{\mu}(k)}=\frac{M_{\mu}(k)}{M_{\mu}(k)^{2}}$$

我们有

$$-\frac{1}{2}\sum_{k}\frac{\hat{K}(k)+m^{2}}{M_{\mu}(k)^{2}}+\frac{1}{2}\sum_{k}\frac{1}{M_{\mu}(k)}=\frac{1}{2}\sum_{k}\frac{M_{\mu}(k)-[\tilde{K}(k)+m^{2}]}{M_{\mu}(k)^{2}}$$

并且由于

$$M_{\mu}(k)-[\hat{K}(k)+m^{2}]=[\hat{K}(k)+\mu]-[\hat{K}(k)+m^{2}]=\mu-m^{2},$$

我们得到

$$-\frac{1}{2}\sum_{k}\frac{\tilde{K}(k)+m^{2}}{M_{\mu}(k)^{2}}+\frac{1}{2}\sum_{k}\frac{1}{M_{\mu}(k)}=\frac{1}{2}\sum_{k}\frac{\mu-m^{2}}{M_{\mu}(k)^{2}}$$

同时，由于

$$\frac{\partial G_{\mu}(0)}{\partial\mu}=-\frac{1}{2V}\sum_{k}\frac{1}{M_{\mu}(k)^{2}}$$

所以

$$\sum_{k}\frac{1}{M_{\mu}(k)^{2}}=-2V\frac{\partial G_{\mu}(0)}{\partial\mu}.$$

因此

$$\frac{\partial D_{KL}}{\partial\mu}=V\frac{\partial G_{\mu}(0)}{\partial\mu}[m^{2}-\mu+6\lambda G_{\mu}(0)].$$

如果 KL 极小值位于允许集合 $\mathcal{A}$ 的内部，它必须满足：

$$\frac{\partial D_{KL}}{\partial\mu}=0.$$

由于 $\frac{\partial G_{\mu}(0)}{\partial\mu}\ne0$，我们得到方程：

$$m^{2}-\mu+6\lambda G_{\mu}(0)=0.$$

这意味着

$$\mu=m^{2}+6\lambda G_{\mu}(0).$$

换回 $\mu=m_{free}^{2}$，最佳自由场质量参数满足：

$$m_{free}^{2}=m^{2}+6\lambda G(0).$$

其中

$$G(0)=\frac{1}{V}\sum_{k}\frac{1}{2[\hat{K}(k)+m_{free}^{2}]}.$$

因此，第一和第二原则将初始先验分布限制在一组动量空间高斯分布中；第三原则通过最小化初始 KL 散度从该族中找到最佳质量参数。

### 5.2.4 $L=14$ 的数值解 (Numerical Solution for $L=14$)

对于：

$$L=14, \quad m^{2}=-4, \quad \lambda=5.113$$

格点体积为

$$V=L^{2}=196$$

现在我们需要求解

$$\mu=-4+6\times5.113\times\frac{1}{196}\sum_{n_{1}=0}^{13}\sum_{n_{2}=0}^{13}\frac{1}{2[\hat{K}(n_{1},n_{2})+\mu]}.$$

其中

$$\hat{K}(n_{1},n_{2})=4\sin^{2}\left(\frac{\pi n_{1}}{14}\right)+4\sin^{2}\left(\frac{\pi n_{2}}{14}\right)$$

通过数值求解该方程，我们得到：

$$\mu_{*}=m_{free,*}^{2}\approx0.6005269985.$$

对应的零距离两点函数为：

$$G(0)\approx0.1499617641.$$

验证自洽方程的右侧：

$$-4+6\times5.113\times0.1499617641\approx0.6005269985.$$

因此，对于参数集 $L=14$，$m^{2}=-4$，$\lambda=5.113$，在我们当前的作用量归一化约定下，在 KL 意义上的最佳自由场先验质量参数为：

$$m_{free,*}^{2}\approx0.6005.$$

### 5.2.5 不同初始分布下的梯度下降过程与接受率对比

根据我们的计算，当 $m_{free,*}^{2}\approx0.6005$ 时，初始阶段的 KL 散度应该比选择任何其他 $m_{free}$ 都小。我们对比了不同初始分布下的loss下降历史与MH接受率：



## 5.3 $\mathbb{Z}_{2}$ 对称性的引入 (Introduction of $\mathbb{Z}_{2}$ Symmetry)

二维 $\phi^{4}$ 理论具有明显的全局 $\mathbb{Z}_{2}$ 对称性，由以下变换定义：

$$\phi(x)\rightarrow-\phi(x).$$

因为目标作用量只包含场的偶数次幂，即 $S[\phi]=S[-\phi]$，所以目标概率分布自然满足：

$$p(\phi)=p(-\phi).$$

对于一个理想的生成模型，我们期望生成的分布也满足：

$$q_{\theta}(\phi)=q_{\theta}(-\phi).$$

如果模型未能完全学会这种对称性，训练后的分布可能会偏向正值或者负值。

最直接的解决方法是将这种对称性硬编码到网络架构中，强迫生成映射成为一个奇函数：

$$f_{\theta}(-z)=-f_{\theta}(z).$$

如果先验分布本身是对称的，即 $r(z)=r(-z)$，并且映射满足这种奇函数约束，那么生成的分布将自然满足 $q_{\theta}(\phi)=q_{\theta}(-\phi)$。
然而，在 Real NVP 风格的仿射耦合层中，变换通常写为：

$$\phi_{b}^{\prime}=(\phi_{b}-t_{\theta}(\phi_{a}))\exp[-s_{\theta}(\phi_{a})].$$

为了使整个映射满足奇函数性质，我们必须对函数 $s_{\theta}$ 和 $t_{\theta}$ 施加严格的奇偶性约束。如果输入被全局取反（$\phi_{a}\rightarrow-\phi_{a},\phi_{b}\rightarrow-\phi_{b}$），我们要求输出也全局取反。这意味着缩放函数 $s_{\theta}$ 必须是一个偶函数：

$$s_{\theta}(-\phi_{a})=s_{\theta}(\phi_{a}),$$

而平移函数 $t_{\theta}$ 必须是一个奇函数：

$$t_{\theta}(-\phi_{a})=-t_{\theta}(\phi_{a}).$$

只有这样，仿射耦合变换才能在全局满足 $f_{\theta}(-\phi)=-f_{\theta}(\phi)$。

将这些奇偶性质强加给真正的神经网络会带来巨大的惩罚。为了使 $s_{\theta}$ 成为偶函数，通常的技巧是让它依赖于 $\phi_{a}^{2}$ 或其他偶函数特征。为了使 $t_{\theta}$ 成为奇函数，你需要约束网络架构或使用特定的激活函数组合（如 tanh）。我们尝试通过激进的通道分裂和通过特定激活函数手动设计奇/偶函数来硬编码 $\mathbb{Z}_{2}$ 对称性，但训练结果非常糟糕。我们遇到的主要问题是：

1. 网络表达能力急剧下降；
2. 缩放和平移函数可用的自由度受到严重限制；
3. 早期训练阶段的梯度传播恶化；
4. 模型无法灵活地修正自由场先验与真实目标分布之间的差异。

因此，尽管硬编码 $\mathbb{Z}_{2}$ 对称性在纸面上看起来很优雅，但在我们现有的模型规模和训练设置下，表达能力的损失远远超过了对称性约束带来的任何好处。

一种更温和的替代方案是使用**软惩罚（Soft Penalty）**。我们可以不改动网络架构，而是向损失函数中添加一个与全局磁化强度（global magnetization）相关的额外惩罚项。将单个配置的平均场（或磁化强度）定义为：

$$\overline{\phi}=\frac{1}{V}\sum_{x}\phi(x)$$

我们可以添加以下对称性惩罚项：

$$\mathcal{L}_{sym}=V\lambda_{sym}\left(\frac{1}{V}\sum_{x}\phi(x)\right)^{2}=V\lambda_{sym}\overline{\phi}^{2}.$$

该惩罚项将每个独立生成的配置的平均场推向接近于零，试图抑制模型偏向特定磁化方向的倾向。总损失随后变为：

$$\mathcal{L}_{total}=\mathcal{L}_{KL}+\mathcal{L}_{sym}.$$

然而，我们的实验结果清楚地表明，这种软惩罚方法也是非常次优的，并且会直接损害训练稳定性。为了定量评估软 $\mathbb{Z}_{2}$ 惩罚的影响，我们在训练期间追踪了马尔可夫链蒙特卡洛（MCMC）的接受率和平均场的期望值。从信息的角度来看，我们的自由场先验通过注入目标分布的部分信息显着提高了训练效率。这表明简单地添加软 $\mathbb{Z}_{2}$ 惩罚并不能为模型提供足够的、有效的、可用的结构信息。与直接改变输入分布并预编码动能结构的自由场先验相比，软惩罚仅仅改变了优化目标，并且很容易与主要的 KL 任务发生竞争。因此，在我们当前的实验设置下，它并没有带来预期的收益。展望未来，我们可能需要探索其他注入 $\mathbb{Z}_{2}$ 对称性信息的方法，也许可以通过研究目标分布满足的其他方程来寻找更容易被网络学习的对称性约束。

我们追踪了实际的 MCMC 接受率以及场幂次 $\langle\phi^{1}\rangle$ 到 $\langle\phi^{5}\rangle$ 的期望值。结果汇总在表 1 中。

**表 1：不同训练里程碑处的 MCMC 接受率和场幂次期望值（无显式强化 $\mathbb{Z}_{2}$ 对称性）**

| L    | m²   | lam   | m_free²  |
| ---- | ---- | ----- | -------- |
| 6    | -4   | 6.975 | 1.118882 |
| 8    | -4   | 6.008 | 0.849136 |
| 10   | -4   | 5.55  | 0.721314 |
| 12   | -4   | 5.276 | 0.645392 |
| 14   | -4   | 5.113 | 0.600527 |

---

# 6 物理观测量与自相关时间的验证，以及基于流的模型与传统算法之间的计算开销对比 (Verification of Physical Observables and Autocorrelation Time, and Computational Overhead Comparison between Flow-based and Traditional Algorithms)

## 6.1 物理观测量 (Physical Observables)

## 6.2 自相关时间 (Autocorrelation Time)

## 6.3 计算开销对比 (Computational Overhead Comparison)

# 7 总结 (Summary)

# 参考文献 (References)