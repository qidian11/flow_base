absract: 我们分析了**Flow-based generative models for Markov chain Monte Carlo in lattice field theory**这篇论文中关于的针对φ4理论的**Flow-based generative models**，验证了作者对于Flow-based generative models ** for Markov chain Monte Carlo**和传统方法HMC和local metropolis自相关时间的对比和相关物理观测量的计算；并且我们仔细对比了Flow-based generative models和HMC两种方法采样的计算量。我们还基于原论文中Flow-based generative models从标准高斯分布到φ4相互作用场分布映射的采样方法，提出了基于自由场分布到φ4相互作用场分布映射的采样方法，避免了Flow-based generative models前期对于φ4场中的动能项学习的局限性，提升了模型的学习效率。此外我们对该论文中的模型还尝试引入了平移等变形和z_2对称性，比较了引入物理先验后的模型与原模型的采样效率。

introduction:在LQCD中传统的采样方法比如仍然存在一些限制，比如：临界慢化和拓扑冻结。临界减速指的是当参数空间接近临界点时，相关自相关时间的发散。这种行为会显著增加这些参数区域内模拟的计算成本 []。本文研究的Flow-based generative models for Markov chain Monte Carlo method[]，



MCMC(Markov Chain Monte Carlo)算法

Monte Carlo

在物理学计算中，我们通常需要计算某个可观测量$O(\phi)$在特定分布$p(\phi)$下的期望值：

$$\langle O \rangle = \int O(\phi) p(\phi) D\phi$$

$\phi$ 通常是高维空间中的一点，在高维情况下，由于维度限制，通常直接求积分是很困难的。蒙特卡罗方法提出，如果我们能在$p(\phi)$ 分布下，抽取独立的$N$个样本$\phi_i$，那么随着N的增大，我们可以近似得到：

$$\langle O \rangle \approx \frac{1}{N} \sum_{i=1}^{N} O(\phi_i)$$



Markov Chain

而从目标分布中抽取样本正是Markov Chain的工作，具体来说，Markov Chain通过构造一个状态序列$\phi_1 \to \phi_2 \to \dots \to \phi_n$。在这个序列中，下一步状态$\phi_{t+1}$出现的概率只依赖于当前状态$\phi_{t}$,我们称为转移概率$T(\phi' | \phi)$，对任意$\phi$ 和$\phi'$，$T(\phi' | \phi) \neq 0$，也即任意两个状态之间都有概率相互转移

具体来说假设当前系统处于各个状态的概率分布为 $P_t(\phi)$，经过一次转移概率 $T(\phi' | \phi)$ 的演化后，第 $t+1$ 步处于状态 $\phi'$ 的概率 $P_{t+1}(\phi')$，等于从**所有可能的上一状态 $\phi$** 转移过来的概率之和：

$$P_{t+1}(\phi') = \int P_t(\phi) T(\phi' | \phi) d\phi$$

而当系统处于平稳状态时，即系统在每一步的概率分布都相等，$P_t(\phi)$=$P_{t+1}(\phi)$=$P(\phi)$，此时：

$$P(\phi') = \int P(\phi) T(\phi' | \phi) d\phi$$

为了让这个序列链最终能收敛到我们想要的状态分布$p(\phi)$，状态之间的转移概率，我们需要人为构造细致平衡条件：

$$p(\phi) T(\phi' | \phi) = p(\phi') T(\phi | \phi')$$

将此式左右两边同时对$\phi$ 积分我们就能得到平稳状态下的分布



MH 算法（Metropolis-Hastings）

Metropolis-Hastings算法将转移概率$T(\phi' | \phi)$拆分成了两部分：

$$T(\phi' | \phi) = g(\phi' | \phi) \times A(\phi' | \phi)$$

提议分布 $g(\phi' | \phi)$和接受概率 $A(\phi' | \phi)$，

提议分布 $g(\phi' | \phi)$是由我们自己确定的，例如我们可以等概率的随机游走，此时$g(\phi' | \phi) = g(\phi | \phi')$.

对于接受概率 $A(\phi' | \phi)$,MH 算法巧妙地设计了：

$$A(\phi' | \phi) = \min\left(1, \frac{p^*(\phi') g(\phi | \phi')}{p^*(\phi) g(\phi' | \phi)}\right)$$

带入细致平衡条件： 左边：$p^*(\phi) T(\phi' | \phi) = p^*(\phi) g(\phi' | \phi) A(\phi' | \phi)$  右边：$p^*(\phi') T(\phi | \phi') = p^*(\phi') g(\phi | \phi') A(\phi | \phi')$

将A(\phi' | \phi)带入之后我们会发现，左边完美等于右边，完美满足细致平衡条件