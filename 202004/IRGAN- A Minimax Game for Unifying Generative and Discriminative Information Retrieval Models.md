# IRGAN: A Minimax Game for Unifying Generative and Discriminative Information Retrieval Models

SIGIR 2017 paper [arxiv](https://arxiv.org/pdf/1705.10513.pdf)



## ABSTRACT
文中对两种信息检索中的建模方式进行了统一讨论：1. 生成模型：给出query，输出相关的doc；2. 判别模型：给定query和doc，输出两者相关性。文中提出了一种一种博弈论中的 ***极小极大博弈*** 来迭代优化两个模型。一方面，判别模型可以从标记和未标记的数据中挖掘信号，从而为训练生成模型提供指导，使生成模型在给定query的情况下，更好去学习潜在相关的doc分布；另一方面，生成模型可以作为作为当前判别模型的attacker，通过最小化判别模型的目标，用对抗的方式生成很难分别的样本。通过两种模型相互竞争，可以证明，这种统一框架可以利用两种思想流派：(i)生成模型 通过判别模型给过来信号，可以更好的学习给定query下doc的相关性分布；(ii)判别模型 通过生成模型select的未标记doc，可以探索更多的未标记数据，从而更好的对doc进行rank估计。文中的实验结果，在web搜索，item推荐，问答等应用上，相比strong baseline，有显著提升了23.96% on Precision@5 和 15.50% on MAP，

## Note

### 整体算法
* 最终目标  
(其中，$\phi$是判别模型参数，$\theta$是生成模型参数)
![object](https://raw.githubusercontent.com/celia01/papernotes/master/202004/pic/1.png)
![discriminative_func](https://raw.githubusercontent.com/celia01/papernotes/master/202004/pic/2.png)

* 锁定生成模型时，判别模型目标
![d_obj](https://raw.githubusercontent.com/celia01/papernotes/master/202004/pic/3.png)

* 锁定判别模型时，生成模型目标
![g_obj](https://raw.githubusercontent.com/celia01/papernotes/master/202004/pic/4.png)

* 训练流程
![algorithm](https://raw.githubusercontent.com/celia01/papernotes/master/202004/pic/6.png)
解释：  
1：初始化参数  
2：生成模型和判别模型各自用训练数据预训练  
3-7：固定判别模型，训练生成模型(会用到RL中的policy gradient)  
8-11：固定生成模型，训练判别模型 

### RL的思想
（用RL的思想学习生成模型中参数）  

* 连续分布的目标与求导
(推导中$\log(1 − D_\phi(G_\theta(z)))$会用$\log D_\phi(G_\theta(z))$代替，NIPS2014中有提到这个近似方法，但在WGAN中证明了，两者不是完全等价的，按一般都这样，下同)
 ![g_obj](https://raw.githubusercontent.com/celia01/papernotes/master/202004/pic/12.png)
 ![g_obj](https://raw.githubusercontent.com/celia01/papernotes/master/202004/pic/13.png)
 ![g_obj](https://raw.githubusercontent.com/celia01/papernotes/master/202004/pic/10.png)
 这里关键是如何求 $\nabla_{G_\theta(z)}f_\phi(G_\theta(z))$这个梯度，对生成模型的分布求梯度，只能用于连续的图像和音频数据中，离散的分布，这个梯度无法求解
  
* 离散分布的目标与求导
 ![d_obj](https://raw.githubusercontent.com/celia01/papernotes/master/202004/pic/4.png)
 ![g_obj](https://raw.githubusercontent.com/celia01/papernotes/master/202004/pic/5.png)
 这里求梯度，要使用RL的思想来求，可以把当做 $V(d,q_n)\equiv log \sigma(f_\phi(d,q_n)) $ 价值函数(value function)，即reward
 ![g_obj](https://raw.githubusercontent.com/celia01/papernotes/master/202004/pic/11.png)
 然而，把 $V(d,q_n)\equiv log \sigma(f_\phi(d,q_n)) $  当做价值函数是有问题的，因为在训练初期，判别模型对于生成模型生成的样本，会给予一个很低的得分，所以$\sigma(f_\phi(d,q_n))$会趋于0，这样reward会趋于负无穷，负无穷有大的值，会导致梯度爆炸的问题，但其实在连续分布梯度中就没有这个问题，因为 $\sigma(f_\phi(G_\theta(z)))$趋于0时，梯度函数的系数 $(1-\sigma(f_\phi(G_\theta(z))))$是趋于1的（主要是log函数在连续型分布的梯度中已经拆解了）

* 为什么要用RL的思想求解（在文章appendix中有讨论）  
在NIPS2014：Generative Adversarial Nets 这篇paper中有提到对于连续型分布的GAN，但在离散型分布中梯度不可导，所以用了RL的方法。而$V(d,q_n)\equiv log \sigma(f_\phi(d,q_n)) $会有梯度爆炸的问题，所以具体实现中奖励会设计成$V(d,q_n)\equiv \sigma(f_\phi(d,q_n)) $，这篇论文实际中设计的是 $V(d,q_n)\equiv 2\sigma(f_\phi(d,q_n))-1 $，这里减去的1，相当于是policy gradient中的baseline，是为了降低variance的

### 具体实现
这篇paper在github上有具体的code，以下是截取了item_recommendation中的code  

* 判别模型
 ![g_obj](https://raw.githubusercontent.com/celia01/papernotes/master/202004/pic/7.png)

* 生成模型
![g_obj](https://raw.githubusercontent.com/celia01/papernotes/master/202004/pic/8.png)

* 联合训练
 ![g_obj](https://raw.githubusercontent.com/celia01/papernotes/master/202004/pic/9.png)



## Comment
在信息检索中使用GAN，主要是在离散分布中使用gan，文末附录，对比了在连续空间和离散空间比较。

1、提出了minimax在信息检索中的frame  
2、拓展了pairwise形式的训练样本（pairwise任务在信息检索中很常见）  
3、配有相关code  


## Other
[code](https://github.com/geek-ai/irgan)

[只用生成器能否打败IRGAN? 读完Code再读论文的一些体会](https://medium.com/@yaoyaowd/%E5%8F%AA%E7%94%A8%E7%94%9F%E6%88%90%E5%99%A8%E8%83%BD%E5%90%A6%E6%89%93%E8%B4%A5irgan-%E8%AF%BB%E5%AE%8Ccode%E5%86%8D%E8%AF%BB%E8%AE%BA%E6%96%87%E7%9A%84%E4%B8%80%E4%BA%9B%E4%BD%93%E4%BC%9A-4b3f7c29b477)



