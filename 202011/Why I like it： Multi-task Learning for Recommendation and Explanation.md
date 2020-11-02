# Why I like it: Multi-task Learning for Recommendation and Explanation
recys 2018 paper [arxiv](https://dl.acm.org/doi/pdf/10.1145/3240323.3240365)



## ABSTRACT
文中提出了一个新颖的多任务推荐模型，它可以同时输出 用户评分预测，和可解释推荐。
对于推荐模型采用的是矩阵分解，对于评分预测任务，采用的是seq2seq的对抗生成模型
最后在真实的数据集上采用这个方法进行评分预测，对于sota的方法，都显著提升

## Note

### 模型图
这是整体的模型图
![](https://raw.githubusercontent.com/celia01/papernotes/master/202011/pic/1.png)


### 1 Adversarial Sequence-to-Sequence Learning 对抗的seq2seq模型
对于评论数据集，会有用户 $user$ 和物品 $item$，比如，用户$user$针对物品$item$可能会产生review doc  

* user reivew document：指用户$i$ 的所有评论过的文档集合是 $d_{u,i}$   
* item review document：指物品 $j$的所有被评价的文档集合是$d_{v,j}$  
* 如果 $d_{u,i} \cap d_{v,j} \neq \emptyset $，表示用户$i$评论过物品$j$  

我们任务一个用户的评论文档，可以表示一个用户的某些preference，比如用户评论中经常出现"价格"，说明用户购买时很容易受到价格影响，如果用户评论中经常出现"低质"，说明用户很关注item质量问题。所以我们采用了seq2seq的encoder-decoder模型来表示用户和商品，用户和商品，采用的是类似的模型，下面主要对用户模型进行举例描述

#### 1.1 Recurrent Review Generator.递归评论生成器
生成器主要包含Encdoer和Decoder两部分  

* Encoder部分：  
	1. 对于user $i$的每一条评论 $d_{u,i}$：我们可以获得这条评论的所有word$(w_1,w_2,...,w_T)$,把这些word喂入一个双向GRU中，最终取出Bi-GRU两端的向量concat $h_T=[\overrightarrow{h_T},\overleftarrow{h_T}]$ 组成输出 ，得到了评论文档$d_{u,i}$级别的向量$h_T$；  其中词向量会有word2vec预训练得到，最后在训练模型中继续finetune.  
	2. 把用户$i$评论过的所有文档$d_{u,i}$的向量ave-pooling，就可以得到用户级别的隐向量 $\widetilde{U}_i$，即模型图中 User i textual feature  

* Decoder部分：  
<center>$P(y_{i,1},...,y_{i,T^{'}}|\widetilde{U}_i)=\sum_{t=1}^{T^{'}}p(y_{i,t}|\widetilde{U}_i,y_{i,1},...,y_{i,t-1})$</center>   
用户根据用户text feature $\widetilde{U}_i$，和已经已知的前序评论词，预测用户的评论，这里使用的是单向GRU，逐字逐句地生成评论。其中$y_{i,0}$都是SOS token，$h_0$都是0向量，末尾都是EOS token表示评论结束

#### 1.2 Convolutional Review Discriminator 卷积评论判别器
判别器主要使用卷积网络实现，主要功能是区分真假评论：给定特定的评论和用户，判别器会判断这条评论是否是这个用户撰写的。其实判别器学习到的有两方面： 
 
1. 主要是这条评论是否由人类撰写，或是由机器生产，如果这条评论的撰写方式一般不会出现在人工评论中，那么就很可能被判断为假评论；  
2. 加入用户信息，判断这条评论是不是由特定的用户撰写的，这就保证了生成器不仅仅要产出人类可读的评论，还需要生产出和目标用户有关联的评论  

这里使用了cnn的模型，主要如图
![](https://raw.githubusercontent.com/celia01/papernotes/master/202011/pic/2.png)  
主要参考的就是cnn用于文本分类的经典之作《Convolutional neural networks for sentence classification》中的模型，不过这儿额外加入了query embedding，表示用户的embedding信息  ，
word embedding部分主要是特定评论中的word集合，是使用word2vec预训练的向量+finetune
query embedding是只用户的特定向量，是在模型中和其他参数一起学习得到的  
卷积核同时使用了，1-5 gram

#### 1.3 Adversarial Training for Review Generation with REINFORCE 对抗训练
这儿使用GAN模型与《Adversarial learning for neural dialogue generation》中的模型和类似，在自动对话机器人学习中区分机器还是人工生成的句子，使用了生成对抗的框架进行联合训练，目标函数$J(\theta)$  
<center>$max_\phi E_{Y\sim{p_{data}}} [\log{D_\phi(Y)}]+E_{Y_{'}\sim{G_{\theta}}}[\log{(1-D_\phi(Y^{'}))}]$</center>  
$Y\sim{p_{data}}$是ground-truth，$Y^{'}\sim{G_{\theta}}$是生成器生成的分布，这块用了gan框架常见的训练方式  
值得一说的是，这儿最终的$L(\theta)$并不只是$-J(\theta)$，而是$L(\theta)=-J(\theta)+\lambda{||U-\widetilde U||}_F$，这里的$\widetilde U$是生成模型autoendoce中的隐向量，而$U$是评分矩阵中矩阵分解得到的的user向量，$\lambda$是控制 协同过滤和gan，两个模型之间重要性的超参数，具体的协同过滤-矩阵分解模型，会在下一部分详细介绍

### 2 Generating Personalized Explanation 生成个性化的可解释性
从上一部分可以知道，通过$d_{u,i}$和$d_{v,j}$可以得到user $i$和item $j$的隐向量表示 $\widetilde U_i$和$\widetilde V_j$。如果将 $\widetilde U_i$和$\widetilde V_j$拼接起来，喂入decoder，也可以生成特定user-item组合的评论。

### 3 Context-aware Matrix Factorization for Rating Prediction 评分预测时的上下文形式的矩阵分解
这儿矩阵分解使用的是PMF model（本质也是会有latent model）  
<center>$p(R|U,V,\sigma^2)=\prod_{i=1}^N{ \prod_{j=1}^M [\mathcal N(R_{ij}|u_i^T·v_j,\sigma^2)]^{I_{ij}}}$</center>
继续拆分，假设$U$,$V$的先验分布就是之前得到的textual features，则$U$和$V$的先验分布如下：  
$p(U|\widetilde U, \sigma_U^2)=\prod \mathcal N(U_i|\widetilde U_i,\sigma_U^2I)$  
$p(V|\widetilde V,\sigma_V^2)=\prod \mathcal N(V_j|\widetilde V_j,\sigma_V^2I)$  
最后目标是要 $\max_{U,V} {p(U,V|R)}$，也就是
<center>$\max_{U,V} {p(U,V,R|\widetilde U, \widetilde V, \sigma^2, \sigma_U^2, \sigma_V^2)}$</center>



### 4 Optimization Methodology 算法优化
根据最大似然MLE和最大后验MAP的关系 
<center>$\max p(U,V,R)=\max p(R|U,V)*P(U,V)$</center>
所以，最终目标是求 
<center>$\max p(U,V,R)=\max_{U,V}p(R|U,V,\sigma^2)p(U,V|\widetilde U,\widetilde V,\sigma_U^2, \sigma_V^2)=\max_{U,V}[p(R|U,V,\sigma^2)p(U|\widetilde U,\sigma_U^2)p(V|\widetilde V,\sigma_V^2)]$</center>  
这儿也是用了经典的EM算法求解 $\widetilde U$和$\widetilde V$, 假定他们先验分布服从正态分布.  
最后等价于求解 
<center>$\mathcal L(U,V|R,\widetilde U, \widetilde V)=\frac12\sum_i^N\sum_j^MI_{ij}(R_{ij}-U_i^TV_j)^2+\frac{\lambda_U}{2}||U-\widetilde U||_F^2+\frac{\lambda_V}{2}||V-\widetilde V||_F^2$</center>  
可以使用最小二乘法（ALS）依次优化$U,V$其中一个：  
${U_i=(VI_iV^T+\lambda_UI_K)^{-1}(VR_i+\lambda_U\widetilde U_i)}$  
${V_j=(UI_jU^T+\lambda_VI_K)^{-1}(UR_j+\lambda_V\widetilde V_j)}$  

在训练过程中，用到了teacher focing方法，就是如果直接利用RL来训练gen model，会非常不稳定，因为通过reward来返回gen model生成的句子，不能得到直接的反馈，gen model知道自己生成的句子很烂，但不知道好的生成应该是怎么样的，dis model并没有告诉gen model。为了缓和这种情况，可以使用teacher forcing机制：  

1. 在gen model更新时，不仅惩罚差的生成句子，也等比例的喂入真实的句子，放入gen model中，而真实的句子对应的reward=1（ps，这个很正常）;  
2. dis model对真实的句子打分，当reward超过了一定阈值后，再更新gen model
这里的teacher focing，其实很类似强化学习学习中很常见的，先使用监督学习的方式去预训练模型。这里的话就是说gen model训练时，刚开始会使用一部分的ground turth掺杂进入gen model的训练中，这部分真实的正例的reward直接为1，到后面开始全都用gen model送入dis model的样本。但对于某个单一模型，其实teacher forcing技术也很容易exposure bias。如果放入在gan框架中，因为有两个模型，会有所缓解  

最终模型如下  
![](https://raw.githubusercontent.com/celia01/papernotes/master/202011/pic/3.png)  


### 对比的baseline 和 指标
在**rating prediction**任务中主要对比了： 
  
* PMF: Probabilistic Matrix Factorization (PMF)   
* HFT:HiddenFactorasTopics(HFT)  
* CTR: Collaborative Topic Regression (CTR)   
* JMARS: Jointly Modeling Aspects, Ratings, and Sentiments (JMARS)   
* ConvMF+:ConvolutionalMatrixFactorization(ConvMF)

都是比较简单的浅层算法，论文中提出的算法，主要有三个，MT，MT-encoder（只有encoder），MT-decoder（只有decoder），MT都是表现最好的。  
![](https://raw.githubusercontent.com/celia01/papernotes/master/202011/pic/4.png)  
ps，其实整体框架很类似SDNE中的那种，结构对称性有个autoencoder，保证rating的合理性有个MF


### 解释指标
这块关键在如何评估模型的解释质量。其实也有点类似于用户满意度，需要用户调研，才能得到比较好的评估对比。
这里的可解释性，主要是生成的评论的可解释性。这里使用了 困惑度指标（perplexity metric ）来评估解释质量，使用tf-idf（tf-idf similarity metric）来评估生成的评论和真实评论之间的相关性. 
  
* 困惑度：句子概率越大，语言模型越好，迷惑度越小（ps，这个困惑度只能近似表示可解释性，最直接的还是要用户问卷等）  
* tf-idf：对文档中的词语使用tf-idf后表示评论，最后用评论向量相似性，来衡量真假  

最后效果如下，MT-P 困惑度最小，相似性最高
![](https://raw.githubusercontent.com/celia01/papernotes/master/202011/pic/5.png)  


## Comment


## Other
[使用卷积神经网络进行文本分类](http://www.o9z.net/kj/191124063744887.html)  
[关于Teacher Forcing 和Exposure Bias的碎碎念](https://zhuanlan.zhihu.com/p/93030328)  
[通俗解释困惑度 (Perplexity)-评价语言模型的好坏](https://zhuanlan.zhihu.com/p/44107044)  
<!--[参考code](https://github.com/wziji/deep_ctr/tree/master/BST)-->
[我为什么喜欢它？带有解释的推荐系统第二弹](https://mp.weixin.qq.com/s/T-lBZtW0TxUbxH4XruGnpw)



