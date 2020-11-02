# Behavior Sequence Transformer for E-commerce Recommendation in Alibaba
kdd 2019 paper [arxiv](https://arxiv.org/pdf/1905.06874.pdf)



## ABSTRACT
深度学习已经广泛应用在了推荐系统中，之前的工作主要是利用了Embedding和MLP：一系列特征会通过embedding变成低维度的向量，接着低维向量经过MLP，作为最终的推荐。但是，大部分工作都只是concat了不同特征，忽略了用户行为的有序性。这篇paper中，我们使用了powerful的transformer去刻画用户在alibaba中的行为序列。相比淘宝目前部署在线上的两组baseline，大大提升了效果。

## Note

### 模型图
这是在淘宝搜索中的模型，主要的大体模型如图
![](https://raw.githubusercontent.com/celia01/papernotes/master/202009/pic/1.png)
模型没有特别大的创新，主要是套了一个transformer，主要的序列是用户的点击行为序列，包含点击序列中item自身特征+position id
position id。

### position 建模
使用 $pos(v_i)=t(v_t)-t(v_i)$表示position value，$v_t$表示当前item，$v_i$表示历史点击中的item。  
$t(v_t)$表示推荐当前item $v_t$的时间（即曝光时间），$t(v_i)$表示用户历史行为中点击$v_i$的时间
相当于每个item的position value表示点击该item时的时间与当前曝光时间的时间差

### transformer layer & loss
主要就是普通的带self-attention的transformer，还有常规的多任务交叉熵loss

### 对比的baseline 和 指标
主要对比了 WDL，WDL(+Seq)，DIN三个模型，只用的 Offline AUC 和 Online CTR作为指标。
同时也比较了多stack attention的方式，发现最后只是用一层就已经达到了不错的效果




## Comment


## Other
[参考code](https://github.com/wziji/deep_ctr/tree/master/BST)
<!--[论文翻译](http://d0evi1.com/mind/)-->



