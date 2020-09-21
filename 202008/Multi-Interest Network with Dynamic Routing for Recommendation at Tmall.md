# Multi-Interest Network with Dynamic Routing for Recommendation at Tmall

cikm 2019 paper [arxiv](https://arxiv.org/pdf/1904.08030.pdf)



## ABSTRACT
推荐系统有召回和排序两阶段，建模用户兴趣，在两个阶段都很重要。现在深度模型中刻画用户兴趣，都是用一个向量表示，这不能捕获用户不断变化的兴趣。在论文中，我们用多个向量表示用户兴趣，来描述不同方面的用户兴趣。
提出了MIND方法，来处理召回阶段用户不同的兴趣。这是基于Capsule routing的方法，这可以聚集用户行为，并提取用户兴趣。
还发展了 label-aware attention 来帮助学习用户多元表示向量。在公开数据集，和Tmall数据上测试，MIND效果很显著。MIND已经部署在了Tmall APP的主页上

## Note

### 模型图
主要的大体模型如图
![](https://raw.githubusercontent.com/celia01/papernotes/master/202008/pic/1.png)
中间收集feature向量，没有直接使用concat或attention聚合成1个向量，而是使用的capsule，变成多个向量。
因为最后会有多个向量，最后会把label和多个聚类向量，做attention，求加权和，最为用户最终向量。  
用户向量 $\overrightarrow {v}_i^k$  
用户向量组 $V_u=(\overrightarrow {v}_u^1, ..., \overrightarrow {v}_u^K)$, K是聚类capsule的维度
item向量，item就是label，$\overrightarrow {e}_i$  
线上求topN时，对用户 u 和物品 i 打分，$f_{score}(V_u, \overrightarrow {e}_i)=\max \limits_{1\le k \le K} \overrightarrow {e}_i^T, \overrightarrow {v}_u^k,$，即求一个最大的得分



## Comment


## Other
<!--[code](https://github.com/geek-ai/irgan)-->
[论文翻译](http://d0evi1.com/mind/)



