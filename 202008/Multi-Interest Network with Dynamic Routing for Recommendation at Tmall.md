# Multi-Interest Network with Dynamic Routing for Recommendation at Tmall

cikm 2019 paper [arxiv](https://arxiv.org/pdf/1904.08030.pdf)



## ABSTRACT
推荐系统有召回和排序两阶段，建模用户兴趣，在两个阶段都很重要。现在深度模型中刻画用户兴趣，都是用一个向量表示，这不能捕获用户不断变化的兴趣。在论文中，我们用多个向量表示用户兴趣，来描述不同方面的用户兴趣。
提出了MIND方法，来处理召回阶段用户不同的兴趣。这是基于Capsule routing的方法，这可以聚集用户行为，并提取用户兴趣。
还发展了 label-aware attention 来帮助学习用户多元表示向量。在公开数据集，和Tmall数据上测试，MIND效果很显著。MIND已经部署在了Tmall APP的主页上

## Note

### 模型图
主要的大体模型如图
![](https://raw.githubusercontent.com/celia01/papernotes/master/202009/pic/1.png)
中间收集feature向量，没有直接使用concat或attention聚合成1个向量，而是使用的capsule，变成多个向量。
因为最后会有



### Bloom Embedding
x是一个d为的one-hot向量，$x=[x_1, ..., x_d], x_i\in\{0,1\} $，实际上非零元素集合为 $p=\{p_i\}_{i=1}^c, p_i\in N_{\le d}$，c是非零元素的个数，数量远远小于d，  
最后需要映射到 $u=[u_1,...,u_m], u_i\in \{0,1\}$  
有k个相互独立的hash函数 $H_j, j=1,...,k$，设置 $u_{H_j(p_i)}=1$



### 具体实现

  
###  pairwise方式


### 其他讨论



## Comment


## Other
[code](https://github.com/geek-ai/irgan)

[只用生成器能否打败IRGAN? 读完Code再读论文的一些体会](https://medium.com/@yaoyaowd/%E5%8F%AA%E7%94%A8%E7%94%9F%E6%88%90%E5%99%A8%E8%83%BD%E5%90%A6%E6%89%93%E8%B4%A5irgan-%E8%AF%BB%E5%AE%8Ccode%E5%86%8D%E8%AF%BB%E8%AE%BA%E6%96%87%E7%9A%84%E4%B8%80%E4%BA%9B%E4%BD%93%E4%BC%9A-4b3f7c29b477)



