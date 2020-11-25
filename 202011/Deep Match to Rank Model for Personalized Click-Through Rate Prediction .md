# Deep Match to Rank Model for Personalized Click-Through Rate Prediction

aaai 2020 paper [arxiv](https://arxiv.org/pdf/2008.02974.pdf)
这是阿里的一篇paper


## ABSTRACT
主要是一般推荐系统都分召回和排序两个阶段，大部分ctr任务，都主要focus在用户表示上，不会太注意用户和item之间的相关性（这个相关系主要表示了用户对item的倾向程度）。所以这儿提出了Deep Match to Rank（DMR），在DMR中，会直接设置 User-Item网络和Item-Item网络，两种形式表示相关性。

* User-Item的网络中，使用内积来表示user和item的相关性。同时建立了一个辅助的match网络来监督训练过程，让内部的产品变现相关性更大
* Item-Item的网络中，通过attention机制计算了用户交互的item和目标item的相似度，最后求和得到另一种形式的user-item的相关性

最后大量的实验等等，都表现得好

## Note
总体的模型图，如下图  
![](https://raw.githubusercontent.com/celia01/papernotes/master/202011/pic/8.png)

### 1、base model
### User-Item 网络
### Item-Item 网络


## Comment


## Other
[一文搞懂阿里DMR排序模型——Deep Match to Rank](https://zhuanlan.zhihu.com/p/158497063)  
[[阿里]DMR：Matching和Ranking相结合的点击率预估模型](https://www.jianshu.com/p/60eed27e06d4)    
[code](https://github.com/lvze92/DMR)





