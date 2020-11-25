# MiNet: Mixed Interest Network for Cross-Domain Click-Through Rate Prediction
cikm 2020 paper [arxiv](https://arxiv.org/pdf/2008.02974.pdf)
这是阿里UC头条的一篇paper，主要讨论的多域多任务学习


## ABSTRACT
点击率(CTR)预估是在线广告系统的一个关键人物。目前的工作主要聚焦于单域的广告CTR预估，模型主要关注交互，用户行为历史，和上下文信息。但是，广告展示经常和正常的推荐内容一样，这就为跨域点击率预估提供了基础（即可以任务是内容一样的item）。在论文中，我们解决了单域的问题，并且利用源域的新行为来提升目标域ctr的预估。论文的工作主要是在uc头条上的一个工作，这里的源域是news，目标域是广告。我们提出了 MiNet模型可以把下面三类兴趣一起用户建模：

1. Long-term interest across domains（跨域的长期兴趣）
2. Short-term interest from the source domain（源域的短期兴趣）
3. short-term interest in the target domain（目标域的短期兴趣）

MiNet包含两层attention，item级别的attention主要是可以从点击过的news/ads中，自适应的提炼有用的信息；Interest级别的attention主要是可以适应性地融合不同的兴趣表达。离线实验中MiNet比sota的模型表现的都好，同时MiNet也在UC头条中部署上线了，从A/B Test上看，也提升很多。目前UC头条就是使用的MiNet模型


## Note

### 1、背景
* Cross Domain：包含多个域的推荐  
使用用户的profile表示这一部分特征
* Source Domain：辅助推荐结果的其他域  

* Target Domain：最终要提升推荐效果的目标域

* 跨域的



## Comment


## Other
[CIKM20-MiNet：阿里|跨域点击率预估混合兴趣模型](https://zhuanlan.zhihu.com/p/221719082)  




