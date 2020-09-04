# PSGAN: A Minimax Game for Personalized Search with Limited and Noisy Click Data

SIGIR 2019 paper  [paper](http://playbigdata.ruc.edu.cn/dou/publication/2019_sigir_psgan.pdf)



## ABSTRACT
个性化搜索是使搜索排序具有个性化，是搜索结果适应用户兴趣。传统方法是从历史数据中提取点击和代表性feature的方法，构建用户画像。最近几年，深度学习因为有自动学习特征的性质，已经在个性化搜索中成功应用。然而，在学习相关和不相干的个性化边界中，少部分噪音的用户数据给深度模型带来了挑战。在这篇论文中，我们提出了PSGAN，一个用于个性化搜索的生成对抗网络(GAN)框架。通过对抗学习，我们强迫模型更多的关注那些更难区分的训练数据。我们使用判别模型去评估文档的个性化相关性，使用生成器去学习相关文档的分布。我们测试了两种生成器的构造方法：基于当前query or 基于一组生成查询。在搜索引擎中应用的数据实验表明，我们模型有重大优化，超过了state-of-the-art的模型

## Note

* 最终目标
![object](https://raw.githubusercontent.com/celia01/papernotes/master/202004/pic/1.png)
![discriminative_func](https://raw.githubusercontent.com/celia01/papernotes/master/202004/pic/2.png)

* 锁定生成模型时，判别模型目标
![d_obj](https://raw.githubusercontent.com/celia01/papernotes/master/202004/pic/3.png)

* 锁定判别模型时，生成模型目标
![g_obj](https://raw.githubusercontent.com/celia01/papernotes/master/202004/pic/4.png)
![g_obj2](https://raw.githubusercontent.com/celia01/papernotes/master/202004/pic/5.png)

* 训练流程
![algorithm](https://raw.githubusercontent.com/celia01/papernotes/master/202004/pic/6.png)



## Comment
在信息检索中使用GAN，主要是在离散分布中使用gan，文末附录，对比了在连续空间和离散空间比较。
1、提出了minimax在信息检索中的frame
2、拓展了pairwise形式的训练样本（pairwise任务在信息检索中很常见）
3、配有相关code


## Other
[code](https://github.com/geek-ai/irgan)

[只用生成器能否打败IRGAN? 读完Code再读论文的一些体会](https://medium.com/@yaoyaowd/%E5%8F%AA%E7%94%A8%E7%94%9F%E6%88%90%E5%99%A8%E8%83%BD%E5%90%A6%E6%89%93%E8%B4%A5irgan-%E8%AF%BB%E5%AE%8Ccode%E5%86%8D%E8%AF%BB%E8%AE%BA%E6%96%87%E7%9A%84%E4%B8%80%E4%BA%9B%E4%BD%93%E4%BC%9A-4b3f7c29b477)



