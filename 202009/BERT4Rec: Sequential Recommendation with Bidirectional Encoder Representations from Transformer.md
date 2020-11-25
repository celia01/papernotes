# Behavior Sequence Transformer for E-commerce Recommendation in Alibaba
kdd 2019 paper [arxiv](https://arxiv.org/pdf/1904.06690.pdf)



## ABSTRACT
通过用户行为历史，建模动态的用户属性，是推荐系统中关键的一环。之前的方法主要采用 序列神经网络，按时间从前往后，来encode用户历史互动（比如rnn）。尽管这样很有效果，但我们认为，这样从前往后的单向建模只能收获到局部最优的方法，主要有如下限制：a）单向结构限制了用户行为序列中隐藏的特征；b）这种单向结构，会假设这些行为都是一个严格的顺序，实际上许多行为并不是有顺序的（eg 没有因果关系的）。为了规避这种限制，我们提出了双向的self-attention来建模用户行为序列。
为了避免双向模型中的信息泄露，我们采用了 Cloze任务进行目标推荐，通过对用户前后的上下文随机masked。我们训练了一个双向模型，这个模型允许用户通过前后两侧信息融合，来表示用户历史行为，从而进行推荐。我们在4个基础数据集上做了丰富的实验，我们的模型比sofa的模型都好。

## Note

### 模型图
这是在阿里的paper，主要的大体模型如图
![](https://raw.githubusercontent.com/celia01/papernotes/master/202009/pic/2.png)
主要有两个对比模型：SASRec和RNN based rec。
SASRec是一个和Bert4Rec最类似的模型，是一个单向的Transformer模型


### Cloze Task



## Comment


## Other
[参考code](https://github.com/wziji/deep_ctr/tree/master/BST)
<!--[论文翻译](http://d0evi1.com/mind/)-->



