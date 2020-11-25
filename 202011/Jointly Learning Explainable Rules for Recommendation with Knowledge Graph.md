# Jointly Learning Explainable Rules for Recommendation with Knowledge Graph
www 2019 paper [dl.acm](https://dl.acm.org/doi/pdf/10.1145/3308558.3313607)



## ABSTRACT
可解释性和有效性，是建设推荐系统的关键。之前的工作主要都focus在如何取得更好的推荐效果（即有效性）。但这些方法有一些问题：  
（1）基于神经网络的embedding方法很难解释和debug（debug可以理解为调参）；  
（2）基于图形的方法（eg，基于meta path的方法）需要许多人工干预和领域知识，但却忽略了item之间的类型关联（eg）。  
这篇论文提出了一类新颖的联合学习框架（文中称为 RuleRec），可以 从知识图中归纳出可解释性的规则，并构建规则指导的神经网络推荐模型。这个框架鼓励两个模块相互互补，以达到有效和可解释的推荐。主要有这两个模块：  
（1）从以item为中心的只是图中提取归纳规则，总结出多跳的关系pattern，从而推断出不同的item联系，为模型预测提供可解释性  
（2）推荐模块可以通过归纳好的推责进行拓展，从而对于冷启动问题有更好的泛化能力。  
大量实验表明，我们提出的方法在许多真实数据集上都去的了很答的提升，也有很好的鲁棒性	

## Note

### 1、背景
* 物品推荐  
User U 和 Item I
* 知识图谱  
实体entity和关系relation  
head entity $E_1$，relation type $r_1$，tail entity $E_2$
* 知识图谱上的归纳规则  
在两个实体之间会存在许多路径，（eg，$P_k=E_1r_1E_2r_2E3$就是$E_1$到$E_3$的路径，$P_k$这条路径的规则就可以定义为$R=r_1r_2$）。注意这里路径$P$（主要关注实体），和规则$R$（主要关注relation type）不一样
具体异构网络，可以参考示意图
![](https://raw.githubusercontent.com/celia01/papernotes/master/202011/pic/6.png)
* 问题定义  
已知：User U，Item I，user-item的互动，item之间的联系，知识图谱  
目标：  
（1）在item图上学习规则，  
（2）基于规则$R$和用户互动历史$I_u$，得到一个推荐系统：给user $u$推荐item $I_u^{'}$  
最终模型会同时给出 一些列规则的集合 与 用户推荐列表

### 2、base model
* Bayesian Personalized Ranking Matrix Factorization (BPRMF)  
	实际上就是基于bpr loss 的矩阵分解：
	
	* 矩阵分解：$S_{u,i}=U_u^T·I_i$，最朴素的矩阵分解，向量直接相乘 
	* bpr loss：$O_{BPRMF}=\sum_{u\in U}\sum_{p\in I_u,n\notin I_u}(S_{u,p}-S_{u,n})$  
	说明：$S_{u,p}$是正例，$S_{u,n }$是随机采样的负例
* Neural Collaborative Filtering (NCF)  
	* generalized matrix factorization (GMF)：$h_{u,i}=U_u^T·I_i$，有latent的的矩阵分解
	* multi-layer percep- tion (MLP) ：$g_{u,i}=\phi_n(...\phi_2(\phi_1(z_1)))$，  
	其中 $z_1=\phi_0(U_u\oplus I_i)$，$\oplus $表示vector concat，$\phi(z_{k-1})=\phi(W_k^Tz_{k-1}+b_{k-1}) $
	* 最终 $h_{u,i}$和$g_{u,i}$共同表示最终的$S_{u,i}$
	<center>$S_{u,i}=\phi(\alpha·h_{u,i}\oplus(1-\alpha)·g_{u,i})$</center>
	<center>$O_{NCF}=\sigma(\sum_{u\in U}\sum_{p\in I_u,n\notin I_u}(S_{u,p}-S_{u,n}))$</center>
	
两个base model 都比较弱

### 3、RuleRec 框架
这儿带有规则的推荐主要包含两个子任务：

1.  基于item关系，从知识图中学习规则
2. 基于用户购买历史$I_u$和1中得到的规则R，给用户推荐商品

基于上述1，2，将问题建模为多任务，$O_r,V$表示推荐任务的目标和参数，$O_l,W$表示规则学习任务的目标和参数，最终目标就是
<center>$\min_{V,W}O=\min_{V,W}{\{O_r+\lambda O_l\}}$</center>

#### 规则学习
$P(b|a,R)$表示商品对(a,b)，从a到b的概率为P，ab之间的关系为R，那么

* $P(b|a,R)=\sum_{e\in N(a,R^{'})}P(e|a,R^{'}·P(b|e,r_k))$，其中$R^{'}=r_1...r_{k-1}$，$P(b|e,r_k)=\frac{I(r_k(e,b))}{\sum_iI(r_k|e,i)}$，这个概率表示通过关系$r_k$从e出发，最终走到b的概率

形象化举例：  
![](https://raw.githubusercontent.com/celia01/papernotes/master/202011/pic/7.png)  
$P(b|a,R)=P(c|a,r_1)·P(b|c,r_2)+P(d|a,r_1)·P(b|d,r_2)$，这里$R=r_1·r_2$   
实际上从a到b之间可以存在许多规则，可以定义为向量$x(a,b)=[P(b|a,R_1),...,P(b|a,R_n)]^T$

#### 规则选择
为了选择最有用的规则，需要进行规则选择  

* Hard Selection Method  
  设置超参数，指定我们需要选择的规则条数，接着使用chi-square OR 模型建模（回归or分类）来选择最好的规则，其余规则直接删除 
* Soft Selection Method  
  通过模型软选择，这样不会删除任何规则，可以灵活组合，也是本文的方法
  
#### item 推荐
* 基本模型： $S_{u,i}^{'}=f_w(S_{u,i}, \sum_{k\in I_u}F_{(i,k|R)})$，即从与user有交互的item集合的关系图中学习到$S_{u,i}$，可以简写为 $S_{u,i}^{'}=f_w(S_{u,i}, F_{(u,I_u|R)})$
* 最终目标：$O_r=\sum_{u\in U}\sum_{p\in I_u,n\notin I_u}(S_{u,p}^{'}-S_{u,n}^{'})=\sum_{u\in U}\sum_{p\in I_u,n\notin I_u}(f_w(S_{u,p},F_{(p,I_u|R)})-f_w(S_{n,p},F_{(n,I_u|R)}))$

这个框架下$S_{u,i}$的预估学习$f_w$，可以把之前的许多推荐模型融入进来

#### 多任务学习
在训练过程中，规则侧的权重是共享的，即$O_r,O_l$共用一套规则权重w

## Comment


## Other
[csdn上的一个解读](https://blog.csdn.net/qq_41621342/article/details/104188843)  
[如何让你的推荐系统具有可解释性？](https://mp.weixin.qq.com/s/arnp_ZsxY5wiGlyZsSzUNw) 
[code](https://github.com/THUIR/RuleRec)



