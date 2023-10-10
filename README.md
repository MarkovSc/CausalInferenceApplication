# Causal Inference Application

这个库是集成现有的github 项目上有价值的几个项目，从应用的角度进行集成方便进行相关因果建模的开发。



| 相关的库     | 基础的介绍                                                   | support causal inference or causal discovery |      |      |
| ------------ | ------------------------------------------------------------ | -------------------------------------------- | ---- | ---- |
| EcomML       | 基于观测数据或者实验数据来估计 individualized causal responses。 跟进因果机器学习的最新进展 ，并通过结合机器学习和可解释的因果模型，提高预测可靠性，并且更加容易理解 | inference                                    |      |      |
| CausalML     | 基于最近2020前的进展实现的基于机器学习的因果推断套件库， 基于观测数据预估 CATE或者ITE。 Essentially, it estimates the causal impact of intervention `T` on outcome `Y` for users with observed features `X`, without strong assumptions on the model form. | inference                                    |      |      |
| Doubleml     | 基于Chernozhukov et al. (2018)  采用Python and R package 实现的 double / debiased machine learning framework 。 | inference                                    |      |      |
| GRF          | 基于Paper"Generalized Random Forests" 使用R语言实现的库。并进行了一些GRF 的扩展。 | inference                                    |      |      |
| DoWhy        | DoWhy是用于*因果推断*的*Python*库，它支持对因果假设进行显式建模和验证。DoWhy基于用于因果推理的统一语言，结合了因果图模型和潜结果框架 | discovery                                    |      |      |
| Causal-learn | Causal-learn，由CMU张坤老师主导，多个团队（CMU因果研究团队、DMIR实验室、宫明明老师团队和Shohei Shimizu老师团队）联合开发出品的因果发现算法平台。 | discovery                                    |      |      |
| UpliftML     | Booking.com 做的一个库，基于Spark 做的开发                   | Inference                                    |      |      |