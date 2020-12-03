# pretrained-unilm-Chinese
中文版unilm预训练模型
## 简介
最近由于公司业务的需要，做了一些基于预训练seq2seq模型的文本生成式任务，研究了[MASS](https://github.com/microsoft/MASS)、[T5](https://github.com/google-research/text-to-text-transfer-transformer)、[UNILM](https://github.com/microsoft/unilm)之后，发现unilm这风格的seq2seq非常优雅。但是作者只开源了英文版的预训练模型，在git上也没找到合适的中文版unilm的预训练模型以及预训练代码，于是自己写了一个tensorflow版本。本项中预训练基于tensorflow-gpu==1.14.0，后续的微调任务基于[bert4keras](https://github.com/bojone/bert4keras)。

## 预训练
### 数据
简体中文维基百科数据，处理成一行一行句子对的形式。
![pic/image-20201123144313297.png](https://github.com/zhongerqiandan/pretrained-unilm-Chinese/blob/master/pic/image-20201123144313297.png)
### 预训练细节
1. 在一条数据中随机mask15%的token，被mask的token中80%用[MASK]表示，10%从vocab中随机选择一个token，10%不变。e.g. 一条可能的数据如下：[CLS] A1 A2 [MASK] A4 [SEP] B1 B2 B3 [MASK] B5 [SEP]，其中A1-A4是句子1的token，B1-B5是句子2的token，A3和B4被mask。
2. 根据1中masked input的结果，生成不同的attention mask，unilm原文中说有1/3的数据采用seq2seq式的attention mask策略，1/3的数据采用bert式的attention mask策略，1/6数据采用left2right的language model式的attention mask，1/6数据采用right2left的language model式的attention mask。seq2seq其实就是对应的casual with prefix attention mask(下图，其他token在这里不可以看到被mask位置的符号):
![pic/image-20201126141626762.png](https://github.com/zhongerqiandan/pretrained-unilm-Chinese/blob/master/pic/image-20201126141626762.png)
bert式对应的就是fully-visible attention mask(下图):
![pic/image-20201126141822288.png](https://github.com/zhongerqiandan/pretrained-unilm-Chinese/blob/master/pic/image-20201126141822288.png)
left2right LM对应的就是casual attention mask，每个token只能attend它和它左边的token（下图）:
![pic/image-20201126141922904.png](https://github.com/zhongerqiandan/pretrained-unilm-Chinese/blob/master/pic/image-20201126141922904.png)
right2left LM与上面相反:
![pic/image-20201126142013570.png](https://github.com/zhongerqiandan/pretrained-unilm-Chinese/blob/master/pic/image-20201126142013570.png)
3. 以上就是整个数据处理过程，数据处理完之后就可以开始训练模型了，训练方法和原文一样，就是预测被mask的token，从谷歌原版的中文bert-base模型初始化。我们使用4块v100，每块卡上的batch size为6，学习率和原文保持一致，持续训练100万步。
## 下游任务
我们做了四个下游任务，分别是论文标题生成(csl)，webqa，微博新闻摘要和相似问句生成，其中前三个任务参考[CLUEbencmark/CLGE](https://github.com/CLUEbenchmark/CLGE)
前三个任务中，我们对比了载入google原版bert权重和我们预训练的unilm权重，结果如下
|            | csl(bleu,rouge-L) | webqa(bleu,rouge-L) | 微博新闻标题生成(bleu,rouge-L) | 相似问句生成(bleu) |
| ---------- | ----------------- | ------------------- | ------------------------------ | ------------------ |
| Unilm-base | 0.476,  0.648     | 0.358,  0.708       | 0.108, 0.265                   |                    |
|            |                   |                     |                                |                    |
| Bert-base  | 0.452,  0.640     | 0.342,  0.676       | 0.110, 0.267                   |                    |
### unilm生成论文标题
![pic/image-20201029110703723.png](https://github.com/zhongerqiandan/pretrained-unilm-Chinese/blob/master/pic/image-20201029110703723.png)
### unilm生成式问答(webqa)
![pic/image-20201118100107676.png](https://github.com/zhongerqiandan/pretrained-unilm-Chinese/blob/master/pic/image-20201118100107676.png)
### unilm微博新闻标题生成
![pic/image-20201119142146145.png](https://github.com/zhongerqiandan/pretrained-unilm-Chinese/blob/master/pic/image-20201119142146145.png)
### unilm相似问句生成
完全端到端的构建相似句，可以用于基于faq的智能问答系统、预料扩充等。这个任务的数据暂时无法公开，但是可以说这种端到端的生成相似句的效果已经非常不错了，不用维护复杂的规则，但是需要较好的语料。我司现在线上就在使用这个模型。
![pic/image-20201126145937390.png](https://github.com/zhongerqiandan/pretrained-unilm-Chinese/blob/master/pic/image-20201126145937390.png)