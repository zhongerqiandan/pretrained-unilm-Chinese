# pretrained-unilm-Chinese
中文版unilm预训练模型
## 简介
最近由于公司业务的需要，做了一些基于预训练seq2seq模型的文本生成式任务，研究了[MASS](https://github.com/microsoft/MASS)、[T5](https://github.com/google-research/text-to-text-transfer-transformer)、[UNILM](https://github.com/microsoft/unilm)之后，发现unilm这风格的seq2seq非常优雅。但是作者只开源了英文版的预训练模型，在git上也没找到合适的中文版unilm的预训练模型以及预训练代码，于是自己写了一个tensorflow版本。本项中预训练基于tensorflow-gpu==1.14.0，后续的微调任务基于[bert4keras](https://github.com/bojone/bert4keras)。

## 预训练
### 数据
简体中文维基百科数据，处理成一行一行句子对的形式。
![](https://github.com/zhongerqiandan/pretrained-unilm-Chinese/blob/master/pic/image-20201123144313297.png)