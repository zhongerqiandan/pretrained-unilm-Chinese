# pretrained-unilm-chinese
中文版unilm预训练语言模型

## Table of Contents

- [Background](#background)
- [Pretraining Details](#pretraining details)
- [Download](#install)
- [Usage](#usage)
- [Experiment](#experiment)
- [TODO](#todo)
- [Maintainers](#maintainers)
- [Contributing](#contributing)
- [License](#license)

## Background
最近由于公司业务的需要，做了一些基于预训练seq2seq模型的文本生成式任务，研究了[MASS](https://github.com/microsoft/MASS)、[T5](https://github.com/google-research/text-to-text-transfer-transformer)、[UNILM](https://github.com/microsoft/unilm)之后，发现unilm这风格的seq2seq非常优雅。但是作者只开源了英文版的预训练模型，在git上也没找到合适的中文版unilm的预训练模型以及预训练代码，于是自己写了一个tensorflow版本。本项中预训练基于tensorflow-gpu==1.14.0，后续的微调任务基于[bert4keras](https://github.com/bojone/bert4keras)。
## Pretraining Details
### Training Data
简体中文维基百科数据，处理成一行一行句子对的形式。
### Input Mask And Attention Mask
1. 在一条数据中随机mask15%的token，被mask的token中80%用[MASK]表示，10%从vocab中随机选择一个token，10%不变。e.g. 一条可能的数据如下：[CLS] A1 A2 [MASK] A4 [SEP] B1 B2 B3 [MASK] B5 [SEP]，其中A1-A4是句子1的token，B1-B5是句子2的token，A3和B4被mask。
2. 根据1中masked input的结果，生成不同的attention mask，unilm原文中说有1/3的数据采用seq2seq式的attention mask策略，1/3的数据采用bert式的attention mask策略，1/6数据采用left2right的language model式的attention mask，1/6数据采用right2left的language model式的attention mask。seq2seq其实就是对应的casual with prefix attention mask(下图，其他token在这里不可以看到被mask位置的符号):
#### casual with prefix attention mask
![pic/image-20201126141626762.png](https://github.com/zhongerqiandan/pretrained-unilm-Chinese/blob/master/pic/image-20201126141626762.png)

bert式对应的就是fully-visible attention mask(下图):
#### fully-visible attention mask
![pic/image-20201126141822288.png](https://github.com/zhongerqiandan/pretrained-unilm-Chinese/blob/master/pic/image-20201126141822288.png)

left2right LM对应的就是casual attention mask，每个token只能attend它和它左边的token（下图）:
#### casual attention mask
![pic/image-20201126141922904.png](https://github.com/zhongerqiandan/pretrained-unilm-Chinese/blob/master/pic/image-20201126141922904.png)

right2left LM与上面相反:
#### reverse casual attention mask
![pic/image-20201126142013570.png](https://github.com/zhongerqiandan/pretrained-unilm-Chinese/blob/master/pic/image-20201126142013570.png)
## Download
我们开源了预训练好的模型、代码以及预训练用到的数据，同时为了方便大家finetune，这里一同给出下游任务数据的链接
| 链接 | 提取码 |
|----------------------------------------------------------------|---|
|[tensorflow版](https://pan.baidu.com/s/1x9eRJMt76bEPQ5nNkOkPZQ) |jfb3|
|[pytorch版](https://pan.baidu.com/s/1FKjieHoXr-LBWK89EnMdZw)|x2wf|
|[整理好的中文wiki预训练数据](https://pan.baidu.com/s/1XGkhwUePsIR3lP_quiXlCQ)|p75b|
|[论文标题生成数据csl](https://pan.baidu.com/s/1AzTupql6EwW1j_kI4qmQkA)|kd9h|
|[webqa](https://pan.baidu.com/s/1OOwOtBzZ11b6Bw1X8tY6Tg)|kteo|
|[微博新闻摘要](https://pan.baidu.com/s/186qUGq_HIiOXgMfl3QRwKw)|cdtc|

## Usage
### pretrain
1. 首先确保机器上有python3的环境，推荐安装anaconda3。
2. conda install tensorflow-gpu=1.14.0
3. pip install keras4bert
4. 下载google原版的中文版bert，chinese_L-12_H-768_A-12
5. 下载数据集，https://pan.baidu.com/s/1XGkhwUePsIR3lP_quiXlCQ，提取码：p75b
6. 修改base/data_load.py文件中dict_path的值，将其更改为你的bert文件夹中vocab.txt的路径
7. 修改run_pretraining_google.py中37、38行，你使用几块gpu就更改为对应的值
8. 
```
python run_pretraining_google.py \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --input_file=$DATA_BASE_DIR/wiki_sent_pair.txt \
  --output_dir=$OUT_PUT_BASE_DIR/checkpoint
```
### fine-tune
1. 首先确保机器上有python3的环境，推荐安装anaconda3。
2. conda install tensorflow-gpu=1.14.0
3. pip install keras4bert rouge nltk
4. 下载数据集，放到dataset/目录下
4. 下载预训练好的tensorflow版本的模型，https://pan.baidu.com/s/1x9eRJMt76bEPQ5nNkOkPZQ，提取码jfb3
5. 以task/task_summary.py为例，将文件中config_path、checkpoint_path、dict_path改为上一步下载好的模型目录中的相关路径,model_save_path 改为自己的模型保存路径
6. 
```
python task_summary.py
```
## Experiment
我们做了四个下游任务，分别是论文标题生成(csl)，webqa，微博新闻摘要和相似问句生成，其中前三个任务参考[CLUEbencmark/CLGE](https://github.com/CLUEbenchmark/CLGE)
前三个任务中，我们对比了载入google原版bert权重和我们预训练的unilm权重，结果如下
|            | csl(bleu,rouge-L) | webqa(bleu,rouge-L) | 微博新闻标题生成(bleu,rouge-L) | 相似问句生成(bleu) |
| ---------- | ----------------- | ------------------- | ------------------------------ | ------------------ |
| Unilm-base | 0.476,  0.648     | 0.358,  0.708       | 0.108, 0.265                   |                    |
|            |                   |                     |                                |                    |
| Bert-base  | 0.452,  0.640     | 0.342,  0.676       | 0.110, 0.267                   |                    |
## TODO
1. pretrain和fine tune的pytorch版本，基于hugging face的transformers库

## Maintainers

[@zhongerqiandan](https://github.com/zhongerqiandan)

## Contributing

See [the contributing file](contributing.md)!

PRs accepted.

Small note: If editing the README, please conform to the [standard-readme](https://github.com/RichardLitt/standard-readme) specification.

## License

MIT © 2018 Richard McRichface
