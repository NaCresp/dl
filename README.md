# 第一届辐射成像大赛

## 示例代码介绍

### [`Makefile`](./Makefile)

- 使用以下命令可以完成从数据集下载到评分的全过程
```bash
make all
```
- 使用以下命令可以清除包含模型参数与原始和预测数据在内的所有数据
```bash
make clean
```
- 使用以下命令可以完成从模型训练到评分的全过程(注意会删除之前保存的模型与预测结果)
```bash
make score
```

#### 数据集下载

- 下载数据集, 可修改 `TRAIN_START`, `TRAIN_END`, `TEST_START`, `TEST_END` 这四个参数调整训练集与评分集的范围.
```bash
make data
```
- 删除数据集
```bash
make cleanData
```

#### 模型训练到评分过程

- 模型训练(注意运行后会删除之前保存的模型)
```bash
make train
```
- 预测结果(注意运行后会删除之前保存的预测结果)
```bash
make predict
```
- 评分
```bash
make grade
```

### 模型训练代码 [`train.py`](./train.py)

本代码仅供参考, 仅保证可运行, 选手还需要**对模型进行优化调整**, 直接运行的总得分会较低, 且得到的预测结果图标签值均较小(可能都在 `3` 以下, 选手可思考原因并针对此展开第一个优化调整).

选手也可以采用不同的网络类型完成任务(甚至不使用神经网络完成任务, 如果能做到的话~).

本代码介绍可参考本赛事[第一次宣讲会](https://ri.thudep.com/talk/presentation1), 相较于宣讲会给出的代码有所改动.

### 预测结果代码 [`predict.py`](./predict.py)

本代码与示例的 [`train.py`](./train.py) 适配, 采用训练好的模型, 把原始 CT 图像进行 $[-1, 1]$ 的归一化后输入, 输出 $[-1, 1]$ 的归一化结果后, 再重新调整为 $0 \sim 15$ 的整数标签。

如果在 [`train.py`](./train.py) 中修改了模型的输入输出, **不要忘记将本代码改动以适配**.

### 评分代码 [`grade.py`](./grade.py)

本代码将得到的预测结果和真实的标签进行对比, 转化为 [one-hot 编码](https://en.wikipedia.org/wiki/One-hot)后使用 Dice 系数进行评分, 选手**不能更改**.

## 注意事项

- 参赛选手**不能修改**评分代码 [`grade.py`](./grade.py) , 如果发现此代码存在 bug , 请向赛事组委会反馈, 组委会将统一说明并修改.
- 参赛选手请**尽量不要改动** [`Makefile`](./Makefile) 中下载命令的框架(可以修改第6-22行的变量值), 后续进行隐藏测例的评判时, 组委会将统一修改路径. 如果不得已需要修改, 请在[报告](./report.md)中说明.
- 详细的评分规则见[辐射成像大赛官网-规则](https://ri.thudep.com/rules), 在此不再赘述.

## 通过 release 发布 unet.pth 文件

CI 已经设置好了通过 release 发布 unet.pth 文件，触发方法是将你选取好的 commit 使用 `git tag` 打上 tag，随后
```bash
git push --tag
```
推送 tag 后，即可触发 create-release job
