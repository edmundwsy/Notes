# Single Shot MultiBox Detector

[TOC]

```latex
bibtex from arxiv
```

## 摘要部分

发现了使用单神经网络进行detection的方法，输出的是不同长宽比不同尺度的一组默认框

预测时：生成每类物体每类框的一组分数，生成调整更好的配合框

将具有不同分辨率的多层feature map上的预测组合在一起，**从而可以处理不同尺度的物体**

相对于 object proposals 的简单，因为消除了 proposal generation 和 resampling 阶段， 并且用一个网络实现

容易训练

**快**

## Introduction

### 现状

目前大家都有的三个方法：

1.  hypothesize bounding boxes
2. resample pixels or features for each box
3. apply a high-quality classifier

虽然准确，但是这些方法都太慢了（计算成本很高）

目前，提速是要消耗准确率的，不好做 trade-off

 

### 创新点

1. 不会在bbox的假设上 resample ，仍然保证了准确率

2. 提速来源于消除了proposals 和 resample stage

3. 使用小conv filter去预测目标种类和bbox的偏移

4. 使用单独的filter去检测长宽比

5. 将这些filter应用于后面多层用于多尺度检测

   *（平移不变性如何实现*？卷积过程就实现了？）
###  特点

1. 比YOLO快
2. 和Faster R-CNN 一样准
3. 在feature map上面用小卷积核预测出 scores 和 bbox offsets
4. 实现了不同尺度
5. end-to-end 
6. 在低分辨率输入也work

   

## SSD算法介绍

![1559574075141](assets/SSD-1)

- 标准结构：前边几层，使用VGG-16，最后一层截掉 （base network）

- 多尺度特征检测网络：在 base network 后面加入conv layer

- 预测器（卷积核）：每个feature layer 使用一个固定系列的卷积核

  - 3 * 3 * p  输出score 和 offset
  - 输出与默认box 的position和 feature map 的location有关

- 默认框和长宽比：一系列默认框

  - 默认框做卷积  平铺

  - 每个框里面预测 与形状大小有关的偏移量 （4个参数）

  - 每个框预测是否有class ，并且评分 （c类）

  - 每个位置如果需要k组不同大小的box， 总共需要 $(c + 4)k​$ 个卷积核

  - 如果$m\times n$ feature map 需要  $(c + 4)kmn$ 个filter

    

### 训练

#### Matching Strategy

需要清楚哪个默认框对应着哪个ground truth box，然后对应的训练

- 将ground truth 和最佳的 jaccard overlap 对应
- 找 jaccard overlap 大于 0.5 的ground truth
- 因此不局限于一个，可以选择较大的多个训练

#### Training Objective

$$
L(x, c, l, g)=\frac{1}{N}\left(L_{\operatorname{con} f}(x, c)+\alpha L_{l o c}(x, l, g)\right)
$$

location loss 和 confidence loss 的 加权平均

location loss：
$$
L_{l o c}(x, l, g)=\sum_{i \in P o s}^{N} \sum_{m \in\{c x, c y, w, h\}} x_{i j}^{k} \text { smooth }_{\mathrm{Ll}}\left(l_{i}^{m}-\hat{g}_{j}^{m}\right)
$$

> The localization loss is a Smooth L1 loss [6] between the predicted box (l) and the ground truth box (g) parameters.
>
> We regress to offsets for the center (cx, cy) of the default bounding box (d) and for its width (w) and height (h).

confidence loss
$$
L_{c o n f}(x, c)=-\sum_{i \in P o s}^{N} x_{i j}^{p} \log \left(\hat{c}_{i}^{p}\right)-\sum_{i \in N e g} \log \left(\hat{c}_{i}^{0}\right) \quad \text { where } \hat{c}_{i}^{p}=\frac{\exp \left(c_{i}^{p}\right)}{\sum_{p} \exp \left(c_{i}^{p}\right)}
$$

#### Choosing scales and aspect ratios for default boxes