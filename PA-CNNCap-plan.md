# PA-CNNCap 执行计划

## 0. 当前目标

PA-CNNCap 的目标是在原始 CNNCap 的 2D coupling capacitance 预测框架基础上，引入 SSRN 中的 physics-inspired 网络结构。

### 输入

```text
2D extraction window 下全部导体的 CNNCap grid representation
```

其中当前数据编码规则为：

```text
master conductor: ratio > 1
target conductor: ratio < 0
ordinary conductor: 0 < ratio <= 1
```

### 输出

```text
指定 master-target 导体对的 coupling capacitance Cij
```

### 当前约束

```text
不预测 residual
不引入额外 field solver baseline
不先做 3D
baseline 使用原始 CNNCap
先做版本 A，再做版本 B
```

---

## 1. Phase 1：复现原始 CNNCap baseline

### 1.1 目标

先跑通原始 CNNCap 的 2D env/coupling capacitance 预测模型，作为后续 PA-CNNCap 的 baseline。

### 1.2 关键文件

```text
dataset.py
train.py
test.py
resnet.py
```

### 1.3 数据流

在 `goal='env'` 时，数据应使用：

```text
item['env_data'][k]['x_compress'] -> x_grid
item['env_data'][k]['y'] -> coupling capacitance label
item['env_data'][k]['dim'] -> grid dimension
```

原始输入张量：

```text
x_grid shape = [in_channel, dim]
```

其中：

```text
in_channel = 金属层数
dim = 横向网格长度
```

### 1.4 Baseline 模型

原始 CNNCap 使用：

```text
x_grid -> ResNet1D -> AdaptiveAvgPool1d -> FC -> Cij
```

对应模型：

```text
resnet34
```

### 1.5 需要记录的指标

```text
mean relative error
median relative error
95th percentile relative error
max relative error
<1% ratio
<5% ratio
<10% ratio
runtime
```

相对误差建议定义为：

```text
relative error = |C_pred - C_true| / (|C_true| + eps)
```

---

## 2. Phase 2：版本 A，CNNCap + SSRN-style Head

版本 A 不修改 dataset，只修改模型结构。

### 2.1 目标

在原始 CNNCap 的 ResNet1D backbone 后面，将普通 FC prediction head 替换为 SSRN-inspired prediction head。

### 2.2 建议新增文件

```text
pa_resnet.py
```

建议保持原始 `resnet.py` 不动，便于 baseline 对比。

### 2.3 PA-CNNCap-A 总结构

原始 CNNCap：

```text
Input grid
  -> Conv1d
  -> ResNet blocks
  -> AdaptiveAvgPool1d
  -> FC
  -> Cij
```

PA-CNNCap-A：

```text
Input grid
  -> Conv1d
  -> ResNet blocks
  -> AdaptiveAvgPool1d
  -> feature z
  -> SSRNHead
  -> Cij
```

### 2.4 SSRNHead 设计

SSRNHead 借鉴 SSRN 中的：

```text
linear transformation
power(-1) transformation
power(-2) transformation
BN
SELU
residual / fusion structure
```

推荐第一版结构：

```text
z
├── FC(z)
├── FC(1 / (abs(z) + eps))
└── FC(1 / ((abs(z) + eps)^2))
        ↓
      concat
        ↓
       BN
        ↓
      SELU
        ↓
       FC
        ↓
      Cij
```

第一版推荐使用 `concat`，比 `add` 更稳定。

### 2.5 推荐模型命名

```text
pa_resnet34
```

用于和原始 CNNCap 对比：

```text
CNNCap baseline: resnet34
PA-CNNCap-A: pa_resnet34
```

### 2.6 训练入口修改

建议在 `train.py` 和 `test.py` 中支持模型选择参数：

```text
--model resnet34
--model pa_resnet34
```

逻辑示例：

```python
if args.model == "pa_resnet34":
    model = pa_resnet34(in_channel=dataset.in_channel)
else:
    model = resnet34(in_channel=dataset.in_channel)
```

### 2.7 Phase 2 实验对比

| Method | Input | Model structure | Output |
|---|---|---|---|
| CNNCap | grid | ResNet1D + FC | Cij |
| PA-CNNCap-A | grid | ResNet1D + SSRNHead | Cij |

目标：验证仅替换 prediction head 后，SSRN-style transformation 是否能提升 coupling capacitance 预测精度。

---

## 3. Phase 3：版本 B，CNN branch + Pair Geometry SSRN Branch

版本 B 在版本 A 基础上修改 dataset，使模型额外接收 pair geometry parameters。

### 3.1 目标

从 `x_compress` 中解析 master conductor 和 target conductor 的几何参数，并通过 SSRN-style parameter branch 与 CNN spatial feature 融合。

### 3.2 从 x_compress 解析 conductor

每条压缩记录格式：

```text
[layer_id, l, r, ratio]
```

根据 ratio 判断导体类型：

```text
ratio > 1: master conductor
ratio < 0: target conductor
0 < ratio <= 1: ordinary conductor
```

对于 master 和 target，如果一个 conductor 被拆成多个 segment，第一版可以使用：

```text
l = min(all_l)
r = max(all_r)
layer_id = 对应 layer
```

### 3.3 Pair geometry features

第一版建议提取以下参数：

```text
w_master
w_target
center_master
center_target
spacing
left_edge_space
right_edge_space
layer_master
layer_target
layer_distance
same_layer
```

计算方式：

```text
w_master = r_m - l_m + 1
w_target = r_t - l_t + 1

center_master = (l_m + r_m) / 2
center_target = (l_t + r_t) / 2

spacing = max(0, max(l_m, l_t) - min(r_m, r_t) - 1)

left_edge_space = min(l_m, l_t)
right_edge_space = dim - max(r_m, r_t)

layer_distance = abs(layer_m - layer_t)
same_layer = 1 if layer_m == layer_t else 0
```

### 3.4 参数归一化

建议第一版在 dataset 中做简单归一化：

```text
w_master / dim
w_target / dim
center_master / dim
center_target / dim
spacing / dim
left_edge_space / dim
right_edge_space / dim
layer_master / in_channel
layer_target / in_channel
layer_distance / in_channel
same_layer 保持 0/1
```

### 3.5 Dataset 返回值

保持兼容原始 CNNCap。

原始模式：

```python
return x_grid, y
```

启用 pair 参数模式：

```python
return x_grid, pair_param, y
```

建议添加参数：

```text
use_pair_param = False / True
```

### 3.6 PairGeometryBranch 结构

```text
pair_param
  ├── FC(pair_param)
  ├── FC(1 / (pair_param + eps))
  └── FC(1 / ((pair_param + eps)^2))
        ↓
      concat
        ↓
       BN
        ↓
      SELU
        ↓
       FC
        ↓
  geometry feature
```

注意：对可能为 0 的参数使用 eps，避免除零。

### 3.7 PA-CNNCap-B 总结构

```text
x_grid
  -> CNNCap ResNet1D backbone
  -> spatial feature
                     \
                      concat -> fusion MLP -> Cij
                     /
pair_param
  -> SSRN parameter branch
  -> geometry feature
```

推荐模型名：

```text
pa_resnet34_pair
```

### 3.8 训练入口修改

建议增加参数：

```text
--use-pair-param
```

训练逻辑：

```python
if args.use_pair_param:
    x_grid, pair_param, y = batch
    pred = model(x_grid, pair_param)
else:
    x_grid, y = batch
    pred = model(x_grid)
```

---

## 4. Phase 4：实验矩阵

### 4.1 主实验

| Method | Grid input | Pair params | SSRN structure | Output |
|---|---|---|---|---|
| CNNCap | yes | no | no | Cij |
| PA-CNNCap-A | yes | no | yes, head only | Cij |
| PA-CNNCap-B | yes | yes | yes, head + parameter branch | Cij |

### 4.2 消融实验

#### Ablation 1：SSRNHead 是否有效

```text
CNNCap vs PA-CNNCap-A
```

#### Ablation 2：pair geometry 参数是否有效

```text
PA-CNNCap-A vs PA-CNNCap-B
```

#### Ablation 3：power transformation 是否有效

```text
PA-CNNCap-B without power branch
PA-CNNCap-B with power(-1)
PA-CNNCap-B with power(-1) + power(-2)
```

#### Ablation 4：融合方式

第一版只做 concat，后续可比较：

```text
concat fusion
add fusion
gated fusion
```

---

## 5. Phase 5：结果整理

### 5.1 误差指标

统一使用：

```text
relative error = |C_pred - C_true| / (|C_true| + eps)
```

记录：

```text
mean
median
95th percentile
max
<1%
<5%
<10%
```

### 5.2 可视化

建议绘制：

```text
predicted vs reference scatter plot
relative error histogram
error vs capacitance magnitude
CNNCap vs PA-CNNCap-B per-sample error comparison
```

---

## 6. Phase 6：论文方法部分结构

后续论文方法部分可以按以下结构展开：

```text
3. Proposed PA-CNNCap
   3.1 Problem Formulation
   3.2 Pair-aware Grid Representation
   3.3 Baseline CNNCap Model
   3.4 SSRN-inspired Prediction Head
   3.5 Pair Geometry Enhanced Branch
   3.6 Training Objective
```

重点表述：

```text
PA-CNNCap 不改变 CNNCap 的 pair-aware 输入定义；
PA-CNNCap 在 CNNCap 基础上引入 SSRN 的 physics-inspired transformation；
PA-CNNCap 直接输出 coupling capacitance；
原始 CNNCap 作为 baseline。
```

---

## 7. 最终执行顺序

```text
1. 跑通原始 CNNCap env baseline
2. 新增 SSRNHead
3. 实现 PA-CNNCap-A
4. 训练并和 baseline 对比
5. 从 x_compress 解析 master/target 几何参数
6. 修改 dataset 支持 pair_param
7. 实现 PairGeometryBranch
8. 实现 PA-CNNCap-B
9. 做主实验和消融实验
10. 整理表格、图和论文方法描述
```
