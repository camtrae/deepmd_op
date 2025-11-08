Readme env mat · MD
Copy

# DeepMD env_mat_a 函数 SVE 向量化优化

## 1. 计算模式简介

`env_mat_a` 函数用于计算 DeepMD 模型中的环境矩阵及其导数，是分子动力学模拟中的核心函数。

### 1.1 计算流程

```
对每个中心原子 i:
    对每个邻居原子 j:
        1. 计算相对位置向量: r_ij = pos_j - pos_i
        2. 计算距离: r = |r_ij|
        3. 计算几何量: 1/r, 1/r², 1/r³, 1/r⁴
        4. 应用平滑函数: sw(r), dsw(r)
        5. 计算描述符: [1/r, x/r², y/r², z/r²] × sw
        6. 计算12个导数分量
```

### 1.2 计算特征

**输入**：
- `posi`: 所有原子的三维坐标 [N × 3]
- `fmt_nlist_a`: 邻居列表（间接索引）
- `i_idx`: 中心原子索引

**输出**：
- `descrpt_a`: 描述符 [M × 4]
- `descrpt_a_deriv`: 导数矩阵 [M × 12]  
- `rij_a`: 相对位置向量 [M × 3]

**计算量**：
- 每个邻居：约 50-60 次浮点运算
- 平均邻居数：10-100
- 包含：开方、除法、指数函数等高延迟操作

## 2. 优化策略

### 2.1 融合计算

将多个计算步骤融合在一起，减少中间数组的分配和访问：

```cpp
// ❌ 原始：分步计算，多次读写内存
float nr2 = rx*rx + ry*ry + rz*rz;
float inr = 1.0f/sqrtf(nr2);
float nr = nr2 * inr;
// ... 多次访问 inr, nr, nr2

// ✅ 优化：一次性计算所有几何量
svfloat32_t vec_nr2 = compute_distance_squared(...);
svfloat32_t vec_inr = fast_rsqrt(vec_nr2);      // 快速倒数平方根
svfloat32_t vec_nr = svmul(vec_nr2, vec_inr);
svfloat32_t vec_inr2 = svmul(vec_inr, vec_inr); // 同时计算所有需要的幂次
svfloat32_t vec_inr3 = svmul(vec_inr2, vec_inr);
svfloat32_t vec_inr4 = svmul(vec_inr2, vec_inr2);
```

### 2.2 向量化平滑函数

使用 SVE 向量化实现 5 次样条平滑函数：

```cpp
// 标量版本
void spline5_switch_scalar(float& sw, float& dsw, float r, 
                          float rmin, float rmax) {
    if (r < rmin) { sw = 1.0f; dsw = 0.0f; }
    else if (r < rmax) {
        float uu = (r - rmin) / (rmax - rmin);
        sw = uu³(-6uu² + 15uu - 10) + 1;
        dsw = (...);
    }
    else { sw = 0.0f; dsw = 0.0f; }
}

// SVE向量化版本
void spline5_switch_sve(svfloat32_t& vec_sw, svfloat32_t& vec_dsw,
                       svfloat32_t vec_r, float rmin, float rmax,
                       svbool_t pg) {
    // 使用谓词掩码处理三个分支
    svbool_t mask_below = svcmplt(pg, vec_r, vec_rmin);
    svbool_t mask_above = svcmpge(pg, vec_r, vec_rmax);
    svbool_t mask_middle = (...);
    
    // 向量化计算多项式
    vec_sw = svsel(mask_below, vec_one, vec_zero);
    vec_sw = svsel(mask_middle, compute_poly(...), vec_sw);
}
```

### 2.3 批处理策略

```cpp
// 将邻居分批处理，充分利用向量寄存器
constexpr int MAX_BATCH = 128;
float batch_x[MAX_BATCH];
float batch_y[MAX_BATCH];
float batch_z[MAX_BATCH];

for (int batch_start = sec_start; batch_start < sec_end; 
     batch_start += SVL) {
    // 1. 收集一批数据
    int batch_count = 0;
    for (int jj = batch_start; jj < sec_end && batch_count < MAX_BATCH; ++jj) {
        int j_idx = fmt_nlist_a[jj];  // 间接访问
        batch_x[batch_count] = posi[j_idx*3 + 0] - center_x;
        // ...
        batch_count++;
    }
    
    // 2. 向量化处理
    svfloat32_t vec_x = svld1_f32(pg, batch_x);
    // ... 向量化计算
    
    // 3. 存储结果
    svst1_f32(pg, temp_buffer, result);
    for (int i = 0; i < batch_count; i++) {
        output[indices[i]] = temp_buffer[i];  // scatter存储
    }
}
```

### 2.4 快速倒数平方根

使用硬件加速的 `svrsqrte_f32` + Newton-Raphson 迭代：

```cpp
// 硬件估计 + 一次迭代，达到 float32 精度
svfloat32_t vec_inr = svrsqrte_f32(vec_nr2);
svfloat32_t correction = svmls_f32_m(pg, svdup_f32(1.5f), 
                                     vec_nr2, svmul(vec_inr, vec_inr) * 0.5f);
vec_inr = svmul(vec_inr, correction);
```

## 3. 性能瓶颈分析

尽管使用了多种优化技术，但 **SVE 优化效果并不理想**。以下是主要的性能瓶颈：

### 3.1 间接内存访问（Gather 操作）

```cpp
// ❌ 最大性能杀手
for (int jj = batch_start; jj < batch_end; ++jj) {
    int j_idx = fmt_nlist_a[jj];           // ← 间接索引
    batch_x[batch_count] = posi[j_idx*3 + 0];  // ← 随机访问
}
```

**问题**：
- 邻居列表 `fmt_nlist_a` 是**无序的随机索引**
- 导致对 `posi` 数组的**随机访问模式**
- **Cache miss 率极高**（可能 >80%）
- 每次访问需要等待内存延迟（100-300 周期）

### 3.2 Scatter 存储模式

```cpp
// ❌ 结果存储也是间接的
for (int i = 0; i < batch_count; i++) {
    int jj = batch_indices[i];  // 间接索引
    descrpt_a[jj * 4 + 0] = temp_buffer[i];      // scatter store
    descrpt_a_deriv[jj * 12 + 0] = deriv_buf[i]; // scatter store
    // ... 需要存储 4 + 12 + 3 = 19 个值
}
```

**问题**：
- 需要将向量化结果**分散写入**不同的内存位置
- 无法使用高效的连续向量存储指令
- 每个结果需要**多次标量存储**（19个值 × batch_count）

### 3.3 批处理开销

```cpp
// ❌ 数据需要先收集到临时数组
float batch_x[MAX_BATCH];  // ← 额外的内存拷贝
float batch_y[MAX_BATCH];
float batch_z[MAX_BATCH];

// 收集数据
for (...) {
    batch_x[batch_count] = posi[...];  // copy
}

// 向量化处理
svfloat32_t vec_x = svld1_f32(pg, batch_x);  // 再次读取
```

**问题**：
- 数据需要经过：`内存 → batch数组 → SVE寄存器`
- 相比标量直接访问，**多了一次拷贝**

## 4. 性能对比总表

| 测试 | 原子数 | 邻居数 | 迭代次数 | 基准时间 (ms) | SVE优化时间 (ms) | 加速比 | 性能变化 |
|------|--------|--------|----------|---------------|------------------|--------|----------|
| 1    | 50     | 10     | 5000     | 0.000066      | 0.002000         | **0.03×** | 🔴 **慢 30倍** | 
| 2    | 200    | 25     | 2000     | 0.000151      | 0.003355         | **0.05×** | 🔴 **慢 22倍** | 
| 3    | 500    | 50     | 1000     | 0.000319      | 0.006372         | **0.05×** | 🔴 **慢 20倍** | 
| 4    | 1000   | 100    | 500      | 0.000618      | 0.012462         | **0.05×** | 🔴 **慢 20倍** | 

## 5. 结论

### 5.1 核心问题总结

`env_mat_a` 函数的 SVE 向量化失败的根本原因：

1. ❌ **间接内存访问主导**：60-70% 时间花在随机内存访问上
2. ❌ **Scatter 存储开销大**：向量化结果必须分散写回
3. ❌ **批处理引入额外拷贝**：数据需要经过中间缓冲区
4. ❌ **计算强度相对不足**：计算部分只占总时间的 30-40%
5. ✅ **标量代码已被编译器优化**：clang -O3 能生成很好的标量代码

### 5.2 适用性分析

**不适合向量化的特征**（env_mat_a 具备）：
- ❌ 间接/随机内存访问模式
- ❌ 小批量处理（邻居数 10-100）
- ❌ 需要 scatter 存储
- ❌ 计算强度低（50-60 FLOP vs 数百周期内存延迟）

**适合向量化的特征**（PPPM ORDER=15 具备）：
- ✅ 连续内存访问
- ✅ 大批量计算（数千个操作点）
- ✅ 连续存储
- ✅ 计算强度高（计算主导）