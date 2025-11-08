# PPPM fieldforce_ik 函数 SVE 向量化优化

## 1. 计算模式简介

PPPM (Particle-Particle Particle-Mesh) 的 `fieldforce_ik` 函数用于计算粒子在网格场中受到的静电力。其核心计算模式为：

```
对每个粒子 i:
    for n in [NLOWER, NUPPER]:      # 外层循环
        for m in [NLOWER, NUPPER]:  # 中层循环
            for l in [NLOWER, NUPPER]:  # 内层循环 ← 向量化目标
                ekx -= weight[l] * grid_vdx[mz][my][mx]
                eky -= weight[l] * grid_vdy[mz][my][mx]
                ekz -= weight[l] * grid_vdz[mz][my][mx]
```

**关键特征**：
- 三重嵌套循环，插值阶数 ORDER = NUPPER - NLOWER + 1
- 最内层循环长度固定为 ORDER
- 内层循环执行连续的向量化乘加运算（FMA操作）
- 每个粒子需要计算 ORDER³ 个网格点的贡献

## 2. 优化策略

### 2.1 向量化方案

对**最内层循环**进行 SVE 向量化：

```c
// 原始标量代码
for (int l = NLOWER; l <= NUPPER; l++) {
    float x0 = y0 * weights.rho1d[0][l - NLOWER];
    ekx -= x0 * grid->vdx[mz][my][mx];
    eky -= x0 * grid->vdy[mz][my][mx];
    ekz -= x0 * grid->vdz[mz][my][mx];
}

// SVE向量化代码
svfloat32_t acc_x = svdup_f32(0.0f);
svfloat32_t acc_y = svdup_f32(0.0f);
svfloat32_t acc_z = svdup_f32(0.0f);

for (int l_base = 0; l_base < vec_len; l_base += SVL) {
    svbool_t pg = svwhilelt_b32(l_base, vec_len);
    
    // 向量化读取权重和网格数据
    svfloat32_t vec_rho = svld1(pg, &weights.rho1d[0][l_idx_start]);
    svfloat32_t vec_x0 = svmul_m(pg, vec_y0, vec_rho);
    svfloat32_t vec_gx = svld1(pg, &grid->vdx[mz][my][mx_idx_start]);
    
    // 向量化FMA: acc -= x0 * grid
    acc_x = svmls_m(pg, acc_x, vec_x0, vec_gx);
}

// 规约求和
ekx += svaddv(svptrue_b32(), acc_x);
```

### 2.2 关键技术点

1. **谓词掩码**：使用 `svwhilelt_b32` 处理边界情况
2. **向量化FMA**：使用 `svmls_m` (multiply-subtract) 指令
3. **水平规约**：使用 `svaddv` 将向量寄存器累加为标量
4. **内存对齐**：确保连续内存访问以获得最佳性能

## 3. 性能测试结果

测试环境：
- **SVE向量长度**：16 个 float32 元素 (512-bit)
- **测试平台**：ARM v9 + SME2

### 3.1 ORDER = 7 的情况（范围 -3 到 3）

| 粒子数 | 迭代次数 | 标量时间 | SVE时间 | 加速比 | 性能 |
|--------|---------|---------|---------|--------|------|
| 100    | 100     | 0.057 ms | 0.076 ms | **0.75×** | ❌ **25% 性能下降** |
| 500    | 100     | 0.287 ms | 0.382 ms | **0.75×** | ❌ **25% 性能下降** |
| 1,000  | 100     | 0.576 ms | 0.767 ms | **0.75×** | ❌ **25% 性能下降** |
| 5,000  | 50      | 2.839 ms | 3.849 ms | **0.74×** | ❌ **26% 性能下降** |
| 10,000 | 20      | 5.668 ms | 7.722 ms | **0.73×** | ❌ **27% 性能下降** |

**分析**：
- ❌ **向量化反而变慢**
- 向量利用率低：7/16 = **43.75%**
- 向量化开销大于收益：
  - 谓词掩码开销
  - 水平规约开销
  - 循环控制开销

### 3.2 ORDER = 15 的情况（范围 -7 到 7）

| 粒子数 | 迭代次数 | 标量时间 | SVE时间 | 加速比 | 性能提升 |
|--------|---------|---------|---------|--------|----------|
| 100    | 100     | 0.485 ms | 0.285 ms | **1.70×** | ✅ **70%** |
| 500    | 100     | 2.417 ms | 1.424 ms | **1.70×** | ✅ **70%** |
| 1,000  | 100     | 4.809 ms | 2.843 ms | **1.69×** | ✅ **69%** |
| 2,000  | 50      | 9.656 ms | 5.707 ms | **1.69×** | ✅ **69%** |
| 5,000  | 50      | 23.711 ms | 14.121 ms | **1.68×** | ✅ **68%** |
| 10,000 | 20      | 49.700 ms | 28.072 ms | **1.77×** | ✅ **77%** |

**分析**：
- ✅ **稳定的 1.7× 加速比**
- 向量利用率高：15/16 = **93.75%**
- 计算强度增加：15³ = 3,375 个操作点 vs 7³ = 343 个操作点
- 向量化开销相对于计算量更小