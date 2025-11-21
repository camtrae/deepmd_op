// pppm_fieldforce_optimization.c
// PPPM fieldforce_ik 函数优化测试
// 编译: clang -O3 -march=armv9-a+sme2 -o pppm_opt
// pppm_fieldforce_optimization.c -lm

#include <arm_sme.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// ============================================================================
// 配置参数
// ============================================================================
#define ORDER 15
#define NLOWER (-7)
#define NUPPER 7

#define GRID_SIZE 64
#define MAX_PARTICLES 10000

// ============================================================================
// 数据结构
// ============================================================================
typedef struct {
    float x, y, z;
    float q;
    int nx, ny, nz;
} Particle;

typedef struct {
    float*** vdx;
    float*** vdy;
    float*** vdz;
    int nz, ny, nx;
} GridData;

typedef struct {
    float rho1d[3][ORDER];
} InterpolationWeights;

// ============================================================================
// 辅助函数
// ============================================================================
double get_time_seconds() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

float*** allocate_3d_array(int nz, int ny, int nx) {
    float*** arr = (float***)malloc(nz * sizeof(float**));
    for (int i = 0; i < nz; i++) {
        arr[i] = (float**)malloc(ny * sizeof(float*));
        for (int j = 0; j < ny; j++) {
            arr[i][j] = (float*)aligned_alloc(64, nx * sizeof(float));
            memset(arr[i][j], 0, nx * sizeof(float));
        }
    }
    return arr;
}

void free_3d_array(float*** arr, int nz, int ny) {
    for (int i = 0; i < nz; i++) {
        for (int j = 0; j < ny; j++) {
            free(arr[i][j]);
        }
        free(arr[i]);
    }
    free(arr);
}

void init_grid_data(GridData* grid) {
    srand(42);
    for (int k = 0; k < grid->nz; k++) {
        for (int j = 0; j < grid->ny; j++) {
            for (int i = 0; i < grid->nx; i++) {
                grid->vdx[k][j][i] =
                    ((float)rand() / (float)RAND_MAX - 0.5f) * 2.0f;
                grid->vdy[k][j][i] =
                    ((float)rand() / (float)RAND_MAX - 0.5f) * 2.0f;
                grid->vdz[k][j][i] =
                    ((float)rand() / (float)RAND_MAX - 0.5f) * 2.0f;
            }
        }
    }
}

void init_particles(Particle* particles, int nparticles, GridData* grid) {
    srand(43);
    for (int i = 0; i < nparticles; i++) {
        particles[i].x = (float)rand() / (float)RAND_MAX * (grid->nx - 10) + 5;
        particles[i].y = (float)rand() / (float)RAND_MAX * (grid->ny - 10) + 5;
        particles[i].z = (float)rand() / (float)RAND_MAX * (grid->nz - 10) + 5;
        particles[i].q = ((float)rand() / (float)RAND_MAX - 0.5f) * 2.0f;

        particles[i].nx = (int)particles[i].x;
        particles[i].ny = (int)particles[i].y;
        particles[i].nz = (int)particles[i].z;
    }
}

void compute_rho1d(InterpolationWeights* weights, float dx, float dy,
                   float dz) {
    float shift = 0.5f;

    for (int i = NLOWER; i <= NUPPER; i++) {
        int idx = i - NLOWER;
        float rx = (float)i - dx + shift;
        float ry = (float)i - dy + shift;
        float rz = (float)i - dz + shift;

        weights->rho1d[0][idx] = expf(-rx * rx * 0.5f);
        weights->rho1d[1][idx] = expf(-ry * ry * 0.5f);
        weights->rho1d[2][idx] = expf(-rz * rz * 0.5f);
    }

    // 归一化
    float sum_x = 0, sum_y = 0, sum_z = 0;
    for (int i = 0; i < ORDER; i++) {
        sum_x += weights->rho1d[0][i];
        sum_y += weights->rho1d[1][i];
        sum_z += weights->rho1d[2][i];
    }
    for (int i = 0; i < ORDER; i++) {
        weights->rho1d[0][i] /= sum_x;
        weights->rho1d[1][i] /= sum_y;
        weights->rho1d[2][i] /= sum_z;
    }
}

// ============================================================================
// 共享辅助函数：预计算和提取
// ============================================================================

// 预计算权重数组
void precompute_weights(const InterpolationWeights* weights,
                        float* weights_flat) {
    int idx = 0;
    for (int n = NLOWER; n <= NUPPER; n++) {
        float z0 = weights->rho1d[2][n - NLOWER];
        for (int m = NLOWER; m <= NUPPER; m++) {
            float y0 = z0 * weights->rho1d[1][m - NLOWER];
            for (int l = NLOWER; l <= NUPPER; l++) {
                weights_flat[idx] = y0 * weights->rho1d[0][l - NLOWER];
                idx++;
            }
        }
    }
}

// 提取网格数据到一维数组
void extract_grid_data(const GridData* grid, const Particle* p, float* vdx_flat,
                       float* vdy_flat, float* vdz_flat) {
    int idx = 0;
    for (int n = NLOWER; n <= NUPPER; n++) {
        int mz = n + p->nz;

        for (int m = NLOWER; m <= NUPPER; m++) {
            int my = m + p->ny;

            for (int l = NLOWER; l <= NUPPER; l++) {
                int mx = l + p->nx;

                // 边界检查：越界填充0
                if (mz >= 0 && mz < grid->nz && my >= 0 && my < grid->ny &&
                    mx >= 0 && mx < grid->nx) {
                    vdx_flat[idx] = grid->vdx[mz][my][mx];
                    vdy_flat[idx] = grid->vdy[mz][my][mx];
                    vdz_flat[idx] = grid->vdz[mz][my][mx];
                } else {
                    vdx_flat[idx] = 0.0f;
                    vdy_flat[idx] = 0.0f;
                    vdz_flat[idx] = 0.0f;
                }
                idx++;
            }
        }
    }
}

// ============================================================================
// 版本1: 标量基准版本
// ============================================================================
void fieldforce_ik_baseline(const Particle* particles, int nparticles,
                            const GridData* grid, float* fele,
                            float qqrd2e_scale) {
    InterpolationWeights weights;

    for (int i = 0; i < nparticles; i++) {
        const Particle* p = &particles[i];

        float dx = p->nx + 0.5f - p->x;
        float dy = p->ny + 0.5f - p->y;
        float dz = p->nz + 0.5f - p->z;

        compute_rho1d(&weights, dx, dy, dz);

        float ekx = 0.0f, eky = 0.0f, ekz = 0.0f;

        for (int n = NLOWER; n <= NUPPER; n++) {
            int mz = n + p->nz;
            if (mz < 0 || mz >= grid->nz)
                continue;
            float z0 = weights.rho1d[2][n - NLOWER];

            for (int m = NLOWER; m <= NUPPER; m++) {
                int my = m + p->ny;
                if (my < 0 || my >= grid->ny)
                    continue;
                float y0 = z0 * weights.rho1d[1][m - NLOWER];

                for (int l = NLOWER; l <= NUPPER; l++) {
                    int mx = l + p->nx;
                    if (mx < 0 || mx >= grid->nx)
                        continue;
                    float x0 = y0 * weights.rho1d[0][l - NLOWER];

                    ekx -= x0 * grid->vdx[mz][my][mx];
                    eky -= x0 * grid->vdy[mz][my][mx];
                    ekz -= x0 * grid->vdz[mz][my][mx];
                }
            }
        }

        float qfactor = qqrd2e_scale * p->q;
        fele[i * 3 + 0] = qfactor * ekx;
        fele[i * 3 + 1] = qfactor * eky;
        fele[i * 3 + 2] = qfactor * ekz;
    }
}

// ============================================================================
// 版本2: SVE 向量化最内层循环
// ============================================================================
__arm_locally_streaming void
fieldforce_ik_sve_inner(const Particle* particles, int nparticles,
                        const GridData* grid, float* fele, float qqrd2e_scale) {
    InterpolationWeights weights;
    uint64_t SVL = svcntsw();

    for (int i = 0; i < nparticles; i++) {
        const Particle* p = &particles[i];

        float dx = p->nx + 0.5f - p->x;
        float dy = p->ny + 0.5f - p->y;
        float dz = p->nz + 0.5f - p->z;

        compute_rho1d(&weights, dx, dy, dz);

        float ekx = 0.0f, eky = 0.0f, ekz = 0.0f;

        for (int n = NLOWER; n <= NUPPER; n++) {
            int mz = n + p->nz;
            if (mz < 0 || mz >= grid->nz)
                continue;
            float z0 = weights.rho1d[2][n - NLOWER];

            for (int m = NLOWER; m <= NUPPER; m++) {
                int my = m + p->ny;
                if (my < 0 || my >= grid->ny)
                    continue;
                float y0 = z0 * weights.rho1d[1][m - NLOWER];

                int l_start = NLOWER;
                int l_end = NUPPER;
                int mx_start = l_start + p->nx;

                if (mx_start < 0) {
                    l_start -= mx_start;
                    mx_start = 0;
                }
                if (mx_start + (l_end - l_start + 1) > grid->nx) {
                    l_end = l_start + (grid->nx - mx_start - 1);
                }

                int vec_len = l_end - l_start + 1;
                if (vec_len <= 0)
                    continue;

                svfloat32_t acc_x = svdup_f32(0.0f);
                svfloat32_t acc_y = svdup_f32(0.0f);
                svfloat32_t acc_z = svdup_f32(0.0f);
                svfloat32_t vec_y0 = svdup_f32(y0);

                for (int l_base = 0; l_base < vec_len; l_base += SVL) {
                    svbool_t pg = svwhilelt_b32(l_base, vec_len);

                    int l_idx_start = l_start + l_base - NLOWER;
                    int mx_idx_start = mx_start + l_base;

                    svfloat32_t vec_rho =
                        svld1(pg, &weights.rho1d[0][l_idx_start]);
                    svfloat32_t vec_x0 = svmul_m(pg, vec_y0, vec_rho);

                    svfloat32_t vec_gx =
                        svld1(pg, &grid->vdx[mz][my][mx_idx_start]);
                    svfloat32_t vec_gy =
                        svld1(pg, &grid->vdy[mz][my][mx_idx_start]);
                    svfloat32_t vec_gz =
                        svld1(pg, &grid->vdz[mz][my][mx_idx_start]);

                    acc_x = svmls_m(pg, acc_x, vec_x0, vec_gx);
                    acc_y = svmls_m(pg, acc_y, vec_x0, vec_gy);
                    acc_z = svmls_m(pg, acc_z, vec_x0, vec_gz);
                }

                ekx += svaddv(svptrue_b32(), acc_x);
                eky += svaddv(svptrue_b32(), acc_y);
                ekz += svaddv(svptrue_b32(), acc_z);
            }
        }

        float qfactor = qqrd2e_scale * p->q;
        fele[i * 3 + 0] = qfactor * ekx;
        fele[i * 3 + 1] = qfactor * eky;
        fele[i * 3 + 2] = qfactor * ekz;
    }
}

// ============================================================================
// 版本3: 纯SVE多向量组（无ZA）- 三重循环完全展开
// ============================================================================

// SVE多向量核心计算函数
__arm_locally_streaming void
compute_field_force_sve_multi(const float* weights, const float* vdx_flat,
                              const float* vdy_flat, const float* vdz_flat,
                              uint64_t size, float* ekx_out, float* eky_out,
                              float* ekz_out) {
    uint64_t SVL = svcntsw();

    // 使用4组累加器处理每个分量，提高ILP（指令级并行）
    svfloat32_t acc_x0 = svdup_f32(0.0f);
    svfloat32_t acc_x1 = svdup_f32(0.0f);
    svfloat32_t acc_x2 = svdup_f32(0.0f);
    svfloat32_t acc_x3 = svdup_f32(0.0f);

    svfloat32_t acc_y0 = svdup_f32(0.0f);
    svfloat32_t acc_y1 = svdup_f32(0.0f);
    svfloat32_t acc_y2 = svdup_f32(0.0f);
    svfloat32_t acc_y3 = svdup_f32(0.0f);

    svfloat32_t acc_z0 = svdup_f32(0.0f);
    svfloat32_t acc_z1 = svdup_f32(0.0f);
    svfloat32_t acc_z2 = svdup_f32(0.0f);
    svfloat32_t acc_z3 = svdup_f32(0.0f);

    // 主循环：每次处理 4*SVL 个元素
    for (uint64_t i = 0; i < size; i += 4 * SVL) {
        svcount_t pc = svwhilelt_c32(i, size, 4);

        // 加载权重（4个向量组）
        svfloat32x4_t w = svld1_x4(pc, &weights[i]);

        // 加载vdx数据
        svfloat32x4_t vdx = svld1_x4(pc, &vdx_flat[i]);

        // 加载vdy数据
        svfloat32x4_t vdy = svld1_x4(pc, &vdy_flat[i]);

        // 加载vdz数据
        svfloat32x4_t vdz = svld1_x4(pc, &vdz_flat[i]);

        // 分解向量组（每组包含4个向量）
        svfloat32_t w0 = svget4(w, 0);
        svfloat32_t w1 = svget4(w, 1);
        svfloat32_t w2 = svget4(w, 2);
        svfloat32_t w3 = svget4(w, 3);

        svfloat32_t vdx0 = svget4(vdx, 0);
        svfloat32_t vdx1 = svget4(vdx, 1);
        svfloat32_t vdx2 = svget4(vdx, 2);
        svfloat32_t vdx3 = svget4(vdx, 3);

        svfloat32_t vdy0 = svget4(vdy, 0);
        svfloat32_t vdy1 = svget4(vdy, 1);
        svfloat32_t vdy2 = svget4(vdy, 2);
        svfloat32_t vdy3 = svget4(vdy, 3);

        svfloat32_t vdz0 = svget4(vdz, 0);
        svfloat32_t vdz1 = svget4(vdz, 1);
        svfloat32_t vdz2 = svget4(vdz, 2);
        svfloat32_t vdz3 = svget4(vdz, 3);

        // 创建全真谓词（用于无条件执行）
        svbool_t pg = svptrue_b32();

        // 累加 ekx（使用负号实现减法）
        acc_x0 = svmls_x(pg, acc_x0, w0, vdx0); // acc -= w * vdx
        acc_x1 = svmls_x(pg, acc_x1, w1, vdx1);
        acc_x2 = svmls_x(pg, acc_x2, w2, vdx2);
        acc_x3 = svmls_x(pg, acc_x3, w3, vdx3);

        // 累加 eky
        acc_y0 = svmls_x(pg, acc_y0, w0, vdy0);
        acc_y1 = svmls_x(pg, acc_y1, w1, vdy1);
        acc_y2 = svmls_x(pg, acc_y2, w2, vdy2);
        acc_y3 = svmls_x(pg, acc_y3, w3, vdy3);

        // 累加 ekz
        acc_z0 = svmls_x(pg, acc_z0, w0, vdz0);
        acc_z1 = svmls_x(pg, acc_z1, w1, vdz1);
        acc_z2 = svmls_x(pg, acc_z2, w2, vdz2);
        acc_z3 = svmls_x(pg, acc_z3, w3, vdz3);
    }

    // 合并4组累加器
    svbool_t pg = svptrue_b32();
    svfloat32_t acc_x = svadd_x(pg, acc_x0, acc_x1);
    acc_x = svadd_x(pg, acc_x, acc_x2);
    acc_x = svadd_x(pg, acc_x, acc_x3);

    svfloat32_t acc_y = svadd_x(pg, acc_y0, acc_y1);
    acc_y = svadd_x(pg, acc_y, acc_y2);
    acc_y = svadd_x(pg, acc_y, acc_y3);

    svfloat32_t acc_z = svadd_x(pg, acc_z0, acc_z1);
    acc_z = svadd_x(pg, acc_z, acc_z2);
    acc_z = svadd_x(pg, acc_z, acc_z3);

    // 水平规约求和
    *ekx_out = svaddv(pg, acc_x);
    *eky_out = svaddv(pg, acc_y);
    *ekz_out = svaddv(pg, acc_z);
}

// 完整的fieldforce函数（版本3）
void fieldforce_ik_sve_multi(const Particle* particles, int nparticles,
                             const GridData* grid, float* fele,
                             float qqrd2e_scale) {
    InterpolationWeights weights;
    const int total_size = ORDER * ORDER * ORDER; // 3375

    // ✅ 修正：向上对齐到64字节边界
    size_t alloc_size = total_size * sizeof(float); // 13500
    alloc_size = ((alloc_size + 63) / 64) * 64;     // 向上对齐到 13504

    float* weights_flat = (float*)aligned_alloc(64, alloc_size);
    float* vdx_flat = (float*)aligned_alloc(64, alloc_size);
    float* vdy_flat = (float*)aligned_alloc(64, alloc_size);
    float* vdz_flat = (float*)aligned_alloc(64, alloc_size);

    if (weights_flat == NULL)
        fprintf(stderr,
                "内存分配失败：aligned_alloc 未能成功分配 %zu 字节对齐内存\n",
                total_size * sizeof(float));

    for (int i = 0; i < nparticles; i++) {
        const Particle* p = &particles[i];

        // 计算插值权重
        float dx = p->nx + 0.5f - p->x;
        float dy = p->ny + 0.5f - p->y;
        float dz = p->nz + 0.5f - p->z;
        compute_rho1d(&weights, dx, dy, dz);

        // 阶段1：预计算权重
        precompute_weights(&weights, weights_flat);

        // 阶段2：提取网格数据
        extract_grid_data(grid, p, vdx_flat, vdy_flat, vdz_flat);

        // // 阶段3：SVE多向量计算
        float ekx = 0, eky = 0, ekz = 0;
        compute_field_force_sve_multi(weights_flat, vdx_flat, vdy_flat,
                                      vdz_flat, total_size, &ekx, &eky, &ekz);

        // 转换为力
        float qfactor = qqrd2e_scale * p->q;
        fele[i * 3 + 0] = qfactor * ekx;
        fele[i * 3 + 1] = qfactor * eky;
        fele[i * 3 + 2] = qfactor * ekz;
    }

    free(weights_flat);
    free(vdx_flat);
    free(vdy_flat);
    free(vdz_flat);
}

// ============================================================================
// 版本4: SME2 策略B - 3个ZA数组并行三分量
// ============================================================================

// SME2核心计算函数：使用3个ZA数组
__arm_new("za") __arm_locally_streaming void compute_field_force_sme2_3za(
    const float* weights, const float* vdx_flat, const float* vdy_flat,
    const float* vdz_flat, uint64_t size, float* ekx_out, float* eky_out,
    float* ekz_out) {
    svzero_za();
    uint64_t SVL = svcntsw();

    // 使用3个ZA数组分别累加 ekx, eky, ekz
    for (uint64_t i = 0; i < size; i += 4 * SVL) {
        svcount_t pc = svwhilelt_c32(i, size, 4);

        // 加载权重（只加载一次）
        svfloat32x4_t weight_vec = svld1_x4(pc, &weights[i]);

        // 加载vdx数据并累加到ZA[0]
        svfloat32x4_t vdx_vec = svld1_x4(pc, &vdx_flat[i]);
        svmla_za32_f32_vg1x4(0, weight_vec, vdx_vec);

        // 加载vdy数据并累加到ZA[1]（复用weights）
        svfloat32x4_t vdy_vec = svld1_x4(pc, &vdy_flat[i]);
        svmla_za32_f32_vg1x4(1, weight_vec, vdy_vec);

        // 加载vdz数据并累加到ZA[2]（复用weights）
        svfloat32x4_t vdz_vec = svld1_x4(pc, &vdz_flat[i]);
        svmla_za32_f32_vg1x4(2, weight_vec, vdz_vec);
    }

    // // 从ZA[0]读取ekx的累加结果（读取4行并求和）
    // svfloat32_t ekx_row0 =
    //     svread_hor_za32_f32_m(svundef_f32(), svptrue_b32(), 0, 0);
    // svfloat32_t ekx_row1 =
    //     svread_hor_za32_f32_m(svundef_f32(), svptrue_b32(), 0, 4);
    // svfloat32_t ekx_row2 =
    //     svread_hor_za32_f32_m(svundef_f32(), svptrue_b32(), 0, 8);
    // svfloat32_t ekx_row3 =
    //     svread_hor_za32_f32_m(svundef_f32(), svptrue_b32(), 0, 12);
    // svfloat32_t ekx_sum = svadd_x(svptrue_b32(), ekx_row0, ekx_row1);
    // ekx_sum = svadd_x(svptrue_b32(), ekx_sum, ekx_row2);
    // ekx_sum = svadd_x(svptrue_b32(), ekx_sum, ekx_row3);
    // *ekx_out = -svaddv(svptrue_b32(), ekx_sum);

    // // 从ZA[1]读取eky的累加结果
    // svfloat32_t eky_row0 =
    //     svread_hor_za32_f32_m(svundef_f32(), svptrue_b32(), 1, 0);
    // svfloat32_t eky_row1 =
    //     svread_hor_za32_f32_m(svundef_f32(), svptrue_b32(), 1, 4);
    // svfloat32_t eky_row2 =
    //     svread_hor_za32_f32_m(svundef_f32(), svptrue_b32(), 1, 8);
    // svfloat32_t eky_row3 =
    //     svread_hor_za32_f32_m(svundef_f32(), svptrue_b32(), 1, 12);
    // svfloat32_t eky_sum = svadd_x(svptrue_b32(), eky_row0, eky_row1);
    // eky_sum = svadd_x(svptrue_b32(), eky_sum, eky_row2);
    // eky_sum = svadd_x(svptrue_b32(), eky_sum, eky_row3);
    // *eky_out = -svaddv(svptrue_b32(), eky_sum);

    // // 从ZA[2]读取ekz的累加结果
    // svfloat32_t ekz_row0 =
    //     svread_hor_za32_f32_m(svundef_f32(), svptrue_b32(), 2, 0);
    // svfloat32_t ekz_row1 =
    //     svread_hor_za32_f32_m(svundef_f32(), svptrue_b32(), 2, 4);
    // svfloat32_t ekz_row2 =
    //     svread_hor_za32_f32_m(svundef_f32(), svptrue_b32(), 2, 8);
    // svfloat32_t ekz_row3 =
    //     svread_hor_za32_f32_m(svundef_f32(), svptrue_b32(), 2, 12);
    // svfloat32_t ekz_sum = svadd_x(svptrue_b32(), ekz_row0, ekz_row1);
    // ekz_sum = svadd_x(svptrue_b32(), ekz_sum, ekz_row2);
    // ekz_sum = svadd_x(svptrue_b32(), ekz_sum, ekz_row3);
    // *ekz_out = -svaddv(svptrue_b32(), ekz_sum);
}

// 完整的fieldforce函数（版本4）
void fieldforce_ik_sme2_3za(const Particle* particles, int nparticles,
                            const GridData* grid, float* fele,
                            float qqrd2e_scale) {
    InterpolationWeights weights;
    const int total_size = ORDER * ORDER * ORDER;

    size_t alloc_size = total_size * sizeof(float); // 13500
    alloc_size = ((alloc_size + 63) / 64) * 64;     // 向上对齐到 13504

    float* weights_flat = (float*)aligned_alloc(64, alloc_size);
    float* vdx_flat = (float*)aligned_alloc(64, alloc_size);
    float* vdy_flat = (float*)aligned_alloc(64, alloc_size);
    float* vdz_flat = (float*)aligned_alloc(64, alloc_size);

    for (int i = 0; i < nparticles; i++) {
        const Particle* p = &particles[i];

        // 计算插值权重
        float dx = p->nx + 0.5f - p->x;
        float dy = p->ny + 0.5f - p->y;
        float dz = p->nz + 0.5f - p->z;
        compute_rho1d(&weights, dx, dy, dz);

        // 阶段1：预计算权重
        precompute_weights(&weights, weights_flat);

        // 阶段2：提取网格数据
        extract_grid_data(grid, p, vdx_flat, vdy_flat, vdz_flat);

        // 阶段3：SME2向量化计算
        float ekx, eky, ekz;
        compute_field_force_sme2_3za(weights_flat, vdx_flat, vdy_flat, vdz_flat,
                                     total_size, &ekx, &eky, &ekz);

        // 转换为力
        float qfactor = qqrd2e_scale * p->q;
        fele[i * 3 + 0] = qfactor * ekx;
        fele[i * 3 + 1] = qfactor * eky;
        fele[i * 3 + 2] = qfactor * ekz;
    }

    free(weights_flat);
    free(vdx_flat);
    free(vdy_flat);
    free(vdz_flat);
}

// ============================================================================
// 验证和性能测试
// ============================================================================
float compute_max_error(const float* a, const float* b, int n) {
    float max_err = 0.0f;
    for (int i = 0; i < n; i++) {
        float err = fabsf(a[i] - b[i]);
        if (err > max_err)
            max_err = err;
    }
    return max_err;
}

void run_benchmark(const char* name,
                   void (*func)(const Particle*, int, const GridData*, float*,
                                float),
                   const Particle* particles, int nparticles,
                   const GridData* grid, float* fele, float qqrd2e_scale,
                   int iterations, const float* reference_result) {
    printf("%-45s", name);

    // 预热
    func(particles, nparticles, grid, fele, qqrd2e_scale);

    // 正式测试
    double start = get_time_seconds();
    for (int i = 0; i < iterations; i++) {
        func(particles, nparticles, grid, fele, qqrd2e_scale);
    }
    double end = get_time_seconds();

    double avg_time = (end - start) / iterations;

    // 计算 GFLOPS
    long long ops_per_particle = (long long)(NUPPER - NLOWER + 1) *
                                 (NUPPER - NLOWER + 1) * (NUPPER - NLOWER + 1) *
                                 9;
    double gflops = (nparticles * ops_per_particle / 1e9) / avg_time;

    printf(" %8.3f ms  %7.2f GFLOPS", avg_time * 1000.0, gflops);

    // 验证正确性
    if (reference_result != NULL) {
        float max_err =
            compute_max_error(fele, reference_result, nparticles * 3);
        printf("  误差: %.2e", max_err);

        if (max_err < 1e-3) {
            printf(" ✓");
        } else {
            printf(" ✗ WARNING");
        }
    }
    printf("\n");
}

// ============================================================================
// 主函数
// ============================================================================
int main(int argc, char* argv[]) {
    if (!__arm_has_sme()) {
        printf("错误：系统不支持 SME\n");
        return 1;
    }

    uint64_t SVL = svcntsw();

    printf("==================================================================="
           "=============\n");
    printf("          PPPM fieldforce_ik 优化性能测试 (SVE多向量 vs SME2)\n");
    printf("==================================================================="
           "=============\n");
    printf("SVE向量长度: %llu 个FP32元素\n", (unsigned long long)SVL);
    printf("插值阶数:    %d (范围: %d 到 %d)\n", ORDER, NLOWER, NUPPER);
    printf("模板点数:    %d (= %d^3)\n", ORDER * ORDER * ORDER, ORDER);
    printf("Grid尺寸:    %d × %d × %d\n", GRID_SIZE, GRID_SIZE, GRID_SIZE);
    printf("每次处理:    4*SVL = %llu 个元素\n", (unsigned long long)(4 * SVL));
    printf("==================================================================="
           "=============\n\n");

    // 测试不同粒子数
    int test_sizes[] = {100, 500, 1000, 2000, 5000, 10000};
    int num_tests = sizeof(test_sizes) / sizeof(test_sizes[0]);

    // 初始化 grid
    GridData grid;
    grid.nx = grid.ny = grid.nz = GRID_SIZE;
    grid.vdx = allocate_3d_array(GRID_SIZE, GRID_SIZE, GRID_SIZE);
    grid.vdy = allocate_3d_array(GRID_SIZE, GRID_SIZE, GRID_SIZE);
    grid.vdz = allocate_3d_array(GRID_SIZE, GRID_SIZE, GRID_SIZE);
    init_grid_data(&grid);

    for (int test_idx = 0; test_idx < num_tests; test_idx++) {
        int nparticles = test_sizes[test_idx];
        int iterations =
            (nparticles <= 1000) ? 100 : ((nparticles <= 5000) ? 50 : 20);

        printf("\n测试 %d/%d: %d 个粒子 (迭代 %d 次)\n", test_idx + 1,
               num_tests, nparticles, iterations);
        printf("---------------------------------------------------------------"
               "-----------------\n");
        printf("%-45s %11s  %15s  %s\n", "方法", "时间", "性能", "验证");
        printf("---------------------------------------------------------------"
               "-----------------\n");

        // 初始化粒子
        size_t alloc_size = nparticles * sizeof(Particle);
        alloc_size = ((alloc_size + 63) / 64) * 64;
        Particle* particles = (Particle*)aligned_alloc(64, alloc_size);
        if (particles == NULL) {
            printf("错误：内存分配失败\n");
            continue;
        }
        init_particles(particles, nparticles, &grid);

        // 分配输出数组
        size_t fele_size = nparticles * 3 * sizeof(float);
        fele_size = ((fele_size + 63) / 64) * 64;
        float* fele_baseline = (float*)aligned_alloc(64, fele_size);
        float* fele_sve_inner = (float*)aligned_alloc(64, fele_size);
        float* fele_sve_multi = (float*)aligned_alloc(64, fele_size);
        float* fele_sme2 = (float*)aligned_alloc(64, fele_size);

        if (fele_baseline == NULL || fele_sve_inner == NULL ||
            fele_sve_multi == NULL || fele_sme2 == NULL) {
            printf("错误：输出数组内存分配失败\n");
            free(particles);
            if (fele_baseline)
                free(fele_baseline);
            if (fele_sve_inner)
                free(fele_sve_inner);
            if (fele_sve_multi)
                free(fele_sve_multi);
            if (fele_sme2)
                free(fele_sme2);
            break;
        }

        float qqrd2e_scale = 1.0f;

        // 测试所有版本
        run_benchmark("版本1: 标量基准", fieldforce_ik_baseline, particles,
                      nparticles, &grid, fele_baseline, qqrd2e_scale,
                      iterations, NULL);

        run_benchmark("版本2: SVE向量化(最内层循环)", fieldforce_ik_sve_inner,
                      particles, nparticles, &grid, fele_sve_inner,
                      qqrd2e_scale, iterations, fele_baseline);

        run_benchmark("版本3: SVE多向量组(三重循环展开,无ZA)",
                      fieldforce_ik_sve_multi, particles, nparticles, &grid,
                      fele_sve_multi, qqrd2e_scale, iterations, fele_baseline);

        run_benchmark("版本4: SME2策略B(3ZA并行三分量)", fieldforce_ik_sme2_3za,
                      particles, nparticles, &grid, fele_sme2, qqrd2e_scale,
                      iterations, fele_baseline);

        // 清理
        free(particles);
        free(fele_baseline);
        free(fele_sve_inner);
        free(fele_sve_multi);
        free(fele_sme2);
    }

    // 清理 grid
    free_3d_array(grid.vdx, GRID_SIZE, GRID_SIZE);
    free_3d_array(grid.vdy, GRID_SIZE, GRID_SIZE);
    free_3d_array(grid.vdz, GRID_SIZE, GRID_SIZE);

    printf("\n================================================================="
           "===============\n");
    printf("测试完成！\n");
    printf("\n关键对比：\n");
    printf("  版本3 (SVE多向量): 无ZA开销，使用12个累加器提高ILP\n");
    printf("  版本4 (SME2 3ZA):  ZA开销，但指令更少，硬件加速\n");
    printf("\n预期结果：\n");
    printf("  - 小规模 (< 1000粒子): 版本3可能略快（避免ZA开销）\n");
    printf("  - 大规模 (> 2000粒子): 版本4应该更快（指令优势）\n");
    printf("==================================================================="
           "=============\n");

    return 0;
}