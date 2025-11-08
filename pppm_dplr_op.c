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
#define ORDER 7
#define NLOWER (-3)
#define NUPPER 3
// #define ORDER 15
// #define NLOWER (-7)
// #define NUPPER 7

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

        // 使用高斯函数作为插值权重
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
// 版本1: 标量基准版本（与原始代码完全一致）
// ============================================================================
void fieldforce_ik_baseline(const Particle* particles, int nparticles,
                            const GridData* grid, float* fele,
                            float qqrd2e_scale) {
    InterpolationWeights weights;

    for (int i = 0; i < nparticles; i++) {
        const Particle* p = &particles[i];

        // 计算相对位置
        float dx = p->nx + 0.5f - p->x;
        float dy = p->ny + 0.5f - p->y;
        float dz = p->nz + 0.5f - p->z;

        // 计算插值权重
        compute_rho1d(&weights, dx, dy, dz);

        float ekx = 0.0f, eky = 0.0f, ekz = 0.0f;

        // 三重循环：原始实现
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

        // 转换为力
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

        // 外两层循环保持标量
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

                // 最内层循环向量化
                // 计算有效的 l 范围
                int l_start = NLOWER;
                int l_end = NUPPER;
                int mx_start = l_start + p->nx;

                // 边界检查
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

                // SVE 向量化处理
                svfloat32_t acc_x = svdup_f32(0.0f);
                svfloat32_t acc_y = svdup_f32(0.0f);
                svfloat32_t acc_z = svdup_f32(0.0f);
                svfloat32_t vec_y0 = svdup_f32(y0);

                for (int l_base = 0; l_base < vec_len; l_base += SVL) {
                    svbool_t pg = svwhilelt_b32(l_base, vec_len);

                    int l_idx_start = l_start + l_base - NLOWER;
                    int mx_idx_start = mx_start + l_base;

                    // 向量化读取 rho1d[0]
                    svfloat32_t vec_rho =
                        svld1(pg, &weights.rho1d[0][l_idx_start]);

                    // 计算 x0 = y0 * rho1d[0][l]
                    svfloat32_t vec_x0 = svmul_m(pg, vec_y0, vec_rho);

                    // 向量化读取网格数据
                    svfloat32_t vec_gx =
                        svld1(pg, &grid->vdx[mz][my][mx_idx_start]);
                    svfloat32_t vec_gy =
                        svld1(pg, &grid->vdy[mz][my][mx_idx_start]);
                    svfloat32_t vec_gz =
                        svld1(pg, &grid->vdz[mz][my][mx_idx_start]);

                    // 向量化累加: ekx -= x0 * vdx
                    acc_x = svmls_m(pg, acc_x, vec_x0, vec_gx);
                    acc_y = svmls_m(pg, acc_y, vec_x0, vec_gy);
                    acc_z = svmls_m(pg, acc_z, vec_x0, vec_gz);
                }

                // 规约求和
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
    printf("%-35s", name);

    // 预热
    func(particles, nparticles, grid, fele, qqrd2e_scale);

    // 正式测试
    double start = get_time_seconds();
    for (int i = 0; i < iterations; i++) {
        func(particles, nparticles, grid, fele, qqrd2e_scale);
    }
    double end = get_time_seconds();

    double avg_time = (end - start) / iterations;

    // 计算 GFLOPS（每个粒子约 343*9 次浮点运算）
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

    printf(
        "================================================================\n");
    printf("          PPPM fieldforce_ik 优化性能测试\n");
    printf(
        "================================================================\n");
    printf("SVE向量长度: %llu 个FP32元素\n", (unsigned long long)SVL);
    printf("插值阶数:    %d (范围: %d 到 %d)\n", ORDER, NLOWER, NUPPER);
    printf("模板点数:    %d (= %d^3)\n", ORDER * ORDER * ORDER, ORDER);
    printf("Grid尺寸:    %d × %d × %d\n", GRID_SIZE, GRID_SIZE, GRID_SIZE);
    printf(
        "================================================================\n\n");

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
               "-\n");
        printf("%-35s %11s  %15s  %s\n", "方法", "时间", "性能", "验证");
        printf("---------------------------------------------------------------"
               "-\n");

        // 初始化粒子
        // MacOS 要求 size 必须是 alignment 的整数倍
        size_t alloc_size = nparticles * sizeof(Particle);
        alloc_size = ((alloc_size + 63) / 64) * 64; // 向上对齐到 64 字节
        Particle* particles = (Particle*)aligned_alloc(64, alloc_size);
        if (particles == NULL) {
            printf("错误：内存分配失败\n");
            continue;
        }
        init_particles(particles, nparticles, &grid);

        // 分配输出数组
        // MacOS 要求 size 必须是 alignment 的整数倍
        size_t fele_size = nparticles * 3 * sizeof(float);
        fele_size = ((fele_size + 63) / 64) * 64; // 向上对齐到 64 字节
        float* fele_baseline = (float*)aligned_alloc(64, fele_size);
        float* fele_sve = (float*)aligned_alloc(64, fele_size);
        float* fele_sme2 = (float*)aligned_alloc(64, fele_size);

        if (fele_baseline == NULL || fele_sve == NULL || fele_sme2 == NULL) {
            printf("错误：输出数组内存分配失败\n");
            free(particles);
            if (fele_baseline)
                free(fele_baseline);
            if (fele_sve)
                free(fele_sve);
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
                      particles, nparticles, &grid, fele_sve, qqrd2e_scale,
                      iterations, fele_baseline);

        // 清理
        free(particles);
        free(fele_baseline);
        free(fele_sve);
        free(fele_sme2);
    }

    // 清理 grid
    free_3d_array(grid.vdx, GRID_SIZE, GRID_SIZE);
    free_3d_array(grid.vdy, GRID_SIZE, GRID_SIZE);
    free_3d_array(grid.vdz, GRID_SIZE, GRID_SIZE);

    return 0;
}