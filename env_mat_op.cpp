// env_mat_a_practical_optimized.cpp
// 实用优化版本 - 融合计算 + 向量化平滑函数 + 最小内存开销
// 编译: clang++ -O3 -march=armv9-a+sve2 -o benchmark_practical env_mat_a_practical_optimized.cpp

#include <arm_sme.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <vector>

double get_time_in_seconds() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

// ============================================================================
// 向量化的平滑函数（关键优化）
// ============================================================================

inline void spline5_switch_scalar(float& sw, float& dsw, float r, float rmin, float rmax) {
    if (r < rmin) {
        sw = 1.0f;
        dsw = 0.0f;
    } else if (r < rmax) {
        float uu = (r - rmin) / (rmax - rmin);
        float du = 1.0f / (rmax - rmin);
        sw = uu * uu * uu * (-6.0f * uu * uu + 15.0f * uu - 10.0f) + 1.0f;
        dsw = (3.0f * uu * uu * (-6.0f * uu * uu + 15.0f * uu - 10.0f) + 
               uu * uu * uu * (-12.0f * uu + 15.0f)) * du;
    } else {
        sw = 0.0f;
        dsw = 0.0f;
    }
}

__arm_locally_streaming
inline void spline5_switch_sve(
    svfloat32_t& vec_sw,
    svfloat32_t& vec_dsw,
    svfloat32_t vec_r,
    float rmin,
    float rmax,
    svbool_t pg)
{
    svfloat32_t vec_rmin = svdup_f32(rmin);
    svfloat32_t vec_rmax = svdup_f32(rmax);
    svfloat32_t vec_one = svdup_f32(1.0f);
    svfloat32_t vec_zero = svdup_f32(0.0f);
    
    // 三个区域的mask
    svbool_t mask_below = svand_z(pg, pg, svcmplt(pg, vec_r, vec_rmin));
    svbool_t mask_above = svand_z(pg, pg, svcmpge(pg, vec_r, vec_rmax));
    svbool_t mask_middle = svbic_z(pg, svbic_z(pg, pg, mask_below), mask_above);
    
    // r < rmin: sw = 1, dsw = 0
    vec_sw = svsel(mask_below, vec_one, vec_zero);
    vec_dsw = vec_zero;
    
    // rmin <= r < rmax: 计算多项式
    if (svptest_any(pg, mask_middle)) {
        svfloat32_t vec_range = svsub_x(pg, vec_rmax, vec_rmin);
        svfloat32_t vec_du = svdiv_x(mask_middle, vec_one, vec_range);
        svfloat32_t vec_uu = svmul_x(mask_middle, svsub_x(mask_middle, vec_r, vec_rmin), vec_du);
        
        svfloat32_t uu2 = svmul_x(mask_middle, vec_uu, vec_uu);
        svfloat32_t uu3 = svmul_x(mask_middle, uu2, vec_uu);
        
        // sw = uu³(-6uu² + 15uu - 10) + 1
        svfloat32_t poly = svmad_x(mask_middle, vec_uu, svdup_f32(15.0f), svdup_f32(-10.0f));
        poly = svmad_x(mask_middle, uu2, svdup_f32(-6.0f), poly);
        svfloat32_t sw_mid = svmad_x(mask_middle, uu3, poly, vec_one);
        vec_sw = svsel(mask_middle, sw_mid, vec_sw);
        
        // dsw = (3uu²×poly + uu³×(-12uu + 15)) × du
        svfloat32_t term1 = svmul_x(mask_middle, svmul_x(mask_middle, svdup_f32(3.0f), uu2), poly);
        svfloat32_t term2 = svmad_x(mask_middle, vec_uu, svdup_f32(-12.0f), svdup_f32(15.0f));
        term2 = svmul_x(mask_middle, uu3, term2);
        svfloat32_t dsw_mid = svmul_x(mask_middle, svadd_x(mask_middle, term1, term2), vec_du);
        vec_dsw = svsel(mask_middle, dsw_mid, vec_dsw);
    }
}

// ============================================================================
// Baseline实现
// ============================================================================

void env_mat_a_cpu_baseline(
    std::vector<float>& descrpt_a,
    std::vector<float>& descrpt_a_deriv,
    std::vector<float>& rij_a,
    const std::vector<float>& posi,
    const std::vector<int>& type,
    const int& i_idx,
    const std::vector<int>& fmt_nlist_a,
    const std::vector<int>& sec_a,
    const float& rmin,
    const float& rmax)
{
    descrpt_a.resize(sec_a.back() * 4, 0.0f);
    descrpt_a_deriv.resize(sec_a.back() * 4 * 3, 0.0f);
    rij_a.resize(sec_a.back() * 3, 0.0f);
    
    for (int sec_iter = 0; sec_iter < int(sec_a.size()) - 1; ++sec_iter) {
        for (int nei_iter = sec_a[sec_iter]; nei_iter < sec_a[sec_iter + 1]; ++nei_iter) {
            if (fmt_nlist_a[nei_iter] < 0) break;
            
            int j_idx = fmt_nlist_a[nei_iter];
            float rx = posi[j_idx * 3 + 0] - posi[i_idx * 3 + 0];
            float ry = posi[j_idx * 3 + 1] - posi[i_idx * 3 + 1];
            float rz = posi[j_idx * 3 + 2] - posi[i_idx * 3 + 2];
            
            float nr2 = rx * rx + ry * ry + rz * rz;
            float inr = 1.0f / sqrtf(nr2);
            float nr = nr2 * inr;
            float inr2 = inr * inr;
            float inr3 = inr2 * inr;
            float inr4 = inr2 * inr2;
            
            float sw, dsw;
            spline5_switch_scalar(sw, dsw, nr, rmin, rmax);
            
            float desc0 = inr;
            float desc1 = rx * inr2;
            float desc2 = ry * inr2;
            float desc3 = rz * inr2;
            
            rij_a[nei_iter * 3 + 0] = rx;
            rij_a[nei_iter * 3 + 1] = ry;
            rij_a[nei_iter * 3 + 2] = rz;
            
            descrpt_a[nei_iter * 4 + 0] = desc0 * sw;
            descrpt_a[nei_iter * 4 + 1] = desc1 * sw;
            descrpt_a[nei_iter * 4 + 2] = desc2 * sw;
            descrpt_a[nei_iter * 4 + 3] = desc3 * sw;
            
            // 12个导数分量
            descrpt_a_deriv[nei_iter * 12 + 0] = rx * inr3 * sw - desc0 * dsw * rx * inr;
            descrpt_a_deriv[nei_iter * 12 + 1] = ry * inr3 * sw - desc0 * dsw * ry * inr;
            descrpt_a_deriv[nei_iter * 12 + 2] = rz * inr3 * sw - desc0 * dsw * rz * inr;
            
            descrpt_a_deriv[nei_iter * 12 + 3] = (2.0f * rx * rx * inr4 - inr2) * sw - desc1 * dsw * rx * inr;
            descrpt_a_deriv[nei_iter * 12 + 4] = (2.0f * rx * ry * inr4) * sw - desc1 * dsw * ry * inr;
            descrpt_a_deriv[nei_iter * 12 + 5] = (2.0f * rx * rz * inr4) * sw - desc1 * dsw * rz * inr;
            
            descrpt_a_deriv[nei_iter * 12 + 6] = (2.0f * ry * rx * inr4) * sw - desc2 * dsw * rx * inr;
            descrpt_a_deriv[nei_iter * 12 + 7] = (2.0f * ry * ry * inr4 - inr2) * sw - desc2 * dsw * ry * inr;
            descrpt_a_deriv[nei_iter * 12 + 8] = (2.0f * ry * rz * inr4) * sw - desc2 * dsw * rz * inr;
            
            descrpt_a_deriv[nei_iter * 12 + 9] = (2.0f * rz * rx * inr4) * sw - desc3 * dsw * rx * inr;
            descrpt_a_deriv[nei_iter * 12 + 10] = (2.0f * rz * ry * inr4) * sw - desc3 * dsw * ry * inr;
            descrpt_a_deriv[nei_iter * 12 + 11] = (2.0f * rz * rz * inr4 - inr2) * sw - desc3 * dsw * rz * inr;
        }
    }
}

// ============================================================================
// 实用优化版本：融合计算 + 最小内存开销
// ============================================================================

__arm_locally_streaming
void env_mat_a_cpu_practical_optimized(
    std::vector<float>& descrpt_a,
    std::vector<float>& descrpt_a_deriv,
    std::vector<float>& rij_a,
    const std::vector<float>& posi,
    const std::vector<int>& type,
    const int& i_idx,
    const std::vector<int>& fmt_nlist_a,
    const std::vector<int>& sec_a,
    const float& rmin,
    const float& rmax)
{
    const uint64_t SVL = svcntw();
    const float center_x = posi[i_idx * 3 + 0];
    const float center_y = posi[i_idx * 3 + 1];
    const float center_z = posi[i_idx * 3 + 2];
    
    descrpt_a.resize(sec_a.back() * 4, 0.0f);
    descrpt_a_deriv.resize(sec_a.back() * 4 * 3, 0.0f);
    rij_a.resize(sec_a.back() * 3, 0.0f);
    
    // 栈上预分配缓冲区（避免动态分配）
    constexpr int MAX_BATCH = 128;
    float batch_x[MAX_BATCH];
    float batch_y[MAX_BATCH];
    float batch_z[MAX_BATCH];
    int batch_indices[MAX_BATCH];
    
    for (int sec_iter = 0; sec_iter < int(sec_a.size()) - 1; ++sec_iter) {
        int sec_start = sec_a[sec_iter];
        int sec_end = sec_a[sec_iter + 1];
        
        // 批处理：每次处理 SVL 个邻居
        for (int batch_start = sec_start; batch_start < sec_end; batch_start += SVL) {
            // 1. 收集一个批次的数据
            int batch_count = 0;
            for (int jj = batch_start; jj < sec_end && batch_count < MAX_BATCH && jj < batch_start + SVL; ++jj) {
                if (fmt_nlist_a[jj] < 0) break;
                
                int j_idx = fmt_nlist_a[jj];
                batch_x[batch_count] = posi[j_idx * 3 + 0] - center_x;
                batch_y[batch_count] = posi[j_idx * 3 + 1] - center_y;
                batch_z[batch_count] = posi[j_idx * 3 + 2] - center_z;
                batch_indices[batch_count] = jj;
                batch_count++;
            }
            
            if (batch_count == 0) break;
            
            svbool_t pg = svwhilelt_b32(0, batch_count);
            
            // 2. 向量化计算（全部在寄存器中）
            svfloat32_t vec_x = svld1_f32(pg, batch_x);
            svfloat32_t vec_y = svld1_f32(pg, batch_y);
            svfloat32_t vec_z = svld1_f32(pg, batch_z);
            
            // nr2 = x² + y² + z²
            svfloat32_t vec_nr2 = svmul_f32_x(pg, vec_x, vec_x);
            vec_nr2 = svmla_f32_m(pg, vec_nr2, vec_y, vec_y);
            vec_nr2 = svmla_f32_m(pg, vec_nr2, vec_z, vec_z);
            
            // inr = 1/sqrt(nr2)
            svfloat32_t vec_inr = svrsqrte_f32(vec_nr2);
            // Newton-Raphson: inr = inr * (1.5 - 0.5 * nr2 * inr²)
            svfloat32_t half = svdup_f32(0.5f);
            svfloat32_t one_half = svdup_f32(1.5f);
            svfloat32_t inr_sq = svmul_f32_x(pg, vec_inr, vec_inr);
            svfloat32_t correction = svmls_f32_m(pg, one_half, vec_nr2, svmul_f32_x(pg, inr_sq, half));
            vec_inr = svmul_f32_x(pg, vec_inr, correction);
            
            // 几何量
            svfloat32_t vec_nr = svmul_f32_x(pg, vec_nr2, vec_inr);
            svfloat32_t vec_inr2 = svmul_f32_x(pg, vec_inr, vec_inr);
            svfloat32_t vec_inr3 = svmul_f32_x(pg, vec_inr2, vec_inr);
            svfloat32_t vec_inr4 = svmul_f32_x(pg, vec_inr2, vec_inr2);
            
            // 3. 向量化平滑函数（关键优化！）
            svfloat32_t vec_sw, vec_dsw;
            spline5_switch_sve(vec_sw, vec_dsw, vec_nr, rmin, rmax, pg);
            
            // 4. 计算描述符
            svfloat32_t vec_desc0 = vec_inr;
            svfloat32_t vec_desc1 = svmul_f32_x(pg, vec_x, vec_inr2);
            svfloat32_t vec_desc2 = svmul_f32_x(pg, vec_y, vec_inr2);
            svfloat32_t vec_desc3 = svmul_f32_x(pg, vec_z, vec_inr2);
            
            // 5. 计算所有12个导数分量
            svfloat32_t vec_two = svdup_f32(2.0f);
            
            // 导数 0-2: ∂(1/r)/∂{x,y,z}
            svfloat32_t common_0 = svmul_f32_x(pg, svmul_f32_x(pg, vec_desc0, vec_dsw), vec_inr);
            svfloat32_t term1_0 = svmul_f32_x(pg, vec_inr3, vec_sw);
            
            svfloat32_t deriv_0 = svmls_f32_m(pg, svmul_f32_x(pg, vec_x, term1_0), common_0, vec_x);
            svfloat32_t deriv_1 = svmls_f32_m(pg, svmul_f32_x(pg, vec_y, term1_0), common_0, vec_y);
            svfloat32_t deriv_2 = svmls_f32_m(pg, svmul_f32_x(pg, vec_z, term1_0), common_0, vec_z);
            
            // 导数 3-5: ∂(x/r²)/∂{x,y,z}
            svfloat32_t common_1 = svmul_f32_x(pg, svmul_f32_x(pg, vec_desc1, vec_dsw), vec_inr);
            
            svfloat32_t term_xx = svsub_f32_x(pg, 
                svmul_f32_x(pg, svmul_f32_x(pg, svmul_f32_x(pg, vec_x, vec_x), vec_inr4), vec_two), 
                vec_inr2);
            svfloat32_t deriv_3 = svmls_f32_m(pg, svmul_f32_x(pg, term_xx, vec_sw), common_1, vec_x);
            
            svfloat32_t term_xy = svmul_f32_x(pg, svmul_f32_x(pg, svmul_f32_x(pg, vec_x, vec_y), vec_inr4), vec_two);
            svfloat32_t deriv_4 = svmls_f32_m(pg, svmul_f32_x(pg, term_xy, vec_sw), common_1, vec_y);
            
            svfloat32_t term_xz = svmul_f32_x(pg, svmul_f32_x(pg, svmul_f32_x(pg, vec_x, vec_z), vec_inr4), vec_two);
            svfloat32_t deriv_5 = svmls_f32_m(pg, svmul_f32_x(pg, term_xz, vec_sw), common_1, vec_z);
            
            // 导数 6-8: ∂(y/r²)/∂{x,y,z}
            svfloat32_t common_2 = svmul_f32_x(pg, svmul_f32_x(pg, vec_desc2, vec_dsw), vec_inr);
            
            svfloat32_t term_yx = svmul_f32_x(pg, svmul_f32_x(pg, svmul_f32_x(pg, vec_y, vec_x), vec_inr4), vec_two);
            svfloat32_t deriv_6 = svmls_f32_m(pg, svmul_f32_x(pg, term_yx, vec_sw), common_2, vec_x);
            
            svfloat32_t term_yy = svsub_f32_x(pg,
                svmul_f32_x(pg, svmul_f32_x(pg, svmul_f32_x(pg, vec_y, vec_y), vec_inr4), vec_two),
                vec_inr2);
            svfloat32_t deriv_7 = svmls_f32_m(pg, svmul_f32_x(pg, term_yy, vec_sw), common_2, vec_y);
            
            svfloat32_t term_yz = svmul_f32_x(pg, svmul_f32_x(pg, svmul_f32_x(pg, vec_y, vec_z), vec_inr4), vec_two);
            svfloat32_t deriv_8 = svmls_f32_m(pg, svmul_f32_x(pg, term_yz, vec_sw), common_2, vec_z);
            
            // 导数 9-11: ∂(z/r²)/∂{x,y,z}
            svfloat32_t common_3 = svmul_f32_x(pg, svmul_f32_x(pg, vec_desc3, vec_dsw), vec_inr);
            
            svfloat32_t term_zx = svmul_f32_x(pg, svmul_f32_x(pg, svmul_f32_x(pg, vec_z, vec_x), vec_inr4), vec_two);
            svfloat32_t deriv_9 = svmls_f32_m(pg, svmul_f32_x(pg, term_zx, vec_sw), common_3, vec_x);
            
            svfloat32_t term_zy = svmul_f32_x(pg, svmul_f32_x(pg, svmul_f32_x(pg, vec_z, vec_y), vec_inr4), vec_two);
            svfloat32_t deriv_10 = svmls_f32_m(pg, svmul_f32_x(pg, term_zy, vec_sw), common_3, vec_y);
            
            svfloat32_t term_zz = svsub_f32_x(pg,
                svmul_f32_x(pg, svmul_f32_x(pg, svmul_f32_x(pg, vec_z, vec_z), vec_inr4), vec_two),
                vec_inr2);
            svfloat32_t deriv_11 = svmls_f32_m(pg, svmul_f32_x(pg, term_zz, vec_sw), common_3, vec_z);
            
            // 应用平滑函数到描述符
            vec_desc0 = svmul_f32_x(pg, vec_desc0, vec_sw);
            vec_desc1 = svmul_f32_x(pg, vec_desc1, vec_sw);
            vec_desc2 = svmul_f32_x(pg, vec_desc2, vec_sw);
            vec_desc3 = svmul_f32_x(pg, vec_desc3, vec_sw);
            
            // 6. 批量存储结果
            float temp_buffer[MAX_BATCH];
            
            // 存储描述符
            svst1_f32(pg, temp_buffer, vec_desc0);
            for (int i = 0; i < batch_count; i++) {
                int jj = batch_indices[i];
                descrpt_a[jj * 4 + 0] = temp_buffer[i];
                rij_a[jj * 3 + 0] = batch_x[i];
            }
            
            svst1_f32(pg, temp_buffer, vec_desc1);
            for (int i = 0; i < batch_count; i++) {
                int jj = batch_indices[i];
                descrpt_a[jj * 4 + 1] = temp_buffer[i];
                rij_a[jj * 3 + 1] = batch_y[i];
            }
            
            svst1_f32(pg, temp_buffer, vec_desc2);
            for (int i = 0; i < batch_count; i++) {
                int jj = batch_indices[i];
                descrpt_a[jj * 4 + 2] = temp_buffer[i];
                rij_a[jj * 3 + 2] = batch_z[i];
            }
            
            svst1_f32(pg, temp_buffer, vec_desc3);
            for (int i = 0; i < batch_count; i++) {
                int jj = batch_indices[i];
                descrpt_a[jj * 4 + 3] = temp_buffer[i];
            }
            
            // 存储所有12个导数分量
            svst1_f32(pg, temp_buffer, deriv_0);
            for (int i = 0; i < batch_count; i++) {
                descrpt_a_deriv[batch_indices[i] * 12 + 0] = temp_buffer[i];
            }
            
            svst1_f32(pg, temp_buffer, deriv_1);
            for (int i = 0; i < batch_count; i++) {
                descrpt_a_deriv[batch_indices[i] * 12 + 1] = temp_buffer[i];
            }
            
            svst1_f32(pg, temp_buffer, deriv_2);
            for (int i = 0; i < batch_count; i++) {
                descrpt_a_deriv[batch_indices[i] * 12 + 2] = temp_buffer[i];
            }
            
            svst1_f32(pg, temp_buffer, deriv_3);
            for (int i = 0; i < batch_count; i++) {
                descrpt_a_deriv[batch_indices[i] * 12 + 3] = temp_buffer[i];
            }
            
            svst1_f32(pg, temp_buffer, deriv_4);
            for (int i = 0; i < batch_count; i++) {
                descrpt_a_deriv[batch_indices[i] * 12 + 4] = temp_buffer[i];
            }
            
            svst1_f32(pg, temp_buffer, deriv_5);
            for (int i = 0; i < batch_count; i++) {
                descrpt_a_deriv[batch_indices[i] * 12 + 5] = temp_buffer[i];
            }
            
            svst1_f32(pg, temp_buffer, deriv_6);
            for (int i = 0; i < batch_count; i++) {
                descrpt_a_deriv[batch_indices[i] * 12 + 6] = temp_buffer[i];
            }
            
            svst1_f32(pg, temp_buffer, deriv_7);
            for (int i = 0; i < batch_count; i++) {
                descrpt_a_deriv[batch_indices[i] * 12 + 7] = temp_buffer[i];
            }
            
            svst1_f32(pg, temp_buffer, deriv_8);
            for (int i = 0; i < batch_count; i++) {
                descrpt_a_deriv[batch_indices[i] * 12 + 8] = temp_buffer[i];
            }
            
            svst1_f32(pg, temp_buffer, deriv_9);
            for (int i = 0; i < batch_count; i++) {
                descrpt_a_deriv[batch_indices[i] * 12 + 9] = temp_buffer[i];
            }
            
            svst1_f32(pg, temp_buffer, deriv_10);
            for (int i = 0; i < batch_count; i++) {
                descrpt_a_deriv[batch_indices[i] * 12 + 10] = temp_buffer[i];
            }
            
            svst1_f32(pg, temp_buffer, deriv_11);
            for (int i = 0; i < batch_count; i++) {
                descrpt_a_deriv[batch_indices[i] * 12 + 11] = temp_buffer[i];
            }
        }
    }
}

// ============================================================================
// 测试代码
// ============================================================================

void generate_test_data(
    std::vector<float>& posi,
    std::vector<int>& type,
    std::vector<int>& fmt_nlist_a,
    std::vector<int>& sec_a,
    int& i_idx,
    int num_atoms,
    int avg_neighbors)
{
    srand(42);
    posi.resize(num_atoms * 3);
    type.resize(num_atoms);
    
    for (int i = 0; i < num_atoms; i++) {
        posi[i * 3 + 0] = (float)rand() / RAND_MAX * 10.0f;
        posi[i * 3 + 1] = (float)rand() / RAND_MAX * 10.0f;
        posi[i * 3 + 2] = (float)rand() / RAND_MAX * 10.0f;
        type[i] = rand() % 2;
    }
    
    i_idx = num_atoms / 2;
    
    int type1_count = avg_neighbors / 2;
    int type2_count = avg_neighbors - type1_count;
    
    sec_a.push_back(0);
    sec_a.push_back(type1_count);
    sec_a.push_back(type1_count + type2_count);
    
    fmt_nlist_a.resize(sec_a.back(), -1);
    
    int idx = 0;
    for (int i = 0; i < num_atoms && idx < type1_count; i++) {
        if (i != i_idx && type[i] == 0) {
            fmt_nlist_a[idx++] = i;
        }
    }
    
    idx = type1_count;
    for (int i = 0; i < num_atoms && idx < type1_count + type2_count; i++) {
        if (i != i_idx && type[i] == 1) {
            fmt_nlist_a[idx++] = i;
        }
    }
}

bool verify_results(
    const std::vector<float>& result1,
    const std::vector<float>& result2,
    const char* name,
    float tolerance = 1e-3f)
{
    if (result1.size() != result2.size()) {
        printf("  ❌ %s: Size mismatch (%zu vs %zu)\n", name, result1.size(), result2.size());
        return false;
    }
    
    float max_diff = 0.0f;
    int diff_count = 0;
    
    for (size_t i = 0; i < result1.size(); i++) {
        float diff = fabs(result1[i] - result2[i]);
        if (diff > tolerance) diff_count++;
        max_diff = fmax(max_diff, diff);
    }
    
    printf("  %s %s: max_err=%.2e, errors=%d/%zu\n", 
           diff_count == 0 ? "✅" : "⚠️", name, max_diff, diff_count, result1.size());
    
    return diff_count == 0;
}

void run_benchmark(int num_atoms, int avg_neighbors, int iterations) {
    printf("\n========================================\n");
    printf("Atoms=%d, Neighbors=%d, Iterations=%d\n", num_atoms, avg_neighbors, iterations);
    printf("========================================\n");
    
    std::vector<float> posi;
    std::vector<int> type, fmt_nlist_a, sec_a;
    int i_idx;
    
    generate_test_data(posi, type, fmt_nlist_a, sec_a, i_idx, num_atoms, avg_neighbors);
    
    float rmin = 0.5f, rmax = 6.0f;
    
    // Baseline
    std::vector<float> desc_base, deriv_base, rij_base;
    double t1 = get_time_in_seconds();
    for (int i = 0; i < iterations; i++) {
        env_mat_a_cpu_baseline(desc_base, deriv_base, rij_base,
                               posi, type, i_idx, fmt_nlist_a, sec_a, rmin, rmax);
    }
    double t2 = get_time_in_seconds();
    double baseline_time = (t2 - t1) / iterations * 1000.0;
    
    // Optimized
    std::vector<float> desc_opt, deriv_opt, rij_opt;
    t1 = get_time_in_seconds();
    for (int i = 0; i < iterations; i++) {
        env_mat_a_cpu_practical_optimized(desc_opt, deriv_opt, rij_opt,
                                          posi, type, i_idx, fmt_nlist_a, sec_a, rmin, rmax);
    }
    t2 = get_time_in_seconds();
    double opt_time = (t2 - t1) / iterations * 1000.0;
    
    printf("\n性能结果:\n");
    printf("  Baseline:  %.6f ms\n", baseline_time);
    printf("  Optimized: %.6f ms\n", opt_time);
    printf("  加速比:    %.2fx\n", baseline_time / opt_time);
    
    printf("\n正确性验证:\n");
    verify_results(desc_base, desc_opt, "描述符");
    verify_results(deriv_base, deriv_opt, "导数");
    verify_results(rij_base, rij_opt, "距离向量");
}

int main() {
    printf("=================================================\n");
    printf("       实用SVE优化性能测试\n");
    printf("=================================================\n");
    
    run_benchmark(50, 10, 5000);
    run_benchmark(200, 25, 2000);
    run_benchmark(500, 50, 1000);
    run_benchmark(1000, 100, 500);
    
    printf("\n=================================================\n");
    printf("测试完成！\n");
    printf("=================================================\n");
    
    return 0;
}