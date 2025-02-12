/*Author: 8891689/ChatGPT */

#ifndef SECP256K1_CUH
#define SECP256K1_CUH

#include <cstdio>
#include <cstring>
#include <stdint.h>
#include <cuda_runtime.h>

//=============================================================================
// 1. 数据结构定义
//=============================================================================

// 256位大整数（8个32位无符号整数，小端存储，即 data[0] 为最低位）
struct BigInt {
    uint32_t data[8];
};

// ECC 点（仿射坐标）
struct ECPoint {
    BigInt x;
    BigInt y;
    bool infinity;  // true 表示无穷远点
};

//=============================================================================
// 2. __constant__ 内存：存放 SECP256k1 的模数 p = 2^256 - 2^32 - 977
//=============================================================================
__constant__ BigInt const_p;

//=============================================================================
// 3. 辅助函数：多精度运算及模运算函数
//=============================================================================

// 初始化：将 BigInt 清零，并设置第 0 个字为 val
__host__ __device__ __forceinline__ void init_bigint(BigInt &x, uint32_t val) {
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        x.data[i] = 0;
    }
    x.data[0] = val;
}

// 内联复制
__host__ __device__ __forceinline__ void copy_bigint(BigInt &dest, const BigInt &src) {
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        dest.data[i] = src.data[i];
    }
}

// 内联比较：返回 1 表示 a > b，-1 表示 a < b，0 表示相等
__host__ __device__ __forceinline__ int compare_bigint(const BigInt &a, const BigInt &b) {
    for (int i = 7; i >= 0; i--) {
        if (a.data[i] > b.data[i])
            return 1;
        else if (a.data[i] < b.data[i])
            return -1;
    }
    return 0;
}

// 判断是否为 0
__host__ __device__ __forceinline__ bool is_zero(const BigInt &a) {
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        if (a.data[i] != 0)
            return false;
    }
    return true;
}

// 内联取位（从低位开始计数）
__host__ __device__ __forceinline__ bool get_bit(const BigInt &a, int i) {
    int word = i / 32;
    int bit = i % 32;
    return (a.data[word] >> bit) & 1;
}

//------------------------------------------------------------------------------
// 辅助函数：简单 256 位加法与减法
//------------------------------------------------------------------------------

__host__ __device__ inline void ptx_u256Add(BigInt &res, const BigInt &a, const BigInt &b) {
    uint32_t carry = 0;
    for (int i = 0; i < 8; i++) {
         uint64_t sum = (uint64_t)a.data[i] + b.data[i] + carry;
         res.data[i] = (uint32_t)(sum & 0xFFFFFFFF);
         carry = (uint32_t)(sum >> 32);
    }
}

__host__ __device__ inline void ptx_u256Sub(BigInt &res, const BigInt &a, const BigInt &b) {
    uint32_t borrow = 0;
    for (int i = 0; i < 8; i++) {
         uint64_t ai = a.data[i];
         uint64_t bi = b.data[i];
         uint64_t diff = ai - bi - borrow;
         res.data[i] = (uint32_t)(diff & 0xFFFFFFFF);
         borrow = (ai < bi + borrow) ? 1 : 0;
    }
}

//------------------------------------------------------------------------------
// 以下为辅助函数，用于辅助实现“特化归约”：
// 1. 将 256 位 BigInt 乘以一个 32 位常数，结果存入 9 个32位字（低位在前）
//------------------------------------------------------------------------------
__host__ __device__ inline void multiply_bigint_by_const(const BigInt &a, uint32_t c, uint32_t result[9]) {
    uint64_t carry = 0;
    for (int i = 0; i < 8; i++) {
        uint64_t prod = (uint64_t)a.data[i] * c + carry;
        result[i] = (uint32_t)(prod & 0xFFFFFFFFUL);
        carry = prod >> 32;
    }
    result[8] = (uint32_t)carry;
}

//------------------------------------------------------------------------------
// 将 BigInt 左移 32 位（即一个字），结果存入 9 个32位字
//------------------------------------------------------------------------------
__host__ __device__ inline void shift_left_word(const BigInt &a, uint32_t result[9]) {
    result[0] = 0;
    for (int i = 0; i < 8; i++) {
        result[i+1] = a.data[i];
    }
}

//------------------------------------------------------------------------------
// 辅助：9字数组加法（共9个32位字，低位在前）
//------------------------------------------------------------------------------
__host__ __device__ inline void add_9word(uint32_t r[9], const uint32_t addend[9]) {
    uint64_t carry = 0;
    for (int i = 0; i < 9; i++) {
        uint64_t sum = (uint64_t)r[i] + addend[i] + carry;
        r[i] = (uint32_t)(sum & 0xFFFFFFFFUL);
        carry = sum >> 32;
    }
}

//------------------------------------------------------------------------------
// 辅助：将 9字数组（低8字）转回 BigInt
//------------------------------------------------------------------------------
__host__ __device__ inline void convert_9word_to_bigint(const uint32_t r[9], BigInt &res) {
    for (int i = 0; i < 8; i++) {
        res.data[i] = r[i];
    }
}

//------------------------------------------------------------------------------
// 修正后的 256 位模乘：
// 计算 a * b 得到 512 位乘积，然后利用 p = 2^256 - 2^32 - 977 的特殊结构归约：
//   设  X = L + H*2^256，则有 X ≡ L + H*(2^32+977) (mod p)
//------------------------------------------------------------------------------
__host__ __device__ inline void mul_mod(BigInt &res, const BigInt &a, const BigInt &b, const BigInt &p) {
    // 计算512位乘积，存入16个32位字
    uint32_t prod[16] = {0};
    for (int i = 0; i < 16; i++) prod[i] = 0;
    for (int i = 0; i < 8; i++) {
        uint64_t carry = 0;
        for (int j = 0; j < 8; j++) {
            uint64_t tmp = (uint64_t)prod[i+j] + (uint64_t)a.data[i]*b.data[j] + carry;
            prod[i+j] = (uint32_t)(tmp & 0xFFFFFFFFUL);
            carry = tmp >> 32;
        }
        prod[i+8] += (uint32_t)carry;
    }
    // 将乘积分为低半部 L 和高半部 H
    BigInt L, H;
    for (int i = 0; i < 8; i++) {
        L.data[i] = prod[i];
        H.data[i] = prod[i+8];
    }
    // 按照公式：X mod p = L + H*(2^32+977)
    uint32_t Rext[9] = {0};
    for (int i = 0; i < 8; i++) {
        Rext[i] = L.data[i];
    }
    Rext[8] = 0;
    uint32_t H977[9] = {0};
    multiply_bigint_by_const(H, 977, H977);
    add_9word(Rext, H977);
    uint32_t Hshift[9] = {0};
    shift_left_word(H, Hshift);
    add_9word(Rext, Hshift);
    if (Rext[8] != 0) {
        uint32_t extra[9] = {0};
        BigInt extraBI;
        init_bigint(extraBI, Rext[8]);
        Rext[8] = 0;
        uint32_t extra977[9] = {0}, extraShift[9] = {0};
        multiply_bigint_by_const(extraBI, 977, extra977);
        shift_left_word(extraBI, extraShift);
        uint32_t fold[9] = {0};
        for (int i = 0; i < 9; i++) fold[i] = extra977[i];
        add_9word(fold, extraShift);
        add_9word(Rext, fold);
    }
    BigInt R_temp;
    convert_9word_to_bigint(Rext, R_temp);
    while ((Rext[8] != 0) || (compare_bigint(R_temp, p) >= 0)) {
        BigInt temp;
        ptx_u256Sub(temp, R_temp, p);
        copy_bigint(R_temp, temp);
        for (int i = 0; i < 8; i++) Rext[i] = R_temp.data[i];
        Rext[8] = 0;
    }
    copy_bigint(res, R_temp);
}

//------------------------------------------------------------------------------
// 以下为模归约、模指数、模逆函数（与原代码基本相同）
//------------------------------------------------------------------------------
__host__ __device__ __forceinline__ void efficient_mod(BigInt &r, const BigInt &a, const BigInt &p) {
    copy_bigint(r, a);
    if (compare_bigint(r, p) >= 0) {
         BigInt temp;
         ptx_u256Sub(temp, r, p);
         if (compare_bigint(temp, p) >= 0)
              ptx_u256Sub(temp, temp, p);
         copy_bigint(r, temp);
    }
}

__host__ __device__ __forceinline__ void mod_generic(BigInt &r, const BigInt &a, const BigInt &p) {
    efficient_mod(r, a, p);
}

__host__ __device__ __forceinline__ void sub_mod(BigInt &res, const BigInt &a, const BigInt &b, const BigInt &p) {
    BigInt temp;
    if (compare_bigint(a, b) < 0) {
         BigInt sum;
         ptx_u256Add(sum, a, p);
         ptx_u256Sub(temp, sum, b);
    } else {
         ptx_u256Sub(temp, a, b);
    }
    mod_generic(res, temp, p);
}

__host__ __device__ __forceinline__ void add_mod(BigInt &res, const BigInt &a, const BigInt &b, const BigInt &p) {
    BigInt temp;
    ptx_u256Add(temp, a, b);
    mod_generic(res, temp, p);
}

__host__ __device__ __forceinline__ void modexp(BigInt &res, const BigInt &base, const BigInt &exp, const BigInt &p) {
    BigInt result;
    init_bigint(result, 1);
    BigInt b;
    copy_bigint(b, base);
    for (int i = 0; i < 256; i++) {
         if (get_bit(exp, i)) {
              BigInt temp;
              mul_mod(temp, result, b, p);
              copy_bigint(result, temp);
         }
         BigInt temp;
         mul_mod(temp, b, b, p);
         copy_bigint(b, temp);
    }
    copy_bigint(res, result);
}

__host__ __device__ __forceinline__ void mod_inverse(BigInt &res, const BigInt &a, const BigInt &p) {
    BigInt p_minus_2;
    copy_bigint(p_minus_2, p);
    BigInt two;
    init_bigint(two, 2);
    BigInt temp;
    ptx_u256Sub(temp, p_minus_2, two);
    copy_bigint(p_minus_2, temp);
    modexp(res, a, p_minus_2, p);
}

//------------------------------------------------------------------------------
// ECC 运算：点复制、设置无穷远、点加、点加倍
//------------------------------------------------------------------------------
__host__ __device__ __forceinline__ void point_set_infinity(ECPoint &P) {
    P.infinity = true;
}

__host__ __device__ __forceinline__ void point_copy(ECPoint &dest, const ECPoint &src) {
    copy_bigint(dest.x, src.x);
    copy_bigint(dest.y, src.y);
    dest.infinity = src.infinity;
}

__device__ __forceinline__ void point_add(ECPoint &R, const ECPoint &P, const ECPoint &Q, const BigInt &p) {
    if (P.infinity) { point_copy(R, Q); return; }
    if (Q.infinity) { point_copy(R, P); return; }
    BigInt diffY, diffX, inv_diffX, lambda, lambda2, temp;
    sub_mod(diffY, Q.y, P.y, p);
    sub_mod(diffX, Q.x, P.x, p);
    mod_inverse(inv_diffX, diffX, p);
    mul_mod(lambda, diffY, inv_diffX, p);
    mul_mod(lambda2, lambda, lambda, p);
    sub_mod(temp, lambda2, P.x, p);
    sub_mod(R.x, temp, Q.x, p);
    sub_mod(temp, P.x, R.x, p);
    mul_mod(R.y, lambda, temp, p);
    sub_mod(R.y, R.y, P.y, p);
    R.infinity = false;
}

__device__ __forceinline__ void double_point(ECPoint &R, const ECPoint &P, const BigInt &p) {
    if (P.infinity || is_zero(P.y)) {
         point_set_infinity(R);
         return;
    }
    BigInt x2, numerator, denominator, inv_den, lambda, lambda2, two, two_x;
    mul_mod(x2, P.x, P.x, p);
    BigInt three; init_bigint(three, 3);
    mul_mod(numerator, three, x2, p);
    init_bigint(two, 2);
    mul_mod(denominator, two, P.y, p);
    mod_inverse(inv_den, denominator, p);
    mul_mod(lambda, numerator, inv_den, p);
    mul_mod(lambda2, lambda, lambda, p);
    mul_mod(two_x, two, P.x, p);
    sub_mod(R.x, lambda2, two_x, p);
    sub_mod(numerator, P.x, R.x, p);
    mul_mod(R.y, lambda, numerator, p);
    sub_mod(R.y, R.y, P.y, p);
    R.infinity = false;
}

//------------------------------------------------------------------------------
// 内核：批量 Montgomery ladder 标量乘法，每个线程计算 Q = d * G
// 同时利用原子加计数器记录处理的密钥数
//------------------------------------------------------------------------------
__global__ void kernel_montgomery_ladder_batch_optimized(const BigInt *d_keys, 
                                                         const ECPoint G, 
                                                         const BigInt p, 
                                                         ECPoint *Q_keys, 
                                                         int n,
                                                         unsigned long long *d_processedCounter)
{
    extern __shared__ ECPoint sG[];
    if (threadIdx.x == 0) {
         sG[0] = G;
    }
    __syncthreads();
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
         BigInt d;
         copy_bigint(d, d_keys[idx]);
         ECPoint R0, R1, Q;
         point_set_infinity(R0);
         point_copy(R1, sG[0]);
         for (int i = 255; i >= 0; i--) {
              if (get_bit(d, i)) {
                   ECPoint temp;
                   point_add(temp, R0, R1, p);
                   point_copy(R0, temp);
                   ECPoint temp2;
                   double_point(temp2, R1, p);
                   point_copy(R1, temp2);
              } else {
                   ECPoint temp;
                   point_add(temp, R0, R1, p);
                   point_copy(R1, temp);
                   ECPoint temp2;
                   double_point(temp2, R0, p);
                   point_copy(R0, temp2);
              }
         }
         point_copy(Q, R0);
         Q_keys[idx] = Q;
         
         // 计数：每处理一个密钥进行一次原子累加
         atomicAdd(d_processedCounter, 1ULL);
    }
}

//------------------------------------------------------------------------------
// 以下为字节序转换相关辅助函数
//------------------------------------------------------------------------------
__host__ __device__ inline unsigned int swap_uint32(unsigned int x) {
    return ((x & 0x000000FFU) << 24) |
           ((x & 0x0000FF00U) << 8)  |
           ((x & 0x00FF0000U) >> 8)  |
           ((x & 0xFF000000U) >> 24);
}

__host__ __device__ inline void toBigEndianWords(const BigInt &in, unsigned int out[8]) {
    for (int i = 0; i < 8; i++) {
         out[i] = in.data[7 - i];
    }
}

#endif // SECP256K1_CUH

