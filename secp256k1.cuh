// Author: 8891689
#ifndef SECP256K1_CUH
#define SECP256K1_CUH

#include <cuda_runtime.h>
#include <stdint.h>
#include <string.h>

#define BIGINT_WORDS 8

// CUDA 错误检查宏
#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)


// ============================================================================
// 1. 数据结构 (与 C 版本相同, 可在 Host 和 Device 间传递)
// ============================================================================
struct BigInt {
    uint32_t data[BIGINT_WORDS];
};

struct ECPoint {
    BigInt x, y;
    bool infinity;
};

struct ECPointJac {
    BigInt X, Y, Z;
    bool infinity;
};

// ============================================================================
// 常量参数 (在 GPU 的 __constant__ 内存中)
// ============================================================================
__constant__ BigInt const_p;
__constant__ ECPointJac const_G_jacobian;
__constant__ BigInt const_n;


// ============================================================================
// 3. 移植的 C 函数 (添加 __host__ __device__ __forceinline__ 等限定符)
// ============================================================================

// --- BigInt 基本运算 ---
__host__ __device__ __forceinline__ void init_bigint(BigInt *x, uint32_t val) {
    x->data[0] = val;
    for (int i = 1; i < BIGINT_WORDS; i++) x->data[i] = 0;
}

__host__ __device__ __forceinline__ void copy_bigint(BigInt *dest, const BigInt *src) {
    for (int i = 0; i < BIGINT_WORDS; i++) {
        dest->data[i] = src->data[i];
    }
}

__host__ __device__ __forceinline__ int compare_bigint(const BigInt *a, const BigInt *b) {
    for (int i = BIGINT_WORDS - 1; i >= 0; i--) {
        if (a->data[i] > b->data[i]) return 1;
        if (a->data[i] < b->data[i]) return -1;
    }
    return 0;
}

__host__ __device__ __forceinline__ bool is_zero(const BigInt *a) {
    for (int i = 0; i < BIGINT_WORDS; i++) {
        if (a->data[i]) return false;
    }
    return true;
}

__host__ __device__ __forceinline__ int get_bit(const BigInt *a, int i) {
    int word_idx = i >> 5; // i / 32
    int bit_idx = i & 31;  // i % 32
    if (word_idx >= BIGINT_WORDS) return 0;
    return (a->data[word_idx] >> bit_idx) & 1;
}

__host__ __device__ __forceinline__ void ptx_u256Add(BigInt *res, const BigInt *a, const BigInt *b) {
    uint64_t carry = 0;
    for (int i = 0; i < BIGINT_WORDS; ++i) {
        uint64_t sum = (uint64_t)a->data[i] + b->data[i] + carry;
        res->data[i] = (uint32_t)sum;
        carry = (sum >> 32);
    }
}

__host__ __device__ __forceinline__ void ptx_u256Sub(BigInt *res, const BigInt *a, const BigInt *b) {
    uint32_t borrow = 0;
    for (int i = 0; i < BIGINT_WORDS; ++i) {
        // 先做 a[i] - borrow
        uint64_t tmp = (uint64_t)a->data[i] - borrow;
        // 判断 tmp < b[i] 来决定是否需要再次借位
        borrow = tmp < b->data[i] ? 1u : 0u;
        // 真正的差值
        res->data[i] = (uint32_t)(tmp - b->data[i]);
    }
}

// -- 模乘所需的辅助函数 --
__device__ __forceinline__ void multiply_bigint_by_const(const BigInt *a, uint32_t c, uint32_t result[9]) {
    uint64_t carry = 0;
    for (int i = 0; i < BIGINT_WORDS; i++) {
        uint64_t prod = (uint64_t)a->data[i] * c + carry;
        result[i] = (uint32_t)prod;
        carry = prod >> 32;
    }
    result[8] = (uint32_t)carry;
}

__device__ __forceinline__ void shift_left_word(const BigInt *a, uint32_t result[9]) {
    result[0] = 0;
    for (int i = 0; i < BIGINT_WORDS; i++) {
        result[i+1] = a->data[i];
    }
}

__device__ __forceinline__ void add_9word(uint32_t r[9], const uint32_t addend[9]) {
    uint64_t carry = 0;
    for (int i = 0; i < 9; i++) {
        uint64_t sum = (uint64_t)r[i] + addend[i] + carry;
        r[i] = (uint32_t)sum;
        carry = sum >> 32;
    }
}

__device__ __forceinline__ void convert_9word_to_bigint(const uint32_t r[9], BigInt *res) {
    for (int i = 0; i < BIGINT_WORDS; i++) {
        res->data[i] = r[i];
    }
}

// 忠实移植 C 语言版本的 mul_mod_old
__device__ __forceinline__ void mul_mod_device(BigInt *res, const BigInt *a, const BigInt *b) {
    uint32_t prod[2 * BIGINT_WORDS] = {0};
    for (int i = 0; i < BIGINT_WORDS; i++) {
        uint64_t carry = 0;
        for (int j = 0; j < BIGINT_WORDS; j++) {
            uint64_t tmp = (uint64_t)prod[i + j] + (uint64_t)a->data[i] * b->data[j] + carry;
            prod[i + j] = (uint32_t)tmp;
            carry = tmp >> 32;
        }
        prod[i + BIGINT_WORDS] += (uint32_t)carry;
    }
    
    BigInt L, H;
    for (int i = 0; i < BIGINT_WORDS; i++) {
        L.data[i] = prod[i];
        H.data[i] = prod[i + BIGINT_WORDS];
    }
    
    uint32_t Rext[9] = {0};
    for (int i = 0; i < BIGINT_WORDS; i++) Rext[i] = L.data[i];
    Rext[8] = 0;
    
    uint32_t H977[9] = {0};
    multiply_bigint_by_const(&H, 977, H977);
    add_9word(Rext, H977);

    uint32_t Hshift[9] = {0};
    shift_left_word(&H, Hshift);
    add_9word(Rext, Hshift);
    
    if (Rext[8]) {
        uint32_t extra[9] = {0};
        BigInt extraBI;
        init_bigint(&extraBI, Rext[8]);
        Rext[8] = 0;
        
        uint32_t extra977[9] = {0}, extraShift[9] = {0};
        multiply_bigint_by_const(&extraBI, 977, extra977);
        shift_left_word(&extraBI, extraShift);
        
        for (int i = 0; i < 9; i++) extra[i] = extra977[i];
        add_9word(extra, extraShift);
        add_9word(Rext, extra);
    }
    
    BigInt R_temp;
    convert_9word_to_bigint(Rext, &R_temp);
    
    // Final subtractions to ensure result < p
    if (Rext[8] || compare_bigint(&R_temp, &const_p) >= 0) {
        ptx_u256Sub(&R_temp, &R_temp, &const_p);
    }
    if (compare_bigint(&R_temp, &const_p) >= 0) {
        ptx_u256Sub(&R_temp, &R_temp, &const_p);
    }
    
    copy_bigint(res, &R_temp);
}

// 其他模运算函数
__device__ __forceinline__ void sub_mod_device(BigInt *res, const BigInt *a, const BigInt *b) {
    BigInt temp;
    if (compare_bigint(a, b) < 0) {
         BigInt sum;
         ptx_u256Add(&sum, a, &const_p);
         ptx_u256Sub(&temp, &sum, b);
    } else {
         ptx_u256Sub(&temp, a, b);
    }
    copy_bigint(res, &temp);
}

// 将 a mod n，结果放到 res（因为 a < 2^256, 所以最多减一次 n 即可）
__device__ __forceinline__ void scalar_mod_n(BigInt *res, const BigInt *a) {
    if (compare_bigint(a, &const_n) >= 0) {
        // a >= n, 做一次减法
        ptx_u256Sub(res, a, &const_n);
    } else {
        copy_bigint(res, a);
    }
}


__device__ __forceinline__ void add_mod_device(BigInt *res, const BigInt *a, const BigInt *b) {
    BigInt sum_ab;
    uint64_t carry = 0;
    for (int i = 0; i < BIGINT_WORDS; ++i) {
         uint64_t word_sum = (uint64_t)a->data[i] + b->data[i] + carry;
         sum_ab.data[i] = (uint32_t)word_sum;
         carry = word_sum >> 32;
    }
    if (carry || compare_bigint(&sum_ab, &const_p) >= 0) {
        ptx_u256Sub(res, &sum_ab, &const_p);
    } else {
        copy_bigint(res, &sum_ab);
    }
}

__device__ void modexp(BigInt *res, const BigInt *base, const BigInt *exp) {
    BigInt result;
    init_bigint(&result, 1);
    BigInt b;
    copy_bigint(&b, base);
    for (int i = 0; i < 256; i++) {
         if (get_bit(exp, i)) {
              mul_mod_device(&result, &result, &b);
         }
         mul_mod_device(&b, &b, &b);
    }
    copy_bigint(res, &result);
}

__device__ void mod_inverse(BigInt *res, const BigInt *a) {
    BigInt p_minus_2, two;
    init_bigint(&two, 2);
    ptx_u256Sub(&p_minus_2, &const_p, &two);
    modexp(res, a, &p_minus_2);
}



// --- 雅可比坐标点运算 (Device 版本) ---
__device__ __forceinline__ void point_set_infinity_jac(ECPointJac *P) {
    P->infinity = true;
}

__device__ __forceinline__ void point_copy_jac(ECPointJac *dest, const ECPointJac *src) {
    copy_bigint(&dest->X, &src->X);
    copy_bigint(&dest->Y, &src->Y);
    copy_bigint(&dest->Z, &src->Z);
    dest->infinity = src->infinity;
}

__device__ void double_point_jac(ECPointJac *R, const ECPointJac *P); // 声明
__device__ void add_point_jac(ECPointJac *R, const ECPointJac *P, const ECPointJac *Q); // 声明

__device__ void double_point_jac(ECPointJac *R, const ECPointJac *P) {
    if (P->infinity || is_zero(&P->Y)) {
        point_set_infinity_jac(R);
        return;
    }
    BigInt A, B, C, D, X3, Y3, Z3, temp, temp2;
    mul_mod_device(&A, &P->Y, &P->Y);
    mul_mod_device(&temp, &P->X, &A);
    init_bigint(&temp2, 4);
    mul_mod_device(&B, &temp, &temp2);
    mul_mod_device(&temp, &A, &A);
    init_bigint(&temp2, 8);
    mul_mod_device(&C, &temp, &temp2);
    mul_mod_device(&temp, &P->X, &P->X);
    init_bigint(&temp2, 3);
    mul_mod_device(&D, &temp, &temp2);
    BigInt D2, two, twoB;
    mul_mod_device(&D2, &D, &D);
    init_bigint(&two, 2);
    mul_mod_device(&twoB, &B, &two);
    sub_mod_device(&X3, &D2, &twoB);
    sub_mod_device(&temp, &B, &X3);
    mul_mod_device(&temp, &D, &temp);
    sub_mod_device(&Y3, &temp, &C);
    init_bigint(&temp, 2);
    mul_mod_device(&temp, &temp, &P->Y);
    mul_mod_device(&Z3, &temp, &P->Z);
    copy_bigint(&R->X, &X3);
    copy_bigint(&R->Y, &Y3);
    copy_bigint(&R->Z, &Z3);
    R->infinity = false;
}

__device__ void add_point_jac(ECPointJac *R, const ECPointJac *P, const ECPointJac *Q) {
    if (P->infinity) { point_copy_jac(R, Q); return; }
    if (Q->infinity) { point_copy_jac(R, P); return; }

    BigInt Z1Z1, Z2Z2, U1, U2, S1, S2, H, R_big, H2, H3, U1H2, X3, Y3, Z3, temp;
    mul_mod_device(&Z1Z1, &P->Z, &P->Z);
    mul_mod_device(&Z2Z2, &Q->Z, &Q->Z);
    mul_mod_device(&U1, &P->X, &Z2Z2);
    mul_mod_device(&U2, &Q->X, &Z1Z1);
    BigInt Z2_cubed, Z1_cubed;
    mul_mod_device(&temp, &Z2Z2, &Q->Z); copy_bigint(&Z2_cubed, &temp);
    mul_mod_device(&temp, &Z1Z1, &P->Z); copy_bigint(&Z1_cubed, &temp);
    mul_mod_device(&S1, &P->Y, &Z2_cubed);
    mul_mod_device(&S2, &Q->Y, &Z1_cubed);

    if (compare_bigint(&U1, &U2) == 0) {
        if (compare_bigint(&S1, &S2) != 0) {
            point_set_infinity_jac(R);
            return;
        } else {
            double_point_jac(R, P);
            return;
        }
    }
    sub_mod_device(&H, &U2, &U1);
    sub_mod_device(&R_big, &S2, &S1);
    mul_mod_device(&H2, &H, &H);
    mul_mod_device(&H3, &H2, &H);
    mul_mod_device(&U1H2, &U1, &H2);
    BigInt R2, two, twoU1H2;
    mul_mod_device(&R2, &R_big, &R_big);
    init_bigint(&two, 2);
    mul_mod_device(&twoU1H2, &U1H2, &two);
    sub_mod_device(&temp, &R2, &H3);
    sub_mod_device(&X3, &temp, &twoU1H2);
    sub_mod_device(&temp, &U1H2, &X3);
    mul_mod_device(&temp, &R_big, &temp);
    mul_mod_device(&Y3, &S1, &H3);
    sub_mod_device(&Y3, &temp, &Y3);
    mul_mod_device(&temp, &P->Z, &Q->Z);
    mul_mod_device(&Z3, &temp, &H);
    copy_bigint(&R->X, &X3);
    copy_bigint(&R->Y, &Y3);
    copy_bigint(&R->Z, &Z3);
    R->infinity = false;
}

__device__ void jacobian_to_affine(ECPoint *R, const ECPointJac *P) {
    if (P->infinity) {
        // 标记为无穷远点，并把坐标全部置零
        R->infinity = true;
        init_bigint(&R->x, 0);
        init_bigint(&R->y, 0);
        return;
    }
    BigInt Zinv, Zinv2, Zinv3;
    mod_inverse(&Zinv, &P->Z);
    mul_mod_device(&Zinv2, &Zinv, &Zinv);
    mul_mod_device(&Zinv3, &Zinv2, &Zinv);
    mul_mod_device(&R->x, &P->X, &Zinv2);
    mul_mod_device(&R->y, &P->Y, &Zinv3);
    R->infinity = false;
}


// --- 核心Device函数: 标量乘法 ---
__device__ void scalar_multiply_jac_device(ECPointJac *result, const ECPointJac *point, const BigInt *scalar) {
    ECPointJac res;
    point_set_infinity_jac(&res);

    int highest_bit = BIGINT_WORDS * 32 - 1;
    for (; highest_bit >= 0; highest_bit--) {
        if (get_bit(scalar, highest_bit)) break;
    }

    if (highest_bit < 0) {
        point_copy_jac(result, &res);
        return;
    }

    ECPointJac p_copy;
    point_copy_jac(&p_copy, point);
    
    for (int i = highest_bit; i >= 0; i--) {
        double_point_jac(&res, &res);
        if (get_bit(scalar, i)) {
            add_point_jac(&res, &res, &p_copy);
        }
    }
    point_copy_jac(result, &res);
}

// ============================================================================
// 4. CUDA Kernel
// ============================================================================
__global__ void private_to_public_key_batch_kernel(
    const BigInt *d_private_keys,
    ECPoint *d_public_keys,
    int num_keys) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_keys) return;

    // 1. 先读取并做 mod n
    BigInt priv;
    copy_bigint(&priv, &d_private_keys[idx]);
    scalar_mod_n(&priv, &priv);

    // 2. 标量乘: Q = priv * G
    ECPointJac result_jac;
    scalar_multiply_jac_device(&result_jac, &const_G_jacobian, &priv);

    // 3. 转换坐标
    ECPoint public_key;
    jacobian_to_affine(&public_key, &result_jac);

    // 4. 写回
    d_public_keys[idx] = public_key;
}

#endif // SECP256K1_CUH
