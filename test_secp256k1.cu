#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <getopt.h>
#include <string>
#include <fstream>
#include <iostream>
#include <vector>
#include <sstream>
#include <cassert>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <cuda_runtime.h>

#include "secp256k1.cuh"

// 简单错误检查宏
#define CHECK_CUDA(call)                                                     \
    {                                                                        \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,  \
                    cudaGetErrorString(err));                                \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    }

// 打印 BigInt（以大端顺序显示）
void print_bigint(const BigInt &b) {
    for (int i = 7; i >= 0; i--) {
        printf("%08x", b.data[i]);
    }
    printf("\n");
}

// 16进制字符串转换为 BigInt
void hex_to_bigint(const char* hex, BigInt &b) {
    memset(b.data, 0, sizeof(b.data));
    int len = (int)strlen(hex);
    for (int i = 0; i < 8; i++) {
        int start = len - (i+1)*8;
        if (start < 0) break;
        char temp[9] = {0};
        strncpy(temp, hex + start, 8);
        b.data[i] = (uint32_t)strtoul(temp, nullptr, 16);
    }
}

// 以 unsigned long long 初始化 BigInt（低位在前）
void set_bigint_from_ull(BigInt &b, unsigned long long val) {
    memset(b.data, 0, sizeof(b.data));
    b.data[0] = (uint32_t)(val & 0xFFFFFFFFULL);
    b.data[1] = (uint32_t)((val >> 32) & 0xFFFFFFFFULL);
}

int main(int argc, char* argv[]) {
    // ========================= 性能与批次优化建议 =========================
    // 1. 尽量使用较大的批次（currentBatchSize）以降低内核启动开销；
    // 2. 推荐使用 pinned 内存（cudaHostAlloc）或异步数据传输与流重叠；
    // 3. 如有可能，将多精度运算部分改为 Montgomery 模乘等更高效算法。
    // 下面示例中为了演示，我们将 currentBatchSize 设置为 1,000,000，
    // 但实际内核可能只处理其中一部分（例如在示例中处理了 65536 个）。
    // =======================================================================

    unsigned long long currentBatchSize = 1000000; // 可根据实际情况调大
    BigInt *d_keys = nullptr;
    ECPoint *d_Q_keys = nullptr;
    CHECK_CUDA(cudaMallocManaged(&d_keys, currentBatchSize * sizeof(BigInt)));
    CHECK_CUDA(cudaMallocManaged(&d_Q_keys, currentBatchSize * sizeof(ECPoint)));

    // 初始化私钥（示例中私钥依次递增，从 1 开始）
    for (unsigned long long i = 0; i < currentBatchSize; i++) {
        unsigned long long val = 1 + i;
        set_bigint_from_ull(d_keys[i], val);
    }

    // 初始化 secp256k1 参数
    BigInt p;
    hex_to_bigint("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F", p);
    BigInt Gx, Gy;
    hex_to_bigint("79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798", Gx);
    hex_to_bigint("483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8", Gy);
    // 将模数 p 写入常量内存
    CHECK_CUDA(cudaMemcpyToSymbol(const_p, &p, sizeof(BigInt), 0, cudaMemcpyHostToDevice));

    ECPoint G;
    copy_bigint(G.x, Gx);
    copy_bigint(G.y, Gy);
    G.infinity = false;

    // 分配计数器（使用统一内存），初始化为 0
    unsigned long long *d_processedCounter = nullptr;
    CHECK_CUDA(cudaMallocManaged(&d_processedCounter, sizeof(unsigned long long)));
    *d_processedCounter = 0;

    // 设置内核配置
    int blocks = 256, threads = 256;

    // 创建 CUDA 事件计时
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start, 0));

    // 调用 ECC 内核：第三个参数设置共享内存大小为 sizeof(ECPoint)
    kernel_montgomery_ladder_batch_optimized<<<blocks, threads, sizeof(ECPoint)>>>(d_keys, G, p, d_Q_keys, currentBatchSize, d_processedCounter);
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaEventRecord(stop, 0));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float elapsed_ms = 0;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start, stop));

    // 根据实际处理的密钥数（计数器 *d_processedCounter）来打印最后 5 个结果
    unsigned long long processed = *d_processedCounter;
    unsigned long long startPrint = (processed < 5) ? 0 : processed - 5;
    printf("打印最后 %llu 个私钥和公钥：\n", processed - startPrint);
    for (unsigned long long i = startPrint; i < processed; i++) {
        printf("私钥 (第 %llu 个): ", i);
        print_bigint(d_keys[i]);
        printf("公钥 (第 %llu 个):\n", i);
        printf("  Q.x = ");
        print_bigint(d_Q_keys[i].x);
        printf("  Q.y = ");
        print_bigint(d_Q_keys[i].y);
    }

    // 输出计时和吞吐量
    printf("内核执行时间: %.3f 毫秒\n", elapsed_ms);
    printf("总共处理密钥数: %llu\n", processed);
    double keysPerSec = (double)processed / (elapsed_ms / 1000.0);
    printf("吞吐量: %.2f 个密钥/秒\n", keysPerSec);

    // 清理内存
    cudaFree(d_keys);
    cudaFree(d_Q_keys);
    cudaFree(d_processedCounter);
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    
    return 0;
}

