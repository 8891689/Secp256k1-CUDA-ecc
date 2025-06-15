/*Author: 8891689
 * Assist in creation ：ChatGPT 
 */
#include <iostream>
#include <vector>
#include <string>
#include <iomanip>
#include <stdexcept>
#include <sstream>

// 引用你的頭文件
#include "secp256k1.cuh"

// ============================================================================
// 1. 主機端輔助函數 (用於測試和驗證)
// ============================================================================

/**
 * @brief 將十六進制字符串轉換為 BigInt 結構體 (大端轉小端)
 * @param hex_str 64個字符的十六進制字符串 (不含 "0x")
 * @return BigInt 結構體
 */
BigInt string_to_bigint(const std::string& hex_str) {
    if (hex_str.length() != 64) {
        throw std::runtime_error("Hex string must be 64 characters long.");
    }
    BigInt result = {0};
    for (int i = 0; i < BIGINT_WORDS; ++i) {
        // 從字符串末尾開始讀取，因為 hex 是大端序，而我們的數組是小端序
        std::string word_hex = hex_str.substr(64 - (i + 1) * 8, 8);
        result.data[i] = std::stoul(word_hex, nullptr, 16);
    }
    return result;
}

/**
 * @brief 在主機端打印 BigInt (以大端十六進制字符串形式)
 * @param b 要打印的 BigInt
 */
void print_bigint_host(const BigInt& b) {
    std::cout << "0x";
    for (int i = BIGINT_WORDS - 1; i >= 0; i--) {
        std::cout << std::hex << std::setw(8) << std::setfill('0') << b.data[i];
    }
    std::cout << std::dec; // 恢復十進制輸出
}

/**
 * @brief 比較兩個 ECPoint 是否相等
 * @return 如果 x 和 y 坐標都相等，則返回 true
 */
bool compare_ecpoint(const ECPoint& p1, const ECPoint& p2) {
    if (p1.infinity != p2.infinity) return false;
    if (p1.infinity) return true;
    
    // 使用 memcmp 進行快速比較
    bool x_match = (memcmp(p1.x.data, p2.x.data, sizeof(BigInt)) == 0);
    bool y_match = (memcmp(p1.y.data, p2.y.data, sizeof(BigInt)) == 0);

    return x_match && y_match;
}

// ============================================================================
// 2. 主機端初始化和單次 GPU 計算封裝
// ============================================================================

// 在 Host 端初始化 GPU __constant__ 內存
void init_gpu_constants() {
    // 1) 定义 p_host
    const BigInt p_host = {
        { 0xFFFFFC2F, 0xFFFFFFFE, 0xFFFFFFFF, 0xFFFFFFFF,
          0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF }
    };
    // 2) 定义 G_jacobian_host
    const ECPointJac G_jacobian_host = {
        .X = {{ 0x16F81798, 0x59F2815B, 0x2DCE28D9, 0x029BFCDB,
                0xCE870B07, 0x55A06295, 0xF9DCBBAC, 0x79BE667E }},
        .Y = {{ 0xFB10D4B8, 0x9C47D08F, 0xA6855419, 0xFD17B448,
                0x0E1108A8, 0x5DA4FBFC, 0x26A3C465, 0x483ADA77 }},
        .Z = {{ 1, 0, 0, 0, 0, 0, 0, 0 }},
        .infinity = false
    };
    // 3) 定义 n_host
    const BigInt n_host = {
        { 0xD0364141, 0xBFD25E8C, 0xAF48A03B, 0xBAAEDCE6,
          0xFFFFFFFE, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF }
    };

    // 然后再复制到 __constant__ 内存
    CHECK_CUDA(cudaMemcpyToSymbol(const_p, &p_host, sizeof(BigInt)));
    CHECK_CUDA(cudaMemcpyToSymbol(const_G_jacobian, &G_jacobian_host, sizeof(ECPointJac)));
    CHECK_CUDA(cudaMemcpyToSymbol(const_n, &n_host, sizeof(BigInt)));
}


/**
 * @brief 使用 GPU 計算單個私鑰對應的公鑰 (用於驗證)
 * @param private_key 輸入的私鑰
 * @return 計算得到的公鑰
 */
ECPoint get_public_key_gpu_single(const BigInt& private_key) {
    // 1. 分配 GPU 內存 (僅用於一個 key)
    BigInt* d_private_key;
    ECPoint* d_public_key;
    CHECK_CUDA(cudaMalloc(&d_private_key, sizeof(BigInt)));
    CHECK_CUDA(cudaMalloc(&d_public_key, sizeof(ECPoint)));

    // 2. 將單個私鑰拷貝到 GPU
    CHECK_CUDA(cudaMemcpy(d_private_key, &private_key, sizeof(BigInt), cudaMemcpyHostToDevice));

    // 3. 啟動內核 (僅用一個線程)
    private_to_public_key_batch_kernel<<<1, 1>>>(d_private_key, d_public_key, 1);
    CHECK_CUDA(cudaGetLastError()); // 檢查內核啟動錯誤

    // 4. 等待並將結果拷貝回主機
    ECPoint host_result;
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(&host_result, d_public_key, sizeof(ECPoint), cudaMemcpyDeviceToHost));

    // 5. 釋放內存
    CHECK_CUDA(cudaFree(d_private_key));
    CHECK_CUDA(cudaFree(d_public_key));

    return host_result;
}


// ============================================================================
// 3. 主要測試邏輯
// ============================================================================
int main() {
    try {
        // 打印 GPU 設備信息
        int device_id;
        cudaDeviceProp props;
        CHECK_CUDA(cudaGetDevice(&device_id));
        CHECK_CUDA(cudaGetDeviceProperties(&props, device_id));
        std::cout << "Using GPU: " << props.name << std::endl << std::endl;

        // 初始化 GPU 常量內存
        init_gpu_constants();
        std::cout << "GPU constants initialized." << std::endl;

        // --- A. 正確性驗證 ---
        std::cout << "\n--- Section A: Correctness Verification ---" << std::endl;

        // 建立測試向量 (私鑰 -> 公鑰X, 公鑰Y)
        struct TestVector {
            std::string name;
            std::string priv_key_hex;
            std::string pub_key_x_hex;
            std::string pub_key_y_hex;
        };

        std::vector<TestVector> test_vectors = {
    {
        "Test Case 1: Private key = 1",
        "0000000000000000000000000000000000000000000000000000000000000001",
        "79be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798", // G.x
        "483ada7726a3c4655da4fbfc0e1108a8fd17b448a68554199c47d08ffb10d4b8"  // G.y
    },
    {
        "Test Case 2: Private key = 2",
        "0000000000000000000000000000000000000000000000000000000000000002",
        "c6047f9441ed7d6d3045406e95c07cd85c778e4b8cef3ca7abac09b95c709ee5", // (2*G).x
        "1ae168fea63dc339a3c58419466ceaeef7f632653266d0e1236431a950cfe52a"  // (2*G).y
    },
    {
        "Test Case 3: A large private key",
        "1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef",
        "bb50e2d89a4ed70663d080659fe0ad4b9bc3e06c17a227433966cb59ceee020d",
        "ecddbf6e00192011648d13b1c00af770c0c1bb609d4d3a5c98a43772e0e18ef4"
    },
    {
        "Test Case 4: A very large private key (n-1)",
        // n = FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFE BAAEDCE6 AF48A03B BFD25E8C D0364141
        "fffffffffffffffffffffffffffffffebaaedce6af48a03bbfd25e8cd0364140",
        "79be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798", // (n-1)G = -G, so x is same
        "b7c52588d95c3b9aa25b0403f1eef75702e84bb7597aabe663b82f6f04ef2777"  // p - G.y
    },
    {
        "Test Case 5: Private key = 0 (infinity)",
        "0000000000000000000000000000000000000000000000000000000000000000",
        "0000000000000000000000000000000000000000000000000000000000000000", // Infinity 表示全 0
        "0000000000000000000000000000000000000000000000000000000000000000"
    },
    {
        "Test Case 6: Private key = n",
        // n = FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFE BAAEDCE6 AF48A03B BFD25E8C D0364141
        "fffffffffffffffffffffffffffffffebaaedce6af48a03bbfd25e8cd0364141",
        "0000000000000000000000000000000000000000000000000000000000000000", // n*G = infinity
        "0000000000000000000000000000000000000000000000000000000000000000"
    },
    {
        "Test Case 7: Private key = n + 1",
        // n+1 raw hex = ...4141 + 1 = ...4142，标量对 G 等价于 1
        "fffffffffffffffffffffffffffffffebaaedce6af48a03bbfd25e8cd0364142",
        "79be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798", // same as 1*G
        "483ada7726a3c4655da4fbfc0e1108a8fd17b448a68554199c47d08ffb10d4b8"
    },
    {
        "Test Case 8: Random private key",
        "a1b2c3d4e5f60718293a4b5c6d7e8f90123456789abcdef0fedcba9876543210",
        "4deec3f2cbde40543d691f30a637fdcccde8902d6699c9b8b97f0aef3fe1c313",
        "6f6fdd988c1283f1b880de40b9ba81762b9f06495299c7df45816681bd3aef59"
    },
    {
        "Test Case 9: Private key = n - 2",
        // (n - 2) = ...4141 - 2 = ...413f
        "fffffffffffffffffffffffffffffffebaaedce6af48a03bbfd25e8cd036413f",
        "c6047f9441ed7d6d3045406e95c07cd85c778e4b8cef3ca7abac09b95c709ee5",
        "e51e970159c23cc65c3a7be6b99315110809cd9acd992f1edc9bce55af301705"
    },
};

   

        int failed_tests = 0;
        for (const auto& test : test_vectors) {
        // 1. 准备输入和预期输出
        BigInt priv_key   = string_to_bigint(test.priv_key_hex);
        ECPoint expected_pub_key;
        expected_pub_key.x        = string_to_bigint(test.pub_key_x_hex);
        expected_pub_key.y        = string_to_bigint(test.pub_key_y_hex);
        // 对于私钥字符串全 0 或者等于阶 n，期望都是无穷远
        if (test.priv_key_hex == "0000000000000000000000000000000000000000000000000000000000000000"
        || test.priv_key_hex == "fffffffffffffffffffffffffffffffebaaedce6af48a03bbfd25e8cd0364141") {
           expected_pub_key.infinity = true;
        } else {
           expected_pub_key.infinity = false;
        }


            std::cout << "  Private Key:  "; print_bigint_host(priv_key); std::cout << std::endl;
            
            // 2. 在 GPU 上執行計算
            ECPoint gpu_result = get_public_key_gpu_single(priv_key);

            std::cout << "  Expected PubX: "; print_bigint_host(expected_pub_key.x); std::cout << std::endl;
            std::cout << "  GPU Result X: "; print_bigint_host(gpu_result.x); std::cout << std::endl;
            std::cout << "  Expected PubY: "; print_bigint_host(expected_pub_key.y); std::cout << std::endl;
            std::cout << "  GPU Result Y: "; print_bigint_host(gpu_result.y); std::cout << std::endl;

            // 3. 比較結果
            if (compare_ecpoint(gpu_result, expected_pub_key)) {
                std::cout << "  => [ \033[32mPASS\033[0m ]" << std::endl;
            } else {
                std::cout << "  => [ \033[31mFAIL\033[0m ]" << std::endl;
                failed_tests++;
            }
        }
        
        if (failed_tests > 0) {
            std::cerr << "\nVerification failed. " << failed_tests << " out of " 
                      << test_vectors.size() << " tests failed." << std::endl;
            std::cerr << "Aborting performance test." << std::endl;
            cudaDeviceReset();
            return 1;
        }

        std::cout << "\n\033[32mAll correctness verification tests passed!\033[0m" << std::endl;
        
        // --- B. 性能測試 (只有在驗證通過後才執行) ---
        std::cout << "\n--- Section B: Performance Test ---" << std::endl;
        const int num_keys = 1000 * 1000;
        const int threads_per_block = 256; // 在現代 GPU 上可以嘗試 256 或 512
        const int num_blocks = (num_keys + threads_per_block - 1) / threads_per_block;

        std::cout << "Configuration:" << std::endl;
        std::cout << "  - Keys to generate: " << num_keys << std::endl;
        std::cout << "  - Threads per block: " << threads_per_block << std::endl;
        std::cout << "  - Number of blocks: " << num_blocks << std::endl;
        
        std::vector<BigInt> h_private_keys(num_keys);
        for (int i = 0; i < num_keys; ++i) {
            init_bigint(&h_private_keys[i], i + 1); // 使用簡單私鑰
        }

        BigInt *d_private_keys = nullptr;
        ECPoint *d_public_keys = nullptr;
        CHECK_CUDA(cudaMalloc(&d_private_keys, num_keys * sizeof(BigInt)));
        CHECK_CUDA(cudaMalloc(&d_public_keys, num_keys * sizeof(ECPoint)));
        CHECK_CUDA(cudaMemcpy(d_private_keys, h_private_keys.data(), num_keys * sizeof(BigInt), cudaMemcpyHostToDevice));

        cudaEvent_t start, stop;
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));
        
        CHECK_CUDA(cudaEventRecord(start));
        private_to_public_key_batch_kernel<<<num_blocks, threads_per_block>>>(d_private_keys, d_public_keys, num_keys);
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaDeviceSynchronize());
        
        float milliseconds = 0;
        CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
        
        std::cout << "\n--- Performance Results ---" << std::endl;
        std::cout << "GPU Kernel time: " << std::fixed << std::setprecision(3) << milliseconds << " ms" << std::endl;
        double keys_per_second = (double)num_keys / (milliseconds / 1000.0);
        std::cout << "Throughput: " << std::fixed << std::setprecision(2) << keys_per_second / 1e6 << " MKey/s" << std::endl;

        // 清理
        CHECK_CUDA(cudaFree(d_private_keys));
        CHECK_CUDA(cudaFree(d_public_keys));
        CHECK_CUDA(cudaEventDestroy(start));
        CHECK_CUDA(cudaEventDestroy(stop));

    } catch (const std::exception& e) {
        std::cerr << "An error occurred: " << e.what() << std::endl;
        cudaDeviceReset();
        return 1;
    }

    // 重置設備，清理所有上下文和內存
    CHECK_CUDA(cudaDeviceReset());
    return 0;
}
