### Secp256k1 CUDA Implementation
This repository provides a CUDA-based high-performance implementation of elliptic curve cryptography (ECC) operations using the SECP256k1 curve, which is widely used in blockchain technologies such as Bitcoin.
The implementation is optimized for batch processing and uses low-level GPU operations to accelerate elliptic curve point multiplication and modulo operations.

### Overview

The implementation covers the following key components:
Big Integer (BigInt): Supports multi-precision arithmetic for 256-bit integers, which are the foundation for ECC operations.

Elliptic Curve Point (ECPoint): Supports affine coordinates for elliptic curve points.
Optimized Arithmetic: Includes functions for addition, subtraction, modular multiplication, modular inverse, and modular exponentiation.

Montgomery Ladder: Implements a batch-optimized scalar multiplication using the Montgomery ladder algorithm, which is crucial for efficient ECC point multiplication.

CUDA Kernels: The core of the implementation is designed for GPU acceleration, using custom CUDA kernels to perform elliptic curve operations in parallel across multiple keys.

### Key Features
`
Elliptic Curve Arithmetic: Supports addition, doubling, and scalar multiplication of elliptic curve points over the SECP256k1 field.
Modular Arithmetic: Optimized modular operations, including multiplication and inverse, that respect the SECP256k1 curve's modulus (p = 2^256 - 2^32 - 977).
CUDA Optimization: The code is optimized for CUDA to utilize the power of GPUs for batch scalar multiplications.
Efficient Memory Usage: The implementation leverages constant memory and shared memory to minimize latency and maximize throughput on supported NVIDIA GPUs.

### Requirements
```sh
CUDA-enabled GPU
CUDA Toolkit 11.x or later
NVIDIA Driver 450.x or later
C++11 or later compiler
```
### File Structure
secp256k1.cuh: Contains the core header file with the definitions of the elliptic curve point structure, big integer operations, and CUDA kernel functions.
kernel_montgomery_ladder_batch_optimized: The kernel for performing batch scalar multiplication using the Montgomery ladder.

Helper Functions: Includes functions for multi-precision integer operations
 (addition, subtraction, multiplication), modular arithmetic, and ECC point operations.


### Compilation
To compile this code, you need a CUDA development environment set up. Run the following command:

```sh
nvcc test_secp256k1.cu -o test_secp256k1
```
Example Usage
Once compiled, you can use the code to perform elliptic curve operations on the GPU. The implementation supports batch processing of scalar multiplications, and you can specify the number of keys to process. 

The kernel_montgomery_ladder_batch_optimized kernel performs scalar multiplication using the Montgomery ladder algorithm.

### Functionality
point_add: Adds two elliptic curve points.
double_point: Doubles an elliptic curve point.
modexp: Computes modular exponentiation.
mod_inverse: Computes the modular inverse of a number.
Kernel Launch
To perform a batch operation, launch the kernel_montgomery_ladder_batch_optimized kernel:
```sh
kernel_montgomery_ladder_batch_optimized<<<numBlocks, numThreads, sharedMemorySize>>>(
    d_keys, G, p, Q_keys, n, d_processedCounter
);
```
Where:
```sh
d_keys: Input array of private keys.
G: The generator point for SECP256k1.
p: The modulus for SECP256k1.
Q_keys: Output array of elliptic curve points (Q = d * G).
n: Number of keys to process.
d_processedCounter: Atomic counter to track the number of processed keys.
Contributing
Contributions are welcome! Please open an issue or submit a pull request for bug fixes, improvements, or new features. 

Ensure all contributions adhere to the style and performance requirements of the repository.
```
### Sponsorship
If this project has been helpful to you, please consider sponsoring. 

Your support is greatly appreciated. Thank you!
```sh
-BTC: bc1qt3nh2e6gjsfkfacnkglt5uqghzvlrr6jahyj2k
-ETH: 0xD6503e5994bF46052338a9286Bc43bC1c3811Fa1
-DOGE: DTszb9cPALbG9ESNJMFJt4ECqWGRCgucky
-TRX: TAHUmjyzg7B3Nndv264zWYUhQ9HUmX4Xu4
```
### License
This project is licensed under the MIT License - see the LICENSE file for details.

