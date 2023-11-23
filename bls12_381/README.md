# Wgsl implementaion of bls12_381

The wgsl implementation is called from `fp_all.wgsl`

## Steps on Testing
- `cd bls12_381`
- `cargo test`

## Steps on Benching
- `cd bls12_381`
- `cargo bench`

## GPU information

| Parameter | Value |
| :--- | :--- |
| name    | "Apple M1 GPU"    |
| vendor    | "Apple Inc."    |
| device    | "M1"    |
| device_type    | "IntegratedGpu"    |
| driver    | "Apple"    |
| driver_info    | "Metal"    |
| backend    | "Metal"    |


## Benchmark results

The time cell that contains N/A are either not implemented yet, for that column


| Function                                        | GPU Time               | CPU Time                 |
| ------------------------------------------------|------------------------|--------------------------|
| Bigint multiply                                 | 78.777 ms - 80.046 ms  | N/A                      |
| Bigint add                                      | 83.114 ms - 87.527 ms  | N/A                      |
| Bigint adc                                      | 82.748 ms - 91.893 ms  | 0.0002 ps - 0.0003 ps    |
| Bigint mac                                      | 81.285 ms - 89.749 ms  | 0.0001 ps - 0.0003 ps    |
| Fp multiply                                     | 79.876 ms - 87.774 ms  | 15.271 ns - 15.339 ns    |
| Fp subtract                                     | 78.921 ms - 87.439 ms  | 14.831 ns - 14.896 ns    |
| Fp2 add                                         | 78.539 ms - 85.524 ms  | N/A                      |
| Fp6 add                                         | 79.135 ms - 87.809 ms  | N/A                      |
| G1projective Generator is on curve              | 81.086 ms - 90.459 ms  | 190.36 ns - 206.23 ns    |
| G1Projective identity is on curve               | 84.144 ms - 93.578 ms  | 192.16 ns - 202.12 ns    |
| G1Projective test equality                      | 80.747 ms - 88.283 ms  | N/A                      |
| G1projective test conditionally select affine   | 92.884 ms - 101.86 ms  | 55.251 ns - 55.481 ns    |
| full pairing                                    | N/A                    | 1.4428 ms - 1.5749 ms    |
| G2 preparation for pairing                      | N/A                    | 159.91 µs - 168.94 µs    |
| miller loop for pairing                         | N/A                    | 400.06 µs - 418.87 µs    |
| final exponentiation for pairing                | N/A                    | 894.33 µs - 930.23 µs    |
| G1Affine scalar multiplication                  | N/A                    | 374.31 µs - 379.78 µs    |
| G1Affine subgroup check                         | N/A                    | 79.742 µs - 81.061 µs    |
| G1Affine deserialize compressed point           | N/A                    | 113.59 µs - 126.59 µs    |
| G1Affine deserialize uncompressed point         | N/A                    | 80.260 µs - 81.992 µs    |

The huge time discrepancy in the GPU side is because 
- The time is actually sum of time required to initialize wgpu, run compute shaders, pass the data to CPU.  
```
    let c = pollster::block_on(run(&vec![9, 4], "multiply_test"));
```
- further optimization is needed which includes selecting appropriate workgroup sizes

Accurate time reading will be done, once the core functionalities  are implemented.
