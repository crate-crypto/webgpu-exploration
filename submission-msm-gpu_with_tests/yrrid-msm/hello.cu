#include <iostream>
#include <cuda_runtime.h>

__global__ void vectorAddition(float* a, float* b, float* result, int size) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < size) {
        result[index] = a[index] + b[index];
    }
}

int main() {
    // Створюємо cudaEvent для вимірювання часу
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Запускаємо таймер
    cudaEventRecord(start, 0);

    int size = 1000000;
    int byteSize = size * sizeof(float);

    // Задаємо вектори
    float* hostVectorA = new float[size];
    float* hostVectorB = new float[size];
    float* hostResult = new float[size];

    // Ініціалізуємо вектори
    for (int i = 0; i < size; ++i) {
        hostVectorA[i] = static_cast<float>(i);
        hostVectorB[i] = static_cast<float>(i * 2);
    }

    // Виділяємо пам'ять на пристрої (GPU)
    float* deviceVectorA, * deviceVectorB, * deviceResult;
    cudaMalloc((void**)&deviceVectorA, byteSize);
    cudaMalloc((void**)&deviceVectorB, byteSize);
    cudaMalloc((void**)&deviceResult, byteSize);

    // Копіюємо дані з хоста на пристрій
    cudaMemcpy(deviceVectorA, hostVectorA, byteSize, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceVectorB, hostVectorB, byteSize, cudaMemcpyHostToDevice);

    // Задаємо конфігурацію запуску ядра
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;

    

    // Виклик ядра
    vectorAddition<<<gridSize, blockSize>>>(deviceVectorA, deviceVectorB, deviceResult, size);

    

    // Копіюємо результати обчислень з пристрою на хост
    cudaMemcpy(hostResult, deviceResult, byteSize, cudaMemcpyDeviceToHost);

    // Виводимо результати та час виконання
    // for (int i = 0; i < size; ++i) {
    //     std::cout << hostVectorA[i] << " + " << hostVectorB[i] << " = " << hostResult[i] << std::endl;
    // }
    

    // Вивільняємо пам'ять
    delete[] hostVectorA;
    delete[] hostVectorB;
    delete[] hostResult;
    cudaFree(deviceVectorA);
    cudaFree(deviceVectorB);
    cudaFree(deviceResult);

    // Зупиняємо таймер
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    // Обчислюємо час виконання
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);

    std::cout << "Elapsed Time: " << elapsedTime << " ms" << std::endl;

    return 0;
}
