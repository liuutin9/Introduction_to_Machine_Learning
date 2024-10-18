#include <iostream>
#include <cmath>
#include <ctime>

void radixSort(int arr[], int len, int digit) {
    for (int d = 0; d < digit; d++) {
        int bucket[10][len] = {0}, count[10] = {0}, idx = 0;
        for (int i = 0; i < len; i++) {
            int category = (arr[i] / (int)std::pow(10, d)) % 10;
            bucket[category][count[category]++] = arr[i];
        }
        for (int i = 0; i < 10; i++)
            for (int j = 0; j < count[i]; j++)
                arr[idx++] = bucket[i][j];
    }
}

int main() {
    std::srand(std::time(0));
    int n = 20, arr[n], k = 10000;
    for (int i = 0; i < n; i++) {
        arr[i] = std::rand() % k;
        if (arr[i] < 1000) arr[i] += 1000;
    }
    std::cout << "Before sorting: ";
    for (int i = 0; i < n; i++) std::cout << arr[i] << ' ';
    std::cout << std::endl;
    radixSort(arr, n, 4);
    std::cout << "After sorting:  ";
    for (int i = 0; i < n; i++) std::cout << arr[i] << ' ';
    std::cout << std::endl;
}