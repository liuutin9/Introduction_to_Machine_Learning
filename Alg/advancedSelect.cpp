#include <iostream>
#include <cstdlib>
#include <ctime>
#include <algorithm>

int Select(int A[], int p, int r, int i);
int RandomizedPartition(int A[], int p, int r);
int Partition(int A[], int p, int r);

int main() {
    int N = 100, K, G;
    // std::cin >> N >> K >> G;
    int arr[N], sorted_arr[N];

    // 初始化隨機數生成器
    std::srand(std::time(0));
    K = std::rand() % N + 1;
    // K = 1;

    // 填充陣列
    for (int i = 0; i < N; ++i) {
        arr[i] = std::rand() % N + 1;
        sorted_arr[i] = arr[i];
    }
    std::sort(sorted_arr, sorted_arr + N);

    // 找出第 K 小的元素
    int Kth = Select(arr, 0, N - 1, K);

    // 輸出結果
    std::cout << "Sorted array: ";
    for (int i = 0; i < N; ++i) {
        std::cout << sorted_arr[i] << ' ';
    }
    std::cout << std::endl;
    std::cout << "K is " << K << std::endl;
    std::cout << Kth << ' ' << sorted_arr[K - 1] << std::endl;
    std::cout << (Kth == sorted_arr[K - 1] ? "True" : "False") << std::endl;
}

int Select(int A[], int p, int r, int i) {
    if (p == r) {
        return A[p];
    }
    int q = RandomizedPartition(A, p, r);
    int k = q - p + 1;
    if (i == k) {
        return A[q];
    } else if (i < k) {
        return Select(A, p, q - 1, i);
    } else {
        return Select(A, q + 1, r, i - k);
    }
}

int RandomizedPartition(int A[], int p, int r) {
    int i = std::rand() % (r - p + 1) + p;
    std::swap(A[r], A[i]);
    return Partition(A, p, r);
}

int Partition(int A[], int p, int r) {
    int x = A[r];
    int i = p - 1;
    for (int j = p; j < r; ++j) {
        if (A[j] <= x) {
            std::swap(A[++i], A[j]);
        }
    }
    std::swap(A[i + 1], A[r]);
    return i + 1;
}