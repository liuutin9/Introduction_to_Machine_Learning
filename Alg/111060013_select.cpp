#include <iostream>
#include <cstdlib>
#include <ctime>
#include <algorithm>

int SelectKthSmallest(int A[], int l, int r, int kth, int numsPerGroup);
int RandomizedSelectPivot(int p, int r);
int MedianSelectPivot(int A[], int l, int r, int numsPerGroup);
int Partition(int A[], int p, int r, int i);
void InsertionSortColumn(int A[], int l, int r, int numsOfGroup);
void InsertionSort(int A[], int l, int r);

int arr[100000000], sortedArr[100000000];

int main() {

    bool debug = false;
    std::srand(std::time(0));

    if (debug) {

        int totalExecutionTimes = 50;
        double averageExecutionTime[5] = {0};
        clock_t start[5], end[5], duration[5];
        int methods[5] = {0, 3, 5, 7, 9}, Kth[5] = {0};
        int N = 10000000, K;
        int i, j;

        std::cout << "======== You're in debug mode ========" << std::endl;
        std::cout << "There are 5 methods to choose: 0, 3, 5, 7, 9." << std::endl;
        std::cout << "Note that 0 means picking a pivot randomly." << std::endl;
        std::cout << "Enter the array size N: ";
        std::cin >> N;
        for (int currExecutionTimes = 1; currExecutionTimes <= totalExecutionTimes; currExecutionTimes++) {
            K = std::rand() % N + 1;
            std::cout << "Test " << currExecutionTimes << std::endl;
            std::cout << "Looking for the " << K << "th smallest element in the array." << std::endl;
            std::cout << "=========== Debug Messages ===========" << std::endl;

            for (i = 0; i < N; i++) sortedArr[i] = std::rand() % N + 1;

            for (j = 0; j < 5; j++) {
                for (i = 0; i < N; i++) arr[i] = sortedArr[i];
                start[j] = clock();
                Kth[j] = arr[SelectKthSmallest(arr, 0, N - 1, K, methods[j])];
                end[j] = clock();
                duration[j] = end[j] - start[j];
                averageExecutionTime[j] += (double)duration[j] / CLOCKS_PER_SEC / totalExecutionTimes;
            }

            std::sort(sortedArr, sortedArr + N);

            std::cout << "Answer by std::sort(): " << sortedArr[K - 1] << '.' << std::endl;
            std::cout << "My answers:" << std::endl;
            std::cout << "With method 'Randomized': " << Kth[0] << '.' << std::endl;
            for (j = 1; j < 5; j++) {
                std::cout << "With method 'Group with " << methods[j] <<"': " << Kth[j] << '.' << std::endl;
            }
            std::cout << "Check my answers:" << std::endl;
            std::cout << "Randomized: " << (Kth[0] == sortedArr[K - 1] ? "Correct!" : "Wrong!") << std::endl;
            if (Kth[0] != sortedArr[K - 1]) system("pause");
            for (j = 1; j < 5; j++) {
                std::cout << "Group with " << methods[j] << ": " << (Kth[j] == sortedArr[K - 1] ? "Correct!" : "Wrong!") << std::endl;
                if (Kth[j] != sortedArr[K - 1]) system("pause");
            }
            std::cout << "Time used:" << std::endl;
            std::cout << "Randomized: " << (double)duration[0] / CLOCKS_PER_SEC << "s." << std::endl;
            for (j = 1; j < 5; j++) {
                std::cout << "Group with " << methods[j] << ": " << (double)duration[j] / CLOCKS_PER_SEC << "s." << std::endl;
            }
            std::cout << "======================================" << std::endl;
        }

        std::cout << "Array size: " << N << std::endl;
        std::cout << "Execution times: " << totalExecutionTimes << std::endl;
        std::cout << "Average time used:" << std::endl;
        std::cout << "Randomized: " << averageExecutionTime[0] << "s." << std::endl;
        for (j = 1; j < 5; j++) {
            std::cout << "Group with " << methods[j] << ": " << averageExecutionTime[j] << "s." << std::endl;
        }

    }
    
    else {
        int N, K, G;
        std::cin >> N >> K >> G;
        int arr[N], Kth;
        for (int i = 0; i < N; i++) std::cin >> arr[i];
        Kth = SelectKthSmallest(arr, 0, N - 1, K, G);
        std::cout << arr[Kth] << std::endl;
    }

}

int SelectKthSmallest(int A[], int l, int r, int kth, int numsPerGroup) {
    int size, numsOfGroup, excess, pivotInitPosition;
    if (numsPerGroup == 0) {
        pivotInitPosition = RandomizedSelectPivot(l, r);
    }
    else {
        size = r - l + 1;
        numsOfGroup = (size) / numsPerGroup;
        excess = (size) % numsPerGroup;
        if (excess != 0) numsOfGroup++;
        if (size <= numsPerGroup) {
            InsertionSort(A, l, r);
            pivotInitPosition = l + size / 2;
        }
        else {
            for (int j = l; j <= l + numsOfGroup - 1; j++) {
                InsertionSortColumn(A, j, r, numsOfGroup);
            }
            pivotInitPosition = SelectKthSmallest(A, l + (numsPerGroup / 2) * numsOfGroup, l + (numsPerGroup / 2 + 1) * numsOfGroup - 1, (numsOfGroup % 2 ? numsOfGroup / 2 : numsOfGroup / 2 + 1), numsPerGroup);
        }
    }
    int pivotFinalPosition = Partition(A, l, r, pivotInitPosition);
    int k = pivotFinalPosition - l + 1;
    if (kth == k) return pivotFinalPosition;
    else if (kth < k) return SelectKthSmallest(A, l, pivotFinalPosition - 1, kth, numsPerGroup);
    else return SelectKthSmallest(A, pivotFinalPosition + 1, r, kth - k, numsPerGroup);
}

int RandomizedSelectPivot(int l, int r) {
    return std::rand() % (r - l + 1) + l;
}

int MedianSelectPivot(int A[], int l, int r, int numsPerGroup) {
    int size = r - l + 1;
    int numsOfGroup = (size) / numsPerGroup;
    int excess = (size) % numsPerGroup;
    if (excess != 0) numsOfGroup++;
    if (size < numsPerGroup) {
        InsertionSort(A, l, r);
        return l + size / 2;
    }
    else {
        for (int j = l; j <= l + numsOfGroup - 1; j++) {
            InsertionSortColumn(A, j, r, numsOfGroup);
        }
        return MedianSelectPivot(A, l + (numsPerGroup / 2) * numsOfGroup, l + (numsPerGroup / 2 + 1) * numsOfGroup - 1, numsPerGroup);
    }
}

int Partition(int A[], int l, int r, int i) {
    std::swap(A[r], A[i]);
    int pivotPosition = l;
    for (int j = l; j < r; j++) {
        if (A[j] <= A[r]) std::swap(A[pivotPosition++], A[j]);
    }
    std::swap(A[pivotPosition], A[r]);
    return pivotPosition;
}

void InsertionSortColumn(int A[], int l, int r, int numsOfGroup) {
    int size = (r - l + 1) / numsOfGroup;
    int tmp[size] = {A[l]};
    for (int i = 1; i < size; i++) {
        tmp[i] = A[l + i * numsOfGroup];
        int curr = i;
        while (tmp[curr] < tmp[curr - 1] && curr > 0) {
            std::swap(tmp[curr], tmp[curr - 1]);
            curr--;
        }
    }
    for (int i = 0; i < size; i++) {
        A[l + i * numsOfGroup] = tmp[i];
    }
}

void InsertionSort(int A[], int l, int r) {
    for (int i = l + 1; i <= r; i++) {
        int curr = i;
        while (A[curr] < A[curr - 1] && curr > l) {
            std::swap(A[curr], A[curr - 1]);
            curr--;
        }
    }
}