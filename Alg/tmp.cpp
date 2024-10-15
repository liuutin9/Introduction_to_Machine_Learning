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

int main() {

    std::srand(std::time(0));
    int N, K, G;
    std::cin >> N >> K >> G;
    int arr[N], Kth;
    for (int i = 0; i < N; i++) std::cin >> arr[i];
    Kth = SelectKthSmallest(arr, 0, N - 1, K, G);
    std::cout << Kth << std::endl;

}

int SelectKthSmallest(int A[], int l, int r, int kth, int numsPerGroup) {
    int pivotInitPosition = (numsPerGroup == 0)
        ? RandomizedSelectPivot(l, r)               // 1
        : MedianSelectPivot(A, l, r, numsPerGroup); // MSP(n)
    int pivotFinalPosition = Partition(A, l, r, pivotInitPosition);
    int k = pivotFinalPosition - l + 1;
    if (kth == k) return A[pivotFinalPosition];
    else if (kth < k) return SelectKthSmallest(A, l, pivotFinalPosition - 1, kth, numsPerGroup);
    else return SelectKthSmallest(A, pivotFinalPosition + 1, r, kth - k, numsPerGroup);
}

int RandomizedSelectPivot(int l, int r) {
    return std::rand() % (r - l + 1) + l;
}

int MedianSelectPivot(int A[], int l, int r, int numsPerGroup) {
    int size = r - l + 1;                      // 1
    int numsOfGroup = (size) / numsPerGroup;   // 1
    int excess = (size) % numsPerGroup;        // 1
    if (excess != 0) numsOfGroup++;            // 1
    if (size < numsPerGroup) {                 // 1
        InsertionSort(A, l, r);                // IS(l, r)
        return l + size / 2;                   // 1
    }
    else {
        for (int j = l; j <= l + numsOfGroup - 1; j++) { // l + floor(n / numsPerGroup)
            InsertionSortColumn(A, j, r, numsOfGroup);   // ISC(j, r) * (l + numsOfGroup)
        } // MSP(floor(n / numsPerGroup))
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