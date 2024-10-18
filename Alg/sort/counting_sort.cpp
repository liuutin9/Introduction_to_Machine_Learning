#include <iostream>
#include <ctime>

void countingSort(int arr[], int len, int numsOfTypes) {
    int c[numsOfTypes] = {0};
    for (int i = 0; i < len; i++) c[arr[i]]++;;
    for (int i = numsOfTypes - 1; i > 0; i--)
        for (int j = 0; j < c[i]; j++)
            arr[--len] = i;
}

int main() {
    std::srand(std::time(0));
    int n = 20, arr[n], k = 100;
    for (int i = 0; i < n; i++) arr[i] = std::rand() % k;
    std::cout << "Before sorting: ";
    for (int i = 0; i < n; i++) std::cout << arr[i] << ' ';
    std::cout << std::endl;
    countingSort(arr, n, k);
    std::cout << "After sorting:  ";
    for (int i = 0; i < n; i++) std::cout << arr[i] << ' ';
    std::cout << std::endl;
}