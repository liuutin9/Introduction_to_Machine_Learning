#include <iostream>
#include <ctime>

class Node {
    public:
    int data;
    int id;
    Node(int d, int i) : data(d), id(i) {}
};

void countingSort(Node* arr[], int k) {
    int count[k], count2[k + 1];
    for (int i = 0; i < k; i++) count[i] = 0;
    for (int i = 0; i < 20; i++) count[arr[i]->data]++;
    for (int i = 1; i <= k; i++) count[i] += count[i - 1];
    count2[0] = 0;
    for (int i = 0; i < k; i++) count2[i + 1] = count[i];

    for (int i = 0; i < 20; i++) {
        while (count2[arr[i]->data] != count[arr[i]->data]) {
            int curr = arr[i]->data;
            if (i == count2[arr[i]->data]) {
                count2[arr[i]->data]++;
                break;
            }
            std::swap(arr[i], arr[count2[arr[i]->data]]);
            count2[curr]++;
        }
    }

}

int main() {
    std::srand(std::time(0));
    Node* arr[20];
    for (int i = 0; i < 20; i++) arr[i] = new Node(std::rand() % 100, i);
    for (int i = 0; i < 20; i++) printf("(%d, %d) ", arr[i]->id, arr[i]->data);
    std::cout << std::endl;
    countingSort(arr, 100);
    for (int i = 0; i < 20; i++) printf("(%d, %d) ", arr[i]->id, arr[i]->data);
    std::cout << std::endl;
    for (int i = 0; i < 20; i++) delete arr[i];
}