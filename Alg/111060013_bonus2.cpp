#include <iostream>
#include <vector>
#include <queue>
#include <string>

using namespace std;

bool isValid(const vector<int>& queens, int row, int col) {
    for (int i = 0; i < row; ++i) {
        int placedCol = queens[i];
        if (placedCol == col || abs(placedCol - col) == abs(row - i)) {
            return false;
        }
    }
    return true;
}

int nQueensBFS(int N) {
    int solutions = 0;
    queue<vector<int>> q;
    q.push(vector<int>());

    while (!q.empty()) {
        vector<int> queens = q.front();
        q.pop();
        int row = queens.size();

        if (row == N) {
            solutions++;
            continue;
        }

        for (int col = 0; col < N; ++col) {
            if (isValid(queens, row, col)) {
                vector<int> newQueens = queens;
                newQueens.push_back(col);
                q.push(newQueens);
            }
        }
    }
    return solutions;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " <N>" << endl;
        return 1;
    }

    int N = stoi(argv[1]);
    if (N < 8) {
        cerr << "N must be at least 8." << endl;
        return 1;
    }

    cout << nQueensBFS(N) << endl;
    return 0;
}