#include <iostream>
#include <vector>
#include <unordered_map>
#include <queue>
#include <string>
#include <functional>

const int WIDTH = 10;
const int HEIGHT = 20;

struct TetrisState {
    std::vector<std::vector<int>> board;
    std::vector<std::vector<int>> currentPiece;
    int x, y, rotation;
    std::vector<std::string> moves;

    bool operator==(const TetrisState& other) const {
        return board == other.board && x == other.x && y == other.y && rotation == other.rotation;
    }
};

struct TetrisStateHash {
    std::size_t operator()(const TetrisState& state) const {
        std::size_t seed = 0;
        for (const auto& row : state.board) {
            for (int cell : row) {
                seed ^= std::hash<int>()(cell) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            }
        }
        seed ^= std::hash<int>()(state.x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        seed ^= std::hash<int>()(state.y) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        seed ^= std::hash<int>()(state.rotation) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        return seed;
    }
};

void printBoard(const std::vector<std::vector<int>>& board) {
    for (const auto& row : board) {
        for (int cell : row) {
            std::cout << (cell ? "#" : ".");
        }
        std::cout << std::endl;
    }
}

std::vector<std::vector<int>> rotatePiece(const std::vector<std::vector<int>>& piece, int rotation) {
    int n = piece.size();
    std::vector<std::vector<int>> rotatedPiece(n, std::vector<int>(n, 0));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            switch (rotation) {
                case 1: rotatedPiece[j][n - 1 - i] = piece[i][j]; break;
                case 2: rotatedPiece[n - 1 - i][n - 1 - j] = piece[i][j]; break;
                case 3: rotatedPiece[n - 1 - j][i] = piece[i][j]; break;
                default: rotatedPiece[i][j] = piece[i][j]; break;
            }
        }
    }
    return rotatedPiece;
}

std::vector<std::vector<int>> placePiece(const std::vector<std::vector<int>>& board, const std::vector<std::vector<int>>& piece, int x, int y, int rotation) {
    std::vector<std::vector<int>> newBoard = board;
    auto rotatedPiece = rotatePiece(piece, rotation);
    for (int i = 0; i < rotatedPiece.size(); ++i) {
        for (int j = 0; j < rotatedPiece[0].size(); ++j) {
            if (rotatedPiece[i][j] && y + i < HEIGHT && x + j < WIDTH) {
                newBoard[y + i][x + j] = rotatedPiece[i][j];
            }
        }
    }
    return newBoard;
}

bool isValid(const TetrisState& state) {
    auto rotatedPiece = rotatePiece(state.currentPiece, state.rotation);
    for (int i = 0; i < rotatedPiece.size(); ++i) {
        for (int j = 0; j < rotatedPiece[0].size(); ++j) {
            if (rotatedPiece[i][j]) {
                int newX = state.x + j;
                int newY = state.y + i;
                if (newX < 0 || newX >= WIDTH || newY < 0 || newY >= HEIGHT || state.board[newY][newX]) {
                    return false;
                }
            }
        }
    }
    return true;
}

void bfsAllTetrisStates() {
    std::vector<std::vector<int>> initialBoard(HEIGHT, std::vector<int>(WIDTH, 0));
    std::vector<std::vector<int>> piece = {
        {0, 1, 0},
        {1, 1, 1},
        {0, 0, 0},
        
    };

    TetrisState initialState = {initialBoard, piece, WIDTH / 2, 0, 0, {}};
    std::unordered_map<TetrisState, std::vector<std::string>, TetrisStateHash> uniqueBoards;
    std::queue<TetrisState> q;
    std::vector<TetrisState> finalStates;

    q.push(initialState);

    while (!q.empty()) {
        TetrisState currentState = q.front();
        q.pop();

        if (currentState.y == HEIGHT - 1 || !isValid({currentState.board, currentState.currentPiece, currentState.x, currentState.y + 1, currentState.rotation})) {
            currentState.board = placePiece(currentState.board, currentState.currentPiece, currentState.x, currentState.y, currentState.rotation);
            currentState.moves.push_back("Place");

            auto it = uniqueBoards.find(currentState);
            if (it == uniqueBoards.end() || it->second.size() > currentState.moves.size()) {
                uniqueBoards[currentState] = currentState.moves;
                finalStates.push_back(currentState);
            }
            continue;
        }

        auto it = uniqueBoards.find(currentState);
        if (it == uniqueBoards.end() || it->second.size() > currentState.moves.size()) {
            uniqueBoards[currentState] = currentState.moves;

            TetrisState downState = currentState;
            downState.y += 1;
            downState.moves.push_back("Down");
            if (isValid(downState)) q.push(downState);

            TetrisState leftState = currentState;
            leftState.x -= 1;
            leftState.moves.push_back("Left");
            if (isValid(leftState)) q.push(leftState);

            TetrisState rightState = currentState;
            rightState.x += 1;
            rightState.moves.push_back("Right");
            if (isValid(rightState)) q.push(rightState);

            TetrisState rotateState = currentState;
            rotateState.rotation = (rotateState.rotation + 1) % 4;
            rotateState.moves.push_back("Rotate");
            if (isValid(rotateState)) q.push(rotateState);
        }
    }

    for (const TetrisState& state : finalStates) {
        printBoard(state.board);
        for (const auto& move : state.moves) {
            std::cout << move << " ";
        }
        std::cout << std::endl;
    }
    std::cout << "Number of unique states: " << finalStates.size() << std::endl;
}

int main() {
    bfsAllTetrisStates();
    return 0;
}
