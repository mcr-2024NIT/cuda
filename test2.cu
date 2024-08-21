    #include <iostream>
    #include <Cudacpp/cudaVector.cuh>
    #include "lib/atomicQ.cuh"
    #include "lib/atomicMap.cuh"
    #include <cuda_runtime.h>
    #include <chrono>

    const int WIDTH = 10;
    const int HEIGHT = 20;


    struct TetrisState {
        cudacpp::cudaVector<cudacpp::cudaVector<int>> board;
        int x, y, rotation;
        cudacpp::cudaVector<char> moves;

        __device__ bool operator==(const TetrisState& other) const {
            return board == other.board && x == other.x && y == other.y && rotation == other.rotation;
        }
    };


    struct TetrisStateHash {
        __device__ size_t operator()(const TetrisState& state) const {
            size_t seed = 0;
            for (int i = 0; i < state.board.size(); ++i) {
                for (int j = 0; j < state.board[i].size(); ++j) {
                    seed ^= state.board[i][j] + 0x9e3779b9 + (seed << 6) + (seed >> 2);
                }
            }
            seed ^= state.x + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            seed ^= state.y + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            seed ^= state.rotation + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            return seed;
        }
    };

    __device__ cudacpp::cudaVector<cudacpp::cudaVector<int>> rotatePiece(const cudacpp::cudaVector<cudacpp::cudaVector<int>>& piece, int rotation) {
        int n = piece.size();
        cudacpp::cudaVector<cudacpp::cudaVector<int>> rotatedPiece(n, cudacpp::cudaVector<int>(n, 0));
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                if (rotation == 1) rotatedPiece[j][n - 1 - i] = piece[i][j];
                else if (rotation == 2) rotatedPiece[n - 1 - i][n - 1 - j] = piece[i][j];
                else if (rotation == 3) rotatedPiece[n - 1 - j][i] = piece[i][j];
                else rotatedPiece[i][j] = piece[i][j];
            }
        }
        return rotatedPiece;
    }

    __device__
    bool isValid(const TetrisState& state, const cudacpp::cudaVector<cudacpp::cudaVector<int>>& piece) {
        auto rotatedPiece = rotatePiece(piece, state.rotation);
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

    __device__
    cudacpp::cudaVector<cudacpp::cudaVector<int>> placePiece(cudacpp::cudaVector<cudacpp::cudaVector<int>> board, const cudacpp::cudaVector<cudacpp::cudaVector<int>>& piece, int x, int y, int rotation) {
        auto rotatedPiece = rotatePiece(piece, rotation);
        for (int i = 0; i < rotatedPiece.size(); ++i) {
            for (int j = 0; j < rotatedPiece[0].size(); ++j) {
                if (rotatedPiece[i][j] && y + i < HEIGHT && x + j < WIDTH) {
                    board[y + i][x + j] = rotatedPiece[i][j];
                }
            }
        }
        return board;
    }
    __device__
    void printBoard(cudacpp::cudaVector<cudacpp::cudaVector<int>> board) {
        for (int i = 0; i < board.size(); ++i) {
            for (int j = 0; j < board[i].size(); ++j) {
                printf("%d ", board[i][j]);
            }
            printf("\n");
        }
        printf("\n");
    }
    //各行、上から確認して一番初めに要素が確認されたところでストップしてその行数を出力する
    __device__
    void isTopRowAllZero(TetrisState& state) {
        for (int i = 0; i < state.board.size(); ++i) {
            for (int j = 0; j < state.board[i].size(); ++j) {
                if (state.board[i][j]) {
                    state.y = i - 1;
                    printf("y: %d\n", state.y);
                    return ;
                }
            }
        }
        state.y = state.board.size()-1;
        return ;
    }



    __global__
    void bfsAllTetrisStates(int* devicePiece, int* deviceBoard) {
        int tid = threadIdx.x;
        
            printf("tid: %d\n", tid);        
        extern __shared__ int sharedMem[];

    cudacpp::cudaQueue<TetrisState, 196>* q = (cudacpp::cudaQueue<TetrisState, 196>*)sharedMem;
    size_t queueSize = sizeof(cudacpp::cudaQueue<TetrisState, 196>);
    printf("queueSize: %d\n", queueSize);
    cudacpp::cuda_unordered_map<TetrisState, cudacpp::cudaVector<char>, TetrisStateHash>* uniqueBoards =
        new (&sharedMem[queueSize]) cudacpp::cuda_unordered_map<TetrisState, cudacpp::cudaVector<char>, TetrisStateHash>(256);
        printf("uniqueBoards size: %d\n", 6);
    size_t mapSize = sizeof(cudacpp::cuda_unordered_map<TetrisState, cudacpp::cudaVector<char>, TetrisStateHash>);
    printf("mapSize: %d\n", mapSize);
    cudacpp::cudaVector<TetrisState>* finalStates =
        new (&sharedMem[queueSize + mapSize]) cudacpp::cudaVector<TetrisState>();
        cudacpp::cudaVector<cudacpp::cudaVector<int>> initialBoard(HEIGHT, cudacpp::cudaVector<int>(WIDTH, 0));
        cudacpp::cudaVector<cudacpp::cudaVector<int>> piece(3, cudacpp::cudaVector<int>(3, 0));
        if (tid==0)
        {
            for (int i = 0; i < HEIGHT; ++i) {
                for (int j = 0; j < WIDTH; ++j) {
                    initialBoard[i][j] = deviceBoard[i * WIDTH + j];
                }
            }

            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    piece[i][j] = devicePiece[i * 3 + j];
                }
            }
            TetrisState initialState = {initialBoard, WIDTH / 2-2, 0, 0, {}};

            isTopRowAllZero(initialState);
            initialState.y-=piece.size();
        
            q->push(initialState);
        }
        __syncthreads();


        






        int is = 0;
        while (tid < q->size()) {

                    printf("q size: %d\n", q->size());
            printf("tid: %d\n", tid);
            
            
            TetrisState currentState;
            q->pop_and_front(currentState);
            if (currentState.y == HEIGHT - 1 || !isValid({currentState.board, currentState.x, currentState.y + 1, currentState.rotation, currentState.moves}, piece)) {
                currentState.board = placePiece(currentState.board, piece, currentState.x, currentState.y, currentState.rotation);
                currentState.moves.push_back('P');
                auto it = uniqueBoards->find(currentState);
                if (it== uniqueBoards->end()||it->second.size() > currentState.moves.size()) {
                    (*uniqueBoards)[currentState] = currentState.moves;
                    finalStates->push_back(currentState);
                    for (int i = 0; i < currentState.moves.size(); i++)
                    {
                        printf("%c ", currentState.moves[i]);
                    }
                    
                }
                
                continue;
            }
            auto it = uniqueBoards->find(currentState);
            printf("uniqueBoards size: %d\n", 6);
            if (it== uniqueBoards->end()||it->second.size() > currentState.moves.size()) {

               (*uniqueBoards)[currentState] = currentState.moves;


                TetrisState newState = currentState;
                newState.rotation = (currentState.rotation + 1) % 4;
                newState.moves.push_back('O');
                if (isValid(newState, piece)) {
                    q->push(newState);
                }
                if (currentState.moves.back() != 'R'){

                
                    TetrisState newStateL = currentState;
                    newStateL.x--;
                    newStateL.moves.push_back('L');
                    if (isValid(newStateL, piece)) {
                        q->push(newStateL);
                        
                    }
                }
                if (currentState.moves.back() != 'L'){
                    TetrisState newStateR = currentState;
                    newStateR.x++;
                    newStateR.moves.push_back('R');
                    if (isValid(newStateR, piece)) {
                        q->push(newStateR);
                    }

                }

                TetrisState newStateD = currentState;
                newStateD.y++;
                newStateD.moves.push_back('D');
                if (isValid(newStateD, piece)) {


                    q->push(newStateD);
                }
                printf("q size: %d\n", q->size());
            
            }
            __syncthreads();
            


            
        is++;
        }

        for (const TetrisState& state : *finalStates) {
            //printBoard(state.board);
            
            for (const auto& move : state.moves) {
                printf("%c ", move);
            }
            printf("\n");
        }
        printf("Number of unique states: %d\n", finalStates->size());

    }

    int main() {
        cudacpp::cudaVector<cudacpp::cudaVector<int>> initialBoard(HEIGHT, cudacpp::cudaVector<int>(WIDTH, 0));
    cudacpp::cudaVector<cudacpp::cudaVector<int>> piece = {
            {0, 1, 0},
            {1, 1, 1},
            {0, 0, 0}
        };

        // ホスト側のデータを定義し、データを準備
        cudacpp::cudaVector<int> hostBoard(HEIGHT*WIDTH);
        cudacpp::cudaVector<int> hostPiece(piece.size()*piece[0].size());
        for (int i = 0; i < HEIGHT; ++i) {
            for (int j = 0; j < WIDTH; ++j) {
                hostBoard[i * WIDTH + j] = initialBoard[i][j];
            }
        }

        for (int i = 0; i < piece.size(); ++i) {
            for (int j = 0; j < piece[0].size(); ++j) {
                hostPiece[i * piece[0].size() + j] = piece[i][j];
            }
        }



        // デバイス側のメモリ確保
        int* devicePiece;int* deviceBoard;
        cudaMalloc(&devicePiece, piece.size()*piece[0].size() * sizeof(int));
        cudaMalloc(&deviceBoard, HEIGHT*WIDTH * sizeof(int));

        // ホストからデバイスへのデータ転送
        cudaMemcpyAsync(devicePiece, hostPiece.data(), piece.size() * piece[0].size() * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpyAsync(deviceBoard, hostBoard.data(), HEIGHT * WIDTH * sizeof(int), cudaMemcpyHostToDevice);

        // カーネルの設定（スレッド1つ、ブロック1つ）
        auto start = std::chrono::system_clock::now();
        int size = sizeof(cudacpp::cudaQueue<TetrisState,196>) + sizeof(cudacpp::cuda_unordered_map<TetrisState, cudacpp::cudaVector<char>, TetrisStateHash>)+sizeof(cudacpp::cudaVector<TetrisState>);

        bfsAllTetrisStates<<<1, 1,size>>>(devicePiece, deviceBoard);
            // カーネルの実行を待機
        cudaDeviceSynchronize();
        auto end = std::chrono::system_clock::now();
        auto dur = end - start;
        auto msec = std::chrono::duration_cast<std::chrono::milliseconds>(dur).count();
        std::cout << msec << " ms" << std::endl;



        // デバイスメモリの解放
        cudaFree(devicePiece);
        cudaFree(deviceBoard);


        return 0;
    }
