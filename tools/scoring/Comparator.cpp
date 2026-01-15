#include "Comparator.hpp"

#include <limits.h>

static double get_distance(std::vector<std::vector<size_t>> A,
    std::vector<std::vector<std::size_t>> B, 
    std::vector<std::pair<size_t, size_t>> blocks_A, 
    std::vector<std::pair<size_t, size_t>> blocks_B, 
    std::string method) {
    assert(A.size() && B.size() && "Non-null arrays allowed only.");
    
    

    return INT_MAX;
}

static size_t get_whitespace(std::vector<std::vector<size_t>> M, std::vector<std::pair<size_t, size_t>> blocks) {
    long long array_size       = M.size() * M[0].size();
    long long total_whitespace = 0;
    long long total_overlap    = 0;

    for(int i = 0; i < blocks.size(); i++) {
        
    }
}

static size_t count_nonzero(std::vector<std::vector<size_t>> M, std::pair<size_t, size_t> tl, std::pair<size_t, size_t> br) {
    assert(tl.first && tl.first < M.size() && tl.second && tl.second < M[0].size() && "Top-left vertex outside matrix bounds.");
    assert(br.first && br.first < M.size() && br.second && br.second < M[0].size() && "Bottom-right vertex outside matrix bounds.");

    size_t ws = 0;

    for(int i = tl.first; i < br.first; i++) {
        for(int j = tl.second; j < br.second; j++) {
            if(M[i][j])
                ws++;
        }
    }

    return ws;
}

