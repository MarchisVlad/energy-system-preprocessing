#ifndef COMPARATOR_HPP
#define COMPARATOR_HPP

#include <iostream>
#include <vector>
#include <string>
#include <pair>

static size_t get_distance(std::vector<std::vector<size_t>> A,
    std::vector<std::vector<std::size_t>> B, 
    std::vector<std::pair<size_t, size_t>> blocks_A, 
    std::vector<std::pair<size_t, size_t>> blocks_B, 
    std::string method);


#endif /* COMPARATOR_HPP */