#include "Score.hpp"

#include <set>
#include <numeric>
#include <cmath>
#include <iostream>
#include <unordered_map>


class StructureScore {
protected:
    std::vector<int> equ_blocks;
    std::vector<int> var_blocks;
    double score;
    bool scorecomputed;

public:
    StructureScore(const std::vector<int>& equ_blocks, const std::vector<int>& var_blocks)
        : equ_blocks(equ_blocks), var_blocks(var_blocks), score(0.0), scorecomputed(false) {}

    virtual ~StructureScore() = default;

    double get_score() {
        if (!scorecomputed) {
            score = _compute_score();
            scorecomputed = true;
        }
        return score;
    }

protected:
    virtual double _compute_score() = 0;
};

class WhiteScore : public StructureScore {
public:
    WhiteScore(const std::vector<int>& equ_blocks, const std::vector<int>& var_blocks)
        : StructureScore(equ_blocks, var_blocks) {}

protected:
    double _compute_score() override {
        // Get sort indices for equations and variables
        std::vector<size_t> equ_sorted(equ_blocks.size());
        std::iota(equ_sorted.begin(), equ_sorted.end(), 0);
        std::stable_sort(equ_sorted.begin(), equ_sorted.end(),
            [this](size_t i, size_t j) { return equ_blocks[i] < equ_blocks[j]; });

        std::vector<size_t> var_sorted(var_blocks.size());
        std::iota(var_sorted.begin(), var_sorted.end(), 0);
        std::stable_sort(var_sorted.begin(), var_sorted.end(),
            [this](size_t i, size_t j) { return var_blocks[i] < var_blocks[j]; });

        // Create sorted arrays
        std::vector<int> equ_stage(equ_blocks.size());
        std::vector<int> var_stage(var_blocks.size());
        for (size_t i = 0; i < equ_sorted.size(); ++i) {
            equ_stage[i] = equ_blocks[equ_sorted[i]];
        }
        for (size_t i = 0; i < var_sorted.size(); ++i) {
            var_stage[i] = var_blocks[var_sorted[i]];
        }

        // Compute matrix area
        size_t matrix_area = equ_stage.size() * var_stage.size();
        if (matrix_area == 0) {
            return 1.0;
        }

        // Get unique blocks
        std::set<int> blocks_set;
        for (int val : equ_stage) blocks_set.insert(val);
        for (int val : var_stage) blocks_set.insert(val);
        std::vector<int> blocks(blocks_set.begin(), blocks_set.end());

        int max_block = *std::max_element(blocks.begin(), blocks.end());

        size_t block_area = 0;

        for (int block : blocks) {
            if (block == 1) {
                // Linking variables
                size_t var_count = std::count(var_stage.begin(), var_stage.end(), 1);
                size_t equ_count = std::count_if(equ_stage.begin(), equ_stage.end(),
                    [max_block](int val) { return val < max_block; });
                block_area += var_count * equ_count;
            } else if (block == max_block) {
                // Linking constraints
                size_t var_count = var_stage.size();
                size_t equ_count = std::count(equ_stage.begin(), equ_stage.end(), max_block);
                block_area += var_count * equ_count;
            } else {
                // Regular blocks
                size_t var_count = std::count(var_stage.begin(), var_stage.end(), block);
                size_t equ_count = std::count(equ_stage.begin(), equ_stage.end(), block);
                block_area += var_count * equ_count;
            }
        }

        return 1.0 - static_cast<double>(block_area) / static_cast<double>(matrix_area);
    }
};

// Forward declarations for model structures
struct ModelDump {
    std::map<std::string, std::vector<int>> records;
    std::map<std::string, std::vector<std::vector<std::string>>> A_records;
};

class PipsScore : public StructureScore {
private:
    ModelDump model_dump;
    WhiteScore whitescorer;
    int num_blocks;

public:
    PipsScore(const ModelDump& model_dump, const std::vector<int>& equ_blocks, 
              const std::vector<int>& var_blocks)
        : StructureScore(equ_blocks, var_blocks),
          model_dump(model_dump),
          whitescorer(equ_blocks, var_blocks),
          num_blocks(std::max(*std::max_element(equ_blocks.begin(), equ_blocks.end()),
                              *std::max_element(var_blocks.begin(), var_blocks.end()))) {}

protected:
    double _compute_score() override {
        double whitescore = whitescorer.get_score();

        auto [linking_equ_score, linking_equ_frac] = _linking_score(INPUTBLOCKS::EQUATIONS);
        auto [linking_var_score, linking_var_frac] = _linking_score(INPUTBLOCKS::VARIABLES);

        std::cout << "PIPS Score: " << whitescore << " " << linking_equ_score << " "
                  << linking_equ_frac << " " << linking_var_score << " "
                  << linking_var_frac << std::endl;

        return 0.85 * whitescore + 2 * linking_equ_score - 90 * linking_equ_frac
               - 3 * linking_var_score - 90 * linking_var_frac;
    }

    std::pair<double, double> _linking_score(INPUTBLOCKS linking_block_type) {
        const std::vector<int>& linking_labels = 
            (linking_block_type == INPUTBLOCKS::EQUATIONS) ? equ_blocks : var_blocks;
        const std::vector<int>& target_labels = 
            (linking_block_type == INPUTBLOCKS::EQUATIONS) ? var_blocks : equ_blocks;

        std::string linking_type = 
            (linking_block_type == INPUTBLOCKS::EQUATIONS) ? "i" : "j";
        std::string target_type = 
            (linking_block_type == INPUTBLOCKS::EQUATIONS) ? "j" : "i";

        int label = (linking_block_type == INPUTBLOCKS::EQUATIONS) ? num_blocks : 1;

        // Build elem_to_block mapping
        std::unordered_map<std::string, int> elem_to_block;
        const auto& target_records = model_dump.records.at(target_type);
        for (size_t i = 0; i < target_records.size(); ++i) {
            elem_to_block[std::to_string(target_records[i])] = target_labels[i];
        }

        // Process A matrix grouping (simplified - actual implementation depends on data structure)
        const auto& A_records = model_dump.A_records.at(linking_type);
        
        std::vector<std::vector<int>> linking_target_labels;
        for (size_t i = 0; i < linking_labels.size(); ++i) {
            if (linking_labels[i] == label && i < A_records.size()) {
                std::vector<int> target_blocks;
                for (const auto& elem : A_records[i]) {
                    if (elem_to_block.count(elem)) {
                        target_blocks.push_back(elem_to_block[elem]);
                    }
                }
                linking_target_labels.push_back(target_blocks);
            } else {
                linking_target_labels.push_back({});
            }
        }

        auto num_linking = [](const std::vector<int>& target_blocks) -> int {
            std::set<int> unique_blocks(target_blocks.begin(), target_blocks.end());
            if (unique_blocks.size() == 1 && *unique_blocks.begin() == 1) {
                return 10;
            }
            return unique_blocks.size();
        };

        std::vector<double> linking_count;
        for (const auto& target_blocks : linking_target_labels) {
            linking_count.push_back(target_blocks.empty() ? 0.0 : num_linking(target_blocks));
        }

        size_t total_linking = std::count_if(linking_count.begin(), linking_count.end(),
            [](double val) { return val != 0.0; });

        if (total_linking == 0) {
            return {1.0, 0.0};
        }

        size_t three_plus_linking = std::count_if(linking_count.begin(), linking_count.end(),
            [](double val) { return val >= 3.0; });

        return {
            1.0 - static_cast<double>(three_plus_linking) / static_cast<double>(total_linking),
            static_cast<double>(total_linking) / static_cast<double>(linking_labels.size())
        };
    }
};