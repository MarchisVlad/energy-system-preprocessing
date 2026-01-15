#ifndef SCORE_HPP
#define SCORE_HPP

#include <vector>
#include <map>
#include <string>
#include <algorithm>


enum class INPUTBLOCKS {
    VARIABLES = 0,
    EQUATIONS = 1
};

class StructureScore {
protected:
    std::vector<int> equ_blocks;
    std::vector<int> var_blocks;
    double score;
    bool scorecomputed;

public:
    StructureScore(const std::vector<int>& equ_blocks, const std::vector<int>& var_blocks);
    virtual ~StructureScore() = default;
    double get_score();

protected:
    virtual double _compute_score();
};

class WhiteScore : public StructureScore {
public:
    WhiteScore(const std::vector<int>& equ_blocks, const std::vector<int>& var_blocks);

protected:
    double _compute_score() override;
};

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
    double _compute_score() override;
    std::pair<double, double> _linking_score(INPUTBLOCKS linking_block_type);
};

#endif /* SCORE_HPP */