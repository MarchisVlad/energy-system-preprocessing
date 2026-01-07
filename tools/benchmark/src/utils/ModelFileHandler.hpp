#ifndef MODEL_HPP
#define MODEL_HPP

enum class Format {
    MPS, 
    GDX,
    GMS,
    LP
};

struct ModelMetadata {
    size_t num_lines;
    size_t num_variables;
    size_t num_constraints;
    size_t num_nonzero_elements;
    size_t num_integer_vars;
    size_t num_binary_vars;
    std::string problem_name;
    std::string objective_sense; // "MIN" or "MAX"
    
    ModelMetadata() : num_lines(0), num_variables(0), num_constraints(0),
                      num_nonzero_elements(0), num_integer_vars(0), 
                      num_binary_vars(0), objective_sense("MIN") {}
};

class Model 
#endif /* MODELS_HPP */ 