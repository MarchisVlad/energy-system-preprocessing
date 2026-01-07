#ifndef MODEL_FILE_HANDLER_HPP
#define MODEL_FILE_HANDLER_HPP

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

class FileHandlerException : public std::runtime_error {
public:
    explicit FileHandlerException(const std::string& msg) 
        : std::runtime_error(msg) {}
};

class ModelFileHandler {
public:
    // Constructor
    explicit ModelFileHandler(const std::string& filepath);
    
    // Destructor
    ~ModelFileHandler();
    
    // Delete copy constructor and assignment (use move semantics)
    ModelFileHandler(const ModelFileHandler&) = delete;
    ModelFileHandler& operator=(const ModelFileHandler&) = delete;
    
    // Move constructor and assignment
    ModelFileHandler(ModelFileHandler&&) noexcept;
    ModelFileHandler& operator=(ModelFileHandler&&) noexcept;
    
    // File reading operations
    bool read();
    bool readMetadataOnly();
    
    // Metadata access
    const ModelMetadata& getMetadata() const;
    FileFormat getFormat() const;
    
    // Format conversion
    bool convertTo(const std::string& output_path, FileFormat target_format);
    
    // Query methods
    size_t getNumLines() const { return metadata_.num_lines; }
    size_t getNumConstraints() const { return metadata_.num_constraints; }
    size_t getNumVariables() const { return metadata_.num_variables; }
    size_t getNumNonZeros() const { return metadata_.num_nonzero_elements; }
    
    // Raw data access
    std::optional<std::string> getRawContent() const;
    const std::vector<std::string>& getLines() const { return lines_; }
    
    // Validation
    bool validate() const;
    std::vector<std::string> getValidationErrors() const;
    
    // Static utility methods
    static FileFormat detectFormat(const std::string& filepath);
    static bool isFormatSupported(FileFormat format);
    static std::string formatToString(FileFormat format);
    
private:
    // File information
    std::string filepath_;
    FileFormat format_;
    ModelMetadata metadata_;
    
    // Cached file content
    std::vector<std::string> lines_;
    bool is_loaded_;
    
    // Format-specific readers
    bool readMPS(std::ifstream& file);
    bool readGDX(std::ifstream& file);
    bool readLP(std::ifstream& file);
    
    // Format-specific writers
    bool writeMPS(std::ofstream& file) const;
    bool writeGDX(std::ofstream& file) const;
    bool writeLP(std::ofstream& file) const;
    
    // Metadata extraction helpers
    void extractMPSMetadata();
    void extractGDXMetadata();
    void extractLPMetadata();
    
    // Validation helpers
    bool validateMPS() const;
    bool validateGDX() const;
    bool validateLP() const;
    
    // Utility methods
    void reset();
    std::string getFileExtension(const std::string& path) const;
    bool fileExists(const std::string& path) const;
};

std::unique_ptr<ModelFileHandler> createHandler(const std::string& filepath);

#endif /* MODEL_FILE_HANDLER_HPP */ 