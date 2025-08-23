#include "../ISMSolver.h"
#include "../graph.h"
#include <onnxruntime_cxx_api.h>
#include <vector>
#include <memory>

class MLSolver : public ISMSolver {
private:
    std::unique_ptr<Ort::Env> env;
    std::unique_ptr<Ort::Session> session;
    Ort::MemoryInfo memory_info;
    
    std::vector<const char*> input_names;
    std::vector<const char*> output_names;
    
    static constexpr int MAX_NODES = 32;
    
public:
    MLSolver(const std::string& model_path);
    ~MLSolver() = default;
    
    bool solve(const Graph& subGraph, const Graph& targetGraph) override;
    
private:
    std::vector<int> predict_mapping(const Graph& subGraph, const Graph& targetGraph);
    bool verify_mapping_fast(const std::vector<int>& mapping, 
                            const Graph& subGraph, 
                            const Graph& targetGraph);
    
    std::vector<Ort::Value> create_input_tensors(const Graph& subGraph, const Graph& targetGraph);
    std::vector<int> extract_mapping(const Ort::Value& output_tensor, int sub_size, int target_size);
};