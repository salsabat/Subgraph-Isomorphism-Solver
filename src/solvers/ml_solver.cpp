#include "../ml_solver.h"
#include <iostream>
#include <algorithm>
#include <chrono>

MLSolver::MLSolver(const std::string& model_path) 
    : memory_info(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {
    
    try {
        env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "GraphMatcher");
        
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        
        session = std::make_unique<Ort::Session>(*env, model_path.c_str(), session_options);
        
        input_names = {"subgraph", "target", "sub_mask", "target_mask"};
        output_names = {"logits"};
        
        std::cout << "ML Solver initialized with model: " << model_path << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error initializing ML Solver: " << e.what() << std::endl;
        throw;
    }
}

bool MLSolver::solve(const Graph& subGraph, const Graph& targetGraph) {
    if (subGraph.n > targetGraph.n) {
        return false;
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    
    auto predicted_mapping = predict_mapping(subGraph, targetGraph);
    
    auto mid = std::chrono::high_resolution_clock::now();
    
    bool result = verify_mapping_fast(predicted_mapping, subGraph, targetGraph);
    
    auto end = std::chrono::high_resolution_clock::now();
    
    auto predict_time = std::chrono::duration_cast<std::chrono::microseconds>(mid - start).count();
    auto verify_time = std::chrono::duration_cast<std::chrono::microseconds>(end - mid).count();
    
    std::cout << "ML Solver - Predict: " << predict_time << "μs, Verify: " << verify_time << "μs" << std::endl;
    
    return result;
}

std::vector<int> MLSolver::predict_mapping(const Graph& subGraph, const Graph& targetGraph) {
    try {
        auto input_tensors = create_input_tensors(subGraph, targetGraph);
        
        auto output_tensors = session->Run(
            Ort::RunOptions{nullptr},
            input_names.data(),
            input_tensors.data(),
            input_names.size(),
            output_names.data(),
            output_names.size()
        );
        
        return extract_mapping(output_tensors[0], subGraph.n, targetGraph.n);
        
    } catch (const std::exception& e) {
        std::cerr << "Error in prediction: " << e.what() << std::endl;
        return std::vector<int>(subGraph.n, -1);
    }
}

std::vector<Ort::Value> MLSolver::create_input_tensors(const Graph& subGraph, const Graph& targetGraph) {
    std::vector<Ort::Value> input_tensors;
    
    std::vector<int64_t> tensor_shape = {1, MAX_NODES, MAX_NODES};
    std::vector<int64_t> mask_shape = {1, MAX_NODES};
    
    std::vector<float> subgraph_data(MAX_NODES * MAX_NODES, 0.0f);
    std::vector<float> target_data(MAX_NODES * MAX_NODES, 0.0f);
    std::vector<bool> sub_mask_data(MAX_NODES, false);
    std::vector<bool> target_mask_data(MAX_NODES, false);
    
    for (int i = 0; i < subGraph.n; i++) {
        for (int j = 0; j < subGraph.n; j++) {
            if (subGraph.adj[i][j]) {
                subgraph_data[i * MAX_NODES + j] = 1.0f;
            }
        }
        sub_mask_data[i] = true;
    }
    
    for (int i = 0; i < targetGraph.n; i++) {
        for (int j = 0; j < targetGraph.n; j++) {
            if (targetGraph.adj[i][j]) {
                target_data[i * MAX_NODES + j] = 1.0f;
            }
        }
        target_mask_data[i] = true;
    }
    
    input_tensors.emplace_back(Ort::Value::CreateTensor<float>(
        memory_info, subgraph_data.data(), subgraph_data.size(),
        tensor_shape.data(), tensor_shape.size()));
    
    input_tensors.emplace_back(Ort::Value::CreateTensor<float>(
        memory_info, target_data.data(), target_data.size(),
        tensor_shape.data(), tensor_shape.size()));
    
    input_tensors.emplace_back(Ort::Value::CreateTensor<bool>(
        memory_info, sub_mask_data.data(), sub_mask_data.size(),
        mask_shape.data(), mask_shape.size()));
    
    input_tensors.emplace_back(Ort::Value::CreateTensor<bool>(
        memory_info, target_mask_data.data(), target_mask_data.size(),
        mask_shape.data(), mask_shape.size()));
    
    return input_tensors;
}

std::vector<int> MLSolver::extract_mapping(const Ort::Value& output_tensor, int sub_size, int target_size) {
    const float* logits = output_tensor.GetTensorData<float>();
    
    std::vector<int> mapping(sub_size);
    
    for (int i = 0; i < sub_size; i++) {
        int best_target = -1;
        float best_score = -std::numeric_limits<float>::infinity();
        
        for (int j = 0; j < target_size; j++) {
            float score = logits[i * target_size + j];
            if (score > best_score) {
                best_score = score;
                best_target = j;
            }
        }
        mapping[i] = best_target;
    }
    
    return mapping;
}

bool MLSolver::verify_mapping_fast(const std::vector<int>& mapping, 
                                  const Graph& subGraph, 
                                  const Graph& targetGraph) {
    if (mapping.size() != static_cast<size_t>(subGraph.n)) {
        return false;
    }
    
    for (int u = 0; u < subGraph.n; u++) {
        int v = mapping[u];
        if (v < 0 || v >= targetGraph.n) {
            return false;
        }
        
        for (int u_neighbor = 0; u_neighbor < subGraph.n; u_neighbor++) {
            if (subGraph.adj[u][u_neighbor]) {
                int v_neighbor = mapping[u_neighbor];
                if (v_neighbor < 0 || v_neighbor >= targetGraph.n) {
                    return false;
                }
                if (!targetGraph.adj[v][v_neighbor]) {
                    return false;
                }
            }
        }
    }
    
    std::vector<bool> used(targetGraph.n, false);
    for (int target_node : mapping) {
        if (target_node >= 0) {
            if (used[target_node]) {
                return false;
            }
            used[target_node] = true;
        }
    }
    
    return true;
}