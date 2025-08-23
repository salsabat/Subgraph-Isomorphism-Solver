#include "graph.h"
#include "ISMSolver.h"
#include "solvers/ml_solver.h"
#include <iostream>
#include <memory>
#include <chrono>
#include <vector>

void benchmark_solver(std::unique_ptr<ISMSolver> solver, const std::string& name,
                      const Graph& subgraph, const Graph& target) {
    const int num_runs = 100;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    bool result = false;
    for (int i = 0; i < num_runs; i++) {
        result = solver->solve(subgraph, target);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    
    std::cout << name << ":" << std::endl;
    std::cout << "  Result: " << (result ? "Found" : "Not found") << std::endl;
    std::cout << "  Average time: " << (duration / num_runs) << " μs" << std::endl;
    std::cout << "  Total time: " << duration << " μs" << std::endl;
    std::cout << std::endl;
}

int main(int argc, char* argv[]) {
    Graph subgraph(3);
    subgraph.add_edge(0, 1);
    subgraph.add_edge(1, 2);
    
    Graph target(5);
    target.add_edge(0, 1);
    target.add_edge(1, 2);
    target.add_edge(2, 3);
    target.add_edge(3, 4);
    target.add_edge(4, 0);
    target.add_edge(1, 4);
    
    std::cout << "=== Graph Isomorphism Solver Benchmark ===" << std::endl;
    std::cout << std::endl;
    
    std::cout << "Subgraph (" << subgraph.n << " nodes):" << std::endl;
    subgraph.print();
    std::cout << std::endl;
    
    std::cout << "Target graph (" << target.n << " nodes):" << std::endl;
    target.print();
    std::cout << std::endl;
    
    std::cout << "=== Benchmarking Solvers ===" << std::endl;
    std::cout << std::endl;

    if (argc > 1) {
        std::string model_path = argv[1];
        std::cout << "Using ML model: " << model_path << std::endl;
        
        try {
            auto ml_solver = std::make_unique<MLSolver>(model_path);
            benchmark_solver(std::move(ml_solver), "ML Solver", subgraph, target);
        } catch (const std::exception& e) {
            std::cout << "Failed to load ML solver: " << e.what() << std::endl;
        }
    } else {
        std::cout << "ML Solver: No model path provided (usage: ./solver model.onnx)" << std::endl;
        std::cout << std::endl;
    }

    return 0;
}