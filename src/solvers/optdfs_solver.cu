#include "ISMSolver.h"
#include "../graph.h"
#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <cstdint>

__constant__ GraphGPU const_subgraph;
__constant__ int const_k_levels;
__constant__ int const_target_n;

struct GraphGPU {
    uint64_t* adj;
    int n;    
    int words_per_row;

    __host__ __device__ GraphGPU() : adj(nullptr), n(0), words_per_row(0) {}

    __host__ __device__ void add_edge(int u, int v) {
        int word_u = v / 64;
        int bit_u = v % 64;
        adj[u * words_per_row + word_u] |= (1ULL << bit_u);
        
        int word_v = u / 64;
        int bit_v = u % 64;
        adj[v * words_per_row + word_v] |= (1ULL << bit_v);
    }
};

__device__ __host__ inline bool has_edge_gpu_inline(const GraphGPU* graph, int u, int v) {
    int word_idx = v / 64;
    int bit_pos = v % 64;
    uint64_t word = graph->adj[u * graph->words_per_row + word_idx];
    
    return (word & (1ULL << bit_pos)) != 0;
}

__device__ bool match_device(int depth, const GraphGPU* targetGraph, 
                             int* mapper, bool* used) {
    if (depth == const_subgraph.n) {
        return true;
    }
    
    if (depth >= const_k_levels) {
        return true; 
    }
    
    int u = depth;
    
    for (int v = 0; v < const_target_n; v++) {
        if (used[v]) {
            continue;
        }
        
        bool validNei = true;
        for (int i = 0; i < depth; i++) {
            int mapped_v = mapper[i];
            
            bool sub_has_edge = has_edge_gpu_inline(&const_subgraph, u, i);
            bool target_has_edge = has_edge_gpu_inline(targetGraph, v, mapped_v);
            
            if (sub_has_edge && !target_has_edge) {
                validNei = false;
                break;
            }
        }
        
        if (!validNei) {
            continue;
        }
        
        mapper[u] = v;
        used[v] = true;
        
        if (match_device(depth + 1, targetGraph, mapper, used)) {
            return true;
        }
        
        mapper[u] = -1;
        used[v] = false;
    }
    
    return false;
}

__global__ void parallel_match_kernel(const GraphGPU* target_graph, bool* results) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int local_tid = threadIdx.x;
    
    if (tid >= const_target_n) {
        return;
    }
    
    extern __shared__ char shared_mem[];
    
    int threads_per_block = blockDim.x;
    size_t mapper_offset = 0;
    size_t used_offset = threads_per_block * MAX_NODES * sizeof(int);
    
    int* mapper = (int*)(shared_mem + mapper_offset) + local_tid * MAX_NODES;
    bool* used = (bool*)(shared_mem + used_offset) + local_tid * MAX_NODES;
    
    for (int i = 0; i < const_target_n; i++) {
        used[i] = false;
    }
    
    __syncthreads();
    
    mapper[0] = tid;
    used[tid] = true;
    
    bool found = match_device(1, target_graph, mapper, used);
    
    if (found) {
        atomicOr((int*)results, 1);
    }
}

class OptDFSSolver : public ISMSolver {
private:
    struct MemoryPool {
        GraphGPU* d_subgraph = nullptr;
        GraphGPU* d_targetgraph = nullptr;
        bool* d_result = nullptr;
        size_t subgraph_capacity = 0;
        size_t targetgraph_capacity = 0;
    } memory_pool;

public:
    OptDFSSolver() : memory_pool{} {}
    
    ~OptDFSSolver() {
        cleanup_memory_pool();
    }
    
    bool solve(const Graph& subGraph, const Graph& targetGraph) override {
        return launch_parallel_search(subGraph, targetGraph);
    }

private:
    __host__ GraphGPU* convert_to_gpu_graph(const Graph& cpu_graph) {
        int words_per_row = (cpu_graph.n + 63) / 64;
        size_t adj_size = cpu_graph.n * words_per_row * sizeof(uint64_t);
        
        size_t total_size = sizeof(GraphGPU) + adj_size;
        char* d_memory;
        cudaMalloc(&d_memory, total_size);
        
        GraphGPU* d_graph = (GraphGPU*)d_memory;
        uint64_t* d_adj = (uint64_t*)(d_memory + sizeof(GraphGPU));
        
        std::vector<uint64_t> host_adj(cpu_graph.n * words_per_row, 0);
        
        for (int i = 0; i < cpu_graph.n; i++) {
            for (int j = 0; j < cpu_graph.n; j++) {
                if (cpu_graph.adj[i][j]) {
                    int word_idx = j / 64;
                    int bit_pos = j % 64;
                    host_adj[i * words_per_row + word_idx] |= (1ULL << bit_pos);
                }
            }
        }
        
        GraphGPU host_graph;
        host_graph.adj = d_adj;
        host_graph.n = cpu_graph.n;
        host_graph.words_per_row = words_per_row;
        
        cudaMemcpy(d_graph, &host_graph, sizeof(GraphGPU), cudaMemcpyHostToDevice);
        cudaMemcpy(d_adj, host_adj.data(), adj_size, cudaMemcpyHostToDevice);
        
        return d_graph;
    }
    
    __host__ void free_gpu_graph(GraphGPU* gpu_graph) {
        if (gpu_graph == nullptr) {
            return;
        }
        cudaFree(gpu_graph);
    }
    
    __host__ void cleanup_memory_pool() {
        if (memory_pool.d_subgraph) {
            cudaFree(memory_pool.d_subgraph);
            memory_pool.d_subgraph = nullptr;
        }
        if (memory_pool.d_targetgraph) {
            cudaFree(memory_pool.d_targetgraph);
            memory_pool.d_targetgraph = nullptr;
        }
        if (memory_pool.d_result) {
            cudaFree(memory_pool.d_result);
            memory_pool.d_result = nullptr;
        }
        
        memory_pool.subgraph_capacity = 0;
        memory_pool.targetgraph_capacity = 0;
    }
    
    __host__ GraphGPU* get_pooled_graph_memory(const Graph& cpu_graph, bool is_subgraph) {
        int words_per_row = (cpu_graph.n + 63) / 64;
        size_t required_size = sizeof(GraphGPU) + cpu_graph.n * words_per_row * sizeof(uint64_t);
        
        GraphGPU** pool_ptr;
        size_t* capacity_ptr;
        
        if (is_subgraph) {
            pool_ptr = &memory_pool.d_subgraph;
            capacity_ptr = &memory_pool.subgraph_capacity;
        } else {
            pool_ptr = &memory_pool.d_targetgraph;
            capacity_ptr = &memory_pool.targetgraph_capacity;
        }
        
        if (*pool_ptr == nullptr || *capacity_ptr < required_size) {
            if (*pool_ptr != nullptr) {
                cudaFree(*pool_ptr);
            }
            cudaMalloc(pool_ptr, required_size);
            *capacity_ptr = required_size;
        }
        
        return convert_to_gpu_graph_pooled(cpu_graph, *pool_ptr);
    }
    
    __host__ GraphGPU* convert_to_gpu_graph_pooled(const Graph& cpu_graph, GraphGPU* d_memory) {
        int words_per_row = (cpu_graph.n + 63) / 64;
        size_t adj_size = cpu_graph.n * words_per_row * sizeof(uint64_t);
        
        GraphGPU* d_graph = d_memory;
        uint64_t* d_adj = (uint64_t*)((char*)d_memory + sizeof(GraphGPU));
        
        std::vector<uint64_t> host_adj(cpu_graph.n * words_per_row, 0);
        
        for (int i = 0; i < cpu_graph.n; i++) {
            for (int j = 0; j < cpu_graph.n; j++) {
                if (cpu_graph.adj[i][j]) {
                    int word_idx = j / 64;
                    int bit_pos = j % 64;
                    host_adj[i * words_per_row + word_idx] |= (1ULL << bit_pos);
                }
            }
        }
        
        GraphGPU host_graph;
        host_graph.adj = d_adj;
        host_graph.n = cpu_graph.n;
        host_graph.words_per_row = words_per_row;
        
        cudaMemcpy(d_graph, &host_graph, sizeof(GraphGPU), cudaMemcpyHostToDevice);
        cudaMemcpy(d_adj, host_adj.data(), adj_size, cudaMemcpyHostToDevice);
        
        return d_graph;
    }

    __host__ bool launch_parallel_search(const Graph& subGraph, const Graph& targetGraph) {
        if (subGraph.n > targetGraph.n) {
            return false;
        }
        
        GraphGPU* d_sub_graph = get_pooled_graph_memory(subGraph, true);
        GraphGPU* d_target_graph = get_pooled_graph_memory(targetGraph, false);
        
        if (memory_pool.d_result == nullptr) {
            cudaMalloc(&memory_pool.d_result, sizeof(bool));
        }
        cudaMemset(memory_pool.d_result, 0, sizeof(bool));
        
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        int max_threads_per_block = prop.maxThreadsPerBlock;
        int block_size = std::min(256, max_threads_per_block);
        int grid_size = (targetGraph.n + block_size - 1) / block_size;
        
        int k_levels = std::min(3, subGraph.n);
        
        GraphGPU host_subgraph;
        cudaMemcpy(&host_subgraph, d_sub_graph, sizeof(GraphGPU), cudaMemcpyDeviceToHost);
        cudaMemcpyToSymbol(const_subgraph, &host_subgraph, sizeof(GraphGPU));
        cudaMemcpyToSymbol(const_k_levels, &k_levels, sizeof(int));
        cudaMemcpyToSymbol(const_target_n, &targetGraph.n, sizeof(int));
        
        size_t shared_mem_size = block_size * MAX_NODES * (sizeof(int) + sizeof(bool));
        
        parallel_match_kernel<<<grid_size, block_size, shared_mem_size>>>(
            d_target_graph,
            memory_pool.d_result
        );
        
        cudaDeviceSynchronize();
        
        bool host_result;
        cudaMemcpy(&host_result, memory_pool.d_result, sizeof(bool), cudaMemcpyDeviceToHost);
        
        return host_result;
    }
};