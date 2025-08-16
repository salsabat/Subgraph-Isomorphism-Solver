#include <iostream>
#include <vector>
#include <algorithm>
#include <bitset>

const int MAX_NODES = 32;

struct Graph {
    int n;
    std::vector<std::bitset<MAX_NODES>> adj;
    Graph(int nodes) : n(nodes), adj(nodes) {}
    void add_edge(int u, int v) {
        adj[u].set(v);
        adj[v].set(u);
    }
};

// Solve by backtracking
bool match(int depth, const Graph& subGraph, const Graph& targetGraph, std::vector<int> mapper, std::vector<bool> used) {
    if (depth == subGraph.n) {
        return true;
    }

    int u = depth;
    for (int v = 0; v < targetGraph.n; v++) {
        if (used[v]) {
            continue;
        }
        if (subGraph.adj[u].count() > targetGraph.adj[v].count()) {
            continue;
        }

        bool validNei = true;
        for (int i = 0; i < u; i++) {
            if (mapper[i] == -1) {
                continue;
            }

            std::bitset<MAX_NODES> subGraph_edges = subGraph.adj[u] & std::bitset<MAX_NODES>().set(i);
            std::bitset<MAX_NODES> targetGraph_edges = targetGraph.adj[v] & std::bitset<MAX_NODES>().set(mapper[i]);
            if (subGraph_edges.any() && (!targetGraph_edges.any())) {
                validNei = false;
                break;
            }
        }

        if (!validNei) {
            continue;
        }

        mapper[u] = v;
        used[v] = true;
        if (match(depth + 1, subGraph, targetGraph, mapper, used)) {
            return true;
        }
        mapper[u] = -1;
        used[v] = false;
    }

    return false;
}

bool subgraph_isomorphism(const Graph& subGraph, const Graph& targetGraph) {
    if (subGraph.n > targetGraph.n) return false;
    std::vector<int> mapper(subGraph.n, -1);
    std::vector<bool> used(targetGraph.n, false);
    return match(0, subGraph, targetGraph, mapper, used);
}

void test_case(int case_num, const Graph& sub, const Graph& target, bool expected) {
    bool result = subgraph_isomorphism(sub, target);
    std::cout << "Test " << case_num << ": " << (result == expected ? "PASS" : "FAIL") 
              << " (Expected: " << expected << ", Got: " << result << ")" << std::endl;
}

int main() {
    std::cout << "=== Subgraph Isomorphism Test Cases ===" << std::endl;
    
    // Test 1: Single node subgraph in single node target - TRUE
    Graph sub1(1), target1(1);
    test_case(1, sub1, target1, true);
    
    // Test 2: Single node subgraph in larger target - TRUE  
    Graph sub2(1), target2(3);
    target2.add_edge(0, 1);
    target2.add_edge(1, 2);
    test_case(2, sub2, target2, true);
    
    // Test 3: Edge subgraph in triangle - TRUE
    Graph sub3(2), target3(3);
    sub3.add_edge(0, 1);
    target3.add_edge(0, 1);
    target3.add_edge(1, 2);
    target3.add_edge(0, 2);
    test_case(3, sub3, target3, true);
    
    // Test 4: Triangle subgraph in triangle - TRUE
    Graph sub4(3), target4(3);
    sub4.add_edge(0, 1);
    sub4.add_edge(1, 2);
    sub4.add_edge(0, 2);
    target4.add_edge(0, 1);
    target4.add_edge(1, 2);
    target4.add_edge(0, 2);
    test_case(4, sub4, target4, true);
    
    // Test 5: Path subgraph in cycle - TRUE
    Graph sub5(3), target5(4);
    sub5.add_edge(0, 1);
    sub5.add_edge(1, 2);
    target5.add_edge(0, 1);
    target5.add_edge(1, 2);
    target5.add_edge(2, 3);
    target5.add_edge(3, 0);
    test_case(5, sub5, target5, true);
    
    // Test 6: Star subgraph in larger star - TRUE
    Graph sub6(4), target6(5);
    sub6.add_edge(0, 1);
    sub6.add_edge(0, 2);
    sub6.add_edge(0, 3);
    target6.add_edge(0, 1);
    target6.add_edge(0, 2);
    target6.add_edge(0, 3);
    target6.add_edge(0, 4);
    test_case(6, sub6, target6, true);
    
    // Test 7: Triangle in disconnected components - FALSE
    Graph sub7(3), target7(6);
    sub7.add_edge(0, 1);
    sub7.add_edge(1, 2);
    sub7.add_edge(0, 2);
    target7.add_edge(0, 1);
    target7.add_edge(3, 4);
    target7.add_edge(4, 5);
    test_case(7, sub7, target7, false);
    
    // Test 8: 4-clique in 3-clique - FALSE
    Graph sub8(4), target8(3);
    sub8.add_edge(0, 1);
    sub8.add_edge(0, 2);
    sub8.add_edge(0, 3);
    sub8.add_edge(1, 2);
    sub8.add_edge(1, 3);
    sub8.add_edge(2, 3);
    target8.add_edge(0, 1);
    target8.add_edge(1, 2);
    target8.add_edge(0, 2);
    test_case(8, sub8, target8, false);
    
    // Test 9: Cycle in path - FALSE
    Graph sub9(4), target9(4);
    sub9.add_edge(0, 1);
    sub9.add_edge(1, 2);
    sub9.add_edge(2, 3);
    sub9.add_edge(3, 0);
    target9.add_edge(0, 1);
    target9.add_edge(1, 2);
    target9.add_edge(2, 3);
    test_case(9, sub9, target9, false);
    
    // Test 10: Complex subgraph - TRUE
    Graph sub10(4), target10(6);
    sub10.add_edge(0, 1);
    sub10.add_edge(1, 2);
    sub10.add_edge(2, 3);
    target10.add_edge(0, 1);
    target10.add_edge(1, 2);
    target10.add_edge(2, 3);
    target10.add_edge(3, 4);
    target10.add_edge(4, 5);
    target10.add_edge(0, 5);
    test_case(10, sub10, target10, true);
    
    std::cout << "\n=== Expected Results ===" << std::endl;
    std::cout << "Test 1: TRUE  (single node always matches)" << std::endl;
    std::cout << "Test 2: TRUE  (single node matches any graph)" << std::endl;
    std::cout << "Test 3: TRUE  (edge exists in triangle)" << std::endl;
    std::cout << "Test 4: TRUE  (identical triangles)" << std::endl;
    std::cout << "Test 5: TRUE  (path exists in cycle)" << std::endl;
    std::cout << "Test 6: TRUE  (3-star in 4-star)" << std::endl;
    std::cout << "Test 7: FALSE (no triangle in disconnected)" << std::endl;
    std::cout << "Test 8: FALSE (4-clique > 3-clique)" << std::endl;
    std::cout << "Test 9: FALSE (cycle not in path)" << std::endl;
    std::cout << "Test 10: TRUE (path in larger connected graph)" << std::endl;
    
    return 0;
}