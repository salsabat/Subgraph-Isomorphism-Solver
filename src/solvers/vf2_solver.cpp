#include "../graph.h"
#include "../ISMSolver.h"
#include <iostream>
#include <vector>
#include <algorithm>
#include <bitset>
#include <set>

class VF2State {
public:
    std::vector<int> core_1;
    std::vector<int> core_2;
    std::vector<int> term_1;
    std::vector<int> term_2;
    int depth;
    
    VF2State(int n1, int n2) : core_1(n1, -1), core_2(n2, -1), term_1(n1, 0), term_2(n2, 0), depth(0) {}
    
    void add_pair(int n, int m, const Graph& g1, const Graph& g2) {
        core_1[n] = m;
        core_2[m] = n;
        depth++;
        if (term_1[n] == 0) {
            term_1[n] = depth;
        }
        if (term_2[m] == 0) {
            term_2[m] = depth;
        }
        for (int i = 0; i < g1.n; i++) {
            if (g1.adj[n][i] && core_1[i] == -1 && term_1[i] == 0) {
                term_1[i] = depth;
            }
        }
        for (int i = 0; i < g2.n; i++) {
            if (g2.adj[m][i] && core_2[i] == -1 && term_2[i] == 0) {
                term_2[i] = depth;
            }
        }
    }
};

class VF2Solver : public ISMSolver {
public:
    bool solve(const Graph& subGraph, const Graph& targetGraph) override {
        if (subGraph.n > targetGraph.n) {
            return false;
        }
        if (subGraph.n == 0) {
            return true;
        }
        
        VF2State state(subGraph.n, targetGraph.n);
        return match(state, subGraph, targetGraph);
    }

private:
    bool match(VF2State& state, const Graph& g1, const Graph& g2) {
        if (state.depth == g1.n) {
            return true;
        }
        
        std::pair<int, int> candidate = choose_pair(state, g1, g2);
        if (candidate.first == -1) {
            return false;
        }
        
        int n = candidate.first;
        for (int m = 0; m < g2.n; m++) {
            if (state.core_2[m] != -1) {
                continue;
            }
            
            if (is_feasible(state, n, m, g1, g2)) {
                VF2State new_state = state;
                new_state.add_pair(n, m, g1, g2);
                
                if (match(new_state, g1, g2)) {
                    return true;
                }
            }
        }
        
        return false;
    }
    
    std::pair<int, int> choose_pair(const VF2State& state, const Graph& g1, const Graph& g2) {
        for (int i = 0; i < g1.n; i++) {
            if (state.core_1[i] == -1 && state.term_1[i] > 0) {
                return {i, -1};
            }
        }
        for (int i = 0; i < g1.n; i++) {
            if (state.core_1[i] == -1) {
                return {i, -1};
            }
        }
        
        return {-1, -1};
    }
    
    bool is_feasible(const VF2State& state, int n, int m, const Graph& g1, const Graph& g2) {
        if (!semantic_feasible(n, m, g1, g2)) {
            return false;
        }
        return syntactic_feasible(state, n, m, g1, g2);
    }
    
    bool semantic_feasible(int n, int m, const Graph& g1, const Graph& g2) {
        return g1.adj[n].count() <= g2.adj[m].count();
    }
    
    bool syntactic_feasible(const VF2State& state, int n, int m, const Graph& g1, const Graph& g2) {
        int mapped_neighbors_1 = 0, mapped_neighbors_2 = 0;
        for (int i = 0; i < g1.n; i++) {
            if (state.core_1[i] != -1 && g1.adj[i][n]) {
                if (!g2.adj[state.core_1[i]][m]) {
                    return false;
                }
                mapped_neighbors_1++;
            }
        }

        for (int i = 0; i < g2.n; i++) {
            if (state.core_2[i] != -1 && g2.adj[i][m]) {
                mapped_neighbors_2++;
            }
        }

        if (mapped_neighbors_1 != mapped_neighbors_2) {
            return false;
        }

        int term_1_count = 0, term_2_count = 0;
        for (int i = 0; i < g1.n; i++) {
            if (state.core_1[i] == -1 && state.term_1[i] > 0 && g1.adj[n][i]) {
                term_1_count++;
            }
        }

        for (int i = 0; i < g2.n; i++) {
            if (state.core_2[i] == -1 && state.term_2[i] > 0 && g2.adj[m][i]) {
                term_2_count++;
            }
        }

        if (term_1_count > term_2_count) {
            return false;
        }

        int new_1_count = 0, new_2_count = 0;
        for (int i = 0; i < g1.n; i++) {
            if (state.core_1[i] == -1 && state.term_1[i] == 0 && g1.adj[n][i]) {
                new_1_count++;
            }
        }

        for (int i = 0; i < g2.n; i++) {
            if (state.core_2[i] == -1 && state.term_2[i] == 0 && g2.adj[m][i]) {
                new_2_count++;
            }
        }

        if (new_1_count > new_2_count) {
            return false;
        }
        
        return true;
    }
};
