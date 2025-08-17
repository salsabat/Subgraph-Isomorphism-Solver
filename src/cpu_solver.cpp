#include "graph.h"
#include <iostream>
#include <vector>
#include <algorithm>
#include <bitset>

// DFS + bitset CPU Solver

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