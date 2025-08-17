#pragma once
#include <bitset>
#include <vector>
#include <iostream>

const int MAX_NODES = 32;

// CPU Graph representation
struct Graph {
    int n;
    std::vector<std::bitset<MAX_NODES>> adj;

    Graph(int nodes) : n(nodes), adj(nodes) {}

    void add_edge(int u, int v) {
        adj[u].set(v);
        adj[v].set(u);
    }

    void print() const {
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                std::cout << adj[i][j] << " ";
            }
            std::cout << std::endl;
        }
    }
};
