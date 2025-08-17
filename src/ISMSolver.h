#pragma once
#include "graph.h"

class ISMSolver {
    public:
        virtual ~ISMSolver() = default;

        virtual bool solve(const Graph& subGraph, const Graph& targetGraph) = 0;
};