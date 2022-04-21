/*
 * Copyright 2011 ANAS LOUTER 
 */

#include <emscripten/bind.h>
#include <stdio.h>
#include <vector>
#include "computeff.h"
#include "compute_param.h"

//std::vector<double> computeFF(std::vector<int> const& ph2v, std::vector<double> const& ppoints);
//std::vector<double> compute_param(const std::vector<int>& ph2v, const std::vector<double>& ppoints, const std::vector<double>& ff_angles);

EMSCRIPTEN_BINDINGS(types) {
    emscripten::register_vector<double>("VectorDouble");
    emscripten::register_vector<int>("VectorInt");
    emscripten::function("computeFF", &computeFF);
    emscripten::function("compute_param", &compute_param);
}

