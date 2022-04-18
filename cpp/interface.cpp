/*
 * Copyright 2011 ANAS LOUTER 
 */

#include <emscripten/bind.h>
#include <stdio.h>
#include "computeff.h"

std::string print()
{
    return "hello papy\n"; 
}

EMSCRIPTEN_BINDINGS(types) {
    emscripten::register_vector<double>("VectorDouble");
    emscripten::register_vector<int>("VectorInt");
    emscripten::function("meshMagic", &appel);
    emscripten::function("printTarace", &print);
}

