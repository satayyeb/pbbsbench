// This code is part of the Problem Based Benchmark Suite (PBBS)
// Copyright (c) 2011 Guy Blelloch and the PBBS team
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the
// "Software"), to deal in the Software without restriction, including
// without limitation the rights (to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Software, and to
// permit persons to whom the Software is furnished to do so, subject to
// the following conditions:
//
// The above copyright notice and this permission notice shall be included
// in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
// OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
// LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
// OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
// WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#define NOTMAIN 1

#include <iostream>
#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "common/graph.h"
#include "common/speculative_for.h"
#include "common/get_time.h"
#include "matching.h"

#include <iostream>
#include <math.h>

// Kernel function to add the elements of two arrays
__global__
void add(int n, float *x, float *y) {
    int index = threadIdx.x;
    int stride = blockDim.x;
    for (int i = index; i < n; i += stride)
        y[i] = x[i] + y[i];
}

int main2(void) {
    int N = 1 << 20;
    float *x, *y;

    // Allocate Unified Memory – accessible from CPU or GPU
    cudaMallocManaged(&x, N * sizeof(float));
    cudaMallocManaged(&y, N * sizeof(float));

    // initialize x and y arrays on the host
    for (int i = 0; i < N; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    // Run kernel on 1M elements on the GPU
    add<<<1, 1>>>(N, x, y);

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    // Check for errors (all values should be 3.0f)
    float maxError = 0.0f;
    for (int i = 0; i < N; i++)
        maxError = fmax(maxError, fabs(y[i] - 3.0f));
    std::cout << "Max error: " << maxError << std::endl;

    // Free memory
    cudaFree(x);
    cudaFree(y);

    return 0;
}


using namespace std;

using reservation = pbbs::reservation<edgeId>;

struct matchStep {
    edges const &E;
    parlay::sequence<reservation> &R;
    parlay::sequence<bool> &matched;

    matchStep(edges const &E,
              parlay::sequence<reservation> &R,
              parlay::sequence<bool> &matched)
            : E(E), R(R), matched(matched) {}

    bool reserve(edgeId i) {
        size_t u = E[i].u;
        size_t v = E[i].v;
        if (matched[u] || matched[v] || (u == v)) return 0;
        R[u].reserve(i);
        R[v].reserve(i);
        return 1;
    }

    bool commit(edgeId i) {
        size_t u = E[i].u;
        size_t v = E[i].v;
        if (R[v].check(i)) {
            R[v].reset();
            if (R[u].check(i)) {
                matched[u] = matched[v] = 1;
                return 1;
            }
        } else if (R[u].check(i)) R[u].reset();
        return 0;
    }
};

parlay::sequence<edgeId> maximalMatching(edges const &E) {
//    printf("CUDA starting...");
    main2();
//    printf("CUDA end.");
    size_t n = max(E.numCols, E.numRows);
    size_t m = E.nonZeros;
    timer t("max matching", true);

    parlay::sequence<reservation> R(n);
    parlay::sequence<bool> matched(n, false);
    matchStep mStep(E, R, matched);
    t.next("init");
    pbbs::speculative_for<edgeId>(mStep, 0, m, 10, 0);
    t.next("speculative for");
    parlay::sequence<edgeId> matchingIdx =
            parlay::pack(parlay::delayed_seq<edgeId>(n, [&](size_t i) { return R[i].get(); }),
                         parlay::tabulate(n, [&](size_t i) -> bool { return R[i].reserved(); }));
    t.next("speculative for");
    cout << "number of matches = " << matchingIdx.size() << endl;
    return matchingIdx;
}  
