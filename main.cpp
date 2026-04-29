// CFR demo: time and print the equilibrium strategies for each implementation.
// Algorithm lives in cfr.h.

#include "cfr.h"

#include <cstdio>
#include <chrono>

constexpr int NUM_ITERS = 100'000'000;

static void print_avg_strategy(const float* sum, const char* label) {
    float total = 0.0f;
    for (int a = 0; a < NUM_ACTIONS; ++a) total += sum[a];
    printf("  %s: ", label);
    for (int a = 0; a < NUM_ACTIONS; ++a) {
        printf("%.4f ", total > 0 ? sum[a] / total : 0.0f);
    }
    printf("\n");
}

template <typename Node, typename IterFn>
static double run_and_time(IterFn iter, Node& p1, Node& p2, int iters) {
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int t = 0; t < iters; ++t) iter(p1, p2);
    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

int main() {
    init_payoff();
    printf("CFR on 8-action extended RPS, %d iterations\n", NUM_ITERS);
    printf("Equilibrium should be (0.333, 0.333, 0.333, 0, 0, 0, 0, 0)\n\n");

    {
        NaiveNode p1{}, p2{};
        double ms = run_and_time(naive_iter, p1, p2, NUM_ITERS);
        printf("[naive]       %.3f ms\n", ms);
        print_avg_strategy(p1.strategy_sum, "P1 avg");
    }
    {
        AlignedNode p1{}, p2{};
        double ms = run_and_time(aligned_iter, p1, p2, NUM_ITERS);
        printf("[aligned]     %.3f ms\n", ms);
        print_avg_strategy(p1.strategy_sum, "P1 avg");
    }
    {
        AlignedNode p1{}, p2{};
        double ms = run_and_time(avx2_iter, p1, p2, NUM_ITERS);
        printf("[avx2]        %.3f ms\n", ms);
        print_avg_strategy(p1.strategy_sum, "P1 avg");
    }
    return 0;
}
