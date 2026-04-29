// Convergence tests for the three CFR impls. Asserts the average strategy
// approaches the (1/3, 1/3, 1/3, 0, 0, 0, 0, 0) RPS equilibrium and that the
// three impls agree with each other.
//
// Exits 0 on success, 1 on any failure — suitable as a CI check.

#include "cfr.h"

#include <cstdio>
#include <cmath>

static int failures = 0;

static void check_near(float actual, float expected, float tol, const char* msg) {
    if (std::fabs(actual - expected) > tol) {
        std::fprintf(stderr, "FAIL: %s — got %.6f, expected %.6f (tol %.6f)\n",
                     msg, actual, expected, tol);
        ++failures;
    }
}

template <typename Node, typename IterFn>
static void run_iters(IterFn iter, Node& p1, Node& p2, int iters) {
    for (int t = 0; t < iters; ++t) iter(p1, p2);
}

template <typename Node>
static void check_equilibrium(const Node& p, const char* label, float tol) {
    float total = 0.0f;
    for (int a = 0; a < NUM_ACTIONS; ++a) total += p.strategy_sum[a];
    if (total <= 0.0f) {
        std::fprintf(stderr, "FAIL: %s — strategy_sum total is zero\n", label);
        ++failures;
        return;
    }

    char buf[64];
    for (int a = 0; a < 3; ++a) {
        std::snprintf(buf, sizeof(buf), "%s avg[%d] (RPS)", label, a);
        check_near(p.strategy_sum[a] / total, 1.0f / 3.0f, tol, buf);
    }
    for (int a = 3; a < NUM_ACTIONS; ++a) {
        std::snprintf(buf, sizeof(buf), "%s avg[%d] (dominated)", label, a);
        check_near(p.strategy_sum[a] / total, 0.0f, tol, buf);
    }
}

template <typename NodeA, typename NodeB>
static void check_agree(const NodeA& a, const NodeB& b,
                        const char* label_a, const char* label_b, float tol) {
    float ta = 0.0f, tb = 0.0f;
    for (int i = 0; i < NUM_ACTIONS; ++i) { ta += a.strategy_sum[i]; tb += b.strategy_sum[i]; }
    char buf[96];
    for (int i = 0; i < NUM_ACTIONS; ++i) {
        std::snprintf(buf, sizeof(buf), "%s vs %s avg[%d]", label_a, label_b, i);
        check_near(a.strategy_sum[i] / ta, b.strategy_sum[i] / tb, tol, buf);
    }
}

int main() {
    init_payoff();

    // 200k iterations is enough to get within ~1% of the (1/3, 1/3, 1/3, 0...)
    // equilibrium on 8-action RPS. Cheap enough to run in CI in well under a second.
    constexpr int iters = 200'000;
    constexpr float eq_tol  = 0.01f;   // distance from equilibrium
    constexpr float agree_tol = 1e-4f; // cross-impl agreement

    NaiveNode n1{}, n2{};
    run_iters(naive_iter, n1, n2, iters);
    check_equilibrium(n1, "naive P1", eq_tol);
    check_equilibrium(n2, "naive P2", eq_tol);

    AlignedNode a1{}, a2{};
    run_iters(aligned_iter, a1, a2, iters);
    check_equilibrium(a1, "aligned P1", eq_tol);
    check_equilibrium(a2, "aligned P2", eq_tol);

    AlignedNode v1{}, v2{};
    run_iters(avx2_iter, v1, v2, iters);
    check_equilibrium(v1, "avx2 P1", eq_tol);
    check_equilibrium(v2, "avx2 P2", eq_tol);

    // Cross-impl agreement. Naive and aligned do the math identically in scalar,
    // so they should match very tightly. AVX2 reorders FP ops so it drifts more —
    // we use the looser equilibrium tolerance there.
    check_agree(n1, a1, "naive", "aligned", agree_tol);
    check_agree(n1, v1, "naive", "avx2",    eq_tol);

    if (failures) {
        std::fprintf(stderr, "\n%d test failure(s)\n", failures);
        return 1;
    }
    std::printf("All CFR tests passed (%d iterations per impl)\n", iters);
    return 0;
}
