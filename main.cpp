// Counterfactual Regret Minimization Demo
// Three implementations of vanilla CFR on an 8-action zero-sum normal-form game:
//     1. naive:       AoS-ish, no alignment, scalar
//     2. soa_aligned: SoA, 64-byte aligned, scalar (which lets the compiler auto-vectorize)
//     3. avx2:        SoA, 64-byte aligned, using AVX2 intrinsics
//
// Game: extended Rock-Paper-Scissors. Actions 0,1,2 are Rock, Paper, Scissors with the
// usual payoff matrix. Actions 3..7 are strictly dominated (always lose to 0..2), and always
// tie among themselves).
//
// Nash Equilibrium: (1/3, 1/3, 1/3, 0, 0, 0, 0, 0)

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <chrono>
#include <immintrin.h>

constexpr int NUM_ACTIONS = 8; // exactly one __m256 of floats
constexpr int CACHE_LINE = 64; // in bytes
constexpr int NUM_ITERS = 10'000;

// aligned so we can SIMD-load each row.
alignas(CACHE_LINE) float payoff[NUM_ACTIONS][NUM_ACTIONS];

void init_payoff() {
    // top left 3x3
    int rps[3][3] = {
        { 0, -1, 1 }, // Rock vs R, P, S
        { 1, 0, -1 }, // Paper
        { -1, 1, 0 }, // Scissors
    };

    for (int i = 0; i < NUM_ACTIONS; ++i) {
        for (int j = 0; j < NUM_ACTIONS; ++j) {
            if (i < 3 && j < 3) {
                payoff[i][j] = (float)rps[i][j];
            } else if (i >= 3 && j < 3) {
                payoff[i][j] = -1.0f; // 3..7 are dominated; P2 wins
            } else if (i < 3 && j >= 3) {
                payoff[i][j] = 1.0f; // 3.7 are dominated; P1 wins
            } else {
                payoff[i][j] = 0.0f; // both dominated; tie
            }
        }
    }
}

// ----------------------------
// Naive CFR
// ----------------------------

struct NaiveNode {
    float regret[NUM_ACTIONS];
    float strategy[NUM_ACTIONS];
    float strategy_sum[NUM_ACTIONS];
};

static void naive_compute_strategy(const float* regret, float* strategy) {
    float sum = 0.0f;

    for (int a = 0; a < NUM_ACTIONS; ++a) {
        float pos = regret[a] > 0.0f ? regret[a] : 0.0f;
        strategy[a] = pos;
        sum += pos;
    }

    if (sum > 0.0f) {
        for (int a = 0; a < NUM_ACTIONS; ++a) strategy[a] /= sum;
    } else {
        for (int a = 0; a < NUM_ACTIONS; ++a) strategy[a] = 1.0f / NUM_ACTIONS;
    }
}

static void naive_iter(NaiveNode& p1, NaiveNode& p2) {
    naive_compute_strategy(p1.regret, p1.strategy);
    naive_compute_strategy(p2.regret, p2.strategy);

    float cfv1[NUM_ACTIONS] = {0};
    float cfv2[NUM_ACTIONS] = {0};
    for (int i = 0; i < NUM_ACTIONS; ++i) {
        for (int j = 0; j < NUM_ACTIONS; ++j) {
            cfv1[i] += p2.strategy[j] * payoff[i][j];
            cfv2[j] += p1.strategy[i] * (-payoff[i][j]);
        }
    }

    float ev1 = 0, ev2 = 0;
    for (int a = 0; a < NUM_ACTIONS; ++a) {
        ev1 += p1.strategy[a] * cfv1[a];
        ev2 += p2.strategy[a] * cfv2[a];
    }

    for (int a = 0; a < NUM_ACTIONS; ++a) {
        p1.regret[a] += cfv1[a] - ev1;
        p2.regret[a] += cfv2[a] - ev2;
        p1.strategy_sum[a] += p1.strategy[a];
        p2.strategy_sum[a] += p2.strategy[a];
    }
}

// -------------------------------------
// SoA + Aligned; Scalar code, but can be auto-vectorized
// Even tough each node is only 96 bytes of data (24 floats)
// we pad to a multiple of 64 so an array 9of nodes never has a node
// straddling cache lines.
//
// Interestingly this is the least efficient of the three.
// -------------------------------------

struct alignas(CACHE_LINE) AlignedNode {
    alignas(CACHE_LINE) float regret[NUM_ACTIONS];
    alignas(CACHE_LINE) float strategy[NUM_ACTIONS];
    alignas(CACHE_LINE) float strategy_sum[NUM_ACTIONS];
};

static_assert(sizeof(AlignedNode) % CACHE_LINE == 0, "AlignNode size must be a cache-line multiple");

static void aligned_compute_strategy(const float* regret, float* strategy) {
    float sum = 0.0f;
    for (int a = 0; a < NUM_ACTIONS; ++a) {
        float pos = regret[a] > 0.0f ? regret[a] : 0.0f;
        strategy[a] = pos;
        sum += pos;
    }
    if (sum > 0.0f) {
        for (int a = 0; a < NUM_ACTIONS; ++a) strategy[a] /= sum;
    } else {
        for (int a = 0; a < NUM_ACTIONS; ++a) strategy[a] = 1.0f / NUM_ACTIONS;
    }
}

static void aligned_iter(AlignedNode& p1, AlignedNode& p2) {
    aligned_compute_strategy(p1.regret, p1.strategy);
    aligned_compute_strategy(p2.regret, p2.strategy);

    alignas(CACHE_LINE) float cfv1[NUM_ACTIONS] = {0};
    alignas(CACHE_LINE) float cfv2[NUM_ACTIONS] = {0};

    for (int i = 0; i < NUM_ACTIONS; ++i) {
        for (int j = 0; j < NUM_ACTIONS; ++j) {
            cfv1[i] += p2.strategy[j] * payoff[i][j];
            cfv2[j] -= p1.strategy[i] * payoff[i][j];
        }
    }

    float ev1 = 0, ev2 = 0;
    for (int a = 0; a < NUM_ACTIONS; ++a) {
        ev1 += p1.strategy[a] * cfv1[a];
        ev2 += p2.strategy[a] * cfv2[a];
    }

    for (int a = 0; a < NUM_ACTIONS; ++a) {
        p1.regret[a]       += cfv1[a] - ev1;
        p2.regret[a]       += cfv2[a] - ev2;
        p1.strategy_sum[a] += p1.strategy[a];
        p2.strategy_sum[a] += p2.strategy[a];
    }
}

// ---------------------------------------------------------------------------
//  AVX2 impl. Same algorithm, but every loop is now a few SIMD ops.
// ---------------------------------------------------------------------------

// Horizontal sum of 8 floats in a __m256 -> single float.
// Used twice: in regret matching (sum of positive regrets) and EV computation.
static inline float hsum256(__m256 v) {
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 s  = _mm_add_ps(hi, lo);                  // 4 floats
    s         = _mm_hadd_ps(s, s);                   // 2 floats (each summed twice)
    s         = _mm_hadd_ps(s, s);                   // 1 float (summed 4x, but we only read lane 0)
    return _mm_cvtss_f32(s);
}

static void avx2_compute_strategy(const float* regret, float* strategy) {
    __m256 r    = _mm256_load_ps(regret);
    __m256 zero = _mm256_setzero_ps();
    __m256 pos  = _mm256_max_ps(r, zero);            // branchless max(r, 0)

    float sum = hsum256(pos);

    if (sum > 0.0f) {
        __m256 inv = _mm256_set1_ps(1.0f / sum);
        _mm256_store_ps(strategy, _mm256_mul_ps(pos, inv));
    } else {
        _mm256_store_ps(strategy, _mm256_set1_ps(1.0f / NUM_ACTIONS));
    }
}

// psychotically efficient :)
static void avx2_iter(AlignedNode& p1, AlignedNode& p2) {
    avx2_compute_strategy(p1.regret, p1.strategy);
    avx2_compute_strategy(p2.regret, p2.strategy);

    __m256 s1 = _mm256_load_ps(p1.strategy);
    __m256 s2 = _mm256_load_ps(p2.strategy);

    // cfv1[i] = sum_j s2[j] * payoff[i][j]   (row-dot-vector, 8 separate dots)
    alignas(CACHE_LINE) float cfv1_arr[NUM_ACTIONS];
    for (int i = 0; i < NUM_ACTIONS; ++i) {
        __m256 row  = _mm256_load_ps(payoff[i]);
        __m256 prod = _mm256_mul_ps(row, s2);
        cfv1_arr[i] = hsum256(prod);
    }
    __m256 cfv1 = _mm256_load_ps(cfv1_arr);

    // cfv2[j] = -sum_i s1[i] * payoff[i][j]  (column-sum, much nicer in SIMD:
    //   accumulate full rows scaled by the scalar s1[i], one FMA per row)
    __m256 cfv2 = _mm256_setzero_ps();
    for (int i = 0; i < NUM_ACTIONS; ++i) {
        __m256 row    = _mm256_load_ps(payoff[i]);
        __m256 s1_i   = _mm256_set1_ps(p1.strategy[i]);   // broadcast scalar
        cfv2 = _mm256_fmadd_ps(s1_i, row, cfv2);          // cfv2 += s1[i]*row
    }
    cfv2 = _mm256_sub_ps(_mm256_setzero_ps(), cfv2);      // negate (it's -A for P2)

    // EVs (two horizontal sums)
    float ev1 = hsum256(_mm256_mul_ps(s1, cfv1));
    float ev2 = hsum256(_mm256_mul_ps(s2, cfv2));

    // Regret update: r += cfv - ev
    __m256 r1 = _mm256_load_ps(p1.regret);
    __m256 r2 = _mm256_load_ps(p2.regret);
    r1 = _mm256_add_ps(r1, _mm256_sub_ps(cfv1, _mm256_set1_ps(ev1)));
    r2 = _mm256_add_ps(r2, _mm256_sub_ps(cfv2, _mm256_set1_ps(ev2)));
    _mm256_store_ps(p1.regret, r1);
    _mm256_store_ps(p2.regret, r2);

    // Strategy-sum accumulation
    __m256 ss1 = _mm256_load_ps(p1.strategy_sum);
    __m256 ss2 = _mm256_load_ps(p2.strategy_sum);
    _mm256_store_ps(p1.strategy_sum, _mm256_add_ps(ss1, s1));
    _mm256_store_ps(p2.strategy_sum, _mm256_add_ps(ss2, s2));
}

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

    // 1. Naive
    {
        NaiveNode p1{}, p2{};
        double ms = run_and_time(naive_iter, p1, p2, NUM_ITERS);
        printf("[naive]       %.3f ms\n", ms);
        print_avg_strategy(p1.strategy_sum, "P1 avg");
    }
    // 2. SoA + aligned
    {
        AlignedNode p1{}, p2{};
        double ms = run_and_time(aligned_iter, p1, p2, NUM_ITERS);
        printf("[aligned]     %.3f ms\n", ms);
        print_avg_strategy(p1.strategy_sum, "P1 avg");
    }
    // 3. AVX2
    {
        AlignedNode p1{}, p2{};
        double ms = run_and_time(avx2_iter, p1, p2, NUM_ITERS);
        printf("[avx2]        %.3f ms\n", ms);
        print_avg_strategy(p1.strategy_sum, "P1 avg");
    }
    return 0;
}
