// Counterfactual Regret Minimization on an 8-action zero-sum normal-form game.
// Three implementations (naive / aligned-SoA / AVX2) of vanilla CFR.
//
// Game: extended Rock-Paper-Scissors. Actions 0,1,2 are R/P/S with the usual
// payoffs. Actions 3..7 are strictly dominated (always lose to 0..2, tie among
// themselves).
// Nash equilibrium: (1/3, 1/3, 1/3, 0, 0, 0, 0, 0).

#pragma once

#include <cstdint>
#include <immintrin.h>

constexpr int NUM_ACTIONS = 8; // exactly one __m256 of floats
constexpr int CACHE_LINE  = 64;

alignas(CACHE_LINE) inline float payoff[NUM_ACTIONS][NUM_ACTIONS];

inline void init_payoff() {
    int rps[3][3] = {
        { 0, -1,  1 }, // Rock vs R, P, S
        { 1,  0, -1 }, // Paper
        {-1,  1,  0 }, // Scissors
    };
    for (int i = 0; i < NUM_ACTIONS; ++i) {
        for (int j = 0; j < NUM_ACTIONS; ++j) {
            if (i < 3 && j < 3)        payoff[i][j] = (float)rps[i][j];
            else if (i >= 3 && j < 3)  payoff[i][j] = -1.0f; // P1 dominated
            else if (i < 3 && j >= 3)  payoff[i][j] =  1.0f; // P2 dominated
            else                       payoff[i][j] =  0.0f; // both dominated
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

inline void naive_compute_strategy(const float* regret, float* strategy) {
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

inline void naive_iter(NaiveNode& p1, NaiveNode& p2) {
    naive_compute_strategy(p1.regret, p1.strategy);
    naive_compute_strategy(p2.regret, p2.strategy);

    float cfv1[NUM_ACTIONS] = {0};
    float cfv2[NUM_ACTIONS] = {0};
    for (int i = 0; i < NUM_ACTIONS; ++i) {
        for (int j = 0; j < NUM_ACTIONS; ++j) {
            cfv1[i] += p2.strategy[j] *  payoff[i][j];
            cfv2[j] += p1.strategy[i] * -payoff[i][j];
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

// -------------------------------------
// SoA + Aligned. Scalar code, but auto-vectorizable.
// Padded to a cache-line multiple so an array of nodes never has a node
// straddling cache lines.
// -------------------------------------

struct alignas(CACHE_LINE) AlignedNode {
    alignas(CACHE_LINE) float regret[NUM_ACTIONS];
    alignas(CACHE_LINE) float strategy[NUM_ACTIONS];
    alignas(CACHE_LINE) float strategy_sum[NUM_ACTIONS];
};

static_assert(sizeof(AlignedNode) % CACHE_LINE == 0,
              "AlignedNode size must be a cache-line multiple");

inline void aligned_compute_strategy(const float* regret, float* strategy) {
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

inline void aligned_iter(AlignedNode& p1, AlignedNode& p2) {
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
inline float hsum256(__m256 v) {
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 s  = _mm_add_ps(hi, lo);
    s         = _mm_hadd_ps(s, s);
    s         = _mm_hadd_ps(s, s);
    return _mm_cvtss_f32(s);
}

inline void avx2_compute_strategy(const float* regret, float* strategy) {
    __m256 r    = _mm256_load_ps(regret);
    __m256 zero = _mm256_setzero_ps();
    __m256 pos  = _mm256_max_ps(r, zero);

    float sum = hsum256(pos);

    if (sum > 0.0f) {
        __m256 inv = _mm256_set1_ps(1.0f / sum);
        _mm256_store_ps(strategy, _mm256_mul_ps(pos, inv));
    } else {
        _mm256_store_ps(strategy, _mm256_set1_ps(1.0f / NUM_ACTIONS));
    }
}

inline void avx2_iter(AlignedNode& p1, AlignedNode& p2) {
    avx2_compute_strategy(p1.regret, p1.strategy);
    avx2_compute_strategy(p2.regret, p2.strategy);

    __m256 s1 = _mm256_load_ps(p1.strategy);
    __m256 s2 = _mm256_load_ps(p2.strategy);

    // cfv1[i] = sum_j s2[j] * payoff[i][j]   (8 row-dots)
    alignas(CACHE_LINE) float cfv1_arr[NUM_ACTIONS];
    for (int i = 0; i < NUM_ACTIONS; ++i) {
        __m256 row  = _mm256_load_ps(payoff[i]);
        __m256 prod = _mm256_mul_ps(row, s2);
        cfv1_arr[i] = hsum256(prod);
    }
    __m256 cfv1 = _mm256_load_ps(cfv1_arr);

    // cfv2[j] = -sum_i s1[i] * payoff[i][j]  (column-sum via FMA per row)
    __m256 cfv2 = _mm256_setzero_ps();
    for (int i = 0; i < NUM_ACTIONS; ++i) {
        __m256 row  = _mm256_load_ps(payoff[i]);
        __m256 s1_i = _mm256_set1_ps(p1.strategy[i]);
        cfv2 = _mm256_fmadd_ps(s1_i, row, cfv2);
    }
    cfv2 = _mm256_sub_ps(_mm256_setzero_ps(), cfv2);

    float ev1 = hsum256(_mm256_mul_ps(s1, cfv1));
    float ev2 = hsum256(_mm256_mul_ps(s2, cfv2));

    __m256 r1 = _mm256_load_ps(p1.regret);
    __m256 r2 = _mm256_load_ps(p2.regret);
    r1 = _mm256_add_ps(r1, _mm256_sub_ps(cfv1, _mm256_set1_ps(ev1)));
    r2 = _mm256_add_ps(r2, _mm256_sub_ps(cfv2, _mm256_set1_ps(ev2)));
    _mm256_store_ps(p1.regret, r1);
    _mm256_store_ps(p2.regret, r2);

    __m256 ss1 = _mm256_load_ps(p1.strategy_sum);
    __m256 ss2 = _mm256_load_ps(p2.strategy_sum);
    _mm256_store_ps(p1.strategy_sum, _mm256_add_ps(ss1, s1));
    _mm256_store_ps(p2.strategy_sum, _mm256_add_ps(ss2, s2));
}
