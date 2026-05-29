#pragma once

#include <array>
#include <cstdint>
#include <tuple>

#include "omp/HandEvaluator.h"

static constexpr uint32_t DENOM2 = 3424608u;

// Gate A: AcAd vs KhKd — bootstrap via scripts/bootstrap_sentinel_score2.py
// Bootstrap: python scripts/bootstrap_sentinel_score2.py @ fb3650f75c03
static constexpr uint64_t SENTINEL_KEY_AA_AKO = 12948042205429808ull;
static constexpr uint32_t SENTINEL_SCORE2_AA_AKO = 2806331u;

int card_id(int rank, int suit);

std::array<int, 4> unpack_key(uint64_t key);

std::tuple<int, int, int, int> representative_from_key(uint64_t key);

uint32_t score_from_omp_ranks(uint16_t hr, uint16_t vr);

// No static/thread_local hand caches — Phase 2 proved fresh 7-card construction
// per eval is required. Reusing hole-card Hands across runouts produces wrong equity.
uint16_t eval7_with(omp::HandEvaluator& evaluator, int c0, int c1, int c2, int c3,
                    int c4, int c5, int c6);

uint16_t eval7(int c0, int c1, int c2, int c3, int c4, int c5, int c6);

uint32_t enumerate_score2(int h0, int h1, int v0, int v1);

uint32_t enumerate_score2_one(uint64_t key);

bool verify_sentinel();

uint32_t native_outcome(uint16_t hr, uint16_t vr);
