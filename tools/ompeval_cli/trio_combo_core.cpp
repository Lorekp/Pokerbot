#include "trio_combo_core.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "hu_score2_core.hpp"
#include "trio_combos_data.hpp"
#include "trio_orientation.hpp"

namespace {

constexpr int kNPlayers = 3;
constexpr int kMaxProfileThreads = 256;

struct TrioRowProfileAcc {
    int64_t exact_triple_count = 0;
    int64_t mc_triple_count = 0;
    double exact_triple_ns = 0.0;
    double mc_triple_ns = 0.0;
    double mc_hidden_pair_exact_ns = 0.0;
    int64_t mc_board_evals = 0;
};

static TrioRowProfileAcc g_prof_by_thread[kMaxProfileThreads];

inline TrioRowProfileAcc& prof_acc() {
    int tid = 0;
#ifdef _OPENMP
    tid = omp_get_thread_num();
#endif
    if (tid < 0 || tid >= kMaxProfileThreads) {
        tid = 0;
    }
    return g_prof_by_thread[tid];
}

inline double steady_ns_elapsed(const std::chrono::steady_clock::time_point& t0) {
    const auto t1 = std::chrono::steady_clock::now();
    return static_cast<double>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
}

void record_mc_board_evals(int n_boards) {
    if (n_boards > 0) {
        prof_acc().mc_board_evals += static_cast<int64_t>(n_boards);
    }
}

int binom5(int n) {
    if (n < 5) {
        return 0;
    }
    return (n * (n - 1) * (n - 2) * (n - 3) * (n - 4)) / 120;
}

void build_live_deck(const int hole[6], int n_players, int deck[52], int* nd_out) {
    bool dead[52] = {};
    for (int k = 0; k < 2 * n_players; ++k) {
        dead[hole[k]] = true;
    }
    int nd = 0;
    for (int c = 0; c < 52; ++c) {
        if (!dead[c]) {
            deck[nd++] = c;
        }
    }
    *nd_out = nd;
}

void exact_nway_shares_impl(omp::HandEvaluator& evaluator, const int hole[6], int n_players, double* shares_out) {
    int deck[52];
    int nd = 0;
    build_live_deck(hole, n_players, deck, &nd);
    if (nd < 5) {
        for (int i = 0; i < n_players; ++i) {
            shares_out[i] = 1.0 / n_players;
        }
        return;
    }
    double wins[6] = {};
    for (int i0 = 0; i0 < nd - 4; ++i0) {
        const int c0 = deck[i0];
        for (int i1 = i0 + 1; i1 < nd - 3; ++i1) {
            const int c1 = deck[i1];
            for (int i2 = i1 + 1; i2 < nd - 2; ++i2) {
                const int c2 = deck[i2];
                for (int i3 = i2 + 1; i3 < nd - 1; ++i3) {
                    const int c3 = deck[i3];
                    for (int i4 = i3 + 1; i4 < nd; ++i4) {
                        const int c4 = deck[i4];
                        uint16_t ranks[6];
                        for (int pi = 0; pi < n_players; ++pi) {
                            ranks[pi] = eval7_with(evaluator, hole[2 * pi], hole[2 * pi + 1], c0, c1, c2, c3, c4);
                        }
                        uint16_t best = 0;
                        int nwin = 0;
                        trio::pick_nway_winners(ranks, n_players, &best, &nwin);
                        const double share = 1.0 / nwin;
                        for (int pi = 0; pi < n_players; ++pi) {
                            if (ranks[pi] == best) {
                                wins[pi] += share;
                            }
                        }
                    }
                }
            }
        }
    }
    const double inv = 1.0 / static_cast<double>(binom5(nd));
    for (int i = 0; i < n_players; ++i) {
        shares_out[i] = wins[i] * inv;
    }
}

// SHA256 (minimal, for combo seed parity with Python hashlib.sha256)
struct Sha256Ctx {
    uint32_t state[8];
    uint64_t count;
    uint8_t buffer[64];
};

static uint32_t rotr(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }

static void sha256_transform(Sha256Ctx* ctx, const uint8_t data[64]) {
    static const uint32_t k[64] = {
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
        0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
        0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
        0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};
    uint32_t w[64];
    for (int i = 0; i < 16; ++i) {
        w[i] = (static_cast<uint32_t>(data[i * 4]) << 24) | (static_cast<uint32_t>(data[i * 4 + 1]) << 16) |
               (static_cast<uint32_t>(data[i * 4 + 2]) << 8) | static_cast<uint32_t>(data[i * 4 + 3]);
    }
    for (int i = 16; i < 64; ++i) {
        const uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
        const uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    uint32_t a = ctx->state[0], b = ctx->state[1], c = ctx->state[2], d = ctx->state[3];
    uint32_t e = ctx->state[4], f = ctx->state[5], g = ctx->state[6], h = ctx->state[7];
    for (int i = 0; i < 64; ++i) {
        const uint32_t S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
        const uint32_t ch = (e & f) ^ ((~e) & g);
        const uint32_t temp1 = h + S1 + ch + k[i] + w[i];
        const uint32_t S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
        const uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
        const uint32_t temp2 = S0 + maj;
        h = g;
        g = f;
        f = e;
        e = d + temp1;
        d = c;
        c = b;
        b = a;
        a = temp1 + temp2;
    }
    ctx->state[0] += a;
    ctx->state[1] += b;
    ctx->state[2] += c;
    ctx->state[3] += d;
    ctx->state[4] += e;
    ctx->state[5] += f;
    ctx->state[6] += g;
    ctx->state[7] += h;
}

static void sha256_init(Sha256Ctx* ctx) {
    ctx->state[0] = 0x6a09e667;
    ctx->state[1] = 0xbb67ae85;
    ctx->state[2] = 0x3c6ef372;
    ctx->state[3] = 0xa54ff53a;
    ctx->state[4] = 0x510e527f;
    ctx->state[5] = 0x9b05688c;
    ctx->state[6] = 0x1f83d9ab;
    ctx->state[7] = 0x5be0cd19;
    ctx->count = 0;
}

static void sha256_update(Sha256Ctx* ctx, const uint8_t* data, size_t len) {
    size_t i = 0;
    while (i < len) {
        const size_t idx = ctx->count % 64;
        ctx->buffer[idx] = data[i++];
        ctx->count++;
        if (ctx->count % 64 == 0) {
            sha256_transform(ctx, ctx->buffer);
        }
    }
}

static void sha256_final(Sha256Ctx* ctx, uint8_t out[32]) {
    const size_t idx = ctx->count % 64;
    ctx->buffer[idx] = 0x80;
    if (idx >= 56) {
        std::memset(ctx->buffer + idx + 1, 0, 64 - idx - 1);
        sha256_transform(ctx, ctx->buffer);
        std::memset(ctx->buffer, 0, 56);
    } else {
        std::memset(ctx->buffer + idx + 1, 0, 55 - idx);
    }
    const uint64_t bits = ctx->count * 8;
    for (int i = 0; i < 8; ++i) {
        ctx->buffer[63 - i] = static_cast<uint8_t>(bits >> (8 * i));
    }
    sha256_transform(ctx, ctx->buffer);
    for (int i = 0; i < 8; ++i) {
        out[i * 4] = static_cast<uint8_t>(ctx->state[i] >> 24);
        out[i * 4 + 1] = static_cast<uint8_t>(ctx->state[i] >> 16);
        out[i * 4 + 2] = static_cast<uint8_t>(ctx->state[i] >> 8);
        out[i * 4 + 3] = static_cast<uint8_t>(ctx->state[i]);
    }
}

uint32_t combo_seed(int i, int j, int k) {
    char buf[64];
    const int n = std::snprintf(buf, sizeof(buf), "%d:%d:%d", i, j, k);
    Sha256Ctx ctx;
    sha256_init(&ctx);
    sha256_update(&ctx, reinterpret_cast<const uint8_t*>(buf), static_cast<size_t>(n));
    uint8_t digest[32];
    sha256_final(&ctx, digest);
    uint32_t seed = 0;
    std::memcpy(&seed, digest, 4);
    return seed;
}

void fisher_yates_board(int deck[], int nd, std::mt19937& rng, int board[5]) {
    int work[52];
    for (int i = 0; i < nd; ++i) {
        work[i] = deck[i];
    }
    std::uniform_int_distribution<int> dist;
    for (int bi = 0; bi < 5; ++bi) {
        dist.param(std::uniform_int_distribution<int>::param_type(bi, nd - 1));
        const int j = dist(rng);
        const int tmp = work[bi];
        work[bi] = work[j];
        work[j] = tmp;
        board[bi] = work[bi];
    }
}

}  // namespace

void exact_six_card_combo_metrics(const int hole[6], TrioComboMetrics& out) {
    bool dead[52] = {};
    for (int i = 0; i < 6; ++i) {
        dead[hole[i]] = true;
    }
    int deck[46];
    int nd = 0;
    for (int c = 0; c < 52; ++c) {
        if (!dead[c]) {
            deck[nd++] = c;
        }
    }

    omp::HandEvaluator evaluator;
    double share0 = 0.0, share1 = 0.0, share2 = 0.0;
    double btn_bb_w = 0.0, btn_sb_w = 0.0, sb_bb_w = 0.0;
    double total = 0.0;

    for (int i0 = 0; i0 < nd - 4; ++i0) {
        const int c0 = deck[i0];
        for (int i1 = i0 + 1; i1 < nd - 3; ++i1) {
            const int c1 = deck[i1];
            for (int i2 = i1 + 1; i2 < nd - 2; ++i2) {
                const int c2 = deck[i2];
                for (int i3 = i2 + 1; i3 < nd - 1; ++i3) {
                    const int c3 = deck[i3];
                    for (int i4 = i3 + 1; i4 < nd; ++i4) {
                        const int c4 = deck[i4];
                        const uint16_t hr_btn = eval7_with(evaluator, hole[0], hole[1], c0, c1, c2, c3, c4);
                        const uint16_t hr_sb = eval7_with(evaluator, hole[2], hole[3], c0, c1, c2, c3, c4);
                        const uint16_t hr_bb = eval7_with(evaluator, hole[4], hole[5], c0, c1, c2, c3, c4);
                        uint16_t best = hr_btn;
                        if (trio::rank_beats(hr_sb, best)) {
                            best = hr_sb;
                        }
                        if (trio::rank_beats(hr_bb, best)) {
                            best = hr_bb;
                        }
                        int nwin = 0;
                        if (hr_btn == best) {
                            nwin += 1;
                        }
                        if (hr_sb == best) {
                            nwin += 1;
                        }
                        if (hr_bb == best) {
                            nwin += 1;
                        }
                        const double pot_share = 1.0 / nwin;
                        if (hr_btn == best) {
                            share0 += pot_share;
                        }
                        if (hr_sb == best) {
                            share1 += pot_share;
                        }
                        if (hr_bb == best) {
                            share2 += pot_share;
                        }
                        trio::pair_equity_counters(hr_btn, hr_sb, hr_bb, &btn_bb_w, &btn_sb_w, &sb_bb_w);
                        total += 1.0;
                    }
                }
            }
        }
    }

    if (total <= 0.0) {
        out.shares[0] = out.shares[1] = out.shares[2] = 1.0 / 3.0;
        out.pair_eq[0] = out.pair_eq[1] = out.pair_eq[2] = 0.5;
        return;
    }
    const double inv = 1.0 / total;
    out.shares[0] = share0 * inv;
    out.shares[1] = share1 * inv;
    out.shares[2] = share2 * inv;
    out.pair_eq[0] = btn_bb_w * inv;
    out.pair_eq[1] = btn_sb_w * inv;
    out.pair_eq[2] = sb_bb_w * inv;
}

void mc_nway_shares(const int hole[6], int n_players, int n_boards, uint32_t seed, double* shares_out,
                    double* pair_eq_out) {
    int deck[52];
    int nd = 0;
    build_live_deck(hole, n_players, deck, &nd);
    if (nd < 5) {
        for (int i = 0; i < n_players; ++i) {
            shares_out[i] = 1.0 / n_players;
        }
        if (pair_eq_out != nullptr && n_players == kNPlayers) {
            pair_eq_out[0] = pair_eq_out[1] = pair_eq_out[2] = 0.5;
        }
        return;
    }
    if (n_boards >= binom5(nd)) {
        if (pair_eq_out != nullptr && n_players == kNPlayers) {
            TrioComboMetrics exact{};
            exact_six_card_combo_metrics(hole, exact);
            for (int i = 0; i < n_players; ++i) {
                shares_out[i] = exact.shares[i];
            }
            pair_eq_out[0] = exact.pair_eq[0];
            pair_eq_out[1] = exact.pair_eq[1];
            pair_eq_out[2] = exact.pair_eq[2];
            return;
        }
        exact_nway_shares_impl(omp::HandEvaluator(), hole, n_players, shares_out);
        return;
    }

    std::mt19937 rng(seed & 0xFFFFFFFFu);
    omp::HandEvaluator evaluator;
    double wins[6] = {};
    double btn_bb_w = 0.0;
    double btn_sb_w = 0.0;
    double sb_bb_w = 0.0;
    int board[5];
    record_mc_board_evals(n_boards);
    for (int b = 0; b < n_boards; ++b) {
        fisher_yates_board(deck, nd, rng, board);
        uint16_t ranks[6];
        for (int pi = 0; pi < n_players; ++pi) {
            ranks[pi] = eval7_with(evaluator, hole[2 * pi], hole[2 * pi + 1], board[0], board[1], board[2],
                                   board[3], board[4]);
        }
        double seat_shares[6];
        trio::seat_shares_from_ranks(ranks, n_players, seat_shares);
        for (int pi = 0; pi < n_players; ++pi) {
            wins[pi] += seat_shares[pi];
        }
        if (pair_eq_out != nullptr && n_players == kNPlayers) {
            trio::pair_equity_counters(ranks[0], ranks[1], ranks[2], &btn_bb_w, &btn_sb_w, &sb_bb_w);
        }
    }
    const double inv = 1.0 / n_boards;
    for (int i = 0; i < n_players; ++i) {
        shares_out[i] = wins[i] * inv;
    }
    if (pair_eq_out != nullptr && n_players == kNPlayers) {
        pair_eq_out[0] = btn_bb_w * inv;
        pair_eq_out[1] = btn_sb_w * inv;
        pair_eq_out[2] = sb_bb_w * inv;
    }
}

namespace {

// One showdown's MC shares for arbitrary n_players (2..6), reusing the same OMPEval evaluator,
// Fisher-Yates board sampler, and winner/share arithmetic as ``mc_nway_shares``. Statistically
// identical to that path (same evaluator + sampler), differing only in the RNG seed/stream.
void mc_nway_shares_one(omp::HandEvaluator& evaluator, const int* hole, int n_players, int n_boards,
                        uint32_t seed, double* shares_out) {
    int deck[52];
    int nd = 0;
    {
        bool dead[52] = {};
        for (int k = 0; k < 2 * n_players; ++k) {
            dead[hole[k]] = true;
        }
        for (int c = 0; c < 52; ++c) {
            if (!dead[c]) {
                deck[nd++] = c;
            }
        }
    }
    if (nd < 5) {
        for (int i = 0; i < n_players; ++i) {
            shares_out[i] = 1.0 / n_players;
        }
        return;
    }
    if (n_boards >= binom5(nd)) {
        // Exact enumeration when the sample count meets/exceeds the runout count (matches
        // mc_nway_shares' exact branch; statistically identical and variance-free).
        double wins[6] = {};
        for (int i0 = 0; i0 < nd - 4; ++i0) {
            const int c0 = deck[i0];
            for (int i1 = i0 + 1; i1 < nd - 3; ++i1) {
                const int c1 = deck[i1];
                for (int i2 = i1 + 1; i2 < nd - 2; ++i2) {
                    const int c2 = deck[i2];
                    for (int i3 = i2 + 1; i3 < nd - 1; ++i3) {
                        const int c3 = deck[i3];
                        for (int i4 = i3 + 1; i4 < nd; ++i4) {
                            const int c4 = deck[i4];
                            uint16_t ranks[6];
                            for (int pi = 0; pi < n_players; ++pi) {
                                ranks[pi] = eval7_with(evaluator, hole[2 * pi], hole[2 * pi + 1], c0, c1,
                                                       c2, c3, c4);
                            }
                            uint16_t best = 0;
                            int nwin = 0;
                            trio::pick_nway_winners(ranks, n_players, &best, &nwin);
                            const double share = 1.0 / nwin;
                            for (int pi = 0; pi < n_players; ++pi) {
                                if (ranks[pi] == best) {
                                    wins[pi] += share;
                                }
                            }
                        }
                    }
                }
            }
        }
        const double inv = 1.0 / static_cast<double>(binom5(nd));
        for (int i = 0; i < n_players; ++i) {
            shares_out[i] = wins[i] * inv;
        }
        return;
    }
    std::mt19937 rng(seed & 0xFFFFFFFFu);
    double wins[6] = {};
    int board[5];
    for (int b = 0; b < n_boards; ++b) {
        fisher_yates_board(deck, nd, rng, board);
        uint16_t ranks[6];
        for (int pi = 0; pi < n_players; ++pi) {
            ranks[pi] = eval7_with(evaluator, hole[2 * pi], hole[2 * pi + 1], board[0], board[1], board[2],
                                   board[3], board[4]);
        }
        double seat_shares[6];
        trio::seat_shares_from_ranks(ranks, n_players, seat_shares);
        for (int pi = 0; pi < n_players; ++pi) {
            wins[pi] += seat_shares[pi];
        }
    }
    const double inv = 1.0 / n_boards;
    for (int i = 0; i < n_players; ++i) {
        shares_out[i] = wins[i] * inv;
    }
}

}  // namespace

void mc_nway_shares_batch(const int* holes_flat, const int* sizes, const uint32_t* seeds, int n_show,
                          int n_boards, double* shares_out, int max_players, int threads) {
    // Per-showdown hole offsets (variable player counts -> variable stride into holes_flat).
    std::vector<int64_t> offsets(static_cast<size_t>(n_show) + 1, 0);
    for (int s = 0; s < n_show; ++s) {
        offsets[static_cast<size_t>(s) + 1] = offsets[static_cast<size_t>(s)] + 2 * sizes[s];
    }
#ifdef _OPENMP
    if (threads > 0) {
        omp_set_num_threads(threads);
    }
#pragma omp parallel
    {
        omp::HandEvaluator evaluator;  // one evaluator per thread (cheap, thread-local)
#pragma omp for schedule(dynamic, 64)
        for (int s = 0; s < n_show; ++s) {
            mc_nway_shares_one(evaluator, holes_flat + offsets[static_cast<size_t>(s)], sizes[s], n_boards,
                               seeds[s], shares_out + static_cast<int64_t>(s) * max_players);
        }
    }
#else
    (void)threads;
    omp::HandEvaluator evaluator;
    for (int s = 0; s < n_show; ++s) {
        mc_nway_shares_one(evaluator, holes_flat + offsets[static_cast<size_t>(s)], sizes[s], n_boards,
                           seeds[s], shares_out + static_cast<int64_t>(s) * max_players);
    }
#endif
}

TrioAdaptiveResult eval_combo_adaptive(const int hole[6], uint32_t seed, int max_boards, int min_boards,
                                       double stderr_gate) {
    TrioAdaptiveResult out{};
    out.pair_eq[0] = out.pair_eq[1] = out.pair_eq[2] = 0.5;

    int deck[52];
    int nd = 0;
    build_live_deck(hole, kNPlayers, deck, &nd);
    if (nd < 5) {
        out.shares[0] = out.shares[1] = out.shares[2] = 1.0 / 3.0;
        out.boards_used = 0;
        out.mode = 0;
        return out;
    }
    const int n_runouts = binom5(nd);
    if (n_runouts <= max_boards) {
        mc_nway_shares(hole, kNPlayers, n_runouts, seed, out.shares, nullptr);
        out.boards_used = n_runouts;
        out.mode = 1;
        return out;
    }

    double mean[3] = {};
    int boards = 0;
    int total_boards = 0;
    int step = std::max(min_boards, 64);
    while (boards < max_boards) {
        const int add = std::min(step, max_boards - boards);
        double chunk[3];
        mc_nway_shares(hole, kNPlayers, add, seed + static_cast<uint32_t>(boards), chunk, nullptr);
        boards += add;
        if (total_boards == 0) {
            mean[0] = chunk[0];
            mean[1] = chunk[1];
            mean[2] = chunk[2];
        } else {
            const double old = static_cast<double>(total_boards);
            const double nw = static_cast<double>(boards);
            mean[0] = (mean[0] * old + chunk[0] * add) / nw;
            mean[1] = (mean[1] * old + chunk[1] * add) / nw;
            mean[2] = (mean[2] * old + chunk[2] * add) / nw;
        }
        total_boards = boards;
        const double stderr_bound = 0.5 / (2.0 * std::sqrt(static_cast<double>(total_boards)));
        if (total_boards >= min_boards && stderr_bound <= stderr_gate) {
            out.shares[0] = mean[0];
            out.shares[1] = mean[1];
            out.shares[2] = mean[2];
            out.boards_used = total_boards;
            out.mode = 2;
            return out;
        }
        step = std::min(step * 2, 512);
    }
    if (total_boards >= n_runouts) {
        mc_nway_shares(hole, kNPlayers, n_runouts, seed, out.shares, nullptr);
        out.boards_used = n_runouts;
        out.mode = 4;
        return out;
    }
    out.shares[0] = mean[0];
    out.shares[1] = mean[1];
    out.shares[2] = mean[2];
    out.boards_used = total_boards;
    out.mode = 3;
    return out;
}

int count_combo_triples(int row, int j, int k) {
    const int nc0 = TRIO_BUCKET_NCOMBOS[static_cast<size_t>(row)];
    const int nc1 = TRIO_BUCKET_NCOMBOS[static_cast<size_t>(j)];
    const int nc2 = TRIO_BUCKET_NCOMBOS[static_cast<size_t>(k)];
    const BucketComboPair* b0 = TRIO_BUCKET_COMBOS[static_cast<size_t>(row)];
    const BucketComboPair* b1 = TRIO_BUCKET_COMBOS[static_cast<size_t>(j)];
    const BucketComboPair* b2 = TRIO_BUCKET_COMBOS[static_cast<size_t>(k)];
    int count = 0;
    for (int a = 0; a < nc0; ++a) {
        for (int b = 0; b < nc1; ++b) {
            for (int c = 0; c < nc2; ++c) {
                const int cards[6] = {b0[a].c0, b0[a].c1, b1[b].c0, b1[b].c1, b2[c].c0, b2[c].c1};
                bool ok = true;
                for (int x = 0; x < 6 && ok; ++x) {
                    for (int y = x + 1; y < 6; ++y) {
                        if (cards[x] == cards[y]) {
                            ok = false;
                            break;
                        }
                    }
                }
                if (ok) {
                    count += 1;
                }
            }
        }
    }
    return count;
}

namespace {

bool hole_six_valid(const int hole[6]) {
    for (int x = 0; x < 6; ++x) {
        for (int y = x + 1; y < 6; ++y) {
            if (hole[x] == hole[y]) {
                return false;
            }
        }
    }
    return true;
}

}  // namespace

void exact_pair_eq_for_triple(int row, int j, int k, double* pair_eq_out, int* combo_count_out) {
    const int count = count_combo_triples(row, j, k);
    *combo_count_out = count;
    if (count == 0) {
        pair_eq_out[0] = pair_eq_out[1] = pair_eq_out[2] = 0.5;
        return;
    }

    const int nc0 = TRIO_BUCKET_NCOMBOS[static_cast<size_t>(row)];
    const int nc1 = TRIO_BUCKET_NCOMBOS[static_cast<size_t>(j)];
    const int nc2 = TRIO_BUCKET_NCOMBOS[static_cast<size_t>(k)];
    const BucketComboPair* b0 = TRIO_BUCKET_COMBOS[static_cast<size_t>(row)];
    const BucketComboPair* b1 = TRIO_BUCKET_COMBOS[static_cast<size_t>(j)];
    const BucketComboPair* b2 = TRIO_BUCKET_COMBOS[static_cast<size_t>(k)];

    double btn_bb_sum = 0.0, btn_sb_sum = 0.0, sb_bb_sum = 0.0;
    int valid = 0;
    for (int a = 0; a < nc0; ++a) {
        for (int b = 0; b < nc1; ++b) {
            for (int c = 0; c < nc2; ++c) {
                const int hole[6] = {b0[a].c0, b0[a].c1, b1[b].c0, b1[b].c1, b2[c].c0, b2[c].c1};
                if (!hole_six_valid(hole)) {
                    continue;
                }
                TrioComboMetrics m{};
                exact_six_card_combo_metrics(hole, m);
                btn_bb_sum += m.pair_eq[0];
                btn_sb_sum += m.pair_eq[1];
                sb_bb_sum += m.pair_eq[2];
                valid += 1;
            }
        }
    }
    if (valid <= 0) {
        pair_eq_out[0] = pair_eq_out[1] = pair_eq_out[2] = 0.5;
        return;
    }
    const double inv = 1.0 / valid;
    pair_eq_out[0] = btn_bb_sum * inv;
    pair_eq_out[1] = btn_sb_sum * inv;
    pair_eq_out[2] = sb_bb_sum * inv;
}

void exact_shares_for_triple(int row, int j, int k, double* shares_out, int* combo_count_out) {
    const int count = count_combo_triples(row, j, k);
    *combo_count_out = count;
    if (count == 0) {
        shares_out[0] = shares_out[1] = shares_out[2] = 1.0 / 3.0;
        return;
    }

    const int nc0 = TRIO_BUCKET_NCOMBOS[static_cast<size_t>(row)];
    const int nc1 = TRIO_BUCKET_NCOMBOS[static_cast<size_t>(j)];
    const int nc2 = TRIO_BUCKET_NCOMBOS[static_cast<size_t>(k)];
    const BucketComboPair* b0 = TRIO_BUCKET_COMBOS[static_cast<size_t>(row)];
    const BucketComboPair* b1 = TRIO_BUCKET_COMBOS[static_cast<size_t>(j)];
    const BucketComboPair* b2 = TRIO_BUCKET_COMBOS[static_cast<size_t>(k)];

    double shares_sum[3] = {};
    for (int a = 0; a < nc0; ++a) {
        for (int b = 0; b < nc1; ++b) {
            for (int c = 0; c < nc2; ++c) {
                const int hole[6] = {b0[a].c0, b0[a].c1, b1[b].c0, b1[b].c1, b2[c].c0, b2[c].c1};
                if (!hole_six_valid(hole)) {
                    continue;
                }
                TrioComboMetrics m{};
                exact_six_card_combo_metrics(hole, m);
                shares_sum[0] += m.shares[0];
                shares_sum[1] += m.shares[1];
                shares_sum[2] += m.shares[2];
            }
        }
    }

    const double inv = 1.0 / count;
    shares_out[0] = shares_sum[0] * inv;
    shares_out[1] = shares_sum[1] * inv;
    shares_out[2] = shares_sum[2] * inv;
}

void avg_metrics_for_triple(int row, int j, int k, int mc_boards, int exact_combo_max, int min_boards_per_combo,
                            double stderr_gate, double* shares_out, double* pair_eq_out, int* combo_count_out,
                            const float* pair_eq_lookup, int pair_eq_lookup_stride) {
    const auto triple_t0 = std::chrono::steady_clock::now();
    const int count = count_combo_triples(row, j, k);
    *combo_count_out = count;
    if (count == 0) {
        shares_out[0] = shares_out[1] = shares_out[2] = 1.0 / 3.0;
        pair_eq_out[0] = pair_eq_out[1] = pair_eq_out[2] = 0.5;
        return;
    }

    const int nc0 = TRIO_BUCKET_NCOMBOS[static_cast<size_t>(row)];
    const int nc1 = TRIO_BUCKET_NCOMBOS[static_cast<size_t>(j)];
    const int nc2 = TRIO_BUCKET_NCOMBOS[static_cast<size_t>(k)];
    const BucketComboPair* b0 = TRIO_BUCKET_COMBOS[static_cast<size_t>(row)];
    const BucketComboPair* b1 = TRIO_BUCKET_COMBOS[static_cast<size_t>(j)];
    const BucketComboPair* b2 = TRIO_BUCKET_COMBOS[static_cast<size_t>(k)];

    double shares_sum[3] = {};
    uint32_t sub_seed = combo_seed(row, j, k);
    const int boards_per_combo = mc_boards > 0 ? mc_boards : 128;

    auto process_combo_shares = [&](const int hole[6], bool exact_path) {
        if (exact_path) {
            TrioComboMetrics m{};
            exact_six_card_combo_metrics(hole, m);
            shares_sum[0] += m.shares[0];
            shares_sum[1] += m.shares[1];
            shares_sum[2] += m.shares[2];
        } else {
            TrioAdaptiveResult r =
                eval_combo_adaptive(hole, sub_seed, boards_per_combo, min_boards_per_combo, stderr_gate);
            sub_seed += 1;
            shares_sum[0] += r.shares[0];
            shares_sum[1] += r.shares[1];
            shares_sum[2] += r.shares[2];
        }
    };

    if (count <= exact_combo_max) {
        for (int a = 0; a < nc0; ++a) {
            for (int b = 0; b < nc1; ++b) {
                for (int c = 0; c < nc2; ++c) {
                    const int hole[6] = {b0[a].c0, b0[a].c1, b1[b].c0, b1[b].c1, b2[c].c0, b2[c].c1};
                    if (!hole_six_valid(hole)) {
                        continue;
                    }
                    process_combo_shares(hole, true);
                }
            }
        }
    } else {
        for (int a = 0; a < nc0; ++a) {
            for (int b = 0; b < nc1; ++b) {
                for (int c = 0; c < nc2; ++c) {
                    const int hole[6] = {b0[a].c0, b0[a].c1, b1[b].c0, b1[b].c1, b2[c].c0, b2[c].c1};
                    if (!hole_six_valid(hole)) {
                        continue;
                    }
                    process_combo_shares(hole, false);
                }
            }
        }
    }

    const double inv = 1.0 / count;
    shares_out[0] = shares_sum[0] * inv;
    shares_out[1] = shares_sum[1] * inv;
    shares_out[2] = shares_sum[2] * inv;

    if (pair_eq_lookup != nullptr && pair_eq_lookup_stride > 0) {
        const std::size_t base =
            (static_cast<std::size_t>(j) * static_cast<std::size_t>(pair_eq_lookup_stride) +
             static_cast<std::size_t>(k)) *
            3u;
        pair_eq_out[0] = static_cast<double>(pair_eq_lookup[base]);
        pair_eq_out[1] = static_cast<double>(pair_eq_lookup[base + 1]);
        pair_eq_out[2] = static_cast<double>(pair_eq_lookup[base + 2]);
    } else {
        int dummy = 0;
        exact_pair_eq_for_triple(row, j, k, pair_eq_out, &dummy);
    }

    const double triple_ns = steady_ns_elapsed(triple_t0);
    auto& acc = prof_acc();
    if (count <= exact_combo_max) {
        acc.exact_triple_count += 1;
        acc.exact_triple_ns += triple_ns;
    } else if (count > 0) {
        acc.mc_triple_count += 1;
        acc.mc_triple_ns += triple_ns;
    }
}

void reset_trio_row_profile() {
    for (int i = 0; i < kMaxProfileThreads; ++i) {
        g_prof_by_thread[i] = {};
    }
}

TrioRowProfile get_trio_row_profile() {
    TrioRowProfileAcc sum{};
    for (int i = 0; i < kMaxProfileThreads; ++i) {
        sum.exact_triple_count += g_prof_by_thread[i].exact_triple_count;
        sum.mc_triple_count += g_prof_by_thread[i].mc_triple_count;
        sum.exact_triple_ns += g_prof_by_thread[i].exact_triple_ns;
        sum.mc_triple_ns += g_prof_by_thread[i].mc_triple_ns;
        sum.mc_hidden_pair_exact_ns += g_prof_by_thread[i].mc_hidden_pair_exact_ns;
        sum.mc_board_evals += g_prof_by_thread[i].mc_board_evals;
    }
    TrioRowProfile out{};
    out.exact_triple_count = sum.exact_triple_count;
    out.mc_triple_count = sum.mc_triple_count;
    out.exact_triple_ns = sum.exact_triple_ns;
    out.mc_triple_ns = sum.mc_triple_ns;
    out.mc_hidden_pair_exact_ns = sum.mc_hidden_pair_exact_ns;
    out.mc_board_evals = sum.mc_board_evals;
    return out;
}

bool verify_trio_sentinel() {
    if (!verify_sentinel()) {
        return false;
    }
    // AcAd / AhAs / 3c3d collision probe
    const int hole[6] = {48, 49, 50, 51, 4, 5};
    TrioComboMetrics m{};
    exact_six_card_combo_metrics(hole, m);
    const double sum = m.shares[0] + m.shares[1] + m.shares[2];
    if (std::abs(sum - 1.0) > 1e-9) {
        return false;
    }
    return m.shares[0] >= 0.0 && m.shares[1] >= 0.0 && m.shares[2] >= 0.0;
}
