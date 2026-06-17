#pragma once

#include <array>
#include <cstdint>

struct BucketComboPair {
    int c0;
    int c1;
};

static constexpr int TRIO_N_BUCKETS = 169;
static constexpr int TRIO_MAX_COMBOS_PER_BUCKET = 6;

static constexpr std::array<BucketComboPair, TRIO_MAX_COMBOS_PER_BUCKET> TRIO_BUCKET_0 = { {
    { 0, 1 },
    { 2, 3 },
    { 0, 2 },
    { 1, 3 },
    { 2, 0 },
    { 1, 2 }
} };

static constexpr std::array<BucketComboPair, TRIO_MAX_COMBOS_PER_BUCKET> TRIO_BUCKET_1 = { {
    { 4, 5 },
    { 6, 7 },
    { 4, 6 },
    { 5, 7 },
    { 6, 4 },
    { 5, 6 }
} };

static constexpr std::array<BucketComboPair, TRIO_MAX_COMBOS_PER_BUCKET> TRIO_BUCKET_2 = { {
    { 8, 9 },
    { 10, 11 },
    { 8, 10 },
    { 9, 11 },
    { 10, 8 },
    { 9, 10 }
} };

static constexpr std::array<BucketComboPair, TRIO_MAX_COMBOS_PER_BUCKET> TRIO_BUCKET_3 = { {
    { 12, 13 },
    { 14, 15 },
    { 12, 14 },
    { 13, 15 },
    { 14, 12 },
    { 13, 14 }
} };

static constexpr std::array<BucketComboPair, TRIO_MAX_COMBOS_PER_BUCKET> TRIO_BUCKET_4 = { {
    { 16, 17 },
    { 18, 19 },
    { 16, 18 },
    { 17, 19 },
    { 18, 16 },
    { 17, 18 }
} };

static constexpr std::array<BucketComboPair, TRIO_MAX_COMBOS_PER_BUCKET> TRIO_BUCKET_5 = { {
    { 20, 21 },
    { 22, 23 },
    { 20, 22 },
    { 21, 23 },
    { 22, 20 },
    { 21, 22 }
} };

static constexpr std::array<BucketComboPair, TRIO_MAX_COMBOS_PER_BUCKET> TRIO_BUCKET_6 = { {
    { 24, 25 },
    { 26, 27 },
    { 24, 26 },
    { 25, 27 },
    { 26, 24 },
    { 25, 26 }
} };

static constexpr std::array<BucketComboPair, TRIO_MAX_COMBOS_PER_BUCKET> TRIO_BUCKET_7 = { {
    { 28, 29 },
    { 30, 31 },
    { 28, 30 },
    { 29, 31 },
    { 30, 28 },
    { 29, 30 }
} };

static constexpr std::array<BucketComboPair, TRIO_MAX_COMBOS_PER_BUCKET> TRIO_BUCKET_8 = { {
    { 32, 33 },
    { 34, 35 },
    { 32, 34 },
    { 33, 35 },
    { 34, 32 },
    { 33, 34 }
} };

static constexpr std::array<BucketComboPair, TRIO_MAX_COMBOS_PER_BUCKET> TRIO_BUCKET_9 = { {
    { 36, 37 },
    { 38, 39 },
    { 36, 38 },
    { 37, 39 },
    { 38, 36 },
    { 37, 38 }
} };

static constexpr std::array<BucketComboPair, TRIO_MAX_COMBOS_PER_BUCKET> TRIO_BUCKET_10 = { {
    { 40, 41 },
    { 42, 43 },
    { 40, 42 },
    { 41, 43 },
    { 42, 40 },
    { 41, 42 }
} };

static constexpr std::array<BucketComboPair, TRIO_MAX_COMBOS_PER_BUCKET> TRIO_BUCKET_11 = { {
    { 44, 45 },
    { 46, 47 },
    { 44, 46 },
    { 45, 47 },
    { 46, 44 },
    { 45, 46 }
} };

static constexpr std::array<BucketComboPair, TRIO_MAX_COMBOS_PER_BUCKET> TRIO_BUCKET_12 = { {
    { 48, 49 },
    { 50, 51 },
    { 48, 50 },
    { 49, 51 },
    { 50, 48 },
    { 49, 50 }
} };

static constexpr std::array<BucketComboPair, TRIO_MAX_COMBOS_PER_BUCKET> TRIO_BUCKET_13 = { {
    { 4, 0 },
    { 5, 1 },
    { 6, 2 },
    { 7, 3 },
    { -1, -1 },
    { -1, -1 }
} };

static constexpr std::array<BucketComboPair, TRIO_MAX_COMBOS_PER_BUCKET> TRIO_BUCKET_14 = { {
    { 4, 1 },
    { 6, 3 },
    { 5, 0 },
    { 7, 2 },
    { -1, -1 },
    { -1, -1 }
} };

static constexpr std::array<BucketComboPair, TRIO_MAX_COMBOS_PER_BUCKET> TRIO_BUCKET_15 = { {
    { 8, 0 },
    { 9, 1 },
    { 10, 2 },
    { 11, 3 },
    { -1, -1 },
    { -1, -1 }
} };

static constexpr std::array<BucketComboPair, TRIO_MAX_COMBOS_PER_BUCKET> TRIO_BUCKET_16 = { {
    { 8, 1 },
    { 10, 3 },
    { 9, 0 },
    { 11, 2 },
    { -1, -1 },
    { -1, -1 }
} };

static constexpr std::array<BucketComboPair, TRIO_MAX_COMBOS_PER_BUCKET> TRIO_BUCKET_17 = { {
    { 8, 4 },
    { 9, 5 },
    { 10, 6 },
    { 11, 7 },
    { -1, -1 },
    { -1, -1 }
} };

static constexpr std::array<BucketComboPair, TRIO_MAX_COMBOS_PER_BUCKET> TRIO_BUCKET_18 = { {
    { 8, 5 },
    { 10, 7 },
    { 9, 4 },
    { 11, 6 },
    { -1, -1 },
    { -1, -1 }
} };

static constexpr std::array<BucketComboPair, TRIO_MAX_COMBOS_PER_BUCKET> TRIO_BUCKET_19 = { {
    { 12, 0 },
    { 13, 1 },
    { 14, 2 },
    { 15, 3 },
    { -1, -1 },
    { -1, -1 }
} };

static constexpr std::array<BucketComboPair, TRIO_MAX_COMBOS_PER_BUCKET> TRIO_BUCKET_20 = { {
    { 12, 1 },
    { 14, 3 },
    { 13, 0 },
    { 15, 2 },
    { -1, -1 },
    { -1, -1 }
} };

static constexpr std::array<BucketComboPair, TRIO_MAX_COMBOS_PER_BUCKET> TRIO_BUCKET_21 = { {
    { 12, 4 },
    { 13, 5 },
    { 14, 6 },
    { 15, 7 },
    { -1, -1 },
    { -1, -1 }
} };

static constexpr std::array<BucketComboPair, TRIO_MAX_COMBOS_PER_BUCKET> TRIO_BUCKET_22 = { {
    { 12, 5 },
    { 14, 7 },
    { 13, 4 },
    { 15, 6 },
    { -1, -1 },
    { -1, -1 }
} };

static constexpr std::array<BucketComboPair, TRIO_MAX_COMBOS_PER_BUCKET> TRIO_BUCKET_23 = { {
    { 12, 8 },
    { 13, 9 },
    { 14, 10 },
    { 15, 11 },
    { -1, -1 },
    { -1, -1 }
} };

static constexpr std::array<BucketComboPair, TRIO_MAX_COMBOS_PER_BUCKET> TRIO_BUCKET_24 = { {
    { 12, 9 },
    { 14, 11 },
    { 13, 8 },
    { 15, 10 },
    { -1, -1 },
    { -1, -1 }
} };

static constexpr std::array<BucketComboPair, TRIO_MAX_COMBOS_PER_BUCKET> TRIO_BUCKET_25 = { {
    { 16, 0 },
    { 17, 1 },
    { 18, 2 },
    { 19, 3 },
    { -1, -1 },
    { -1, -1 }
} };

static constexpr std::array<BucketComboPair, TRIO_MAX_COMBOS_PER_BUCKET> TRIO_BUCKET_26 = { {
    { 16, 1 },
    { 18, 3 },
    { 17, 0 },
    { 19, 2 },
    { -1, -1 },
    { -1, -1 }
} };

static constexpr std::array<BucketComboPair, TRIO_MAX_COMBOS_PER_BUCKET> TRIO_BUCKET_27 = { {
    { 16, 4 },
    { 17, 5 },
    { 18, 6 },
    { 19, 7 },
    { -1, -1 },
    { -1, -1 }
} };

static constexpr std::array<BucketComboPair, TRIO_MAX_COMBOS_PER_BUCKET> TRIO_BUCKET_28 = { {
    { 16, 5 },
    { 18, 7 },
    { 17, 4 },
    { 19, 6 },
    { -1, -1 },
    { -1, -1 }
} };

static constexpr std::array<BucketComboPair, TRIO_MAX_COMBOS_PER_BUCKET> TRIO_BUCKET_29 = { {
    { 16, 8 },
    { 17, 9 },
    { 18, 10 },
    { 19, 11 },
    { -1, -1 },
    { -1, -1 }
} };

static constexpr std::array<BucketComboPair, TRIO_MAX_COMBOS_PER_BUCKET> TRIO_BUCKET_30 = { {
    { 16, 9 },
    { 18, 11 },
    { 17, 8 },
    { 19, 10 },
    { -1, -1 },
    { -1, -1 }
} };

static constexpr std::array<BucketComboPair, TRIO_MAX_COMBOS_PER_BUCKET> TRIO_BUCKET_31 = { {
    { 16, 12 },
    { 17, 13 },
    { 18, 14 },
    { 19, 15 },
    { -1, -1 },
    { -1, -1 }
} };

static constexpr std::array<BucketComboPair, TRIO_MAX_COMBOS_PER_BUCKET> TRIO_BUCKET_32 = { {
    { 16, 13 },
    { 18, 15 },
    { 17, 12 },
    { 19, 14 },
    { -1, -1 },
    { -1, -1 }
} };

static constexpr std::array<BucketComboPair, TRIO_MAX_COMBOS_PER_BUCKET> TRIO_BUCKET_33 = { {
    { 20, 0 },
    { 21, 1 },
    { 22, 2 },
    { 23, 3 },
    { -1, -1 },
    { -1, -1 }
} };

static constexpr std::array<BucketComboPair, TRIO_MAX_COMBOS_PER_BUCKET> TRIO_BUCKET_34 = { {
    { 20, 1 },
    { 22, 3 },
    { 21, 0 },
    { 23, 2 },
    { -1, -1 },
    { -1, -1 }
} };

static constexpr std::array<BucketComboPair, TRIO_MAX_COMBOS_PER_BUCKET> TRIO_BUCKET_35 = { {
    { 20, 4 },
    { 21, 5 },
    { 22, 6 },
    { 23, 7 },
    { -1, -1 },
    { -1, -1 }
} };

static constexpr std::array<BucketComboPair, TRIO_MAX_COMBOS_PER_BUCKET> TRIO_BUCKET_36 = { {
    { 20, 5 },
    { 22, 7 },
    { 21, 4 },
    { 23, 6 },
    { -1, -1 },
    { -1, -1 }
} };

static constexpr std::array<BucketComboPair, TRIO_MAX_COMBOS_PER_BUCKET> TRIO_BUCKET_37 = { {
    { 20, 8 },
    { 21, 9 },
    { 22, 10 },
    { 23, 11 },
    { -1, -1 },
    { -1, -1 }
} };

static constexpr std::array<BucketComboPair, TRIO_MAX_COMBOS_PER_BUCKET> TRIO_BUCKET_38 = { {
    { 20, 9 },
    { 22, 11 },
    { 21, 8 },
    { 23, 10 },
    { -1, -1 },
    { -1, -1 }
} };

static constexpr std::array<BucketComboPair, TRIO_MAX_COMBOS_PER_BUCKET> TRIO_BUCKET_39 = { {
    { 20, 12 },
    { 21, 13 },
    { 22, 14 },
    { 23, 15 },
    { -1, -1 },
    { -1, -1 }
} };

static constexpr std::array<BucketComboPair, TRIO_MAX_COMBOS_PER_BUCKET> TRIO_BUCKET_40 = { {
    { 20, 13 },
    { 22, 15 },
    { 21, 12 },
    { 23, 14 },
    { -1, -1 },
    { -1, -1 }
} };

static constexpr std::array<BucketComboPair, TRIO_MAX_COMBOS_PER_BUCKET> TRIO_BUCKET_41 = { {
    { 20, 16 },
    { 21, 17 },
    { 22, 18 },
    { 23, 19 },
    { -1, -1 },
    { -1, -1 }
} };

static constexpr std::array<BucketComboPair, TRIO_MAX_COMBOS_PER_BUCKET> TRIO_BUCKET_42 = { {
    { 20, 17 },
    { 22, 19 },
    { 21, 16 },
    { 23, 18 },
    { -1, -1 },
    { -1, -1 }
} };

static constexpr std::array<BucketComboPair, TRIO_MAX_COMBOS_PER_BUCKET> TRIO_BUCKET_43 = { {
    { 24, 0 },
    { 25, 1 },
    { 26, 2 },
    { 27, 3 },
    { -1, -1 },
    { -1, -1 }
} };

static constexpr std::array<BucketComboPair, TRIO_MAX_COMBOS_PER_BUCKET> TRIO_BUCKET_44 = { {
    { 24, 1 },
    { 26, 3 },
    { 25, 0 },
    { 27, 2 },
    { -1, -1 },
    { -1, -1 }
} };

static constexpr std::array<BucketComboPair, TRIO_MAX_COMBOS_PER_BUCKET> TRIO_BUCKET_45 = { {
    { 24, 4 },
    { 25, 5 },
    { 26, 6 },
    { 27, 7 },
    { -1, -1 },
    { -1, -1 }
} };

static constexpr std::array<BucketComboPair, TRIO_MAX_COMBOS_PER_BUCKET> TRIO_BUCKET_46 = { {
    { 24, 5 },
    { 26, 7 },
    { 25, 4 },
    { 27, 6 },
    { -1, -1 },
    { -1, -1 }
} };

static constexpr std::array<BucketComboPair, TRIO_MAX_COMBOS_PER_BUCKET> TRIO_BUCKET_47 = { {
    { 24, 8 },
    { 25, 9 },
    { 26, 10 },
    { 27, 11 },
    { -1, -1 },
    { -1, -1 }
} };

static constexpr std::array<BucketComboPair, TRIO_MAX_COMBOS_PER_BUCKET> TRIO_BUCKET_48 = { {
    { 24, 9 },
    { 26, 11 },
    { 25, 8 },
    { 27, 10 },
    { -1, -1 },
    { -1, -1 }
} };

static constexpr std::array<BucketComboPair, TRIO_MAX_COMBOS_PER_BUCKET> TRIO_BUCKET_49 = { {
    { 24, 12 },
    { 25, 13 },
    { 26, 14 },
    { 27, 15 },
    { -1, -1 },
    { -1, -1 }
} };

static constexpr std::array<BucketComboPair, TRIO_MAX_COMBOS_PER_BUCKET> TRIO_BUCKET_50 = { {
    { 24, 13 },
    { 26, 15 },
    { 25, 12 },
    { 27, 14 },
    { -1, -1 },
    { -1, -1 }
} };

static constexpr std::array<BucketComboPair, TRIO_MAX_COMBOS_PER_BUCKET> TRIO_BUCKET_51 = { {
    { 24, 16 },
    { 25, 17 },
    { 26, 18 },
    { 27, 19 },
    { -1, -1 },
    { -1, -1 }
} };

static constexpr std::array<BucketComboPair, TRIO_MAX_COMBOS_PER_BUCKET> TRIO_BUCKET_52 = { {
    { 24, 17 },
    { 26, 19 },
    { 25, 16 },
    { 27, 18 },
    { -1, -1 },
    { -1, -1 }
} };

static constexpr std::array<BucketComboPair, TRIO_MAX_COMBOS_PER_BUCKET> TRIO_BUCKET_53 = { {
    { 24, 20 },
    { 25, 21 },
    { 26, 22 },
    { 27, 23 },
    { -1, -1 },
    { -1, -1 }
} };

static constexpr std::array<BucketComboPair, TRIO_MAX_COMBOS_PER_BUCKET> TRIO_BUCKET_54 = { {
    { 24, 21 },
    { 26, 23 },
    { 25, 20 },
    { 27, 22 },
    { -1, -1 },
    { -1, -1 }
} };

static constexpr std::array<BucketComboPair, TRIO_MAX_COMBOS_PER_BUCKET> TRIO_BUCKET_55 = { {
    { 28, 0 },
    { 29, 1 },
    { 30, 2 },
    { 31, 3 },
    { -1, -1 },
    { -1, -1 }
} };

static constexpr std::array<BucketComboPair, TRIO_MAX_COMBOS_PER_BUCKET> TRIO_BUCKET_56 = { {
    { 28, 1 },
    { 30, 3 },
    { 29, 0 },
    { 31, 2 },
    { -1, -1 },
    { -1, -1 }
} };

static constexpr std::array<BucketComboPair, TRIO_MAX_COMBOS_PER_BUCKET> TRIO_BUCKET_57 = { {
    { 28, 4 },
    { 29, 5 },
    { 30, 6 },
    { 31, 7 },
    { -1, -1 },
    { -1, -1 }
} };

static constexpr std::array<BucketComboPair, TRIO_MAX_COMBOS_PER_BUCKET> TRIO_BUCKET_58 = { {
    { 28, 5 },
    { 30, 7 },
    { 29, 4 },
    { 31, 6 },
    { -1, -1 },
    { -1, -1 }
} };

static constexpr std::array<BucketComboPair, TRIO_MAX_COMBOS_PER_BUCKET> TRIO_BUCKET_59 = { {
    { 28, 8 },
    { 29, 9 },
    { 30, 10 },
    { 31, 11 },
    { -1, -1 },
    { -1, -1 }
} };

static constexpr std::array<BucketComboPair, TRIO_MAX_COMBOS_PER_BUCKET> TRIO_BUCKET_60 = { {
    { 28, 9 },
    { 30, 11 },
    { 29, 8 },
    { 31, 10 },
    { -1, -1 },
    { -1, -1 }
} };

static constexpr std::array<BucketComboPair, TRIO_MAX_COMBOS_PER_BUCKET> TRIO_BUCKET_61 = { {
    { 28, 12 },
    { 29, 13 },
    { 30, 14 },
    { 31, 15 },
    { -1, -1 },
    { -1, -1 }
} };

static constexpr std::array<BucketComboPair, TRIO_MAX_COMBOS_PER_BUCKET> TRIO_BUCKET_62 = { {
    { 28, 13 },
    { 30, 15 },
    { 29, 12 },
    { 31, 14 },
    { -1, -1 },
    { -1, -1 }
} };

static constexpr std::array<BucketComboPair, TRIO_MAX_COMBOS_PER_BUCKET> TRIO_BUCKET_63 = { {
    { 28, 16 },
    { 29, 17 },
    { 30, 18 },
    { 31, 19 },
    { -1, -1 },
    { -1, -1 }
} };

static constexpr std::array<BucketComboPair, TRIO_MAX_COMBOS_PER_BUCKET> TRIO_BUCKET_64 = { {
    { 28, 17 },
    { 30, 19 },
    { 29, 16 },
    { 31, 18 },
    { -1, -1 },
    { -1, -1 }
} };

static constexpr std::array<BucketComboPair, TRIO_MAX_COMBOS_PER_BUCKET> TRIO_BUCKET_65 = { {
    { 28, 20 },
    { 29, 21 },
    { 30, 22 },
    { 31, 23 },
    { -1, -1 },
    { -1, -1 }
} };

static constexpr std::array<BucketComboPair, TRIO_MAX_COMBOS_PER_BUCKET> TRIO_BUCKET_66 = { {
    { 28, 21 },
    { 30, 23 },
    { 29, 20 },
    { 31, 22 },
    { -1, -1 },
    { -1, -1 }
} };

static constexpr std::array<BucketComboPair, TRIO_MAX_COMBOS_PER_BUCKET> TRIO_BUCKET_67 = { {
    { 28, 24 },
    { 29, 25 },
    { 30, 26 },
    { 31, 27 },
    { -1, -1 },
    { -1, -1 }
} };

static constexpr std::array<BucketComboPair, TRIO_MAX_COMBOS_PER_BUCKET> TRIO_BUCKET_68 = { {
    { 28, 25 },
    { 30, 27 },
    { 29, 24 },
    { 31, 26 },
    { -1, -1 },
    { -1, -1 }
} };

static constexpr std::array<BucketComboPair, TRIO_MAX_COMBOS_PER_BUCKET> TRIO_BUCKET_69 = { {
    { 32, 0 },
    { 33, 1 },
    { 34, 2 },
    { 35, 3 },
    { -1, -1 },
    { -1, -1 }
} };

static constexpr std::array<BucketComboPair, TRIO_MAX_COMBOS_PER_BUCKET> TRIO_BUCKET_70 = { {
    { 32, 1 },
    { 34, 3 },
    { 33, 0 },
    { 35, 2 },
    { -1, -1 },
    { -1, -1 }
} };

static constexpr std::array<BucketComboPair, TRIO_MAX_COMBOS_PER_BUCKET> TRIO_BUCKET_71 = { {
    { 32, 4 },
    { 33, 5 },
    { 34, 6 },
    { 35, 7 },
    { -1, -1 },
    { -1, -1 }
} };

static constexpr std::array<BucketComboPair, TRIO_MAX_COMBOS_PER_BUCKET> TRIO_BUCKET_72 = { {
    { 32, 5 },
    { 34, 7 },
    { 33, 4 },
    { 35, 6 },
    { -1, -1 },
    { -1, -1 }
} };

static constexpr std::array<BucketComboPair, TRIO_MAX_COMBOS_PER_BUCKET> TRIO_BUCKET_73 = { {
    { 32, 8 },
    { 33, 9 },
    { 34, 10 },
    { 35, 11 },
    { -1, -1 },
    { -1, -1 }
} };

static constexpr std::array<BucketComboPair, TRIO_MAX_COMBOS_PER_BUCKET> TRIO_BUCKET_74 = { {
    { 32, 9 },
    { 34, 11 },
    { 33, 8 },
    { 35, 10 },
    { -1, -1 },
    { -1, -1 }
} };

static constexpr std::array<BucketComboPair, TRIO_MAX_COMBOS_PER_BUCKET> TRIO_BUCKET_75 = { {
    { 32, 12 },
    { 33, 13 },
    { 34, 14 },
    { 35, 15 },
    { -1, -1 },
    { -1, -1 }
} };

static constexpr std::array<BucketComboPair, TRIO_MAX_COMBOS_PER_BUCKET> TRIO_BUCKET_76 = { {
    { 32, 13 },
    { 34, 15 },
    { 33, 12 },
    { 35, 14 },
    { -1, -1 },
    { -1, -1 }
} };

static constexpr std::array<BucketComboPair, TRIO_MAX_COMBOS_PER_BUCKET> TRIO_BUCKET_77 = { {
    { 32, 16 },
    { 33, 17 },
    { 34, 18 },
    { 35, 19 },
    { -1, -1 },
    { -1, -1 }
} };

static constexpr std::array<BucketComboPair, TRIO_MAX_COMBOS_PER_BUCKET> TRIO_BUCKET_78 = { {
    { 32, 17 },
    { 34, 19 },
    { 33, 16 },
    { 35, 18 },
    { -1, -1 },
    { -1, -1 }
} };

static constexpr std::array<BucketComboPair, TRIO_MAX_COMBOS_PER_BUCKET> TRIO_BUCKET_79 = { {
    { 32, 20 },
    { 33, 21 },
    { 34, 22 },
    { 35, 23 },
    { -1, -1 },
    { -1, -1 }
} };

static constexpr std::array<BucketComboPair, TRIO_MAX_COMBOS_PER_BUCKET> TRIO_BUCKET_80 = { {
    { 32, 21 },
    { 34, 23 },
    { 33, 20 },
    { 35, 22 },
    { -1, -1 },
    { -1, -1 }
} };

static constexpr std::array<BucketComboPair, TRIO_MAX_COMBOS_PER_BUCKET> TRIO_BUCKET_81 = { {
    { 32, 24 },
    { 33, 25 },
    { 34, 26 },
    { 35, 27 },
    { -1, -1 },
    { -1, -1 }
} };

static constexpr std::array<BucketComboPair, TRIO_MAX_COMBOS_PER_BUCKET> TRIO_BUCKET_82 = { {
    { 32, 25 },
    { 34, 27 },
    { 33, 24 },
    { 35, 26 },
    { -1, -1 },
    { -1, -1 }
} };

static constexpr std::array<BucketComboPair, TRIO_MAX_COMBOS_PER_BUCKET> TRIO_BUCKET_83 = { {
    { 32, 28 },
    { 33, 29 },
    { 34, 30 },
    { 35, 31 },
    { -1, -1 },
    { -1, -1 }
} };

static constexpr std::array<BucketComboPair, TRIO_MAX_COMBOS_PER_BUCKET> TRIO_BUCKET_84 = { {
    { 32, 29 },
    { 34, 31 },
    { 33, 28 },
    { 35, 30 },
    { -1, -1 },
    { -1, -1 }
} };

static constexpr std::array<BucketComboPair, TRIO_MAX_COMBOS_PER_BUCKET> TRIO_BUCKET_85 = { {
    { 36, 0 },
    { 37, 1 },
    { 38, 2 },
    { 39, 3 },
    { -1, -1 },
    { -1, -1 }
} };

static constexpr std::array<BucketComboPair, TRIO_MAX_COMBOS_PER_BUCKET> TRIO_BUCKET_86 = { {
    { 36, 1 },
    { 38, 3 },
    { 37, 0 },
    { 39, 2 },
    { -1, -1 },
    { -1, -1 }
} };

static constexpr std::array<BucketComboPair, TRIO_MAX_COMBOS_PER_BUCKET> TRIO_BUCKET_87 = { {
    { 36, 4 },
    { 37, 5 },
    { 38, 6 },
    { 39, 7 },
    { -1, -1 },
    { -1, -1 }
} };

static constexpr std::array<BucketComboPair, TRIO_MAX_COMBOS_PER_BUCKET> TRIO_BUCKET_88 = { {
    { 36, 5 },
    { 38, 7 },
    { 37, 4 },
    { 39, 6 },
    { -1, -1 },
    { -1, -1 }
} };

static constexpr std::array<BucketComboPair, TRIO_MAX_COMBOS_PER_BUCKET> TRIO_BUCKET_89 = { {
    { 36, 8 },
    { 37, 9 },
    { 38, 10 },
    { 39, 11 },
    { -1, -1 },
    { -1, -1 }
} };

static constexpr std::array<BucketComboPair, TRIO_MAX_COMBOS_PER_BUCKET> TRIO_BUCKET_90 = { {
    { 36, 9 },
    { 38, 11 },
    { 37, 8 },
    { 39, 10 },
    { -1, -1 },
    { -1, -1 }
} };

static constexpr std::array<BucketComboPair, TRIO_MAX_COMBOS_PER_BUCKET> TRIO_BUCKET_91 = { {
    { 36, 12 },
    { 37, 13 },
    { 38, 14 },
    { 39, 15 },
    { -1, -1 },
    { -1, -1 }
} };

static constexpr std::array<BucketComboPair, TRIO_MAX_COMBOS_PER_BUCKET> TRIO_BUCKET_92 = { {
    { 36, 13 },
    { 38, 15 },
    { 37, 12 },
    { 39, 14 },
    { -1, -1 },
    { -1, -1 }
} };

static constexpr std::array<BucketComboPair, TRIO_MAX_COMBOS_PER_BUCKET> TRIO_BUCKET_93 = { {
    { 36, 16 },
    { 37, 17 },
    { 38, 18 },
    { 39, 19 },
    { -1, -1 },
    { -1, -1 }
} };

static constexpr std::array<BucketComboPair, TRIO_MAX_COMBOS_PER_BUCKET> TRIO_BUCKET_94 = { {
    { 36, 17 },
    { 38, 19 },
    { 37, 16 },
    { 39, 18 },
    { -1, -1 },
    { -1, -1 }
} };

static constexpr std::array<BucketComboPair, TRIO_MAX_COMBOS_PER_BUCKET> TRIO_BUCKET_95 = { {
    { 36, 20 },
    { 37, 21 },
    { 38, 22 },
    { 39, 23 },
    { -1, -1 },
    { -1, -1 }
} };

static constexpr std::array<BucketComboPair, TRIO_MAX_COMBOS_PER_BUCKET> TRIO_BUCKET_96 = { {
    { 36, 21 },
    { 38, 23 },
    { 37, 20 },
    { 39, 22 },
    { -1, -1 },
    { -1, -1 }
} };

static constexpr std::array<BucketComboPair, TRIO_MAX_COMBOS_PER_BUCKET> TRIO_BUCKET_97 = { {
    { 36, 24 },
    { 37, 25 },
    { 38, 26 },
    { 39, 27 },
    { -1, -1 },
    { -1, -1 }
} };

static constexpr std::array<BucketComboPair, TRIO_MAX_COMBOS_PER_BUCKET> TRIO_BUCKET_98 = { {
    { 36, 25 },
    { 38, 27 },
    { 37, 24 },
    { 39, 26 },
    { -1, -1 },
    { -1, -1 }
} };

static constexpr std::array<BucketComboPair, TRIO_MAX_COMBOS_PER_BUCKET> TRIO_BUCKET_99 = { {
    { 36, 28 },
    { 37, 29 },
    { 38, 30 },
    { 39, 31 },
    { -1, -1 },
    { -1, -1 }
} };

static constexpr std::array<BucketComboPair, TRIO_MAX_COMBOS_PER_BUCKET> TRIO_BUCKET_100 = { {
    { 36, 29 },
    { 38, 31 },
    { 37, 28 },
    { 39, 30 },
    { -1, -1 },
    { -1, -1 }
} };

static constexpr std::array<BucketComboPair, TRIO_MAX_COMBOS_PER_BUCKET> TRIO_BUCKET_101 = { {
    { 36, 32 },
    { 37, 33 },
    { 38, 34 },
    { 39, 35 },
    { -1, -1 },
    { -1, -1 }
} };

static constexpr std::array<BucketComboPair, TRIO_MAX_COMBOS_PER_BUCKET> TRIO_BUCKET_102 = { {
    { 36, 33 },
    { 38, 35 },
    { 37, 32 },
    { 39, 34 },
    { -1, -1 },
    { -1, -1 }
} };

static constexpr std::array<BucketComboPair, TRIO_MAX_COMBOS_PER_BUCKET> TRIO_BUCKET_103 = { {
    { 40, 0 },
    { 41, 1 },
    { 42, 2 },
    { 43, 3 },
    { -1, -1 },
    { -1, -1 }
} };

static constexpr std::array<BucketComboPair, TRIO_MAX_COMBOS_PER_BUCKET> TRIO_BUCKET_104 = { {
    { 40, 1 },
    { 42, 3 },
    { 41, 0 },
    { 43, 2 },
    { -1, -1 },
    { -1, -1 }
} };

static constexpr std::array<BucketComboPair, TRIO_MAX_COMBOS_PER_BUCKET> TRIO_BUCKET_105 = { {
    { 40, 4 },
    { 41, 5 },
    { 42, 6 },
    { 43, 7 },
    { -1, -1 },
    { -1, -1 }
} };

static constexpr std::array<BucketComboPair, TRIO_MAX_COMBOS_PER_BUCKET> TRIO_BUCKET_106 = { {
    { 40, 5 },
    { 42, 7 },
    { 41, 4 },
    { 43, 6 },
    { -1, -1 },
    { -1, -1 }
} };

static constexpr std::array<BucketComboPair, TRIO_MAX_COMBOS_PER_BUCKET> TRIO_BUCKET_107 = { {
    { 40, 8 },
    { 41, 9 },
    { 42, 10 },
    { 43, 11 },
    { -1, -1 },
    { -1, -1 }
} };

static constexpr std::array<BucketComboPair, TRIO_MAX_COMBOS_PER_BUCKET> TRIO_BUCKET_108 = { {
    { 40, 9 },
    { 42, 11 },
    { 41, 8 },
    { 43, 10 },
    { -1, -1 },
    { -1, -1 }
} };

static constexpr std::array<BucketComboPair, TRIO_MAX_COMBOS_PER_BUCKET> TRIO_BUCKET_109 = { {
    { 40, 12 },
    { 41, 13 },
    { 42, 14 },
    { 43, 15 },
    { -1, -1 },
    { -1, -1 }
} };

static constexpr std::array<BucketComboPair, TRIO_MAX_COMBOS_PER_BUCKET> TRIO_BUCKET_110 = { {
    { 40, 13 },
    { 42, 15 },
    { 41, 12 },
    { 43, 14 },
    { -1, -1 },
    { -1, -1 }
} };

static constexpr std::array<BucketComboPair, TRIO_MAX_COMBOS_PER_BUCKET> TRIO_BUCKET_111 = { {
    { 40, 16 },
    { 41, 17 },
    { 42, 18 },
    { 43, 19 },
    { -1, -1 },
    { -1, -1 }
} };

static constexpr std::array<BucketComboPair, TRIO_MAX_COMBOS_PER_BUCKET> TRIO_BUCKET_112 = { {
    { 40, 17 },
    { 42, 19 },
    { 41, 16 },
    { 43, 18 },
    { -1, -1 },
    { -1, -1 }
} };

static constexpr std::array<BucketComboPair, TRIO_MAX_COMBOS_PER_BUCKET> TRIO_BUCKET_113 = { {
    { 40, 20 },
    { 41, 21 },
    { 42, 22 },
    { 43, 23 },
    { -1, -1 },
    { -1, -1 }
} };

static constexpr std::array<BucketComboPair, TRIO_MAX_COMBOS_PER_BUCKET> TRIO_BUCKET_114 = { {
    { 40, 21 },
    { 42, 23 },
    { 41, 20 },
    { 43, 22 },
    { -1, -1 },
    { -1, -1 }
} };

static constexpr std::array<BucketComboPair, TRIO_MAX_COMBOS_PER_BUCKET> TRIO_BUCKET_115 = { {
    { 40, 24 },
    { 41, 25 },
    { 42, 26 },
    { 43, 27 },
    { -1, -1 },
    { -1, -1 }
} };

static constexpr std::array<BucketComboPair, TRIO_MAX_COMBOS_PER_BUCKET> TRIO_BUCKET_116 = { {
    { 40, 25 },
    { 42, 27 },
    { 41, 24 },
    { 43, 26 },
    { -1, -1 },
    { -1, -1 }
} };

static constexpr std::array<BucketComboPair, TRIO_MAX_COMBOS_PER_BUCKET> TRIO_BUCKET_117 = { {
    { 40, 28 },
    { 41, 29 },
    { 42, 30 },
    { 43, 31 },
    { -1, -1 },
    { -1, -1 }
} };

static constexpr std::array<BucketComboPair, TRIO_MAX_COMBOS_PER_BUCKET> TRIO_BUCKET_118 = { {
    { 40, 29 },
    { 42, 31 },
    { 41, 28 },
    { 43, 30 },
    { -1, -1 },
    { -1, -1 }
} };

static constexpr std::array<BucketComboPair, TRIO_MAX_COMBOS_PER_BUCKET> TRIO_BUCKET_119 = { {
    { 40, 32 },
    { 41, 33 },
    { 42, 34 },
    { 43, 35 },
    { -1, -1 },
    { -1, -1 }
} };

static constexpr std::array<BucketComboPair, TRIO_MAX_COMBOS_PER_BUCKET> TRIO_BUCKET_120 = { {
    { 40, 33 },
    { 42, 35 },
    { 41, 32 },
    { 43, 34 },
    { -1, -1 },
    { -1, -1 }
} };

static constexpr std::array<BucketComboPair, TRIO_MAX_COMBOS_PER_BUCKET> TRIO_BUCKET_121 = { {
    { 40, 36 },
    { 41, 37 },
    { 42, 38 },
    { 43, 39 },
    { -1, -1 },
    { -1, -1 }
} };

static constexpr std::array<BucketComboPair, TRIO_MAX_COMBOS_PER_BUCKET> TRIO_BUCKET_122 = { {
    { 40, 37 },
    { 42, 39 },
    { 41, 36 },
    { 43, 38 },
    { -1, -1 },
    { -1, -1 }
} };

static constexpr std::array<BucketComboPair, TRIO_MAX_COMBOS_PER_BUCKET> TRIO_BUCKET_123 = { {
    { 44, 0 },
    { 45, 1 },
    { 46, 2 },
    { 47, 3 },
    { -1, -1 },
    { -1, -1 }
} };

static constexpr std::array<BucketComboPair, TRIO_MAX_COMBOS_PER_BUCKET> TRIO_BUCKET_124 = { {
    { 44, 1 },
    { 46, 3 },
    { 45, 0 },
    { 47, 2 },
    { -1, -1 },
    { -1, -1 }
} };

static constexpr std::array<BucketComboPair, TRIO_MAX_COMBOS_PER_BUCKET> TRIO_BUCKET_125 = { {
    { 44, 4 },
    { 45, 5 },
    { 46, 6 },
    { 47, 7 },
    { -1, -1 },
    { -1, -1 }
} };

static constexpr std::array<BucketComboPair, TRIO_MAX_COMBOS_PER_BUCKET> TRIO_BUCKET_126 = { {
    { 44, 5 },
    { 46, 7 },
    { 45, 4 },
    { 47, 6 },
    { -1, -1 },
    { -1, -1 }
} };

static constexpr std::array<BucketComboPair, TRIO_MAX_COMBOS_PER_BUCKET> TRIO_BUCKET_127 = { {
    { 44, 8 },
    { 45, 9 },
    { 46, 10 },
    { 47, 11 },
    { -1, -1 },
    { -1, -1 }
} };

static constexpr std::array<BucketComboPair, TRIO_MAX_COMBOS_PER_BUCKET> TRIO_BUCKET_128 = { {
    { 44, 9 },
    { 46, 11 },
    { 45, 8 },
    { 47, 10 },
    { -1, -1 },
    { -1, -1 }
} };

static constexpr std::array<BucketComboPair, TRIO_MAX_COMBOS_PER_BUCKET> TRIO_BUCKET_129 = { {
    { 44, 12 },
    { 45, 13 },
    { 46, 14 },
    { 47, 15 },
    { -1, -1 },
    { -1, -1 }
} };

static constexpr std::array<BucketComboPair, TRIO_MAX_COMBOS_PER_BUCKET> TRIO_BUCKET_130 = { {
    { 44, 13 },
    { 46, 15 },
    { 45, 12 },
    { 47, 14 },
    { -1, -1 },
    { -1, -1 }
} };

static constexpr std::array<BucketComboPair, TRIO_MAX_COMBOS_PER_BUCKET> TRIO_BUCKET_131 = { {
    { 44, 16 },
    { 45, 17 },
    { 46, 18 },
    { 47, 19 },
    { -1, -1 },
    { -1, -1 }
} };

static constexpr std::array<BucketComboPair, TRIO_MAX_COMBOS_PER_BUCKET> TRIO_BUCKET_132 = { {
    { 44, 17 },
    { 46, 19 },
    { 45, 16 },
    { 47, 18 },
    { -1, -1 },
    { -1, -1 }
} };

static constexpr std::array<BucketComboPair, TRIO_MAX_COMBOS_PER_BUCKET> TRIO_BUCKET_133 = { {
    { 44, 20 },
    { 45, 21 },
    { 46, 22 },
    { 47, 23 },
    { -1, -1 },
    { -1, -1 }
} };

static constexpr std::array<BucketComboPair, TRIO_MAX_COMBOS_PER_BUCKET> TRIO_BUCKET_134 = { {
    { 44, 21 },
    { 46, 23 },
    { 45, 20 },
    { 47, 22 },
    { -1, -1 },
    { -1, -1 }
} };

static constexpr std::array<BucketComboPair, TRIO_MAX_COMBOS_PER_BUCKET> TRIO_BUCKET_135 = { {
    { 44, 24 },
    { 45, 25 },
    { 46, 26 },
    { 47, 27 },
    { -1, -1 },
    { -1, -1 }
} };

static constexpr std::array<BucketComboPair, TRIO_MAX_COMBOS_PER_BUCKET> TRIO_BUCKET_136 = { {
    { 44, 25 },
    { 46, 27 },
    { 45, 24 },
    { 47, 26 },
    { -1, -1 },
    { -1, -1 }
} };

static constexpr std::array<BucketComboPair, TRIO_MAX_COMBOS_PER_BUCKET> TRIO_BUCKET_137 = { {
    { 44, 28 },
    { 45, 29 },
    { 46, 30 },
    { 47, 31 },
    { -1, -1 },
    { -1, -1 }
} };

static constexpr std::array<BucketComboPair, TRIO_MAX_COMBOS_PER_BUCKET> TRIO_BUCKET_138 = { {
    { 44, 29 },
    { 46, 31 },
    { 45, 28 },
    { 47, 30 },
    { -1, -1 },
    { -1, -1 }
} };

static constexpr std::array<BucketComboPair, TRIO_MAX_COMBOS_PER_BUCKET> TRIO_BUCKET_139 = { {
    { 44, 32 },
    { 45, 33 },
    { 46, 34 },
    { 47, 35 },
    { -1, -1 },
    { -1, -1 }
} };

static constexpr std::array<BucketComboPair, TRIO_MAX_COMBOS_PER_BUCKET> TRIO_BUCKET_140 = { {
    { 44, 33 },
    { 46, 35 },
    { 45, 32 },
    { 47, 34 },
    { -1, -1 },
    { -1, -1 }
} };

static constexpr std::array<BucketComboPair, TRIO_MAX_COMBOS_PER_BUCKET> TRIO_BUCKET_141 = { {
    { 44, 36 },
    { 45, 37 },
    { 46, 38 },
    { 47, 39 },
    { -1, -1 },
    { -1, -1 }
} };

static constexpr std::array<BucketComboPair, TRIO_MAX_COMBOS_PER_BUCKET> TRIO_BUCKET_142 = { {
    { 44, 37 },
    { 46, 39 },
    { 45, 36 },
    { 47, 38 },
    { -1, -1 },
    { -1, -1 }
} };

static constexpr std::array<BucketComboPair, TRIO_MAX_COMBOS_PER_BUCKET> TRIO_BUCKET_143 = { {
    { 44, 40 },
    { 45, 41 },
    { 46, 42 },
    { 47, 43 },
    { -1, -1 },
    { -1, -1 }
} };

static constexpr std::array<BucketComboPair, TRIO_MAX_COMBOS_PER_BUCKET> TRIO_BUCKET_144 = { {
    { 44, 41 },
    { 46, 43 },
    { 45, 40 },
    { 47, 42 },
    { -1, -1 },
    { -1, -1 }
} };

static constexpr std::array<BucketComboPair, TRIO_MAX_COMBOS_PER_BUCKET> TRIO_BUCKET_145 = { {
    { 48, 0 },
    { 49, 1 },
    { 50, 2 },
    { 51, 3 },
    { -1, -1 },
    { -1, -1 }
} };

static constexpr std::array<BucketComboPair, TRIO_MAX_COMBOS_PER_BUCKET> TRIO_BUCKET_146 = { {
    { 48, 1 },
    { 50, 3 },
    { 49, 0 },
    { 51, 2 },
    { -1, -1 },
    { -1, -1 }
} };

static constexpr std::array<BucketComboPair, TRIO_MAX_COMBOS_PER_BUCKET> TRIO_BUCKET_147 = { {
    { 48, 4 },
    { 49, 5 },
    { 50, 6 },
    { 51, 7 },
    { -1, -1 },
    { -1, -1 }
} };

static constexpr std::array<BucketComboPair, TRIO_MAX_COMBOS_PER_BUCKET> TRIO_BUCKET_148 = { {
    { 48, 5 },
    { 50, 7 },
    { 49, 4 },
    { 51, 6 },
    { -1, -1 },
    { -1, -1 }
} };

static constexpr std::array<BucketComboPair, TRIO_MAX_COMBOS_PER_BUCKET> TRIO_BUCKET_149 = { {
    { 48, 8 },
    { 49, 9 },
    { 50, 10 },
    { 51, 11 },
    { -1, -1 },
    { -1, -1 }
} };

static constexpr std::array<BucketComboPair, TRIO_MAX_COMBOS_PER_BUCKET> TRIO_BUCKET_150 = { {
    { 48, 9 },
    { 50, 11 },
    { 49, 8 },
    { 51, 10 },
    { -1, -1 },
    { -1, -1 }
} };

static constexpr std::array<BucketComboPair, TRIO_MAX_COMBOS_PER_BUCKET> TRIO_BUCKET_151 = { {
    { 48, 12 },
    { 49, 13 },
    { 50, 14 },
    { 51, 15 },
    { -1, -1 },
    { -1, -1 }
} };

static constexpr std::array<BucketComboPair, TRIO_MAX_COMBOS_PER_BUCKET> TRIO_BUCKET_152 = { {
    { 48, 13 },
    { 50, 15 },
    { 49, 12 },
    { 51, 14 },
    { -1, -1 },
    { -1, -1 }
} };

static constexpr std::array<BucketComboPair, TRIO_MAX_COMBOS_PER_BUCKET> TRIO_BUCKET_153 = { {
    { 48, 16 },
    { 49, 17 },
    { 50, 18 },
    { 51, 19 },
    { -1, -1 },
    { -1, -1 }
} };

static constexpr std::array<BucketComboPair, TRIO_MAX_COMBOS_PER_BUCKET> TRIO_BUCKET_154 = { {
    { 48, 17 },
    { 50, 19 },
    { 49, 16 },
    { 51, 18 },
    { -1, -1 },
    { -1, -1 }
} };

static constexpr std::array<BucketComboPair, TRIO_MAX_COMBOS_PER_BUCKET> TRIO_BUCKET_155 = { {
    { 48, 20 },
    { 49, 21 },
    { 50, 22 },
    { 51, 23 },
    { -1, -1 },
    { -1, -1 }
} };

static constexpr std::array<BucketComboPair, TRIO_MAX_COMBOS_PER_BUCKET> TRIO_BUCKET_156 = { {
    { 48, 21 },
    { 50, 23 },
    { 49, 20 },
    { 51, 22 },
    { -1, -1 },
    { -1, -1 }
} };

static constexpr std::array<BucketComboPair, TRIO_MAX_COMBOS_PER_BUCKET> TRIO_BUCKET_157 = { {
    { 48, 24 },
    { 49, 25 },
    { 50, 26 },
    { 51, 27 },
    { -1, -1 },
    { -1, -1 }
} };

static constexpr std::array<BucketComboPair, TRIO_MAX_COMBOS_PER_BUCKET> TRIO_BUCKET_158 = { {
    { 48, 25 },
    { 50, 27 },
    { 49, 24 },
    { 51, 26 },
    { -1, -1 },
    { -1, -1 }
} };

static constexpr std::array<BucketComboPair, TRIO_MAX_COMBOS_PER_BUCKET> TRIO_BUCKET_159 = { {
    { 48, 28 },
    { 49, 29 },
    { 50, 30 },
    { 51, 31 },
    { -1, -1 },
    { -1, -1 }
} };

static constexpr std::array<BucketComboPair, TRIO_MAX_COMBOS_PER_BUCKET> TRIO_BUCKET_160 = { {
    { 48, 29 },
    { 50, 31 },
    { 49, 28 },
    { 51, 30 },
    { -1, -1 },
    { -1, -1 }
} };

static constexpr std::array<BucketComboPair, TRIO_MAX_COMBOS_PER_BUCKET> TRIO_BUCKET_161 = { {
    { 48, 32 },
    { 49, 33 },
    { 50, 34 },
    { 51, 35 },
    { -1, -1 },
    { -1, -1 }
} };

static constexpr std::array<BucketComboPair, TRIO_MAX_COMBOS_PER_BUCKET> TRIO_BUCKET_162 = { {
    { 48, 33 },
    { 50, 35 },
    { 49, 32 },
    { 51, 34 },
    { -1, -1 },
    { -1, -1 }
} };

static constexpr std::array<BucketComboPair, TRIO_MAX_COMBOS_PER_BUCKET> TRIO_BUCKET_163 = { {
    { 48, 36 },
    { 49, 37 },
    { 50, 38 },
    { 51, 39 },
    { -1, -1 },
    { -1, -1 }
} };

static constexpr std::array<BucketComboPair, TRIO_MAX_COMBOS_PER_BUCKET> TRIO_BUCKET_164 = { {
    { 48, 37 },
    { 50, 39 },
    { 49, 36 },
    { 51, 38 },
    { -1, -1 },
    { -1, -1 }
} };

static constexpr std::array<BucketComboPair, TRIO_MAX_COMBOS_PER_BUCKET> TRIO_BUCKET_165 = { {
    { 48, 40 },
    { 49, 41 },
    { 50, 42 },
    { 51, 43 },
    { -1, -1 },
    { -1, -1 }
} };

static constexpr std::array<BucketComboPair, TRIO_MAX_COMBOS_PER_BUCKET> TRIO_BUCKET_166 = { {
    { 48, 41 },
    { 50, 43 },
    { 49, 40 },
    { 51, 42 },
    { -1, -1 },
    { -1, -1 }
} };

static constexpr std::array<BucketComboPair, TRIO_MAX_COMBOS_PER_BUCKET> TRIO_BUCKET_167 = { {
    { 48, 44 },
    { 49, 45 },
    { 50, 46 },
    { 51, 47 },
    { -1, -1 },
    { -1, -1 }
} };

static constexpr std::array<BucketComboPair, TRIO_MAX_COMBOS_PER_BUCKET> TRIO_BUCKET_168 = { {
    { 48, 45 },
    { 50, 47 },
    { 49, 44 },
    { 51, 46 },
    { -1, -1 },
    { -1, -1 }
} };

static constexpr std::array<int, TRIO_N_BUCKETS> TRIO_BUCKET_NCOMBOS = {
    6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4
};

static constexpr std::array<const BucketComboPair*, TRIO_N_BUCKETS> TRIO_BUCKET_COMBOS = {
    TRIO_BUCKET_0.data(), TRIO_BUCKET_1.data(), TRIO_BUCKET_2.data(), TRIO_BUCKET_3.data(), TRIO_BUCKET_4.data(), TRIO_BUCKET_5.data(), TRIO_BUCKET_6.data(), TRIO_BUCKET_7.data(), TRIO_BUCKET_8.data(), TRIO_BUCKET_9.data(), TRIO_BUCKET_10.data(), TRIO_BUCKET_11.data(), TRIO_BUCKET_12.data(), TRIO_BUCKET_13.data(), TRIO_BUCKET_14.data(), TRIO_BUCKET_15.data(), TRIO_BUCKET_16.data(), TRIO_BUCKET_17.data(), TRIO_BUCKET_18.data(), TRIO_BUCKET_19.data(), TRIO_BUCKET_20.data(), TRIO_BUCKET_21.data(), TRIO_BUCKET_22.data(), TRIO_BUCKET_23.data(), TRIO_BUCKET_24.data(), TRIO_BUCKET_25.data(), TRIO_BUCKET_26.data(), TRIO_BUCKET_27.data(), TRIO_BUCKET_28.data(), TRIO_BUCKET_29.data(), TRIO_BUCKET_30.data(), TRIO_BUCKET_31.data(), TRIO_BUCKET_32.data(), TRIO_BUCKET_33.data(), TRIO_BUCKET_34.data(), TRIO_BUCKET_35.data(), TRIO_BUCKET_36.data(), TRIO_BUCKET_37.data(), TRIO_BUCKET_38.data(), TRIO_BUCKET_39.data(), TRIO_BUCKET_40.data(), TRIO_BUCKET_41.data(), TRIO_BUCKET_42.data(), TRIO_BUCKET_43.data(), TRIO_BUCKET_44.data(), TRIO_BUCKET_45.data(), TRIO_BUCKET_46.data(), TRIO_BUCKET_47.data(), TRIO_BUCKET_48.data(), TRIO_BUCKET_49.data(), TRIO_BUCKET_50.data(), TRIO_BUCKET_51.data(), TRIO_BUCKET_52.data(), TRIO_BUCKET_53.data(), TRIO_BUCKET_54.data(), TRIO_BUCKET_55.data(), TRIO_BUCKET_56.data(), TRIO_BUCKET_57.data(), TRIO_BUCKET_58.data(), TRIO_BUCKET_59.data(), TRIO_BUCKET_60.data(), TRIO_BUCKET_61.data(), TRIO_BUCKET_62.data(), TRIO_BUCKET_63.data(), TRIO_BUCKET_64.data(), TRIO_BUCKET_65.data(), TRIO_BUCKET_66.data(), TRIO_BUCKET_67.data(), TRIO_BUCKET_68.data(), TRIO_BUCKET_69.data(), TRIO_BUCKET_70.data(), TRIO_BUCKET_71.data(), TRIO_BUCKET_72.data(), TRIO_BUCKET_73.data(), TRIO_BUCKET_74.data(), TRIO_BUCKET_75.data(), TRIO_BUCKET_76.data(), TRIO_BUCKET_77.data(), TRIO_BUCKET_78.data(), TRIO_BUCKET_79.data(), TRIO_BUCKET_80.data(), TRIO_BUCKET_81.data(), TRIO_BUCKET_82.data(), TRIO_BUCKET_83.data(), TRIO_BUCKET_84.data(), TRIO_BUCKET_85.data(), TRIO_BUCKET_86.data(), TRIO_BUCKET_87.data(), TRIO_BUCKET_88.data(), TRIO_BUCKET_89.data(), TRIO_BUCKET_90.data(), TRIO_BUCKET_91.data(), TRIO_BUCKET_92.data(), TRIO_BUCKET_93.data(), TRIO_BUCKET_94.data(), TRIO_BUCKET_95.data(), TRIO_BUCKET_96.data(), TRIO_BUCKET_97.data(), TRIO_BUCKET_98.data(), TRIO_BUCKET_99.data(), TRIO_BUCKET_100.data(), TRIO_BUCKET_101.data(), TRIO_BUCKET_102.data(), TRIO_BUCKET_103.data(), TRIO_BUCKET_104.data(), TRIO_BUCKET_105.data(), TRIO_BUCKET_106.data(), TRIO_BUCKET_107.data(), TRIO_BUCKET_108.data(), TRIO_BUCKET_109.data(), TRIO_BUCKET_110.data(), TRIO_BUCKET_111.data(), TRIO_BUCKET_112.data(), TRIO_BUCKET_113.data(), TRIO_BUCKET_114.data(), TRIO_BUCKET_115.data(), TRIO_BUCKET_116.data(), TRIO_BUCKET_117.data(), TRIO_BUCKET_118.data(), TRIO_BUCKET_119.data(), TRIO_BUCKET_120.data(), TRIO_BUCKET_121.data(), TRIO_BUCKET_122.data(), TRIO_BUCKET_123.data(), TRIO_BUCKET_124.data(), TRIO_BUCKET_125.data(), TRIO_BUCKET_126.data(), TRIO_BUCKET_127.data(), TRIO_BUCKET_128.data(), TRIO_BUCKET_129.data(), TRIO_BUCKET_130.data(), TRIO_BUCKET_131.data(), TRIO_BUCKET_132.data(), TRIO_BUCKET_133.data(), TRIO_BUCKET_134.data(), TRIO_BUCKET_135.data(), TRIO_BUCKET_136.data(), TRIO_BUCKET_137.data(), TRIO_BUCKET_138.data(), TRIO_BUCKET_139.data(), TRIO_BUCKET_140.data(), TRIO_BUCKET_141.data(), TRIO_BUCKET_142.data(), TRIO_BUCKET_143.data(), TRIO_BUCKET_144.data(), TRIO_BUCKET_145.data(), TRIO_BUCKET_146.data(), TRIO_BUCKET_147.data(), TRIO_BUCKET_148.data(), TRIO_BUCKET_149.data(), TRIO_BUCKET_150.data(), TRIO_BUCKET_151.data(), TRIO_BUCKET_152.data(), TRIO_BUCKET_153.data(), TRIO_BUCKET_154.data(), TRIO_BUCKET_155.data(), TRIO_BUCKET_156.data(), TRIO_BUCKET_157.data(), TRIO_BUCKET_158.data(), TRIO_BUCKET_159.data(), TRIO_BUCKET_160.data(), TRIO_BUCKET_161.data(), TRIO_BUCKET_162.data(), TRIO_BUCKET_163.data(), TRIO_BUCKET_164.data(), TRIO_BUCKET_165.data(), TRIO_BUCKET_166.data(), TRIO_BUCKET_167.data(), TRIO_BUCKET_168.data()
};
