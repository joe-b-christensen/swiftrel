// swiftrel_engine.cpp -- see swiftrel_engine.h. Part of SwiftRel (LGPL-3.0, derived from vcftools 0.1.17).
#include "swiftrel_engine.h"

#include <sys/mman.h>
#include <sys/uio.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <mutex>
#include <thread>

#include "swiftrel_fmt.h"
#include "swiftrel_input.h"

#if defined(__AVX512F__) && defined(__AVX512VPOPCNTDQ__)
#include <immintrin.h>
#define SWIFTREL_AVX512 1
#if defined(__AVX512DQ__)
#define SWIFTREL_AVX512DQ 1
#endif
#if defined(__AVX512VBMI2__) && defined(__AVX512BW__) && defined(__BMI2__)
#define SWIFTREL_VBMI2 1
#endif
#endif

namespace swiftrel {

int thread_count() {
    const char *s = getenv("SWIFTREL_THREADS");
    int n = 0;
    if (s && *s) n = atoi(s);
    if (n <= 0) n = (int)std::thread::hardware_concurrency();
    if (n <= 0) n = 1;
    if (n > 256) n = 256;
    return n;
}

static const int kTile = 32;    // individuals per tile
static const int kChunkW = 256;  // words per chunk (32 x 512-bit vectors)

#ifdef SWIFTREL_AVX512
// transpose64 on eight zmm registers (R[r] lane l = a[8r + l]): the same six swap stages, in the same order. Stages
// 32, 16 and 8 pair whole registers (r with r + 4, r + 2, r + 1); stages 4, 2 and 1 pair lanes l and l ^ 4, l ^ 2,
// l ^ 1 of one register (partner lanes by permutation, low/high lane by blend).
static inline void tr_cross(__m512i &x, __m512i &y, int j, uint64_t m) {
    const __m512i t = _mm512_and_si512(_mm512_xor_si512(_mm512_srli_epi64(x, (unsigned)j), y), _mm512_set1_epi64((long long)m));
    x = _mm512_xor_si512(x, _mm512_slli_epi64(t, (unsigned)j));
    y = _mm512_xor_si512(y, t);
}
static inline void tr_lane(__m512i &R, const __m512i P, __mmask8 lo, int j, uint64_t m) {
    const __m512i low = _mm512_mask_blend_epi64(lo, P, R), high = _mm512_mask_blend_epi64(lo, R, P);
    const __m512i t = _mm512_and_si512(_mm512_xor_si512(_mm512_srli_epi64(low, (unsigned)j), high), _mm512_set1_epi64((long long)m));
    R = _mm512_xor_si512(R, _mm512_mask_blend_epi64(lo, t, _mm512_slli_epi64(t, (unsigned)j)));
}
static inline void transpose64_avx512(uint64_t a[64]) {
    __m512i R[8];
    for (int r = 0; r < 8; r++) R[r] = _mm512_loadu_si512((const void *)(a + 8 * r));
    for (int r = 0; r < 4; r++) tr_cross(R[r], R[r + 4], 32, 0x00000000FFFFFFFFull);
    for (int r : {0, 1, 4, 5}) tr_cross(R[r], R[r + 2], 16, 0x0000FFFF0000FFFFull);
    for (int r : {0, 2, 4, 6}) tr_cross(R[r], R[r + 1], 8, 0x00FF00FF00FF00FFull);
    for (int r = 0; r < 8; r++) {
        tr_lane(R[r], _mm512_shuffle_i64x2(R[r], R[r], 0x4E), 0x0F, 4, 0x0F0F0F0F0F0F0F0Full);
        tr_lane(R[r], _mm512_permutex_epi64(R[r], 0x4E), 0x33, 2, 0x3333333333333333ull);
        tr_lane(R[r], _mm512_permutex_epi64(R[r], 0xB1), 0x55, 1, 0x5555555555555555ull);
    }
    for (int r = 0; r < 8; r++) _mm512_storeu_si512((void *)(a + 8 * r), R[r]);
}
#endif

void transpose64(uint64_t a[64]) {
    uint64_t m = 0x00000000FFFFFFFFull;
    for (int j = 32; j != 0; j >>= 1, m ^= (m << j)) {
        for (int k = 0; k < 64; k = ((k | j) + 1) & ~j) {
            uint64_t t = ((a[k] >> j) ^ a[k | j]) & m;
            a[k] ^= t << j;
            a[k | j] ^= t;
        }
    }
}

static void *big_alloc(size_t bytes) {
    const size_t HP = 2u << 20;
    size_t n = (bytes + HP - 1) / HP * HP;
    void *p = aligned_alloc(HP, n ? n : HP);
    if (!p) abort();
#ifdef MADV_HUGEPAGE
    madvise(p, n, MADV_HUGEPAGE);
#endif
    return p;
}

RelCounter::RelCounter(int K_) : K(K_), KW((K_ + 63) / 64), total_sites(0) {}

RelCounter::~RelCounter() {
    free(aa);
    free(oo);
    free(tw);
    free(def_planes);
}

void RelCounter::add_site(const uint8_t *codes, const int *alleles) {
    std::vector<uint64_t> m((size_t)3 * KW, 0);
    bool ex = false;
    for (int k = 0; k < K; k++) {
        uint8_t c = codes[k];
        uint64_t bit = 1ull << (k & 63);
        if (c == 1) m[(size_t)(k >> 6)] |= bit;
        else if (c == 2) m[(size_t)KW + (k >> 6)] |= bit;
        else if (c == 3) m[(size_t)2 * KW + (k >> 6)] |= bit;
        else if (c == 4) ex = true;
    }
    owned.push_back(std::move(m));
    sites.push_back(owned.back().data());
    site_ex.push_back(ex ? 1 : 0);
    if (ex) {
        std::vector<int> h(K, -1);
        for (int k = 0; k < K; k++)
            h[k] = codes[k] == 2 ? 0 : codes[k] == 3 ? 1 : codes[k] == 4 ? alleles[k] : -1;
        exotic.push_back(std::move(h));
    }
    total_sites++;
}

void RelCounter::add_site_copy(const uint64_t *masks, const int *hom_alleles, int threads) {
    if (pending_cap == 0) {
        // block size: large enough to amortise the K x K accumulate, small enough to stay in cache
        size_t b = (size_t)K * (size_t)K / 64;
        b = std::max<size_t>(K <= 1024 ? 4096 : 16384, std::min<size_t>(65536, b));  // K <= 1024: only dense sites are pending
        pending_cap = (b + 511) & ~(size_t)511;
        pending.reserve(pending_cap * 3 * KW);
    }
    // With K <= 1024 (matrices <= 8 MB) rare sites are applied right away, overlapping with inflate; larger K keeps
    // them pending so that the few-site fused mode (no matrices) stays possible.
    if (!hom_alleles && K <= 1024 && rare_site(masks)) {
        if (!aa) alloc_zeroed();
        apply_sparse(masks);
        total_sites++;
        return;
    }
    pending.insert(pending.end(), masks, masks + (size_t)3 * KW);
    pending_ex.push_back(hom_alleles ? 1 : 0);
    if (hom_alleles) exotic.emplace_back(hom_alleles, hom_alleles + K);
    total_sites++;
    if (pending.size() >= pending_cap * 3 * (size_t)KW) flush_pending(threads);
}

void RelCounter::alloc_zeroed() {
    const size_t KK = (size_t)K * K;
    aa = (uint32_t *)big_alloc(KK * 4);
    oo = (uint32_t *)big_alloc(KK * 4);
    memset(aa, 0, KK * 4);
    memset(oo, 0, KK * 4);
    nA_rare.assign(K, 0);
}

void RelCounter::flush_pending(int threads, bool defer) {
    const size_t n = pending.size() / (3 * (size_t)KW);
    std::vector<const uint64_t *> ptrs(n);
    for (size_t i = 0; i < n; i++) ptrs[i] = pending.data() + i * 3 * (size_t)KW;
    count_sites(ptrs.data(), n, pending_ex.data(), threads, defer);
    pending.clear();
    pending_ex.clear();
}

void RelCounter::add_site_masks(const uint64_t *masks, const int *hom_alleles) {
    sites.push_back(masks);
    site_ex.push_back(hom_alleles ? 1 : 0);
    if (hom_alleles) exotic.emplace_back(hom_alleles, hom_alleles + K);
    total_sites++;
}

static inline void count_pair_chunk(const uint64_t *pi, const uint64_t *pj, size_t PW, int w0, int nw, uint32_t &raa,
                                    uint32_t &roo) {
#ifdef SWIFTREL_AVX512
    __m512i acc_a = _mm512_setzero_si512(), acc_o = _mm512_setzero_si512();
    for (int w = w0; w < w0 + nw; w += 8) {
        __m512i hi = _mm512_load_si512((const void *)(pi + w));
        __m512i hj = _mm512_load_si512((const void *)(pj + w));
        __m512i ai = _mm512_load_si512((const void *)(pi + PW + w));
        __m512i aj = _mm512_load_si512((const void *)(pj + PW + w));
        __m512i bi = _mm512_load_si512((const void *)(pi + 2 * PW + w));
        __m512i bj = _mm512_load_si512((const void *)(pj + 2 * PW + w));
        acc_a = _mm512_add_epi64(acc_a, _mm512_popcnt_epi64(_mm512_and_si512(hi, hj)));
        acc_o = _mm512_add_epi64(acc_o, _mm512_popcnt_epi64(_mm512_and_si512(ai, bj)));
        acc_o = _mm512_add_epi64(acc_o, _mm512_popcnt_epi64(_mm512_and_si512(bi, aj)));
    }
    raa += (uint32_t)_mm512_reduce_add_epi64(acc_a);
    roo += (uint32_t)_mm512_reduce_add_epi64(acc_o);
#else
    uint64_t sa = 0, so = 0;
    for (int w = w0; w < w0 + nw; w++) {
        sa += __builtin_popcountll(pi[w] & pj[w]);
        so += __builtin_popcountll(pi[PW + w] & pj[2 * PW + w]) + __builtin_popcountll(pi[2 * PW + w] & pj[PW + w]);
    }
    raa += (uint32_t)sa;
    roo += (uint32_t)so;
#endif
}

// Count an RI x RJ block of pairs (rows i, i+1; columns j, j+1) over words [w0, w0+nw) (nw multiple of 8).
// N_AaAa += popcount(het_i & het_j); N_AAaa += popcount((h0_i & h1_j) | (h1_i & h0_j)) -- the two terms are
// disjoint (individual i cannot be hom-0 and hom-1 at one site), so one popcount of the OR equals their sum.
template <int RI, int RJ>
static inline void count_block(const uint64_t *const *pi, const uint64_t *const *pj, size_t PW, size_t w0, size_t nw,
                               uint32_t ra[2][2], uint32_t ro[2][2]) {
#ifdef SWIFTREL_AVX512
    __m512i acc_a[RI][RJ], acc_o[RI][RJ];
    for (int a = 0; a < RI; a++)
        for (int b = 0; b < RJ; b++) acc_a[a][b] = acc_o[a][b] = _mm512_setzero_si512();
    for (size_t w = w0; w < w0 + nw; w += 8) {
        __m512i hi[RI], zi[RI], oi[RI], hj[RJ], zj[RJ], oj[RJ];
        for (int a = 0; a < RI; a++) {
            hi[a] = _mm512_load_si512((const void *)(pi[a] + w));
            zi[a] = _mm512_load_si512((const void *)(pi[a] + PW + w));
            oi[a] = _mm512_load_si512((const void *)(pi[a] + 2 * PW + w));
        }
        for (int b = 0; b < RJ; b++) {
            hj[b] = _mm512_load_si512((const void *)(pj[b] + w));
            zj[b] = _mm512_load_si512((const void *)(pj[b] + PW + w));
            oj[b] = _mm512_load_si512((const void *)(pj[b] + 2 * PW + w));
        }
        for (int a = 0; a < RI; a++)
            for (int b = 0; b < RJ; b++) {
                acc_a[a][b] = _mm512_add_epi64(acc_a[a][b], _mm512_popcnt_epi64(_mm512_and_si512(hi[a], hj[b])));
                // t = (zi & oj) | (oi & zj): and + ternary-logic A | (B & C) = imm 0xF8 (bit index 4A+2B+C)
                __m512i t = _mm512_and_si512(zi[a], oj[b]);
                t = _mm512_ternarylogic_epi64(t, oi[a], zj[b], 0xF8);
                acc_o[a][b] = _mm512_add_epi64(acc_o[a][b], _mm512_popcnt_epi64(t));
            }
    }
    for (int a = 0; a < RI; a++)
        for (int b = 0; b < RJ; b++) {
            ra[a][b] += (uint32_t)_mm512_reduce_add_epi64(acc_a[a][b]);
            ro[a][b] += (uint32_t)_mm512_reduce_add_epi64(acc_o[a][b]);
        }
#else
    for (int a = 0; a < RI; a++)
        for (int b = 0; b < RJ; b++) {
            uint64_t sa = 0, so = 0;
            for (size_t w = w0; w < w0 + nw; w++) {
                sa += (uint64_t)__builtin_popcountll(pi[a][w] & pj[b][w]);
                so += (uint64_t)__builtin_popcountll((pi[a][PW + w] & pj[b][2 * PW + w]) |
                                                     (pi[a][2 * PW + w] & pj[b][PW + w]));
            }
            ra[a][b] += (uint32_t)sa;
            ro[a][b] += (uint32_t)so;
        }
#endif
}

template <class F>
static void parallel_for(size_t n, F f, int threads = 0) {
    int nt = (int)std::min<size_t>((size_t)(threads > 0 ? threads : thread_count()), n);
    if (nt <= 1) {
        for (size_t i = 0; i < n; i++) f(i);
        return;
    }
    std::atomic<size_t> next(0);
    auto w = [&]() {
        for (;;) {
            size_t i = next.fetch_add(1);
            if (i >= n) break;
            f(i);
        }
    };
    std::vector<std::thread> th;
    for (int t = 1; t < nt; t++) th.emplace_back(w);
    w();
    for (auto &x : th) x.join();
}

// Transpose and count S sites (pointers to their het/h0/h1 masks), adding to aa/oo (allocated by the first call).
// Rare sites (few heterozygous, hom-alt and missing individuals) are cheaper to count by direct pair updates than
// by bitplanes. With every individual exactly one of Z (hom-ref), H (het), A (hom-alt), M (missing) at a site
// without odd homozygous alleles, sum_s A_i Z_j = nA_i - |A_i&H_j| - |A_i&A_j| - |A_i&M_j|, so over rare sites
//   N_AAaa(i,j) = nA(i) + nA(j) - [|A_i&H_j| + |H_i&A_j| + 2|A_i&A_j| + |A_i&M_j| + |M_i&A_j|]
//   N_AaAa(i,j) = |H_i&H_j|
// The bracket and N_AaAa need only the small sets H, A, M; nA(i) + nA(j) is added once at the end (finish).
// All arithmetic is on uint32 modulo 2^32; every final count is exact and non-negative.
static inline uint64_t sparse_cost(uint64_t cH, uint64_t cA, uint64_t cM) {
    return cH * cH + 2 * cA * cH + cA * cA + 2 * cA * cM;
}

bool RelCounter::rare_site(const uint64_t *m) const {
    uint64_t cH = 0, cA = 0, cZ = 0;
    for (int w = 0; w < KW; w++) {
        cH += (uint64_t)__builtin_popcountll(m[w]);
        cZ += (uint64_t)__builtin_popcountll(m[KW + w]);
        cA += (uint64_t)__builtin_popcountll(m[2 * KW + w]);
    }
    const uint64_t cM = (uint64_t)K - cH - cA - cZ;
    return cH <= 64 && cA <= 64 && cM <= 64 && sparse_cost(cH, cA, cM) <= (uint64_t)K * (uint64_t)K / 1500;
}

void RelCounter::apply_sparse(const uint64_t *m) {
    const uint64_t *het = m, *h0 = m + KW, *h1 = m + 2 * KW;
    int Hs[64], As[64], Ms[64];
    int nH = 0, nA = 0, nM = 0;
    for (int w = 0; w < KW; w++) {
        uint64_t valid = (w == KW - 1 && (K & 63)) ? ((1ull << (K & 63)) - 1) : ~0ull;
        uint64_t x = het[w];
        while (x) { Hs[nH++] = w * 64 + __builtin_ctzll(x); x &= x - 1; }
        x = h1[w];
        while (x) { As[nA++] = w * 64 + __builtin_ctzll(x); x &= x - 1; }
        x = ~(het[w] | h0[w] | h1[w]) & valid;
        while (x) { Ms[nM++] = w * 64 + __builtin_ctzll(x); x &= x - 1; }
    }
    const size_t Kz = (size_t)K;
    for (int a = 0; a < nH; a++) {
        uint32_t *ra = aa + (size_t)Hs[a] * Kz;
        for (int b = 0; b < nH; b++) ra[Hs[b]] += 1;
    }
    for (int a = 0; a < nA; a++) {
        const int i = As[a];
        nA_rare[i] += 1;
        uint32_t *ri = oo + (size_t)i * Kz;
        for (int b = 0; b < nH; b++) {
            ri[Hs[b]] -= 1;
            oo[(size_t)Hs[b] * Kz + i] -= 1;
        }
        for (int b = 0; b < nA; b++) ri[As[b]] -= 2;
        for (int b = 0; b < nM; b++) {
            ri[Ms[b]] -= 1;
            oo[(size_t)Ms[b] * Kz + i] -= 1;
        }
    }
}

// Count tile pair (I, J), I <= J, of one block of planes: cells (i, j) and (j, i), set (first block) or added.
void RelCounter::count_tile(const uint64_t *planes, size_t PW, size_t nw, int I, int J, bool first) const {
    const int i0 = I * kTile, i1 = std::min(K, i0 + kTile);
    const int j0 = J * kTile, j1 = std::min(K, j0 + kTile);
    uint32_t ta[kTile][kTile], to[kTile][kTile];
    memset(ta, 0, sizeof ta);
    memset(to, 0, sizeof to);
    // few sites (< 32 words): per-pair kernel; otherwise 2x2 register blocking (the per-block horizontal
    // reductions cost more than they save when a pair has only one or two vectors)
    if (nw < 32) {
    for (size_t w0 = 0; w0 < nw; w0 += kChunkW) {
            const int cw = (int)std::min<size_t>(kChunkW, nw - w0);
            for (int i = i0; i < i1; i++) {
                const uint64_t *pi = planes + (size_t)i * 3 * PW;
                for (int j = (I == J ? i : j0); j < j1; j++) {
                    const uint64_t *pj = planes + (size_t)j * 3 * PW;
                    count_pair_chunk(pi, pj, PW, (int)w0, cw, ta[i - i0][j - j0], to[i - i0][j - j0]);
                }
            }
        }
    } else
    for (size_t w0 = 0; w0 < nw; w0 += kChunkW) {
        const size_t cw = std::min<size_t>(kChunkW, nw - w0);
        for (int ib = i0; ib < i1; ib += 2) {
            const int ri = std::min(2, i1 - ib);
            const uint64_t *pi[2] = {planes + (size_t)ib * 3 * PW,
                                     planes + (size_t)(ib + ri - 1) * 3 * PW};
            for (int jb = (I == J ? ib : j0); jb < j1; jb += 2) {
                const int rj = std::min(2, j1 - jb);
                const uint64_t *pj[2] = {planes + (size_t)jb * 3 * PW,
                                         planes + (size_t)(jb + rj - 1) * 3 * PW};
                uint32_t ra[2][2] = {{0, 0}, {0, 0}}, ro[2][2] = {{0, 0}, {0, 0}};
                if (ri == 2 && rj == 2) count_block<2, 2>(pi, pj, PW, w0, cw, ra, ro);
                else if (ri == 2) count_block<2, 1>(pi, pj, PW, w0, cw, ra, ro);
                else if (rj == 2) count_block<1, 2>(pi, pj, PW, w0, cw, ra, ro);
                else count_block<1, 1>(pi, pj, PW, w0, cw, ra, ro);
                for (int a = 0; a < ri; a++)
                    for (int b = 0; b < rj; b++) {
                        ta[ib + a - i0][jb + b - j0] += ra[a][b];
                        to[ib + a - i0][jb + b - j0] += ro[a][b];
                    }
            }
        }
    }
    for (int i = i0; i < i1; i++)
        for (int j = (I == J ? i : j0); j < j1; j++) {
            uint32_t va = ta[i - i0][j - j0], vo = to[i - i0][j - j0];
            size_t x = (size_t)i * K + j, y = (size_t)j * K + i;
            if (first) {
                aa[x] = va; oo[x] = vo; aa[y] = va; oo[y] = vo;
            } else {
                aa[x] += va; oo[x] += vo;
                if (y != x) { aa[y] += va; oo[y] += vo; }
            }
        }
}

void RelCounter::count_sites(const uint64_t *const *all_ptrs, size_t S_all, const uint8_t *ex, int threads, bool defer) {
    const size_t KK = (size_t)K * K;
    bool first = false;
    if (!aa) {
        aa = (uint32_t *)big_alloc(KK * 4);
        oo = (uint32_t *)big_alloc(KK * 4);
        first = true;
        nA_rare.assign(K, 0);
    }
    // split: rare sites (direct pair updates, below) vs the rest (bitplanes)
    std::vector<const uint64_t *> dense, rare;
    dense.reserve(S_all);
    for (size_t i = 0; i < S_all; i++) {
        const uint64_t *m = all_ptrs[i];
        if (!ex[i] && rare_site(m))
            rare.push_back(m);
        else
            dense.push_back(m);
    }
    const uint64_t *const *site_ptrs = dense.data();
    const size_t S = dense.size();
    // Planes: [K][3][PW] individual-major bitplanes for a block of up to 64*PW sites.
    size_t words_total = (S + 63) / 64;
    size_t PW = (words_total + 7) & ~(size_t)7;
    const size_t budget = (size_t)256 << 20;  // bytes for one block of planes
    size_t maxPW = std::max<size_t>(8, (budget / ((size_t)K * 3 * 8)) & ~(size_t)7);
    if (PW > maxPW) PW = maxPW;
    if (PW == 0) PW = 8;
    uint64_t *planes = (uint64_t *)big_alloc((size_t)K * 3 * PW * 8);
    const int ntiles = (K + kTile - 1) / kTile;
    std::vector<std::pair<int, int>> tasks;
    for (int I = 0; I < ntiles; I++)
        for (int J = I; J < ntiles; J++) tasks.push_back({I, J});
    // a deferred block must be the only one and must add to counts of earlier blocks (streaming input; for a single
    // block the matrices would first have to be zeroed, which cost more than the overlap gained)
    const bool defer_now = defer && !first && S > 0 && S <= PW * 64 && ntiles > 1;
    for (size_t s0 = 0; s0 < S || first; s0 += PW * 64) {
        const size_t sb = std::min(S - std::min(S, s0), PW * 64);  // sites in this block
        const size_t nw = (((sb + 63) / 64) + 7) & ~(size_t)7;      // words to count (zero padded)
        // transpose 64-site groups into the planes, parallel over runs of 8 consecutive groups (PW and nw are
        // multiples of 8): for each plane and 64-individual slice, the run's 8 groups are transposed first and then
        // stored as one 64-byte line per individual, instead of 8 separate 8-byte stores per line (the planes are
        // larger than L1, so each separate store would miss it)
        const size_t ngroups = std::max<size_t>(nw, 1);
        const int KW_ = KW;
        parallel_for((ngroups + 7) / 8, [&](size_t run) {
            const size_t g0 = run * 8, gn = std::min<size_t>(8, ngroups - g0);
            alignas(64) uint64_t buf[8][64];
            for (int p = 0; p < 3; p++)
                for (int h = 0; h < KW_; h++) {
                    for (size_t gi = 0; gi < gn; gi++) {
                        const size_t g = g0 + gi;
                        for (int s = 0; s < 64; s++) {
                            size_t si = s0 + g * 64 + (size_t)s;
                            buf[gi][s] = (g * 64 + (size_t)s < sb) ? site_ptrs[si][(size_t)p * KW_ + h] : 0;
                        }
#ifdef SWIFTREL_AVX512
                        transpose64_avx512(buf[gi]);
#else
                        transpose64(buf[gi]);
#endif
                    }
                    int kmax = std::min(64, K - h * 64);
                    for (int k = 0; k < kmax; k++) {
                        uint64_t *dst = planes + ((size_t)(h * 64 + k) * 3 + p) * PW + g0;
                        for (size_t gi = 0; gi < gn; gi++) dst[gi] = buf[gi][k];
                    }
                }
        }, threads);
        if (defer_now) {
            // diagonal tile pairs now (N_Aa(i) = N_AaAa(i, i) is read first); the others when the writer reads them
            parallel_for((size_t)ntiles, [&](size_t I) { count_tile(planes, PW, nw, (int)I, (int)I, false); }, threads);
            def_planes = planes;
            def_PW = PW;
            def_nw = nw;
            def_pending = true;
            break;
        }
        // count tiles; each task owns cells (i,j) and (j,i) of its tile pair
        parallel_for(tasks.size(), [&](size_t t) {
            count_tile(planes, PW, nw, tasks[t].first, tasks[t].second, first);
        }, threads);
        first = false;
        if (s0 + PW * 64 >= S) break;
    }
    if (!defer_now) free(planes);
    for (const uint64_t *m : rare) apply_sparse(m);
}

// Few used sites (<= 2048) and nothing counted yet: keep word-major planes tw[p][w][b] instead of K x K matrices.
bool RelCounter::build_fused() {
    if (aa || !exotic.empty()) return false;
    std::vector<const uint64_t *> ptrs;
    const size_t np = pending.size() / (3 * (size_t)KW);
    for (size_t i = 0; i < np; i++) ptrs.push_back(pending.data() + i * 3 * (size_t)KW);
    for (const uint64_t *q : sites) ptrs.push_back(q);
    const size_t S = ptrs.size();
    if (S > 2048 || K < 64) return false;
    nwf = (S + 63) / 64;
    if (nwf == 0) nwf = 1;
    Kp = ((size_t)K + 7) & ~(size_t)7;
    tw = (uint64_t *)big_alloc(3 * nwf * Kp * 8);
    memset(tw, 0, 3 * nwf * Kp * 8);
    uint64_t buf[64];
    for (size_t g = 0; g < nwf; g++)
        for (int p = 0; p < 3; p++)
            for (int h = 0; h < KW; h++) {
                for (int s = 0; s < 64; s++) {
                    size_t si = g * 64 + (size_t)s;
                    buf[s] = si < S ? ptrs[si][(size_t)p * KW + h] : 0;
                }
                transpose64(buf);
                int kmax = std::min(64, K - h * 64);
                for (int k = 0; k < kmax; k++) tw[((size_t)p * nwf + g) * Kp + (size_t)h * 64 + k] = buf[k];
            }
    naa.assign(K, 0);
    for (int k = 0; k < K; k++) {
        uint32_t c = 0;
        for (size_t g = 0; g < nwf; g++) c += (uint32_t)__builtin_popcountll(tw[g * Kp + k]);
        naa[k] = c;
    }
    pending.clear();
    pending_ex.clear();
    fused = true;
    return true;
}

// Counts of row a against every b: N_AaAa = popcount(het_a & het_b), N_AAaa = popcount((h0_a & h1_b) | (h1_a & h0_b)).
// Vectorised across 8 consecutive b (word-major planes), so no horizontal reductions are needed.
#ifdef SWIFTREL_AVX512
// fused_row for a fixed number of words (NW <= 4): row a's words are broadcast once, outside the column loop.
template <int NW>
static void fused_row_nw(const uint64_t *T0, const uint64_t *T1, const uint64_t *T2, size_t Kp, int a, uint32_t *ra,
                         uint32_t *ro) {
    __m512i ha[NW], za[NW], oa[NW];
    for (int w = 0; w < NW; w++) {
        ha[w] = _mm512_set1_epi64((long long)T0[(size_t)w * Kp + a]);
        za[w] = _mm512_set1_epi64((long long)T1[(size_t)w * Kp + a]);
        oa[w] = _mm512_set1_epi64((long long)T2[(size_t)w * Kp + a]);
    }
    for (size_t b0 = 0; b0 < Kp; b0 += 8) {
        __m512i acc_a = _mm512_setzero_si512(), acc_o = _mm512_setzero_si512();
        for (int w = 0; w < NW; w++) {
            const __m512i hb = _mm512_load_si512((const void *)(T0 + (size_t)w * Kp + b0));
            const __m512i zb = _mm512_load_si512((const void *)(T1 + (size_t)w * Kp + b0));
            const __m512i ob = _mm512_load_si512((const void *)(T2 + (size_t)w * Kp + b0));
            acc_a = _mm512_add_epi64(acc_a, _mm512_popcnt_epi64(_mm512_and_si512(ha[w], hb)));
            __m512i t = _mm512_and_si512(za[w], ob);
            t = _mm512_ternarylogic_epi64(t, oa[w], zb, 0xF8);  // t | (oa & zb)
            acc_o = _mm512_add_epi64(acc_o, _mm512_popcnt_epi64(t));
        }
        // the strips have room for Kp entries
        _mm256_storeu_si256((__m256i *)(ra + b0), _mm512_cvtepi64_epi32(acc_a));
        _mm256_storeu_si256((__m256i *)(ro + b0), _mm512_cvtepi64_epi32(acc_o));
    }
}
#endif

void RelCounter::fused_row(int a, uint32_t *ra, uint32_t *ro) const {
    const uint64_t *T0 = tw, *T1 = tw + nwf * Kp, *T2 = tw + 2 * nwf * Kp;
#ifdef SWIFTREL_AVX512
    switch (nwf) {
        case 1: fused_row_nw<1>(T0, T1, T2, Kp, a, ra, ro); return;
        case 2: fused_row_nw<2>(T0, T1, T2, Kp, a, ra, ro); return;
        case 3: fused_row_nw<3>(T0, T1, T2, Kp, a, ra, ro); return;
        case 4: fused_row_nw<4>(T0, T1, T2, Kp, a, ra, ro); return;
        default: break;
    }
    for (size_t b0 = 0; b0 < Kp; b0 += 8) {
        __m512i acc_a = _mm512_setzero_si512(), acc_o = _mm512_setzero_si512();
        for (size_t w = 0; w < nwf; w++) {
            const __m512i ha = _mm512_set1_epi64((long long)T0[w * Kp + a]);
            const __m512i za = _mm512_set1_epi64((long long)T1[w * Kp + a]);
            const __m512i oa = _mm512_set1_epi64((long long)T2[w * Kp + a]);
            const __m512i hb = _mm512_load_si512((const void *)(T0 + w * Kp + b0));
            const __m512i zb = _mm512_load_si512((const void *)(T1 + w * Kp + b0));
            const __m512i ob = _mm512_load_si512((const void *)(T2 + w * Kp + b0));
            acc_a = _mm512_add_epi64(acc_a, _mm512_popcnt_epi64(_mm512_and_si512(ha, hb)));
            __m512i t = _mm512_and_si512(za, ob);
            t = _mm512_ternarylogic_epi64(t, oa, zb, 0xF8);  // t | (oa & zb)
            acc_o = _mm512_add_epi64(acc_o, _mm512_popcnt_epi64(t));
        }
        _mm256_storeu_si256((__m256i *)(ra + b0), _mm512_cvtepi64_epi32(acc_a));
        _mm256_storeu_si256((__m256i *)(ro + b0), _mm512_cvtepi64_epi32(acc_o));
    }
#else
    for (int b = 0; b < K; b++) {
        uint32_t ca = 0, co = 0;
        for (size_t w = 0; w < nwf; w++) {
            ca += (uint32_t)__builtin_popcountll(T0[w * Kp + a] & T0[w * Kp + b]);
            co += (uint32_t)__builtin_popcountll((T1[w * Kp + a] & T2[w * Kp + b]) | (T2[w * Kp + a] & T1[w * Kp + b]));
        }
        ra[b] = ca;
        ro[b] = co;
    }
#endif
}

void RelCounter::count_deferred(int I, int J) const {
    if (def_pending && I != J) count_tile(def_planes, def_PW, def_nw, I, J, false);
}

void RelCounter::complete_deferred() const {
    if (!def_pending) return;
    const int ntiles = (K + kTile - 1) / kTile;
    std::vector<std::pair<int, int>> tasks;
    for (int I = 0; I < ntiles; I++)
        for (int J = I + 1; J < ntiles; J++) tasks.push_back({I, J});
    parallel_for(tasks.size(), [&](size_t t) { count_tile(def_planes, def_PW, def_nw, tasks[t].first, tasks[t].second, false); });
    def_pending = false;
}

void RelCounter::finish() {
    if (build_fused()) return;
    // the last count_sites call may leave its off-diagonal tile pairs to the writer (see count_deferred)
    const bool defer = thread_count() > 1 && K >= 64 && K <= 1024;
    const bool more = !sites.empty();
    if (!pending.empty()) flush_pending(0, defer && !more);
    if (more || !aa) count_sites(sites.data(), sites.size(), site_ex.data(), 0, defer);
    // rare-site identity, last term: N_AAaa(i,j) += nA(i) + nA(j) (see apply_sparse)
    bool any = false;
    for (int k = 0; k < K && !any; k++) any = nA_rare[k] != 0;
    if (any) {
        for (int i = 0; i < K; i++) {
            uint32_t *ro = oo + (size_t)i * K;
            const uint32_t ni = nA_rare[i];
            for (int j = 0; j < K; j++) ro[j] += ni + nA_rare[j];
        }
    }
    // exact corrections for sites with homozygous alleles outside {0,1}
    for (const std::vector<int> &h : exotic) {
        for (int i = 0; i < K; i++) {
            if (h[i] < 0) continue;
            for (int j = i + 1; j < K; j++) {
                if (h[j] < 0) continue;
                if (h[i] <= 1 && h[j] <= 1) continue;  // counted by the bitplanes
                if (h[i] != h[j]) {
                    oo[(size_t)i * K + j]++;
                    oo[(size_t)j * K + i]++;
                }
            }
        }
    }
}

static bool write_all(int fd, const char *p, size_t n) {
    while (n > 0) {
        ssize_t w = ::write(fd, p, n);
        if (w < 0) {
            if (errno == EINTR) continue;
            return false;
        }
        p += w;
        n -= (size_t)w;
    }
    return true;
}

// A 16-byte phi slot: the string (<= 13 bytes incl. the newline) at 0.., its length in byte 15 and, for the
// compressed-row emitter, its byte mask ((1 << len) - 1) in bytes 13-14.
static inline void slot_lengths(char *slot, char *e) {
    const unsigned len = (unsigned)(e - slot);
    slot[15] = (char)len;
    const uint16_t m16 = (uint16_t)((1u << len) - 1);
    memcpy(slot + 13, &m16, 2);
}

#ifdef SWIFTREL_AVX512DQ
// Writes slot b (16 bytes each, at out + 16 b) for b < 8: the "%.6g\n" string of v_b = fl(num_b / den_b), its
// length in byte 15 and byte mask in bytes 13-14 (as slot_lengths). Requires 0 <= den_b < 2^32, |num_b| < 2^31.
static inline void fmt_phi8(char *out, const int64_t *num, const uint64_t *den) {
    const __m512i vnum = _mm512_loadu_si512((const void *)num);
    const __m512i vden = _mm512_loadu_si512((const void *)den);
    const __m512d dnum = _mm512_cvtepi64_pd(vnum), dden = _mm512_cvtepu64_pd(vden);
    const __m512d v = _mm512_div_pd(dnum, dden);  // the exact IEEE quotient, as in vcftools
    const __m512d a = _mm512_abs_pd(v);
    const __m512i n = _mm512_abs_epi64(vnum);
    // fast lanes: 1e-4 <= a < 1e5 (also excludes 0, inf, nan)
    __mmask8 ok = _mm512_cmp_pd_mask(a, _mm512_set1_pd(1e-4), _CMP_GE_OQ) &
                  _mm512_cmp_pd_mask(a, _mm512_set1_pd(1e5), _CMP_LT_OQ) &
                  _mm512_cmp_epu64_mask(n, _mm512_set1_epi64(1ll << 31), _MM_CMPINT_LT) &
                  _mm512_cmp_epu64_mask(vden, _mm512_set1_epi64(1ll << 32), _MM_CMPINT_LT);
    // E estimate: floor(e2 * log10 2), +1 if a >= 10^E
    const __m512i e2 = _mm512_cvttpd_epi64(_mm512_getexp_pd(a));
    __m512i E = _mm512_srai_epi64(_mm512_mullo_epi64(e2, _mm512_set1_epi64(78913)), 18);
    const __m512d p10lo = _mm512_setr_pd(1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1);
    const __m512d p10hi = _mm512_setr_pd(1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7);
    const __m512i ei = _mm512_add_epi64(E, _mm512_set1_epi64(8));  // index into 1e-8 .. 1e7
    ok &= _mm512_cmp_epi64_mask(ei, _mm512_setzero_si512(), _MM_CMPINT_NLT) &
          _mm512_cmp_epi64_mask(ei, _mm512_set1_epi64(15), _MM_CMPINT_LE);
    const __m512d pe = _mm512_permutex2var_pd(p10lo, ei, p10hi);
    E = _mm512_mask_add_epi64(E, _mm512_cmp_pd_mask(a, pe, _CMP_GE_OQ), E, _mm512_set1_epi64(1));
    // k = 5 - E in [0, 9]; 10^k as integer and as double
    __m512i k = _mm512_sub_epi64(_mm512_set1_epi64(5), E);
    const __m512i p10i_lo = _mm512_setr_epi64(1, 10, 100, 1000, 10000, 100000, 1000000, 10000000);
    const __m512i p10i_hi = _mm512_setr_epi64(100000000, 1000000000, 0, 0, 0, 0, 0, 0);
    ok &= _mm512_cmp_epi64_mask(k, _mm512_setzero_si512(), _MM_CMPINT_NLT) &
          _mm512_cmp_epi64_mask(k, _mm512_set1_epi64(9), _MM_CMPINT_LE);
    __m512i pk = _mm512_permutex2var_epi64(p10i_lo, k, p10i_hi);
    __m512i t = _mm512_mullo_epi64(n, pk);
    const __m512i lo_b = _mm512_mullo_epi64(vden, _mm512_set1_epi64(100000));
    const __m512i hi_b = _mm512_mullo_epi64(vden, _mm512_set1_epi64(1000000));
    const __mmask8 too_small = _mm512_cmp_epu64_mask(t, lo_b, _MM_CMPINT_LT) & ok;
    const __mmask8 too_big = _mm512_cmp_epu64_mask(t, hi_b, _MM_CMPINT_NLT) & ok;
    if (too_small | too_big) {
        k = _mm512_mask_add_epi64(k, too_small, k, _mm512_set1_epi64(1));
        k = _mm512_mask_sub_epi64(k, too_big, k, _mm512_set1_epi64(1));
        E = _mm512_mask_sub_epi64(E, too_small, E, _mm512_set1_epi64(1));
        E = _mm512_mask_add_epi64(E, too_big, E, _mm512_set1_epi64(1));
        ok &= _mm512_cmp_epi64_mask(k, _mm512_setzero_si512(), _MM_CMPINT_NLT) &
              _mm512_cmp_epi64_mask(k, _mm512_set1_epi64(9), _MM_CMPINT_LE);
        pk = _mm512_permutex2var_epi64(p10i_lo, k, p10i_hi);
        t = _mm512_mullo_epi64(n, pk);
        ok &= _mm512_cmp_epu64_mask(t, lo_b, _MM_CMPINT_NLT) & _mm512_cmp_epu64_mask(t, hi_b, _MM_CMPINT_LT);
    }
    // m0 = floor(t / den): estimate a * 10^k, correct with the exact remainder
    const __m512d p10d_lo = _mm512_setr_pd(1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7);
    const __m512d p10d_hi = _mm512_setr_pd(1e8, 1e9, 0, 0, 0, 0, 0, 0);
    const __m512d pkd = _mm512_permutex2var_pd(p10d_lo, k, p10d_hi);
    __m512i m0 = _mm512_cvttpd_epi64(_mm512_mul_pd(a, pkd));
    __m512i rem = _mm512_sub_epi64(t, _mm512_mullo_epi64(m0, vden));
    const __mmask8 neg = _mm512_cmp_epi64_mask(rem, _mm512_setzero_si512(), _MM_CMPINT_LT);
    m0 = _mm512_mask_sub_epi64(m0, neg, m0, _mm512_set1_epi64(1));
    rem = _mm512_mask_add_epi64(rem, neg, rem, vden);
    const __mmask8 over = _mm512_cmp_epi64_mask(rem, vden, _MM_CMPINT_NLT);
    m0 = _mm512_mask_add_epi64(m0, over, m0, _mm512_set1_epi64(1));
    rem = _mm512_mask_sub_epi64(rem, over, rem, vden);
    ok &= _mm512_cmp_epi64_mask(rem, _mm512_setzero_si512(), _MM_CMPINT_NLT) &
          _mm512_cmp_epi64_mask(rem, vden, _MM_CMPINT_LT);
    const __m512i r2 = _mm512_add_epi64(rem, rem);
    ok &= _mm512_cmp_epi64_mask(r2, vden, _MM_CMPINT_NE);  // exact ties: the binary value decides (scalar)
    __m512i D = _mm512_mask_add_epi64(m0, _mm512_cmp_epi64_mask(r2, vden, _MM_CMPINT_NLE), m0, _mm512_set1_epi64(1));
    const __mmask8 carry = _mm512_cmp_epi64_mask(D, _mm512_set1_epi64(1000000), _MM_CMPINT_EQ);
    D = _mm512_mask_mov_epi64(D, carry, _mm512_set1_epi64(100000));
    __m512i X = _mm512_mask_add_epi64(E, carry, E, _mm512_set1_epi64(1));
    ok &= _mm512_cmp_epi64_mask(X, _mm512_set1_epi64(-4), _MM_CMPINT_NLT) &
          _mm512_cmp_epi64_mask(X, _mm512_set1_epi64(6), _MM_CMPINT_LT);
    // six decimal digits of D (100000 <= D <= 999999), most significant in byte 0; q = x / 10 = (x * 0xCCCCCCCD) >> 35
    __m512i dg = _mm512_setzero_si512(), x = D;
    const __m512i c10 = _mm512_set1_epi64(0xCCCCCCCDll), ten = _mm512_set1_epi64(10);
    for (int i = 5; i >= 0; i--) {
        const __m512i q = _mm512_srli_epi64(_mm512_mul_epu32(x, c10), 35);
        const __m512i d = _mm512_sub_epi64(x, _mm512_mullo_epi64(q, ten));
        dg = _mm512_or_si512(dg, _mm512_slli_epi64(d, 8 * i));
        x = q;
    }
    dg = _mm512_add_epi64(dg, _mm512_set1_epi64(0x0000303030303030ll));
    alignas(64) int64_t aX[8], aD[8];
    alignas(64) uint64_t adg[8];
    _mm512_store_si512((void *)aX, X);
    _mm512_store_si512((void *)aD, D);
    _mm512_store_si512((void *)adg, dg);
    for (int b = 0; b < 8; b++) {
        char *slot = out + 16 * b;
        if (!((ok >> b) & 1)) {
            char *e = fmt_phi(slot, num[b], den[b], (double)num[b] / (double)den[b]);
            *e++ = '\n';
            slot_lengths(slot, e);
            continue;
        }
        const uint32_t Dv = (uint32_t)aD[b];
        int nd = 6;
        if (__builtin_expect(Dv % 10u == 0, 0))
            nd = 5 - (Dv % 100u == 0) - (Dv % 1000u == 0) - (Dv % 10000u == 0) - (Dv % 100000u == 0);
        const int Xv = (int)aX[b];
        char *p = slot;
        *p = '-';
        p += (num[b] < 0);
        uint64_t d8 = adg[b] | 0x3030000000000000ull;  // bytes 6, 7 = '0'
        if (Xv < 0) {
            memcpy(p, "0.000000", 8);
            memcpy(p + 1 - Xv, &d8, 8);
            p += 1 - Xv + nd;
        } else {
            const int ip = Xv + 1;
            char dgs[8];
            memcpy(dgs, &d8, 8);
            memcpy(p, dgs, 8);
            if (nd <= ip)
                p += ip;
            else {
                memmove(p + ip + 1, dgs + ip, (size_t)(nd - ip));
                p[ip] = '.';
                p += nd + 1;
            }
        }
        *p++ = '\n';
        slot_lengths(slot, p);
    }
}
#endif

#ifdef SWIFTREL_VBMI2
struct VRowArgs {
    int K;
    const uint32_t *ra, *ro;
    const char *cnt;
    const __m512i *bt;
    const uint64_t *bm;
    const uint32_t *naa_u;
    const double *nAa;
    __m512i atm;
    uint64_t amask;
    uint32_t naa_a;
    double nda;
    char *phi_tab;
    int64_t num_lo, den_n;
    const char *symrow;
};

// One output row in the compressed 64-byte layout (see write_relatedness2). MODE 0: phi from the (num, den) table
// (filled on first use), 1: phi from the symmetric slot grid, 2: phi formatted directly.
// Emit one row in the compressed layout. Slot b's phi string is at base + 16 * (IND ? idx[b] : b). The per-column
// mask bits of name_b / N2_Aa are stored in bytes 16..23 of the column template (that lane is replaced by the count
// pair in the row). No calls in the loop, so its values stay in registers.
template <bool IND>
__attribute__((noinline)) static char *vrow_emit(char *p, int K, const uint32_t *ra, const uint32_t *ro,
                                                 const char *cnt, const __m512i *bt, const char *base,
                                                 const uint32_t *idx, __m512i atm, uint64_t amask) {
    const __mmask8 k_ao = 0x0C, k_ph = 0xC0;
    for (int b = 0; b < K; b++) {
        uint64_t qa, qo, bmv;
        memcpy(&qa, cnt + (size_t)ra[b] * 8, 8);
        memcpy(&qo, cnt + (size_t)ro[b] * 8, 8);
        const char *slot = base + (size_t)(IND ? idx[b] : (uint32_t)b) * 16;
        const __m128i ph = _mm_loadu_si128((const __m128i *)slot);
        uint16_t mph;
        memcpy(&mph, slot + 13, 2);
        memcpy(&bmv, (const char *)&bt[b] + 16, 8);
        const __m128i ao = _mm_set_epi64x((long long)qo, (long long)qa);
        __m512i z = _mm512_or_si512(atm, _mm512_load_si512((const void *)&bt[b]));
        z = _mm512_mask_broadcast_i64x2(z, k_ao, ao);
        z = _mm512_mask_broadcast_i64x2(z, k_ph, ph);
        const uint64_t m = amask | bmv | (_bzhi_u64(0xFFull, (unsigned)(qa >> 56)) << 16) |
                           (_bzhi_u64(0xFFull, (unsigned)(qo >> 56)) << 24) | ((uint64_t)mph << 48);
        _mm512_storeu_si512((void *)p, _mm512_maskz_compress_epi8((__mmask64)m, z));
        p += _mm_popcnt_u64(m);
    }
    return p;
}

// One output row: MODE 0 phi from the (num, den) table (slots filled on first use in a first pass), 1 from the
// symmetric grid, 2 formatted directly (into a per-row scratch in a first pass). scratch: K uint32 + 16 K bytes.
template <int MODE>
static char *vrow_loop(char *p, const VRowArgs &v, uint32_t *idx, char *rowbuf) {
    if (MODE == 0) {
        int b0 = 0;
        if (v.num_lo > -(int64_t)(1 << 30) && v.den_n < ((int64_t)1 << 22)) {
            // 16 columns at a time: idx = (N_AaAa - 2 N_AAaa - num_lo) * den_n + N1_Aa + N2_Aa (< 2^22 entries);
            // the slots' length bytes are gathered to find the few that still need formatting
            const __m512i vnl = _mm512_set1_epi32((int)(-v.num_lo)), vdn = _mm512_set1_epi32((int)v.den_n);
            const __m512i vna = _mm512_set1_epi32((int)v.naa_a);
            for (; b0 < v.K; b0 += 16) {
                const __mmask16 lm = (v.K - b0 >= 16) ? (__mmask16)0xFFFF : (__mmask16)((1u << (v.K - b0)) - 1);
                const __m512i va = _mm512_maskz_loadu_epi32(lm, v.ra + b0);
                const __m512i vo = _mm512_maskz_loadu_epi32(lm, v.ro + b0);
                const __m512i vn = _mm512_maskz_loadu_epi32(lm, v.naa_u + b0);
                const __m512i num = _mm512_sub_epi32(_mm512_add_epi32(va, vnl), _mm512_add_epi32(vo, vo));
                const __m512i ix = _mm512_add_epi32(_mm512_mullo_epi32(num, vdn), _mm512_add_epi32(vna, vn));
                _mm512_mask_storeu_epi32(idx + b0, lm, ix);
                const __m512i g = _mm512_mask_i32gather_epi32(_mm512_setzero_si512(), lm, _mm512_add_epi32(ix, ix),
                                                              (const void *)(v.phi_tab + 12), 8);
                __mmask16 need = _mm512_mask_testn_epi32_mask(lm, g, _mm512_set1_epi32((int)0xFF000000u));
                while (need) {
                    const int t = __builtin_ctz(need);
                    need &= need - 1;
                    const int b = b0 + t;
                    const uint32_t ia = v.ra[b], io = v.ro[b];
                    char *s2 = v.phi_tab + (size_t)idx[b] * 16;
                    if (s2[15] == 0) {  // (a slot may repeat within the 16)
                        double phi = ((double)ia - 2.0 * (double)io) / (v.nda + v.nAa[b]);
                        char *e = fmt_g6(s2, phi);
                        *e++ = '\n';
                        slot_lengths(s2, e);
                    }
                }
            }
        }
        for (int b = b0; b < v.K; b++) {
            const uint32_t ia = v.ra[b], io = v.ro[b];
            const int64_t num = (int64_t)ia - 2 * (int64_t)io;
            const int64_t den = (int64_t)v.naa_a + (int64_t)v.naa_u[b];
            const size_t i = (size_t)((num - v.num_lo) * v.den_n + den);
            idx[b] = (uint32_t)i;
            char *s2 = v.phi_tab + i * 16;
            if (__builtin_expect(s2[15] == 0, 0)) {
                // same double expression as vcftools: (N_AaAa - 2.0*N_AAaa) / (N_Aa[ui] + N_Aa[uj])
                double phi = ((double)ia - 2.0 * (double)io) / (v.nda + v.nAa[b]);
                char *e = fmt_g6(s2, phi);
                *e++ = '\n';
                slot_lengths(s2, e);
            }
        }
        return vrow_emit<true>(p, v.K, v.ra, v.ro, v.cnt, v.bt, v.phi_tab, idx, v.atm, v.amask);
    } else if (MODE == 1) {
        return vrow_emit<false>(p, v.K, v.ra, v.ro, v.cnt, v.bt, v.symrow, nullptr, v.atm, v.amask);
    } else {
        for (int b = 0; b < v.K; b++) {
            char *s2 = rowbuf + (size_t)b * 16;
            double phi = ((double)v.ra[b] - 2.0 * (double)v.ro[b]) / (v.nda + v.nAa[b]);
            char *e = fmt_g6(s2, phi);
            *e++ = '\n';
            slot_lengths(s2, e);
        }
        return vrow_emit<false>(p, v.K, v.ra, v.ro, v.cnt, v.bt, rowbuf, nullptr, v.atm, v.amask);
    }
}
#endif

static bool writev_all(int fd, struct iovec *iov, int n) {
    while (n > 0) {
        ssize_t w = ::writev(fd, iov, n);
        if (w < 0) {
            if (errno == EINTR) continue;
            return false;
        }
        size_t left = (size_t)w;
        while (n > 0 && left >= iov[0].iov_len) {
            left -= iov[0].iov_len;
            iov++;
            n--;
        }
        if (n > 0 && left > 0) {
            iov[0].iov_base = (char *)iov[0].iov_base + left;
            iov[0].iov_len -= left;
        }
    }
    return true;
}

bool write_relatedness2(int fd, const std::vector<std::string> &names, const RelCounter &rc) {
    const int K = rc.K;
    static const char hdr[] = "INDV1\tINDV2\tN_AaAa\tN_AAaa\tN1_Aa\tN2_Aa\tRELATEDNESS_PHI\n";
    if (!write_all(fd, hdr, sizeof(hdr) - 1)) return false;
    if (K == 0) return true;
    // A row is "INDV1\tINDV2\tN_AaAa\tN_AAaa\tN1_Aa\tN2_Aa\tPHI\n". Every piece is precomputed in a fixed-size
    // slot so a row is a handful of constant-size copies:
    //   name slots (32 B): "name\t"           count slots (8 B): "<count>\t" for 0..S (S = max count)
    //   N_Aa slots (16 B): "<N_Aa>\t"         phi slots (16 B, per thread, filled on first use): "<phi>\n",
    //   indexed by (N_AaAa - 2 N_AAaa, N1_Aa + N2_Aa); phi is a function of exactly these two integers.
    // Slots that would not fit (long names, counts >= 1e6, huge tables) fall back to direct formatting.
    std::vector<double> nAa(K);
    uint32_t maxc = 0;
    for (int k = 0; k < K; k++) {
        const uint32_t d = rc.fused ? rc.naa[k] : rc.aa[(size_t)k * K + k];
        nAa[k] = (double)d;
        maxc = std::max(maxc, d);
    }
    std::vector<uint32_t> naa_u(K);
    for (int k = 0; k < K; k++) naa_u[k] = (uint32_t)nAa[k];
    // N_AaAa(i,j) <= min(N_Aa(i), N_Aa(j)) <= maxc; N_AAaa(i,j) <= number of used sites; bound it by a scan.
    const uint32_t S = (uint32_t)std::min<uint64_t>(rc.total_sites, 0xffffffffu);
    const bool slots_ok = S < 1000000u;
    bool long_names = false;
    size_t maxname = 0;
    for (int k = 0; k < K; k++) {
        maxname = std::max(maxname, names[k].size());
        if (names[k].size() + 1 > 31) long_names = true;
    }
    const bool fast = slots_ok && !long_names;
    std::vector<char> name_slot, cnt_slot, naa_slot;
    if (fast) {
        name_slot.assign((size_t)K * 32, 0);
        naa_slot.assign((size_t)K * 16, 0);
        for (int k = 0; k < K; k++) {
            char *d = &name_slot[(size_t)k * 32];
            memcpy(d, names[k].data(), names[k].size());
            d[names[k].size()] = '\t';
            d[31] = (char)(names[k].size() + 1);
            char *c = &naa_slot[(size_t)k * 16];
            char *e = fmt_count(c, nAa[k]);
            *e++ = '\t';
            c[15] = (char)(e - c);
        }
        cnt_slot.assign((size_t)(S + 1) * 8, 0);
        for (uint32_t v = 0; v <= S; v++) {
            char *c = &cnt_slot[(size_t)v * 8];
            char *e = fmt_u32(c, v);
            *e++ = '\t';
            c[7] = (char)(e - c);
        }
    }
    // phi table: num in [-2S, S], den in [0, 2S]
    const int64_t num_lo = -2 * (int64_t)S, num_n = 3 * (int64_t)S + 1, den_n = 2 * (int64_t)S + 1;
    const bool use_phi_tab = fast && (num_n * den_n) <= (int64_t)(1 << 22);
    // Without the phi table (many sites), phi(i,j) == phi(j,i): format each unordered pair once, in parallel, into
    // a K x K grid of 16-byte slots ("phi\n", length in byte 15), and let the row formatter copy slots.
    char *sym = nullptr;
    // With two or more threads the grid is not filled in a separate pass: the output workers compute its tiles in
    // tile-row order between formatting and writing chunks, and a chunk is formatted once the tile-rows of its rows
    // are complete (writing is serial, so the other thread would otherwise idle).
    const bool sym_deferred = thread_count() > 1;
    std::vector<std::pair<int, int>> tp;
    std::vector<size_t> tile_row_end;  // tile-row I is complete once tiles [0, tile_row_end[I]) are done
    std::function<void(size_t)> sym_tile;
    if (fast && !use_phi_tab && !rc.fused && K >= 64 && K <= 1024) {
        sym = (char *)big_alloc((size_t)K * K * 16);
        const int TT = 32, nt = (K + TT - 1) / TT;
        for (int I = 0; I < nt; I++) {
            for (int J = I; J < nt; J++) tp.push_back({I, J});
            tile_row_end.push_back(tp.size());
        }
        sym_tile = [&, TT](size_t t) {
            rc.count_deferred(tp[t].first, tp[t].second);  // TT == kTile
            const int i0 = tp[t].first * TT, i1 = std::min(K, i0 + TT);
            const int j0 = tp[t].second * TT, j1 = std::min(K, j0 + TT);
            alignas(64) char loc[32][32][16];
            for (int i = i0; i < i1; i++) {
                const uint32_t *ra = rc.aa + (size_t)i * K;
                const uint32_t *ro = rc.oo + (size_t)i * K;
                int j = (i0 == j0 ? i : j0);
#ifdef SWIFTREL_AVX512DQ
                // 8 at a time: same quotient fl(num / den) as vcftools' (N_AaAa - 2.0*N_AAaa) / (N_Aa[ui] + N_Aa[uj])
                for (; j + 8 <= j1; j += 8) {
                    alignas(64) int64_t nm[8];
                    alignas(64) uint64_t dn[8];
                    for (int t = 0; t < 8; t++) {
                        nm[t] = (int64_t)ra[j + t] - 2 * (int64_t)ro[j + t];
                        dn[t] = (uint64_t)naa_u[i] + naa_u[j + t];
                    }
                    fmt_phi8(loc[i - i0][j - j0], nm, dn);
                }
#endif
                for (; j < j1; j++) {
                    char *slot = loc[i - i0][j - j0];
                    // same double expression as vcftools: (N_AaAa - 2.0*N_AAaa) / (N_Aa[ui] + N_Aa[uj])
                    double phi = ((double)ra[j] - 2.0 * (double)ro[j]) / (nAa[i] + nAa[j]);
                    char *e = fmt_phi(slot, (int64_t)ra[j] - 2 * (int64_t)ro[j], (uint64_t)naa_u[i] + naa_u[j], phi);
                    *e++ = '\n';
                    slot_lengths(slot, e);
                }
            }
            // rows of the tile, then its transpose, as contiguous 16-byte slot runs
            for (int i = i0; i < i1; i++) {
                const int js = (i0 == j0 ? i : j0);
                memcpy(sym + ((size_t)i * K + js) * 16, loc[i - i0][js - j0], (size_t)(j1 - js) * 16);
            }
            for (int j = j0; j < j1; j++) {
                const int ie = (i0 == j0 ? j : i1);  // rows i < j (diagonal tile) or the whole tile
                char *dst = sym + ((size_t)j * K + i0) * 16;
                for (int i = i0; i < ie; i++) memcpy(dst + (size_t)(i - i0) * 16, loc[i - i0][j - j0], 16);
            }
        };
        if (!sym_deferred) parallel_for(tp.size(), sym_tile);
    }
#ifdef SWIFTREL_VBMI2
    // Compressed rows: when every name is <= 7 bytes, a row fits a fixed 64-byte layout of four 16-byte lanes
    //   [name_a | name_b] [N_AaAa | N_AAaa] [N1_Aa | N2_Aa] [phi]   (8-byte fields, phi 16 bytes)
    // A row is assembled in one zmm register from slots and the bytes beyond each field's length are squeezed out
    // with VPCOMPRESSB (mask built from the slot lengths); one 64-byte store and p += popcount(mask).
    // Per column b the name_b and N2_Aa parts (and their mask bits) are prepared once in a 64-byte template.
    const bool vrows = fast && maxname + 1 <= 8;
    __m512i *btmpl = nullptr;
    uint64_t *bmask = nullptr;
    if (vrows) {
        btmpl = (__m512i *)aligned_alloc(64, (size_t)K * 64);
        bmask = (uint64_t *)malloc((size_t)K * 8);
        for (int b = 0; b < K; b++) {
            alignas(64) unsigned char t[64];
            memset(t, 0, 64);
            const char *nb = &name_slot[(size_t)b * 32];
            const char *sb = &naa_slot[(size_t)b * 16];
            memcpy(t + 8, nb, 8);
            memcpy(t + 40, sb, 8);
            bmask[b] = (_bzhi_u64(0xFFull, (unsigned)nb[31]) << 8) | (_bzhi_u64(0xFFull, (unsigned)sb[15]) << 40);
            memcpy(t + 16, &bmask[b], 8);  // read by vrow_emit; this lane is replaced by the count pair
            btmpl[b] = _mm512_load_si512((const void *)t);
        }
    }
#endif
    const size_t row_max = 2 * (maxname + 1) + 2 * 32 + 2 * 32 + 40;
    // chunks of whole rows, ~1 MB each; a fixed pool of buffers is reused (no zeroing, no page faults)
    size_t rows_per_chunk = std::max<size_t>(1, (size_t)(1u << 20) / (row_max * (size_t)K + 1));
    size_t nchunks = ((size_t)K + rows_per_chunk - 1) / rows_per_chunk;
    // the dedicated-writer layout (below) batches consecutive chunks into one writev of >= 1 MB, so it gets a
    // larger window of buffers
    const bool dedicated_writer = thread_count() > 1 && nchunks > 1 && use_phi_tab && nchunks >= 16;
    const size_t window = std::min<size_t>(nchunks, dedicated_writer ? 16 : 6);
    const size_t cap = rows_per_chunk * row_max * (size_t)K + 64;
    std::vector<char *> pool(window);
    for (auto &b : pool) b = (char *)malloc(cap);
    std::vector<size_t> lens(nchunks, 0);
    std::vector<char> ready(nchunks, 0);
    std::mutex mu;
    std::condition_variable cv_ready, cv_space;
    size_t written = 0;
    std::atomic<size_t> next(0);
    auto fmt_chunk = [&](size_t c, char *phi_tab) {
        size_t r0 = c * rows_per_chunk, r1 = std::min((size_t)K, r0 + rows_per_chunk);
        char *base = pool[c % window];
        char *p = base;
        std::vector<uint32_t> strip_a, strip_o;
        std::vector<char> vscratch;
        if (rc.fused) {
            strip_a.resize((size_t)K + 8);
            strip_o.resize((size_t)K + 8);
        }
        for (size_t a = r0; a < r1; a++) {
            const uint32_t *ra, *ro;
            if (rc.fused) {
                rc.fused_row((int)a, strip_a.data(), strip_o.data());
                ra = strip_a.data();
                ro = strip_o.data();
            } else {
                ra = rc.aa + a * K;
                ro = rc.oo + a * K;
            }
            const double nda = nAa[a];
            if (fast) {
                const char *na = &name_slot[a * 32];
                const size_t lna = (size_t)na[31];
                const char *sa = &naa_slot[a * 16];
                const size_t lsa = (size_t)sa[15];
                const uint32_t naa_a = (uint32_t)nda;
#ifdef SWIFTREL_VBMI2
                if (vrows) {
                    alignas(64) unsigned char t[64];
                    memset(t, 0, 64);
                    memcpy(t, na, 8);
                    memcpy(t + 32, sa, 8);
                    const __m512i atm = _mm512_load_si512((const void *)t);
                    const uint64_t amask = _bzhi_u64(0xFFull, (unsigned)lna) | (_bzhi_u64(0xFFull, (unsigned)lsa) << 32);
                    VRowArgs v{K, ra, ro, cnt_slot.data(), btmpl, bmask, naa_u.data(), nAa.data(), atm, amask, naa_a, nda,
                               phi_tab, num_lo, den_n, sym ? sym + a * (size_t)K * 16 : nullptr};
                    if (vscratch.empty()) vscratch.resize((size_t)K * 20 + 64);
                    uint32_t *vidx = (uint32_t *)vscratch.data();
                    char *vrow = (char *)(((uintptr_t)(vidx + K) + 63) & ~(uintptr_t)63);
                    if (use_phi_tab) p = vrow_loop<0>(p, v, vidx, vrow);
                    else if (sym) p = vrow_loop<1>(p, v, vidx, vrow);
                    else p = vrow_loop<2>(p, v, vidx, vrow);
                    continue;
                }
#endif
                for (int b = 0; b < K; b++) {
                    const char *nb = &name_slot[(size_t)b * 32];
                    memcpy(p, na, 32);
                    p += lna;
                    memcpy(p, nb, 32);
                    p += (size_t)nb[31];
                    const uint32_t ia = ra[b], io = ro[b];
                    const char *ca = &cnt_slot[(size_t)ia * 8];
                    memcpy(p, ca, 8);
                    p += (size_t)ca[7];
                    const char *co = &cnt_slot[(size_t)io * 8];
                    memcpy(p, co, 8);
                    p += (size_t)co[7];
                    memcpy(p, sa, 16);
                    p += lsa;
                    const char *sb = &naa_slot[(size_t)b * 16];
                    memcpy(p, sb, 16);
                    p += (size_t)sb[15];
                    if (use_phi_tab) {
                        const int64_t num = (int64_t)ia - 2 * (int64_t)io;
                        const int64_t den = (int64_t)naa_a + (int64_t)(uint32_t)nAa[b];
                        char *slot = phi_tab + (size_t)(((num - num_lo) * den_n + den) * 16);
                        if (slot[15] == 0) {
                            // same double expression as vcftools: (N_AaAa - 2.0*N_AAaa) / (N_Aa[ui] + N_Aa[uj])
                            double phi = ((double)ia - 2.0 * (double)io) / (nda + nAa[b]);
                            char *e = fmt_g6(slot, phi);
                            *e++ = '\n';
                            slot_lengths(slot, e);
                        }
                        memcpy(p, slot, 16);
                        p += (size_t)slot[15];
                    } else if (sym) {
                        const char *slot = sym + (a * (size_t)K + (size_t)b) * 16;
                        memcpy(p, slot, 16);
                        p += (size_t)slot[15];
                    } else {
                        double phi = ((double)ia - 2.0 * (double)io) / (nda + nAa[b]);
                        p = fmt_g6(p, phi);
                        *p++ = '\n';
                    }
                }
            } else {
                for (int b = 0; b < K; b++) {
                    memcpy(p, names[a].data(), names[a].size());
                    p += names[a].size();
                    *p++ = '\t';
                    memcpy(p, names[b].data(), names[b].size());
                    p += names[b].size();
                    *p++ = '\t';
                    const uint32_t ia = ra[b], io = ro[b];
                    p = fmt_count_u(p, ia);
                    *p++ = '\t';
                    p = fmt_count_u(p, io);
                    *p++ = '\t';
                    p = fmt_count(p, nda);
                    *p++ = '\t';
                    p = fmt_count(p, nAa[b]);
                    *p++ = '\t';
                    // same double expression as vcftools: (N_AaAa - 2.0*N_AAaa) / (N_Aa[ui] + N_Aa[uj])
                    double phi = ((double)ia - 2.0 * (double)io) / (nda + nAa[b]);
                    p = fmt_g6(p, phi);
                    *p++ = '\n';
                }
            }
        }
        lens[c] = (size_t)(p - base);
    };
    // Three layouts: one thread formats then writes; expensive rows -> ordered hand-off; large output with cheap
    // rows -> dedicated writer.
    if (!sym) rc.complete_deferred();  // the sym grid tiles count their own deferred tile pairs
    bool ok = true;
    const int T = thread_count();
    if (sym && sym_deferred && (T <= 1 || nchunks == 1)) parallel_for(tp.size(), sym_tile);  // not interleaved
    if (T <= 1 || nchunks == 1) {
        char *phi_tab = use_phi_tab ? (char *)calloc((size_t)(num_n * den_n), 16) : nullptr;
        for (size_t c = 0; c < nchunks; c++) {
            fmt_chunk(c, phi_tab);
            if (ok) ok = write_all(fd, pool[c % window], lens[c]);
        }
        free(phi_tab);
    } else if (sym && sym_deferred) {
        // Symmetric grid tiles and output chunks interleaved (see sym_deferred). Tiles are taken in tile-row order;
        // done_prefix = number of leading tiles all finished. A chunk of rows [r0, r1) needs tile-rows up to
        // (r1 - 1) / 32, i.e. done_prefix >= tile_row_end[(r1 - 1) / 32]; it is written in order (written == c).
        std::vector<char> tdone(tp.size(), 0);
        size_t next_tile = 0, done_prefix = 0, next_chunk = 0;
        auto chunk_ready = [&](size_t c) {
            const size_t r1 = std::min((size_t)K, (c + 1) * rows_per_chunk);
            return done_prefix >= tile_row_end[(r1 - 1) / 32];
        };
        auto worker = [&]() {
            std::unique_lock<std::mutex> lk(mu);
            for (;;) {
                if (next_chunk < nchunks && next_chunk < written + window && chunk_ready(next_chunk)) {
                    const size_t c = next_chunk++;
                    lk.unlock();
                    fmt_chunk(c, nullptr);
                    lk.lock();
                    // wait for this chunk's turn, computing tiles meanwhile
                    while (written != c) {
                        if (next_tile < tp.size()) {
                            const size_t t = next_tile++;
                            lk.unlock();
                            sym_tile(t);
                            lk.lock();
                            tdone[t] = 1;
                            while (done_prefix < tp.size() && tdone[done_prefix]) done_prefix++;
                            cv_space.notify_all();
                        } else {
                            cv_space.wait(lk);
                        }
                    }
                    lk.unlock();
                    const bool w = write_all(fd, pool[c % window], lens[c]);
                    lk.lock();
                    if (!w) ok = false;
                    written = c + 1;
                    cv_space.notify_all();
                    continue;
                }
                if (next_tile < tp.size()) {
                    const size_t t = next_tile++;
                    lk.unlock();
                    sym_tile(t);
                    lk.lock();
                    tdone[t] = 1;
                    while (done_prefix < tp.size() && tdone[done_prefix]) done_prefix++;
                    cv_space.notify_all();
                    continue;
                }
                if (next_chunk >= nchunks) break;
                cv_space.wait(lk);
            }
        };
        int nt = std::max(1, std::min<int>(T, (int)nchunks));
        std::vector<std::thread> th;
        for (int t = 1; t < nt; t++) th.emplace_back(worker);
        worker();
        for (auto &x : th) x.join();
    } else if (!(use_phi_tab && nchunks >= 16)) {
        // Rows are expensive to format (no phi table) or the output is small: all T threads format, and each
        // writes its own chunk when the earlier chunks are written (ordered hand-off).
        auto worker = [&]() {
            char *phi_tab = nullptr;
            for (;;) {
                size_t c = next.fetch_add(1);
                if (c >= nchunks) break;
                {
                    std::unique_lock<std::mutex> lk(mu);
                    cv_space.wait(lk, [&] { return c < written + window; });
                }
                if (use_phi_tab && !phi_tab) phi_tab = (char *)calloc((size_t)(num_n * den_n), 16);
                fmt_chunk(c, phi_tab);
                {
                    std::unique_lock<std::mutex> lk(mu);
                    cv_space.wait(lk, [&] { return written == c; });
                }
                bool w = write_all(fd, pool[c % window], lens[c]);
                {
                    std::lock_guard<std::mutex> lk(mu);
                    if (!w) ok = false;
                    written = c + 1;
                }
                cv_space.notify_all();
            }
            free(phi_tab);
        };
        int nt = std::max(1, std::min<int>(T, (int)nchunks));
        std::vector<std::thread> th;
        for (int t = 1; t < nt; t++) th.emplace_back(worker);
        worker();
        for (auto &x : th) x.join();
    } else {
        // Large output with cheap rows (phi table): the page-cache copy in write(2) is comparable to formatting,
        // so T-1 threads format and this thread only writes.
        auto formatter = [&]() {
            char *phi_tab = nullptr;
            for (;;) {
                size_t c = next.fetch_add(1);
                if (c >= nchunks) break;
                {
                    std::unique_lock<std::mutex> lk(mu);
                    cv_space.wait(lk, [&] { return c < written + window; });
                }
                if (use_phi_tab && !phi_tab) phi_tab = (char *)calloc((size_t)(num_n * den_n), 16);
                fmt_chunk(c, phi_tab);
                {
                    std::lock_guard<std::mutex> lk(mu);
                    ready[c] = 1;
                }
                cv_ready.notify_all();
            }
            free(phi_tab);
        };
        int nf = std::max(1, std::min<int>(T - 1, (int)nchunks));
        std::vector<std::thread> th;
        for (int t = 0; t < nf; t++) th.emplace_back(formatter);
        // Each write(2) costs ~10 us beyond the copy (238 MB in 190 KB writes: 58 ms, in 1 MB writes: 48 ms), so
        // consecutive formatted chunks are written together with writev once they add up to >= 1 MB.
        for (size_t c = 0; c < nchunks;) {
            struct iovec iov[16];
            int niov = 0;
            size_t total = 0, e = c;
            while (e < nchunks && niov < 16 && e < c + window && total < ((size_t)1 << 20)) {
                {
                    std::unique_lock<std::mutex> lk(mu);
                    cv_ready.wait(lk, [&] { return ready[e] != 0; });
                }
                iov[niov].iov_base = pool[e % window];
                iov[niov].iov_len = lens[e];
                total += lens[e];
                niov++;
                e++;
            }
            if (ok) ok = writev_all(fd, iov, niov);
            {
                std::lock_guard<std::mutex> lk(mu);
                written = e;
            }
            cv_space.notify_all();
            c = e;
        }
        for (auto &x : th) x.join();
    }
    for (auto &b : pool) free(b);
    free(sym);
#ifdef SWIFTREL_VBMI2
    free(btmpl);
    free(bmask);
#endif
    return ok;
}

}  // namespace swiftrel
