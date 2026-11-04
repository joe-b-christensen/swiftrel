// swiftrel_fast.cpp -- the SwiftRel fast path for vcftools --relatedness2.
// Part of SwiftRel (LGPL-3.0, derived from vcftools 0.1.17).
//
// Parses the data lines in place (no per-line string copies), applies the supported site filters with the exact
// semantics and order of vcftools' entry::apply_filters, classifies genotypes like vcf_entry::parse_genotype_entry,
// and feeds used sites to the bit-parallel counter. Anything it cannot reproduce exactly (lines with fewer columns
// than the header -- vcftools then reuses stale fields from earlier lines -- or NUL bytes) makes it return
// "not applicable" before any output, and the caller runs the original vcftools loop instead.
#include "swiftrel_fast.h"

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <thread>

#include "header.h"
#include "output_log.h"
#include "swiftrel_engine.h"

#include <immintrin.h>

extern output_log LOG;

namespace swiftrel {

namespace {

struct Cfg {
    // filters
    bool remove_indels;
    bool remove_filtered_all;
    bool pos_filter;  // single --chr with --from-bp/--to-bp
    std::string pos_chr;
    int start_pos, end_pos;
    std::vector<std::string> chrs;  // --chr set (empty = all)
    bool nalleles_filter;
    int min_alleles, max_alleles;
    double min_quality;
    bool freq_filter;  // filter_sites_by_frequency_and_call_rate active
    double min_maf, max_maf, min_site_call_rate;
    bool mac_filter;  // filter_sites_by_allele_count active
    int min_mac, max_mac, max_missing_call_count;
    int thin;  // min_interSNP_distance
    // samples
    unsigned N_indv;
    const std::vector<bool> *include;
    std::vector<int> kidx;  // ui -> compact index or -1
    std::vector<uint64_t> incw;  // N-bit include mask
    bool all_inc;
    int K;
};

struct Rec {
    const char *chrom;
    uint32_t chrom_len;
    int pos;
    uint8_t reach;      // passed the filters before the genotype-based ones
    uint8_t err_freq;   // raises an error in the MAF/MAC stage
    uint8_t pass_freq;  // passes MAF/call-rate and MAC filters
    uint8_t biallelic;  // N_alleles == 2
    uint8_t poly;       // a polyploid GT -> error when relatedness parses the site
    uint8_t diploid;    // every included individual has ploidy 2
    uint8_t exotic;     // has homozygous alleles outside {0,1}
    int64_t code_off;   // offset into the codes buffer (-1: none)
    int64_t allele_off; // offset into the alleles buffer (-1: none)
};

struct Part {
    const char *b, *e;
    std::vector<Rec> recs;
    std::vector<uint64_t> codes;  // per candidate site: het, h0, h1 masks of KW words each
    std::vector<int> alleles;
    bool bad = false;  // short line or NUL -> fallback
    unsigned n_entries = 0;
};

// glibc atoi == (int)strtol(s, 0, 10) on the NUL-terminated field.
static inline int atoi_field(const char *b, const char *e) {
    // fast path: optional digits only, up to 9 digits
    size_t n = (size_t)(e - b);
    if (n > 0 && n <= 9) {
        int v = 0;
        const char *p = b;
        for (; p < e; p++) {
            unsigned d = (unsigned)(*p - '0');
            if (d > 9) break;
            v = v * 10 + (int)d;
        }
        if (p == e) return v;
    }
    std::string s(b, e);
    return atoi(s.c_str());
}

// header::str2int: "" or "." -> -1, else atoi
static inline int str2int_field(const char *b, const char *e) {
    if (b == e || (e - b == 1 && *b == '.')) return -1;
    return atoi_field(b, e);
}

static inline double str2double_field(const char *b, const char *e) {
    if (b == e || (e - b == 1 && *b == '.')) return -1.0;
    std::string s(b, e);
    return atof(s.c_str());
}

static inline size_t count_tabs(const char *b, const char *e) {
    size_t n = 0;
#if defined(__AVX512BW__)
    const __m512i t = _mm512_set1_epi8('\t');
    while (e - b >= 64) {
        __m512i v = _mm512_loadu_si512((const void *)b);
        n += (size_t)__builtin_popcountll(_mm512_cmpeq_epi8_mask(v, t));
        b += 64;
    }
#endif
    for (; b < e; b++) n += (*b == '\t');
    return n;
}

// Complex GT string (anything but the 3-char "x/y" form): vcf_entry::set_indv_GENOTYPE_and_PHASE.
static inline void parse_gt_complex(const char *b, const char *e, int &a1, int &a2, int &ploidy, bool &poly) {
    size_t n = (size_t)(e - b);
    if (n == 3 && (b[1] == '/' || b[1] == '|')) {
        a1 = b[0] == '.' ? -1 : b[0] - '0';
        a2 = b[2] == '.' ? -1 : b[2] - '0';
        ploidy = 2;
        return;
    }
    const char *sep = nullptr, *last = nullptr;
    for (const char *p = b; p < e; p++)
        if (*p == '/' || *p == '|') {
            if (!sep) sep = p;
            last = p;
        }
    if (!sep) {
        ploidy = 1;
        a1 = str2int_field(b, e);
        a2 = -1;
    } else {
        ploidy = 2;
        a1 = str2int_field(b, sep);
        a2 = str2int_field(sep + 1, e);
        if (last != sep) poly = true;
    }
}

// Index from which vcftools' unchecked allele_counts[a]++ leaves glibc's malloc slack of vector<int>(n).
static inline int slack_index(int n_alleles) {
    const size_t req = (size_t)n_alleles * 4;
    return (int)(std::max<size_t>(24, ((req + 8 + 15) & ~(size_t)15) - 8) / 4);
}

// entry::filter_sites_by_frequency_and_call_rate + entry::filter_sites_by_allele_count, given the counts.
// Sets r.err_freq (1: vcftools error, 2: out-of-bounds write refused) and returns pass/fail.
static inline bool freq_decision(const Cfg &cfg, const std::vector<int> &acount, int N_alleles,
                                 unsigned N_non_missing_chr, unsigned N_chr, bool poly, int GT_idx, bool oob, Rec &r) {
    bool fpass = true;
    if (oob) r.err_freq = 2;
    if (cfg.freq_filter) {
        if (poly || GT_idx == -1) r.err_freq = 1;  // these come first in vcftools
        double maf = std::numeric_limits<double>::max();
        for (int ui = 0; ui < N_alleles; ui++) {
            double freq = acount[(size_t)ui] / (double)N_non_missing_chr;
            double folded_freq = std::min(freq, 1.0 - freq);
            maf = std::min(maf, folded_freq);
        }
        if ((maf < cfg.min_maf) || (maf > cfg.max_maf)) fpass = false;
        double call_rate = N_non_missing_chr / double(N_chr);
        if (call_rate < cfg.min_site_call_rate) fpass = false;
    }
    if (fpass && cfg.mac_filter) {
        if (poly && !r.err_freq) r.err_freq = 1;
        // vcftools passes these limits as double
        if (N_alleles <= 1 && (double)cfg.min_mac > 0) fpass = false;
        int mac = std::numeric_limits<int>::max();
        for (int ui = 0; ui < N_alleles; ui++) mac = std::min(acount[(size_t)ui], mac);
        if (((double)mac < (double)cfg.min_mac) || ((double)mac > (double)cfg.max_mac)) fpass = false;
        if ((double)(N_chr - N_non_missing_chr) > (double)cfg.max_missing_call_count) fpass = false;
    }
    return fpass;
}

#if defined(__AVX512BW__) && defined(__BMI2__)
#define SWIFTREL_SIMD_GT 1
#endif

// A "regular" genotype region: N columns that are exactly "x|y" or "x/y" with x, y in {0, 1, .}, separated by
// single tabs, the last ending at the line end or at a tab. For such lines vcftools' fast GT branch gives
// alleles 0/1/-1, ploidy 2, no polyploidy. Fills N-bit masks: allele a is 0 / 1, allele b is 0 / 1.
static bool regular_gt_line(const char *c, const char *le, unsigned N, uint64_t *A0, uint64_t *A1, uint64_t *B0,
                            uint64_t *B1) {
    const unsigned NW = (N + 63) / 64;
    for (unsigned w = 0; w < NW; w++) A0[w] = A1[w] = B0[w] = B1[w] = 0;
    unsigned ui = 0;
#ifdef SWIFTREL_SIMD_GT
    const __m512i vbar = _mm512_set1_epi8('|'), vsl = _mm512_set1_epi8('/'), vtab = _mm512_set1_epi8('\t');
    const __m512i v0 = _mm512_set1_epi8('0'), v1 = _mm512_set1_epi8('1'), vdot = _mm512_set1_epi8('.');
    const uint64_t APOS = 0x1111111111111111ull, SPOS = 0x2222222222222222ull, BPOS = 0x4444444444444444ull,
                   TPOS = 0x8888888888888888ull;
    while (ui + 16 <= N && c + 64 <= le) {
        __m512i v = _mm512_loadu_si512((const void *)c);
        uint64_t ms = _mm512_cmpeq_epi8_mask(v, vbar) | _mm512_cmpeq_epi8_mask(v, vsl);
        uint64_t mt = _mm512_cmpeq_epi8_mask(v, vtab);
        uint64_t m0 = _mm512_cmpeq_epi8_mask(v, v0), m1 = _mm512_cmpeq_epi8_mask(v, v1);
        uint64_t md = _mm512_cmpeq_epi8_mask(v, vdot);
        if (((ms & SPOS) != SPOS) | ((mt & TPOS) != TPOS) | (((m0 | m1 | md) & (APOS | BPOS)) != (APOS | BPOS)))
            return false;
        const unsigned w = ui >> 6, sh = ui & 63;
        A0[w] |= _pext_u64(m0, APOS) << sh;
        A1[w] |= _pext_u64(m1, APOS) << sh;
        B0[w] |= _pext_u64(m0, BPOS) << sh;
        B1[w] |= _pext_u64(m1, BPOS) << sh;
        ui += 16;
        c += 64;
    }
#endif
    for (; ui < N; ui++) {
        if (c + 3 > le) return false;
        const char x = c[0], sp = c[1], y = c[2];
        if (!(sp == '|' || sp == '/')) return false;
        if (!(x == '0' || x == '1' || x == '.') || !(y == '0' || y == '1' || y == '.')) return false;
        if (c + 3 == le) {
            if (ui != N - 1) return false;  // short line: the general path decides
        } else if (c[3] != '\t')
            return false;
        const unsigned w = ui >> 6;
        const uint64_t bit = 1ull << (ui & 63);
        if (x == '0') A0[w] |= bit;
        if (x == '1') A1[w] |= bit;
        if (y == '0') B0[w] |= bit;
        if (y == '1') B1[w] |= bit;
        c += 4;
    }
    return true;
}

// Append the bits of `src` selected by `sel` (N-bit masks) to a K-bit destination.
static inline void compact_bits(const uint64_t *src, const uint64_t *sel, unsigned NW, bool all, uint64_t *dst,
                                unsigned KW) {
    if (all) {
        for (unsigned w = 0; w < KW; w++) dst[w] = src[w];
        return;
    }
    for (unsigned w = 0; w < KW; w++) dst[w] = 0;
    unsigned pos = 0;
    for (unsigned w = 0; w < NW; w++) {
#ifdef __BMI2__
        uint64_t v = _pext_u64(src[w], sel[w]);
#else
        uint64_t v = 0, m = sel[w];
        for (unsigned b = 0, o = 0; m; b++, m >>= 1)
            if (m & 1) v |= ((src[w] >> b) & 1ull) << o++;
#endif
        unsigned n = (unsigned)__builtin_popcountll(sel[w]);
        if (n == 0) continue;
        dst[pos >> 6] |= v << (pos & 63);
        if ((pos & 63) + n > 64) dst[(pos >> 6) + 1] |= v >> (64 - (pos & 63));
        pos += n;
    }
}

// Position of the first tab in [t, le), or nullptr.
static inline const char *find_tab(const char *t, const char *le) {
#if defined(__AVX2__)
    const __m256i vt = _mm256_set1_epi8('\t');
    while (le - t >= 32) {
        unsigned m = (unsigned)_mm256_movemask_epi8(_mm256_cmpeq_epi8(_mm256_loadu_si256((const __m256i *)t), vt));
        if (m) return t + __builtin_ctz(m);
        t += 32;
    }
#endif
    return (const char *)memchr(t, '\t', (size_t)(le - t));
}

static inline bool is01dot(char x) { return x == '0' || x == '1' || x == '.'; }

// "Semi-regular" genotype region (GT first in FORMAT): every column's GT subfield is either 3 chars "x|y" / "x/y"
// or 1 char "x" (haploid), with x, y in {0, 1, .}, followed by ':' / tab / line end; anything may follow the
// ':' (DP, GQ, AD, PL, ...). vcftools then gives alleles 0/1/-1, ploidy 2 or 1 and never polyploidy. Fills the
// N-bit allele masks plus HP (ploidy-1 columns). Returns false for anything else (the general path decides).
static bool semi_regular_gt_line(const char *c, const char *le, unsigned N, uint64_t *A0, uint64_t *A1, uint64_t *B0,
                                 uint64_t *B1, uint64_t *HP) {
    const unsigned NW = (N + 63) / 64;
    for (unsigned w = 0; w < NW; w++) A0[w] = A1[w] = B0[w] = B1[w] = HP[w] = 0;
    unsigned ui = 0;
#ifdef SWIFTREL_SIMD_GT
    // 64 bytes at a time: column starts are offset 0 and the bytes after tabs; starts at offsets <= 60 are
    // classified with the same rules as below (the last start of the block is redone by the next block).
    {
        const __m512i vbar = _mm512_set1_epi8('|'), vsl = _mm512_set1_epi8('/'), vtab = _mm512_set1_epi8('\t');
        const __m512i vcol = _mm512_set1_epi8(':'), v0 = _mm512_set1_epi8('0'), v1 = _mm512_set1_epi8('1');
        const __m512i vdot = _mm512_set1_epi8('.');
        auto put = [](uint64_t *dst, unsigned pos, uint64_t x, unsigned n) {
            const unsigned w = pos >> 6, sh = pos & 63;
            dst[w] |= x << sh;
            if (sh + n > 64) dst[w + 1] |= x >> (64 - sh);
        };
        // Fixed 64-byte stride while a whole next block lies before the line end: every column start of the block
        // (offsets 0-63) is classified, the 1-3 byte look-ahead of starts near the block end comes from the next
        // block's masks, and a tab in byte 63 makes byte 0 of the next block a start. No loop-carried dependence on
        // where the last column started, so consecutive blocks overlap in the CPU.
        if (ui < N && c + 128 <= le) {
            struct M {
                uint64_t mt, sep, term, m0, m1, D;
            };
            auto masks = [&](const char *p) {
                const __m512i v = _mm512_loadu_si512((const void *)p);
                M m;
                m.mt = _mm512_cmpeq_epi8_mask(v, vtab);
                m.sep = _mm512_cmpeq_epi8_mask(v, vbar) | _mm512_cmpeq_epi8_mask(v, vsl);
                m.term = m.mt | _mm512_cmpeq_epi8_mask(v, vcol);
                m.m0 = _mm512_cmpeq_epi8_mask(v, v0);
                m.m1 = _mm512_cmpeq_epi8_mask(v, v1);
                m.D = m.m0 | m.m1 | _mm512_cmpeq_epi8_mask(v, vdot);
                return m;
            };
            M cur = masks(c);
            uint64_t carry = 1;  // c is a column start
            while (c + 128 <= le) {
                const M nx = masks(c + 64);
                uint64_t U = (cur.mt << 1) | carry;
                const uint64_t ncarry = cur.mt >> 63;
                unsigned n = (unsigned)__builtin_popcountll(U);
                const bool last_block = ui + n >= N;
                if (ui + n > N) {
                    U = _pdep_u64((1ull << (N - ui)) - 1, U);
                    n = N - ui;
                }
                const uint64_t sep1 = (cur.sep >> 1) | (nx.sep << 63);
                const uint64_t term1 = (cur.term >> 1) | (nx.term << 63);
                const uint64_t term3 = (cur.term >> 3) | (nx.term << 61);
                const uint64_t D2 = (cur.D >> 2) | (nx.D << 62);
                const uint64_t dip = U & sep1 & term3;
                const uint64_t hap = U & ~dip & term1;
                if (((dip & cur.D & D2) | (hap & cur.D)) != U) return false;
                if (n) {
                    const uint64_t m02 = (cur.m0 >> 2) | (nx.m0 << 62), m12 = (cur.m1 >> 2) | (nx.m1 << 62);
                    put(A0, ui, _pext_u64(cur.m0, U), n);
                    put(A1, ui, _pext_u64(cur.m1, U), n);
                    put(B0, ui, _pext_u64(m02 & dip, U), n);
                    put(B1, ui, _pext_u64(m12 & dip, U), n);
                    put(HP, ui, _pext_u64(hap, U), n);
                    ui += n;
                }
                if (last_block) return true;  // all N columns classified (as the loop below does when ui reaches N)
                c += 64;
                cur = nx;
                carry = ncarry;
            }
            // c is a block start: a column starts here (carry) or after the tab that ends the last started column
            if (!carry) {
                const char *tab = find_tab(c, le);
                c = tab ? tab + 1 : le + 1;
            }
        }
        while (ui < N && c + 64 <= le) {
            const __m512i v = _mm512_loadu_si512((const void *)c);
            const uint64_t mt = _mm512_cmpeq_epi8_mask(v, vtab);
            const uint64_t sep = _mm512_cmpeq_epi8_mask(v, vbar) | _mm512_cmpeq_epi8_mask(v, vsl);
            const uint64_t term = mt | _mm512_cmpeq_epi8_mask(v, vcol);
            const uint64_t m0 = _mm512_cmpeq_epi8_mask(v, v0), m1 = _mm512_cmpeq_epi8_mask(v, v1);
            const uint64_t D = m0 | m1 | _mm512_cmpeq_epi8_mask(v, vdot);
            const uint64_t S = ((mt << 1) | 1ull) & ((1ull << 61) - 1);
            const int last = 63 - __builtin_clzll(S);
            uint64_t U = S & ~(1ull << last);
            if (U == 0) break;  // a column longer than 60 bytes: the scalar loop takes over
            unsigned n = (unsigned)__builtin_popcountll(U);
            if (ui + n > N) {
                U = _pdep_u64((1ull << (N - ui)) - 1, U);
                n = N - ui;
            }
            const uint64_t dip = U & (sep >> 1) & (term >> 3);
            const uint64_t hap = U & ~dip & (term >> 1);
            if (((dip & D & (D >> 2)) | (hap & D)) != U) return false;
            put(A0, ui, _pext_u64(m0, U), n);
            put(A1, ui, _pext_u64(m1, U), n);
            put(B0, ui, _pext_u64((m0 >> 2) & dip, U), n);
            put(B1, ui, _pext_u64((m1 >> 2) & dip, U), n);
            put(HP, ui, _pext_u64(hap, U), n);
            ui += n;
            c += last;
        }
    }
#endif
    for (; ui < N; ui++) {
        if (c >= le) return false;  // missing or empty column: general path
        const unsigned w = ui >> 6, sh = ui & 63;
        const char *t;
        if (__builtin_expect(c + 4 <= le, 1)) {
            // not at the line end: branch-free classification of c[0..3]
            const char x = c[0], s1 = c[1], y = c[2], s3 = c[3];
            const bool sep = (s1 == '|') | (s1 == '/');
            const bool term3 = (s3 == ':') | (s3 == '\t');
            if (__builtin_expect(sep & term3, 1)) {
                if (!(is01dot(x) & is01dot(y))) return false;
                A0[w] |= (uint64_t)(x == '0') << sh;
                A1[w] |= (uint64_t)(x == '1') << sh;
                B0[w] |= (uint64_t)(y == '0') << sh;
                B1[w] |= (uint64_t)(y == '1') << sh;
                t = c + 3;
            } else if ((s1 == ':') | (s1 == '\t')) {
                if (!is01dot(x)) return false;
                HP[w] |= 1ull << sh;
                A0[w] |= (uint64_t)(x == '0') << sh;
                A1[w] |= (uint64_t)(x == '1') << sh;
                t = c + 1;
            } else
                return false;
        } else {
            // within 3 bytes of the line end: never read past le
            const char x = c[0];
            if (c + 3 <= le && (c[1] == '|' || c[1] == '/') && (c + 3 == le || c[3] == ':' || c[3] == '\t')) {
                const char y = c[2];
                if (!is01dot(x) || !is01dot(y)) return false;
                A0[w] |= (uint64_t)(x == '0') << sh;
                A1[w] |= (uint64_t)(x == '1') << sh;
                B0[w] |= (uint64_t)(y == '0') << sh;
                B1[w] |= (uint64_t)(y == '1') << sh;
                t = c + 3;
            } else if (c + 1 == le || c[1] == ':' || c[1] == '\t') {
                if (!is01dot(x)) return false;
                HP[w] |= 1ull << sh;
                A0[w] |= (uint64_t)(x == '0') << sh;
                A1[w] |= (uint64_t)(x == '1') << sh;
                t = c + 1;
            } else
                return false;
        }
        if (t == le) {
            c = le + 1;
        } else if (*t == '\t') {
            c = t + 1;
        } else {
            const char *tab = find_tab(t, le);
            c = tab ? tab + 1 : le + 1;
        }
    }
    return true;
}

// First '\n' or NUL byte in [p, end): returns the newline (nul = false), or sets nul = true if a NUL byte comes
// first, or returns nullptr if there is neither. Never reads at or beyond end.
static inline const char *find_eol(const char *p, const char *end, bool &nul) {
#ifdef SWIFTREL_SIMD_GT
    const __m512i vnl = _mm512_set1_epi8('\n'), vz = _mm512_setzero_si512();
    while (p < end) {
        const size_t n = (size_t)(end - p);
        const uint64_t valid = n >= 64 ? ~0ull : _bzhi_u64(~0ull, (unsigned)n);
        const __m512i v = _mm512_maskz_loadu_epi8(valid, (const void *)p);
        const uint64_t m = (_mm512_cmpeq_epi8_mask(v, vnl) | _mm512_cmpeq_epi8_mask(v, vz)) & valid;
        if (m) {
            const char *q = p + __builtin_ctzll(m);
            if (*q == '\0') {
                nul = true;
                return nullptr;
            }
            return q;
        }
        p += 64;
    }
    return nullptr;
#else
    const char *nl = (const char *)memchr(p, '\n', (size_t)(end - p));
    const char *le = nl ? nl : end;
    if (memchr(p, '\0', (size_t)(le - p))) nul = true;
    return nl;
#endif
}

static void parse_part(const Cfg &cfg, Part &part) {
    const char *p = part.b;
    const char *end = part.e;
    const unsigned N = cfg.N_indv;
    const int K = cfg.K;
    std::vector<int> a1v(N), a2v(N);
    std::vector<uint8_t> pl(N);
    std::vector<int> acount;
    const unsigned NW = (N + 63) / 64, KW = (unsigned)(K + 63) / 64;
    std::vector<uint64_t> A0(NW + 1), A1(NW + 1), B0(NW + 1), B1(NW + 1), HT(NW + 1), H0(NW + 1), H1(NW + 1),
        HP(NW + 1);
    while (p < end) {
        // the line end and "is there a NUL byte in the line" in one pass (gzgets cuts lines at NUL bytes; such
        // lines are left to vcftools' own loop)
        bool nul = false;
        const char *nl = find_eol(p, end, nul);
        if (nul) {
            part.bad = true;
            return;
        }
        const char *le = nl ? nl : end;
        const char *next = nl ? nl + 1 : end;
        // trim trailing " \t\n\r"
        while (le > p && (le[-1] == ' ' || le[-1] == '\t' || le[-1] == '\r' || le[-1] == '\n')) le--;
        if (le == p) {  // empty line: apply_filters returns 0 and the line is skipped
            p = next;
            continue;
        }
        part.n_entries++;
        // split the 9 fixed fields
        const char *f[10];
        f[0] = p;
        int nf = 1;
        const char *q = p;
        while (nf < 10) {
            const char *t = (const char *)memchr(q, '\t', (size_t)(le - q));
            if (!t) break;
            f[nf++] = t + 1;
            q = t + 1;
        }
        // field i spans [f[i], f[i+1]-1); the 10th start is the first genotype column
        if (nf < 10) {  // fewer than 10 fields: genotype column 0 missing (N >= 1)
            part.bad = true;
            return;
        }
        auto fend = [&](int i) { return f[i + 1] - 1; };
        Rec r;
        r.chrom = f[0];
        r.chrom_len = (uint32_t)(fend(0) - f[0]);
        r.pos = atoi_field(f[1], fend(1));
        r.reach = 0;
        r.err_freq = 0;
        r.pass_freq = 0;
        r.biallelic = 0;
        r.poly = 0;
        r.diploid = 0;
        r.exotic = 0;
        r.code_off = -1;
        r.allele_off = -1;
        const char *gt0 = f[9];
        // ALT alleles
        const char *alt_b = f[4], *alt_e = fend(4);
        int n_alt = 0;
        bool indel = (fend(3) - f[3]) != 1;
        size_t ref_len = (size_t)(fend(3) - f[3]);
        {
            const char *s = alt_b;
            for (;;) {
                const char *c = (const char *)memchr(s, ',', (size_t)(alt_e - s));
                const char *te = c ? c : alt_e;
                if (!(te - s == 1 && *s == '.')) {
                    n_alt++;
                    if ((size_t)(te - s) != ref_len) indel = true;
                }
                if (!c) break;
                s = c + 1;
            }
        }
        int N_alleles = n_alt + 1;
        bool pass = true;
        if (cfg.remove_indels && indel) pass = false;
        if (pass && cfg.remove_filtered_all) {
            const char *fb = f[6], *fe = fend(6);
            if (!(fe - fb == 1 && *fb == '.')) {
                // sorted tokens; keep iff the smallest token is "PASS"
                bool has_pass = false, smaller = false;
                const char *s = fb;
                for (;;) {
                    const char *c = (const char *)memchr(s, ';', (size_t)(fe - s));
                    const char *te = c ? c : fe;
                    size_t L = (size_t)(te - s);
                    if (L == 4 && memcmp(s, "PASS", 4) == 0) has_pass = true;
                    else {
                        size_t m = L < 4 ? L : 4;
                        int cmp = memcmp(s, "PASS", m);
                        if (cmp < 0 || (cmp == 0 && L < 4)) smaller = true;
                    }
                    if (!c) break;
                    s = c + 1;
                }
                if (!(has_pass && !smaller)) pass = false;
            }
        }
        if (pass && cfg.pos_filter) {
            if (r.chrom_len == cfg.pos_chr.size() && memcmp(r.chrom, cfg.pos_chr.data(), r.chrom_len) == 0) {
                if (r.pos < cfg.start_pos || r.pos > cfg.end_pos) pass = false;
            } else
                pass = false;
        }
        if (pass && !cfg.chrs.empty()) {
            bool found = false;
            for (const std::string &c : cfg.chrs)
                if (c.size() == r.chrom_len && memcmp(c.data(), r.chrom, r.chrom_len) == 0) {
                    found = true;
                    break;
                }
            if (!found) pass = false;
        }
        if (pass && cfg.nalleles_filter) {
            if (N_alleles < cfg.min_alleles || N_alleles > cfg.max_alleles) pass = false;
        }
        if (pass && cfg.min_quality >= 0) {
            double QUAL = str2double_field(f[5], fend(5));
            if (QUAL < cfg.min_quality) pass = false;
        }
        bool need_gt = pass && (cfg.freq_filter || cfg.mac_filter || N_alleles == 2);
        if (!need_gt) {
            // still need to know that the line has all N genotype columns (else vcftools would read stale fields)
            if (count_tabs(gt0, le) + 1 < N) {
                part.bad = true;
                return;
            }
            if (pass) r.reach = 1, r.pass_freq = 1;  // reaches thin; not biallelic so never used
            if (pass) part.recs.push_back(r);
            p = next;
            continue;
        }
        r.reach = 1;
        // FORMAT: GT_idx = position of the last "GT" token
        int GT_idx = -1;
        {
            const char *fb = f[8], *fe = fend(8);
            if (fe > fb) {
                int i = 0;
                const char *s = fb;
                for (;;) {
                    const char *c = (const char *)memchr(s, ':', (size_t)(fe - s));
                    const char *te = c ? c : fe;
                    if (te - s == 2 && s[0] == 'G' && s[1] == 'T') GT_idx = i;
                    if (!c) break;
                    s = c + 1;
                    i++;
                }
            }
        }
        // regular lines (all columns "x|y" with x, y in {0,1,.}): SIMD classification into bitmasks
        // and semi-regular lines (GT first, 1- or 3-char GTs over {0,1,.}, other FORMAT fields after ':')
        bool regular = false, semi = false;
        if (GT_idx == 0) {
            regular = regular_gt_line(gt0, le, N, A0.data(), A1.data(), B0.data(), B1.data());
            if (!regular) semi = semi_regular_gt_line(gt0, le, N, A0.data(), A1.data(), B0.data(), B1.data(), HP.data());
        }
        if (regular || semi) {
            unsigned n_hap = 0;  // included ploidy-1 columns
            if (semi)
                for (unsigned w = 0; w < NW; w++) n_hap += (unsigned)__builtin_popcountll(HP[w] & cfg.incw[w]);
            r.poly = 0;
            r.diploid = (n_hap == 0);
            r.exotic = 0;
            r.biallelic = (N_alleles == 2);
            bool fpass = true;
            if (cfg.freq_filter || cfg.mac_filter) {
                unsigned c0 = 0, c1 = 0;
                for (unsigned w = 0; w < NW; w++) {
                    const uint64_t inc = cfg.incw[w];
                    c0 += (unsigned)(__builtin_popcountll(A0[w] & inc) + __builtin_popcountll(B0[w] & inc));
                    c1 += (unsigned)(__builtin_popcountll(A1[w] & inc) + __builtin_popcountll(B1[w] & inc));
                }
                acount.assign((size_t)N_alleles, 0);
                acount[0] = (int)c0;
                if (N_alleles > 1) acount[1] = (int)c1;  // N_alleles == 1: index 1 is inside the slack (ignored)
                fpass = freq_decision(cfg, acount, N_alleles, c0 + c1, 2u * (unsigned)K - n_hap, false, GT_idx, false,
                                      r);
            }
            r.pass_freq = fpass;
            if (fpass && r.biallelic && r.diploid) {
                for (unsigned w = 0; w < NW; w++) {
                    HT[w] = (A0[w] & B1[w]) | (A1[w] & B0[w]);
                    H0[w] = A0[w] & B0[w];
                    H1[w] = A1[w] & B1[w];
                }
                r.code_off = (int64_t)part.codes.size();
                part.codes.resize(part.codes.size() + (size_t)3 * KW);
                uint64_t *het = part.codes.data() + r.code_off;
                compact_bits(HT.data(), cfg.incw.data(), NW, cfg.all_inc, het, KW);
                compact_bits(H0.data(), cfg.incw.data(), NW, cfg.all_inc, het + KW, KW);
                compact_bits(H1.data(), cfg.incw.data(), NW, cfg.all_inc, het + 2 * KW, KW);
            }
            part.recs.push_back(r);
            p = next;
            continue;
        }
        // genotype columns, general path
        bool poly = false;
        bool diploid = true;
        bool exotic = false;
        const char *c = gt0;
        for (unsigned ui = 0; ui < N; ui++) {
            if (c > le) {  // column ui does not exist
                part.bad = true;
                return;
            }
            int a1 = -1, a2 = -1, ploidy = 2;
            const char *ce;  // column end
            if (GT_idx == 0 && c + 3 <= le && (c[1] == '|' || c[1] == '/') && c[0] != '\t' && c[0] != ':' &&
                c[2] != '\t' && c[2] != ':' && (c + 3 == le || c[3] == '\t' || c[3] == ':')) {
                a1 = c[0] == '.' ? -1 : c[0] - '0';
                a2 = c[2] == '.' ? -1 : c[2] - '0';
                if (c + 3 == le || c[3] == '\t')
                    ce = c + 3;
                else {
                    const char *t = (const char *)memchr(c + 3, '\t', (size_t)(le - (c + 3)));
                    ce = t ? t : le;
                }
            } else {
                const char *t = (const char *)memchr(c, '\t', (size_t)(le - c));
                ce = t ? t : le;
                if (GT_idx >= 0) {
                    // find subfield GT_idx within [c, ce)
                    const char *s = c;
                    int i = 0;
                    bool found = true;
                    while (i < GT_idx) {
                        const char *col = (const char *)memchr(s, ':', (size_t)(ce - s));
                        if (!col) {
                            found = false;
                            break;
                        }
                        s = col + 1;
                        i++;
                    }
                    if (found) {
                        const char *col = (const char *)memchr(s, ':', (size_t)(ce - s));
                        const char *ge = col ? col : ce;
                        parse_gt_complex(s, ge, a1, a2, ploidy, poly);
                    }
                }
            }
            a1v[ui] = a1;
            a2v[ui] = a2;
            pl[ui] = (uint8_t)ploidy;
            if ((*cfg.include)[ui]) {
                if (ploidy != 2) diploid = false;
                if (a1 >= 0 && a2 >= 0 && a1 == a2 && a1 > 1) exotic = true;
            }
            c = ce + 1;
        }
        r.poly = poly;
        r.diploid = diploid;
        r.exotic = exotic;
        r.biallelic = (N_alleles == 2);
        // MAF / call rate (entry::filter_sites_by_frequency_and_call_rate) and MAC
        bool fpass = true;
        if (cfg.freq_filter || cfg.mac_filter) {
            acount.assign((size_t)N_alleles, 0);
            unsigned N_non_missing_chr = 0, N_chr = 0;
            // vcftools' get_allele_counts writes allele_counts[a] without a bounds check. Indices inside the
            // malloc slack of the vector<int>(N_alleles) are harmless; beyond it the heap is corrupted and
            // vcftools 0.1.17 crashes (seen as SIGSEGV/SIGABRT in solver/tests/edge_cases.py). We refuse.
            const int safe_idx = slack_index(N_alleles);
            bool oob = false;
            for (unsigned ui = 0; ui < N; ui++) {
                if (!(*cfg.include)[ui]) continue;
                N_chr += pl[ui];
                int a = a1v[ui], b = a2v[ui];
                if (a > -1) {
                    if (a < N_alleles) acount[(size_t)a]++;
                    else if (a >= safe_idx) oob = true;
                    N_non_missing_chr++;
                }
                if (b > -1) {
                    if (b < N_alleles) acount[(size_t)b]++;
                    else if (b >= safe_idx) oob = true;
                    N_non_missing_chr++;
                }
            }
            fpass = freq_decision(cfg, acount, N_alleles, N_non_missing_chr, N_chr, poly, GT_idx, oob, r);
        }
        r.pass_freq = fpass;
        if (fpass && r.biallelic && diploid && !poly) {
            const int KW = (K + 63) / 64;
            r.code_off = (int64_t)part.codes.size();
            part.codes.resize(part.codes.size() + (size_t)3 * KW);
            uint64_t *het = part.codes.data() + r.code_off, *h0 = het + KW, *h1 = h0 + KW;
            uint64_t wh = 0, w0 = 0, w1 = 0;
            int k = 0;
            for (unsigned ui = 0; ui < N; ui++) {
                if (cfg.kidx[ui] < 0) continue;
                int a = a1v[ui], b = a2v[ui];
                uint64_t bit = 1ull << (k & 63);
                bool called = (a >= 0) & (b >= 0);
                wh |= (called & (a != b)) ? bit : 0;
                w0 |= (called & (a == 0) & (b == 0)) ? bit : 0;
                w1 |= (called & (a == 1) & (b == 1)) ? bit : 0;
                k++;
                if ((k & 63) == 0) {
                    het[(k >> 6) - 1] = wh;
                    h0[(k >> 6) - 1] = w0;
                    h1[(k >> 6) - 1] = w1;
                    wh = w0 = w1 = 0;
                }
            }
            if (k & 63) {
                het[k >> 6] = wh;
                h0[k >> 6] = w0;
                h1[k >> 6] = w1;
            }
            if (exotic) {
                r.allele_off = (int64_t)part.alleles.size();
                part.alleles.resize(part.alleles.size() + (size_t)K);
                int *al = part.alleles.data() + r.allele_off;
                for (unsigned ui = 0; ui < N; ui++) {
                    int kk = cfg.kidx[ui];
                    if (kk < 0) continue;
                    int a = a1v[ui], b = a2v[ui];
                    al[kk] = (a >= 0 && b >= 0 && a == b) ? a : -1;
                }
            }
        }
        part.recs.push_back(r);
        p = next;
    }
}

// The sequential part of vcftools' loop, in line order: errors, --thin (static last kept CHROM/POS), used sites.
struct Decider {
    const Cfg &cfg;
    const char *thin_chrom = "";
    uint32_t thin_chrom_len = 0;
    int thin_pos = 0;
    unsigned n_entries = 0, n_kept = 0;
    explicit Decider(const Cfg &c) : cfg(c) {}
    // copy=true: masks are copied into the counter (the part's buffers are reused); threads for block counting
    void run(const Part &pt, RelCounter &rc, bool copy, int threads) {
        n_entries += pt.n_entries;
        for (const Rec &r : pt.recs) {
            if (!r.reach) continue;
            if (r.err_freq == 2 && !r.poly)
                LOG.error("SwiftRel: allele index beyond the site's alleles under an allele-count filter; vcftools "
                          "0.1.17 writes out of bounds here (undefined behaviour, it crashes) -- refusing at " +
                          std::string(r.chrom, r.chrom_len) + ":" + header::int2str(r.pos));
            if (r.err_freq) {
                if (cfg.freq_filter && !r.poly)
                    LOG.error("Require Genotypes in variant file to filter by frequency and/or call rate");
                LOG.error("Polyploidy found, and not supported by vcftools: " + std::string(r.chrom, r.chrom_len) +
                          ":" + header::int2str(r.pos));
            }
            if (!r.pass_freq) continue;
            if (cfg.thin >= 1) {
                bool passed = true;
                if (r.chrom_len == thin_chrom_len && memcmp(r.chrom, thin_chrom, thin_chrom_len) == 0) {
                    // int subtraction as in vcftools, made explicitly wrapping
                    int distance_from_last_SNP = (int)((unsigned)r.pos - (unsigned)thin_pos);
                    if (distance_from_last_SNP < cfg.thin) passed = false;
                }
                if (passed) thin_pos = r.pos;
                thin_chrom = r.chrom;
                thin_chrom_len = r.chrom_len;
                if (!passed) continue;
            }
            n_kept++;
            if (!r.biallelic) {
                LOG.one_off_warning("\tRelatedness: Only using biallelic sites.");
                continue;
            }
            if (r.poly)
                LOG.error("Polyploidy found, and not supported by vcftools: " + std::string(r.chrom, r.chrom_len) +
                          ":" + header::int2str(r.pos));
            if (!r.diploid) {
                LOG.one_off_warning("\tRelatedness: Only using fully diploid sites.");
                continue;
            }
            const uint64_t *m = pt.codes.data() + r.code_off;
            const int *al = r.allele_off >= 0 ? pt.alleles.data() + r.allele_off : nullptr;
            if (copy)
                rc.add_site_copy(m, al, threads);
            else
                rc.add_site_masks(m, al);
        }
    }
};

}  // namespace

bool fast_path_supported(const parameters &params) {
    parameters d(0, 0);
    if (params.stream_in || params.bcf_format || !params.vcf_format) return false;
    if (params.diff_file != "") return false;
    if (!params.chrs_to_exclude.empty()) return false;
    if (!params.snps_to_keep.empty() || params.snps_to_keep_file != "" || params.snps_to_exclude_file != "")
        return false;
    if (params.BED_file != "" || params.positions_file != "" || params.exclude_positions_file != "") return false;
    if (params.positions_overlap_file != "" || params.exclude_positions_overlap_file != "") return false;
    if (params.mask_file != "" || params.keep_only_indels) return false;
    if (!params.site_filter_flags_to_exclude.empty() || !params.site_filter_flags_to_keep.empty()) return false;
    if (!params.site_INFO_flags_to_keep.empty() || !params.site_INFO_flags_to_remove.empty()) return false;
    if (params.min_mean_depth != d.min_mean_depth || params.max_mean_depth != d.max_mean_depth) return false;
    if (params.phased_only) return false;
    if (params.min_genotype_quality != d.min_genotype_quality) return false;
    if (params.min_genotype_depth != d.min_genotype_depth || params.max_genotype_depth != d.max_genotype_depth)
        return false;
    if (!params.geno_filter_flags_to_exclude.empty() || params.remove_all_filtered_genotypes) return false;
    if (params.min_non_ref_af != d.min_non_ref_af || params.max_non_ref_af != d.max_non_ref_af) return false;
    if (params.min_non_ref_af_any != d.min_non_ref_af_any || params.max_non_ref_af_any != d.max_non_ref_af_any)
        return false;
    if (params.min_non_ref_ac != d.min_non_ref_ac || params.max_non_ref_ac != d.max_non_ref_ac) return false;
    if (params.min_non_ref_ac_any != d.min_non_ref_ac_any || params.max_non_ref_ac_any != d.max_non_ref_ac_any)
        return false;
    if (params.min_HWE_pvalue != d.min_HWE_pvalue) return false;
    return true;
}

int run_fast_relatedness(const parameters &params, Input *in, unsigned N_indv, const std::vector<bool> &include_indv,
                         const std::vector<std::string> &indv_names, int out_fd, unsigned &N_entries,
                         unsigned &N_kept_entries) {
    double t0 = now_s();
    Cfg cfg;
    cfg.remove_indels = params.remove_indels;
    cfg.remove_filtered_all = params.remove_all_filtered_sites;
    std::string chr_to_keep = "";
    if (params.chrs_to_keep.size() == 1) chr_to_keep = *(params.chrs_to_keep.begin());
    cfg.pos_filter = !((chr_to_keep == "") ||
                       ((params.start_pos == -1) && (params.end_pos == std::numeric_limits<int>::max())));
    cfg.pos_chr = chr_to_keep;
    cfg.start_pos = params.start_pos;
    cfg.end_pos = params.end_pos;
    for (const std::string &c : params.chrs_to_keep) cfg.chrs.push_back(c);
    cfg.nalleles_filter = !((params.min_alleles <= 0) && (params.max_alleles == std::numeric_limits<int>::max()));
    cfg.min_alleles = params.min_alleles;
    cfg.max_alleles = params.max_alleles;
    cfg.min_quality = params.min_quality;
    cfg.freq_filter = !((params.min_maf <= 0.0) && (params.max_maf >= 1.0) && (params.min_site_call_rate <= 0) &&
                        (params.min_non_ref_af <= 0.0) && (params.max_non_ref_af >= 1.0) &&
                        (params.min_non_ref_af_any <= 0.0) && (params.max_non_ref_af_any >= 1.0));
    cfg.min_maf = params.min_maf;
    cfg.max_maf = params.max_maf;
    cfg.min_site_call_rate = params.min_site_call_rate;
    cfg.mac_filter =
        !((params.min_mac <= 0) && (params.max_mac == std::numeric_limits<int>::max()) &&
          (params.min_non_ref_ac <= 0) && (params.max_non_ref_ac == std::numeric_limits<int>::max()) &&
          (params.min_non_ref_ac_any <= 0) && (params.max_non_ref_ac_any == std::numeric_limits<int>::max()) &&
          (params.max_missing_call_count == std::numeric_limits<int>::max()));
    cfg.min_mac = params.min_mac;
    cfg.max_mac = params.max_mac;
    cfg.max_missing_call_count = params.max_missing_call_count;
    cfg.thin = params.min_interSNP_distance;
    cfg.N_indv = N_indv;
    cfg.include = &include_indv;
    cfg.kidx.assign(N_indv, -1);
    std::vector<std::string> names;
    int K = 0;
    for (unsigned ui = 0; ui < N_indv; ui++)
        if (include_indv[ui]) {
            cfg.kidx[ui] = K++;
            names.push_back(indv_names[ui]);
        }
    cfg.K = K;
    cfg.incw.assign((N_indv + 63) / 64 + 1, 0);
    for (unsigned ui = 0; ui < N_indv; ui++)
        if (include_indv[ui]) cfg.incw[ui >> 6] |= 1ull << (ui & 63);
    cfg.all_inc = (K == (int)N_indv);

    if (profiling()) fprintf(stderr, "[swiftrel] fast path start: avail %zu complete %d streaming %d\n", in->available(), (int)in->complete(), (int)in->streaming());
    if (in->streaming() && !in->complete()) {
        // Streaming: the producer thread is still inflating. Parse complete lines as they become final, decide
        // them in order and count every block of used sites, overlapping all of it with decompression.
        RelCounter rc(K);
        Decider dec(cfg);
        Part part;
        size_t pos = in->pos;
        size_t seen = pos;  // wait until more than this many bytes are final
        double t_wait = 0, t_parse = 0, t_dec = 0;
        for (;;) {
            double tw = now_s();
            size_t a = in->wait_beyond(seen);
            t_wait += now_s() - tw;
            seen = a;
            const bool done = in->complete();
            if (done) a = in->available();
            size_t end = a;
            if (!done) {
                const char *lo = in->data + pos;
                const char *q = (const char *)memrchr(lo, '\n', a - pos);
                if (!q) continue;  // no complete line yet: wait for more data (seen = a)
                q++;
                end = (size_t)(q - in->data);
                // avoid tiny chunks while the producer is far ahead of us
                (void)end;
            }
            if (end > pos) {
                part.b = in->data + pos;
                part.e = in->data + end;
                part.recs.clear();
                part.codes.clear();
                part.alleles.clear();
                part.n_entries = 0;
                double tp0 = now_s();
                parse_part(cfg, part);
                if (part.bad) {
                    in->finish_producer();
                    return 1;  // not applicable: the caller runs the original loop (nothing written yet)
                }
                double tp1 = now_s();
                dec.run(part, rc, true, in->complete() ? 0 : 1);
                t_parse += tp1 - tp0;
                t_dec += now_s() - tp1;
                pos = end;
            }
            if (done && pos >= a) break;
        }
        in->finish_producer();
        N_entries += dec.n_entries;
        N_kept_entries += dec.n_kept;
        double t1 = now_s();
        rc.finish();
        double t2 = now_s();
        bool ok = write_relatedness2(out_fd, names, rc);
        double t3 = now_s();
        if (profiling())
            fprintf(stderr, "[swiftrel] streaming parse+count %.3f s (parse %.3f, decide+count %.3f, waited %.3f s), final count %.3f s, write %.3f s (K=%d)\n",
                    t1 - t0, t_parse, t_dec, t_wait, t2 - t1, t3 - t2, K);
        return ok ? 0 : 2;
    }

    // split the remaining data into parts at line boundaries
    in->finish_producer();
    const char *db = in->data + in->pos, *de = in->data + in->size;
    int T = thread_count();
    size_t total = (size_t)(de - db);
    if (total < (1u << 20)) T = 1;
    std::vector<Part> parts((size_t)T);
    const char *s = db;
    for (int t = 0; t < T; t++) {
        const char *e = (t == T - 1) ? de : db + total * (size_t)(t + 1) / (size_t)T;
        if (e < s) e = s;
        if (t != T - 1 && e < de) {
            const char *nl = (const char *)memchr(e, '\n', (size_t)(de - e));
            e = nl ? nl + 1 : de;
        }
        parts[(size_t)t].b = s;
        parts[(size_t)t].e = e;
        s = e;
    }
    {
        std::vector<std::thread> th;
        for (int t = 1; t < T; t++) th.emplace_back(parse_part, std::cref(cfg), std::ref(parts[(size_t)t]));
        parse_part(cfg, parts[0]);
        for (auto &x : th) x.join();
    }
    for (auto &pt : parts)
        if (pt.bad) return 1;  // not applicable: caller runs the original loop
    double t1 = now_s();

    // sequential pass: errors, thinning, used sites
    RelCounter rc(K);
    double t1b = now_s();
    Decider dec(cfg);
    for (auto &pt : parts) dec.run(pt, rc, false, 0);
    N_entries += dec.n_entries;
    N_kept_entries += dec.n_kept;
    double t1c = now_s();
    rc.finish();
    double t2 = now_s();
    if (profiling()) fprintf(stderr, "[swiftrel] counter alloc %.3f s, decide+scatter %.3f s, count+mirror %.3f s\n", t1b - t1, t1c - t1b, t2 - t1c);
    bool ok = write_relatedness2(out_fd, names, rc);
    double t3 = now_s();
    if (profiling())
        fprintf(stderr, "[swiftrel] parse %.3f s, decide+count %.3f s, write %.3f s (K=%d)\n", t1 - t0, t2 - t1,
                t3 - t2, K);
    if (!ok) return 2;
    return 0;
}

}  // namespace swiftrel
