// swiftrel_engine.h -- bit-parallel KING-robust (vcftools --relatedness2) counting and exact output.
// Part of SwiftRel (LGPL-3.0, derived from vcftools 0.1.17).
//
// Per used site and included individual k the genotype class is one of
//   0 = not counted (missing allele), 1 = het (a != b), 2 = hom allele 0, 3 = hom allele 1,
//   4 = hom with another allele value (only possible with malformed/odd allele codes; handled exactly but slowly).
// N_AaAa(i,j) = popcount(het_i & het_j), N_AAaa(i,j) = popcount(h0_i & h1_j) + popcount(h1_i & h0_j)
// (+ exact pairwise corrections for class-4 sites), N_Aa(i) = N_AaAa(i,i).
#ifndef SWIFTREL_ENGINE_H
#define SWIFTREL_ENGINE_H

#include <cstdint>
#include <string>
#include <vector>

namespace swiftrel {

int thread_count();  // SWIFTREL_THREADS or hardware concurrency

class RelCounter {
   public:
    explicit RelCounter(int K);
    ~RelCounter();
    // Generic path: one used site as per-individual classes (copied). codes[k] in {0,1,2,3,4};
    // alleles[k] is the homozygous allele of class-4 individuals (may be null if there are none).
    void add_site(const uint8_t *codes, const int *alleles);
    // Fast path: one used site as K-bit masks het[KW], h0[KW], h1[KW] (KW = (K+63)/64), not copied: the memory
    // must stay valid until finish(). hom_alleles (K entries, -1 = not homozygous) is required only when some
    // homozygote has an allele other than 0/1 (those individuals are in neither h0 nor h1); otherwise null.
    void add_site_masks(const uint64_t *masks, const int *hom_alleles);
    // Streaming: copy the site; every block of sites is transposed and counted at once with `threads` threads.
    void add_site_copy(const uint64_t *masks, const int *hom_alleles, int threads);
    void finish();  // count what is left, exact corrections

    int K, KW;
    uint64_t total_sites;  // used sites added (bounds every count)
    // results: full K x K row-major matrices
    uint32_t *aa = nullptr, *oo = nullptr;
    // Fused mode (few used sites, no odd homozygous alleles): no K x K matrices; the writer computes each output
    // row's counts from word-major bitplanes right before formatting it (fused_row).
    bool fused = false;
    std::vector<uint32_t> naa;  // N_Aa per individual (fused mode)
    void fused_row(int a, uint32_t *ra, uint32_t *ro) const;
    // Deferred final block (finish, 64 <= K <= 1024, two or more threads): the last block of sites is transposed and
    // its diagonal 32 x 32 tile pairs are counted (so N_Aa is final), but the off-diagonal tile pairs (I < J) are
    // counted only when the writer is about to read them: count_deferred(I, J) for one pair (its cells (i, j) and
    // (j, i)), complete_deferred() for all that are left. Until then those cells lack the last block's counts.
    void count_deferred(int I, int J) const;
    void complete_deferred() const;

   private:
    bool build_fused();
    uint64_t *tw = nullptr;  // fused: [3][nwf][Kp] words
    size_t nwf = 0, Kp = 0;
    void count_sites(const uint64_t *const *site_ptrs, size_t S, const uint8_t *ex, int threads, bool defer = false);
    void count_tile(const uint64_t *planes, size_t PW, size_t nw, int I, int J, bool first) const;
    uint64_t *def_planes = nullptr;  // deferred block: planes, words per plane, words to count
    size_t def_PW = 0, def_nw = 0;
    mutable bool def_pending = false;
    void apply_sparse(const uint64_t *masks);
    bool rare_site(const uint64_t *masks) const;
    void alloc_zeroed();
    std::vector<uint32_t> nA_rare;  // hom-alt count per individual over sites counted by apply_sparse
    std::vector<uint8_t> site_ex, pending_ex;  // per site: has odd homozygous alleles (always counted densely)
    void flush_pending(int threads, bool defer = false);
    std::vector<uint64_t> pending;  // copied masks of sites not yet counted (streaming)
    size_t pending_cap = 0;
    std::vector<const uint64_t *> sites;
    std::vector<std::vector<uint64_t>> owned;          // masks built by add_site
    std::vector<std::vector<int>> exotic;               // hom alleles of sites with odd homozygous alleles
};

// Write header + all K*K rows. names[k] = individual id. fd: output file descriptor.
// Returns false on write error.
bool write_relatedness2(int fd, const std::vector<std::string> &names, const RelCounter &rc);

// 64 x 64 bit transpose in place: afterwards bit s of a[k] is bit k of the old a[s].
void transpose64(uint64_t a[64]);

}  // namespace swiftrel

#endif
