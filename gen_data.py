#!/usr/bin/env python3
"""SwiftRel: deterministic VCF generators.

Two jobs:
  * write_dataset(spec, path): the benchmark VCFs (dev + holdout), seeded and reproducible.
  * fuzz_case(rng): tiny adversarial VCFs + vcftools --relatedness2 options for the random-input check.

Model for the benchmark sets: a Li-Stephens-style haplotype mosaic. Each chromosome has K founder
haplotypes with Beta-distributed allele frequencies. Every sampled haplotype copies one founder and switches
founder between neighbouring sites with probability 1 - exp(-rho * bp_gap), then gains rare mutations.
This gives LD that decays with physical distance, as in real population data. Optional trios: a child copies one
haplotype from each parent per chromosome, so true parent-offspring relationships exist (relatedness2's target
signal). Missing genotypes, unphased calls, haploid calls, extra FORMAT fields, filtered records, multi-allelic
sites and indels are layered on top, because real VCFs carry them.
"""
import gzip
import random

import numpy as np

# Every spec says which VCF to use and which vcftools options the benchmark command uses.
# The benchmark command is: <tool> --vcf|--gzvcf <file> <args...> --out <prefix>   (output: <prefix>.relatedness2)
# Specs with "real" point at a shipped 1000 Genomes subset in data/ (see data/README.txt), pinned by sha256.
# Real sets carry the paper's headline: two family-rich sets from the 30x NYGC release (with parent-offspring trios)
# and four population samples from phase 3. Synthetic sets add shapes the real slices lack.
DEV_SPECS = {
    "r30x_fam_ceu_yri_chs": dict(real="data/r30x_fam_ceu_yri_chs.vcf.gz", sha256="d6c32b5220f8bb6e73c207875e36dd830a9c0dda9ad6d3a46b84fdb64057837a",
                                 args=["--relatedness2"]),
    "r_eur_hap":     dict(real="data/r_eur_hap.vcf.gz",
                          sha256="de9c61c1d9e6a078db3bc0add7d2bb1d46c8e348da878fc12a9f65e728b8772b",
                          args=["--relatedness2", "--remove-indels"]),
    "r_all_hap":     dict(real="data/r_all_hap.vcf.gz",
                          sha256="56cf68945fdbd3312ed93d21376984579c0c63287dcffc957a449a7070762e0b",
                          args=["--relatedness2", "--maf", "0.3"]),
    "r_afr_geno":    dict(real="data/r_afr_geno.vcf.gz",
                          sha256="a75f0df71be4f68d23b9110c28d508bb13dc39752054f512055f44c4ba8ac33c",
                          args=["--relatedness2"]),
    "s_trios_messy": dict(seed=311, n=420, S=6000, chroms=3, L=2_000_000, miss=0.03, unphased=0.5, trios=60,
                          fmt="GT:DP:GQ", multi=0.02, indel=0.03, haploid_chrom=3, args=["--relatedness2"]),
}

# Holdout sets: different samples, regions, seeds and shapes, checked with evaluate.py --holdout.
HOLDOUT_SPECS = {
    "rh30x_fam_gwd_ibs_pjl": dict(real="data/rh30x_fam_gwd_ibs_pjl.vcf.gz", sha256="a4617b0941c564f470d906489e8b08f6e47a70fb3eaefc7708a6d382c41b2489",
                                  args=["--relatedness2", "--max-missing", "0.95"]),
    "rh_eas_hap":    dict(real="data/rh_eas_hap.vcf.gz",
                          sha256="561c77d29925fd456ef2ac37ac5e64443c333e1624a532a29ba3d85d52914bb8",
                          args=["--relatedness2"]),
    "rh_sas_geno":   dict(real="data/rh_sas_geno.vcf.gz",
                          sha256="71d3047fd2be77315dcb20430ee409a84bf48ca4155d065f2b42ed8913376da1",
                          args=["--relatedness2", "--maf", "0.05"]),
    "h_many_samples": dict(seed=411, n=1600, S=600, chroms=1, L=300_000, miss=0.01, unphased=1.0, trios=200,
                           gz=True, args=["--relatedness2"]),
    "h_chr_filter":   dict(seed=412, n=300, S=9000, chroms=4, L=2_000_000, multi=0.02, indel=0.05, trios=30,
                           args=["--relatedness2", "--chr", "chr2", "--chr", "chr4", "--remove-filtered-all"]),
}

_GT_PHASED = ["0|0", "0|1", "1|0", "1|1", ".|."]
_GT_UNPHASED = ["0/0", "0/1", "0/1", "1/1", "./."]
_BASES = "ACGT"


def _extra_fields(fmt, rng, k):
    """k random FORMAT suffixes (everything after GT) for the given FORMAT string."""
    keys = fmt.split(":")[1:]
    out = []
    for _ in range(k):
        vals = []
        for key in keys:
            if key == "DP":
                vals.append(str(rng.integers(3, 60)))
            elif key == "GQ":
                vals.append(str(rng.integers(5, 99)))
            elif key == "AD":
                vals.append(f"{rng.integers(0, 30)},{rng.integers(0, 30)}")
            elif key == "PL":
                vals.append(f"0,{rng.integers(0, 90)},{rng.integers(90, 400)}")
        out.append(":" + ":".join(vals))
    return out


def simulate_chrom(rng, n, S_c, L, spec):
    """Return (positions, haplotypes[2n, S_c] as uint8 0/1)."""
    pos = np.sort(rng.choice(np.arange(1, L + 1), size=S_c, replace=False))
    K = spec.get("founders", 24)
    p = np.clip(rng.beta(0.35, 0.35, S_c), 0.01, 0.99)
    F = (rng.random((K, S_c)) < p).astype(np.uint8)
    gap = np.diff(pos, prepend=pos[0]).astype(np.float64)
    switch_p = 1.0 - np.exp(-spec.get("rho", 2e-5) * gap)
    H = 2 * n
    hap = np.empty((H, S_c), dtype=np.uint8)
    cols = np.arange(S_c)
    for h in range(H):
        sw = rng.random(S_c) < switch_p
        sw[0] = True
        seg = np.cumsum(sw) - 1
        choice = rng.integers(0, K, int(seg[-1]) + 1)
        hap[h] = F[choice[seg], cols]
    hap ^= (rng.random((H, S_c)) < spec.get("mu", 0.002)).astype(np.uint8)
    return pos, hap


def write_dataset(spec, path):
    rng = np.random.default_rng(spec["seed"])
    n, S, chroms, L = spec["n"], spec["S"], spec["chroms"], spec["L"]
    fmt = spec.get("fmt", "GT")
    miss, unphased = spec.get("miss", 0.0), spec.get("unphased", 0.0)
    opener = (lambda p: gzip.open(p, "wb", compresslevel=1)) if spec.get("gz") else (lambda p: open(p, "wb"))
    extras = _extra_fields(fmt, rng, 64) if fmt != "GT" else [""]
    sizes = [S // chroms + (1 if c < S % chroms else 0) for c in range(chroms)]
    with opener(path) as f:
        hdr = ["##fileformat=VCFv4.2", "##source=SwiftX-gen_data"]
        hdr += [f"##contig=<ID=chr{c + 1},length={L}>" for c in range(chroms)]
        hdr += ['##INFO=<ID=DP,Number=1,Type=Integer,Description="Total depth">',
                '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">',
                '##FORMAT=<ID=AD,Number=R,Type=Integer,Description="Allelic depths">',
                '##FORMAT=<ID=DP,Number=1,Type=Integer,Description="Read depth">',
                '##FORMAT=<ID=GQ,Number=1,Type=Integer,Description="Genotype quality">',
                '##FORMAT=<ID=PL,Number=G,Type=Integer,Description="Phred-scaled likelihoods">']
        hdr.append("\t".join(["#CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO", "FORMAT"]
                             + [f"S{i:05d}" for i in range(n)]))
        f.write(("\n".join(hdr) + "\n").encode())
        for c, S_c in enumerate(sizes):
            pos, hap = simulate_chrom(rng, n, S_c, L, spec)
            for t in range(spec.get("trios", 0)):                    # child = one haplotype from each parent
                ch, fa, mo = 3 * t + 2, 3 * t, 3 * t + 1
                hap[2 * ch] = hap[2 * fa + int(rng.integers(0, 2))]
                hap[2 * ch + 1] = hap[2 * mo + int(rng.integers(0, 2))]
            code = hap[0::2].astype(np.int64) * 2 + hap[1::2]          # (n, S_c): 0..3
            haploid = spec.get("haploid_chrom") == c + 1               # e.g. a chrX-like block coded haploid
            if miss:
                code[rng.random(code.shape) < miss] = 4
            unph = rng.random(code.shape) < unphased if unphased else None
            for j in range(S_c):
                ref = _BASES[rng.integers(0, 4)]
                alt = _BASES[(_BASES.index(ref) + rng.integers(1, 4)) % 4]
                kind = rng.random()
                col = code[:, j]
                if kind < spec.get("multi", 0.0):
                    alt = alt + "," + next(b for b in _BASES if b not in (ref, alt))
                elif kind < spec.get("multi", 0.0) + spec.get("indel", 0.0):
                    alt = ref + "".join(_BASES[i] for i in rng.integers(0, 4, rng.integers(1, 4)))
                if haploid:
                    gts = [("0", "1", "0", "1", ".")[k] for k in col]
                elif unph is None:
                    gts = [_GT_PHASED[k] for k in col]
                else:
                    u = unph[:, j]
                    gts = [(_GT_UNPHASED if u[i] else _GT_PHASED)[k] for i, k in enumerate(col)]
                if "," in alt and not haploid:  # a few genotypes carry the second ALT allele
                    for i in rng.integers(0, n, max(1, n // 20)):
                        gts[i] = gts[i][0] + gts[i][1] + "2" if gts[i][0] != "." else gts[i]
                if fmt != "GT":
                    ex = rng.integers(0, len(extras), n)
                    gts = [g + extras[e] for g, e in zip(gts, ex)]
                qual = "." if rng.random() < 0.2 else str(int(rng.integers(20, 999)))
                info = "." if rng.random() < 0.5 else f"DP={int(rng.integers(100, 9000))}"
                vid = "." if rng.random() < 0.7 else f"rs{int(rng.integers(1, 10**8))}"
                filt = "PASS" if rng.random() < 0.97 else "LowQual"
                line = "\t".join([f"chr{c + 1}", str(int(pos[j])), vid, ref, alt, qual, filt, info, fmt] + gts)
                f.write((line + "\n").encode())
    return (n, S)


# ---------------------------------------------------------------- fuzz cases for the random-input check

def _fuzz_gt(rng, phased_p, ploidy, nalt):
    if ploidy == 1:
        return rng.choice(["0", "1", "."] + (["2"] if nalt > 1 else []))
    if ploidy == 3:
        return "0/1/1"
    alle = ["0", "0", "1", "1", "."] + (["2"] if nalt > 1 else [])
    a, b = rng.choice(alle), rng.choice(alle)
    if phased_p < 1.0 and rng.random() < 0.03:
        return "."
    return a + ("|" if rng.random() < phased_p else "/") + b


def fuzz_case(rng: random.Random):
    """Return (vcf_bytes, suffix, args, extra_files). args never contain --vcf/--gzvcf/--out.

    extra_files maps a file name to its bytes (for --keep/--remove); args refer to them by that name.
    """
    n = rng.choice([1, 2, 3, 4, 5, 8, 13, 31, 64, 65])
    S = rng.choice([0, 1, 2, 3, 5, 8, 17, 17, 40, 40, 80, 120])
    chroms = rng.choice([["1"], ["1"], ["chr1", "chr2"], ["2", "10", "X"]])
    phased_p = rng.choice([1.0, 1.0, 1.0, 0.995, 0.9, 0.0])
    haploid_p = rng.choice([0.0, 0.0, 0.0, 0.1, 1.0])
    poly_p = 0.02 if rng.random() < 0.05 else 0.0
    miss_p = rng.choice([0.0, 0.05, 0.3])
    fmt = rng.choice(["GT", "GT", "GT:DP", "DP:GT"])
    names = [f"ind{i}" for i in range(n)]
    lines = ["##fileformat=VCFv4.2",
             '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">',
             '##FORMAT=<ID=DP,Number=1,Type=Integer,Description="Depth">',
             "\t".join(["#CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO", "FORMAT"] + names)]
    sample_ploidy = [1 if rng.random() < haploid_p else (3 if rng.random() < poly_p else 2) for _ in range(n)]
    sites = []
    pos = 0
    ci = 0
    interleave = len(chroms) > 1 and rng.random() < 0.05   # chromosomes revisited out of order
    for j in range(S):
        if interleave and rng.random() < 0.3:
            ci = rng.randrange(len(chroms))
        elif rng.random() < 0.08 and ci < len(chroms) - 1:
            ci += 1
            pos = 0
        r = rng.random()
        pos += 0 if r < 0.05 else rng.choice([1, 2, 10, 100, 1000, 50000])   # 5% duplicate positions
        if rng.random() < 0.02:
            pos = max(1, pos - 500)                                          # rare unsorted input
        sites.append((chroms[ci], max(1, pos)))
    for chrom, p in sites:
        ref = rng.choice("ACGT")
        kind = rng.random()
        if kind < 0.08:
            alt, nalt = rng.choice([b for b in "ACGT" if b != ref]) + "," + "N", 2
        elif kind < 0.12:
            alt, nalt = ".", 0
        elif kind < 0.17:
            alt, nalt = ref + "TT", 1
        else:
            alt, nalt = rng.choice([b for b in "ACGT" if b != ref]), 1
        mono = rng.random() < 0.06         # monomorphic sites give 0/0 variances -> nan/inf rows
        gts = []
        for i in range(n):
            if mono:
                g = "0|0" if sample_ploidy[i] == 2 else "0"
            elif rng.random() < miss_p:
                g = ((".|." if rng.random() < phased_p else rng.choice(["./.", "."]))
                     if sample_ploidy[i] != 1 else ".")
            else:
                g = _fuzz_gt(rng, phased_p, sample_ploidy[i], max(nalt, 1))
            if fmt == "GT:DP":
                g = g + ":" + (str(rng.randrange(50)) if rng.random() < 0.9 else ".")
            elif fmt == "DP:GT":
                g = str(rng.randrange(50)) + ":" + g
            gts.append(g)
        filt = rng.choice(["PASS", "PASS", "PASS", ".", "q10", "LowQual;q10"])
        lines.append("\t".join([chrom, str(p), ".", ref, alt, rng.choice([".", "50", "3.5"]), filt, ".", fmt] + gts))
    text = "\n".join(lines) + "\n"
    if rng.random() < 0.03:
        text = text.rstrip("\n")
    data = text.encode()
    suffix = ".vcf"
    if rng.random() < 0.15:
        data = gzip.compress(data, compresslevel=1)
        suffix = ".vcf.gz"

    args = ["--relatedness2"]
    extra = {}
    if rng.random() < 0.15:
        args += ["--chr", rng.choice(chroms + ["nope"])]
    if rng.random() < 0.1:
        args += ["--maf", rng.choice(["0.05", "0.2", "0.45"])]
    if rng.random() < 0.1:
        args += ["--max-missing", rng.choice(["0.5", "0.9", "1"])]
    if rng.random() < 0.1:
        args.append("--remove-filtered-all")
    if rng.random() < 0.1:
        args.append("--remove-indels")
    if rng.random() < 0.08:
        args += ["--thin", str(rng.choice([1, 5, 1000]))]
    if rng.random() < 0.08:
        args += ["--min-alleles", rng.choice(["1", "2", "3"])]
    if rng.random() < 0.08:
        args += ["--max-alleles", rng.choice(["1", "2", "3"])]
    if rng.random() < 0.08:
        args += ["--mac", rng.choice(["1", "2", "5"])]
    if rng.random() < 0.06:
        args += ["--from-bp", rng.choice(["1", "50", "2000"]), "--to-bp", rng.choice(["100", "60000", "999999"])]
    if rng.random() < 0.06:
        args += ["--minQ", rng.choice(["10", "40"])]
    if rng.random() < 0.05:
        args += ["--max-maf", rng.choice(["0.1", "0.3"])]
    if rng.random() < 0.15:
        sel = [nm for nm in names if rng.random() < 0.5] + (["ghost"] if rng.random() < 0.3 else [])
        flag = rng.choice(["--keep", "--remove"])
        fname = f"ids_{rng.randrange(10**6)}.txt"
        extra[fname] = ("\n".join(sel) + ("\n" if sel else "")).encode()
        args += [flag, fname]
    if rng.random() < 0.15:
        args.append("--stdout")
    head, tail = args[:1], args[1:]
    # shuffle option groups (flag + value pairs stay together)
    groups, k = [], 0
    while k < len(tail):
        if k + 1 < len(tail) and not tail[k + 1].startswith("--"):
            groups.append(tail[k:k + 2])
            k += 2
        else:
            groups.append(tail[k:k + 1])
            k += 1
    rng.shuffle(groups)
    groups.insert(rng.randrange(len(groups) + 1), head)
    return data, suffix, [a for g in groups for a in g], extra
