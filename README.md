# swiftrel

swiftrel is a drop-in replacement for [vcftools](https://github.com/vcftools/vcftools) 0.1.17 `--relatedness2`, the
KING-robust kinship screen. It writes the same `.relatedness2` file, byte for byte, much faster.

On four real 1000 Genomes chromosome 22 sets (503 to 2,504 individuals) it was 470× faster than vcftools, and on
seven sets 6.0× faster than PLINK 2 `--make-king-table` run from the same VCF, which computes a different kinship
estimator (geometric means, 2 threads, AVX-512 CPU). A screen of 503 individuals took vcftools 23 s and swiftrel
33 ms. Without AVX-512 the gain is smaller: 119× at 12 threads on an AVX2 laptop. Details are in the preprint (see
Citation).

## Build

Requirements: Linux on x86-64, gcc/g++ (tested with 13.3.0 on Ubuntu 24.04), GNU make, static zlib (`zlib1g-dev`)
and NASM 2.14 or later, which assembles the vendored ISA-L inflater:

```
sudo apt install build-essential zlib1g-dev nasm
bash solver/build.sh        # produces solver/swiftrel
```

The build uses `-march=native` for the swiftrel sources, so the binary is tuned for the build machine's CPU. The
fastest code paths need AVX-512 VPOPCNTDQ and VBMI2; on other x86-64 CPUs portable paths give the same output, more
slowly. The linker warning about `getaddrinfo` in a static binary concerns vcftools' remote-URL support, which
swiftrel does not use.

## Usage

swiftrel takes the same command line as vcftools:

```
solver/swiftrel --gzvcf input.vcf.gz --relatedness2 --out result         # writes result.relatedness2
solver/swiftrel --vcf input.vcf --maf 0.05 --keep samples.txt --relatedness2 --stdout > result.relatedness2
```

| Group | Options |
|---|---|
| Input / output | `--vcf FILE`, `--gzvcf FILE`, `--out PREFIX`, `--stdout` |
| Mode | `--relatedness2` |
| Site filters | `--chr C` (repeatable), `--maf X`, `--max-maf X`, `--mac N`, `--max-missing X`, `--min-alleles N`, `--max-alleles N`, `--from-bp N` / `--to-bp N`, `--minQ X`, `--remove-filtered-all`, `--remove-indels`, `--thin N` |
| Individual filters | `--keep FILE`, `--remove FILE` |

- The output is byte-identical to vcftools 0.1.17 with the same options, at every thread count: header
  `INDV1 INDV2 N_AaAa N_AAaa N1_Aa N2_Aa RELATEDNESS_PHI`, all N² ordered pairs, six significant digits. Where
  vcftools fails on an input, swiftrel fails as well.
- `SWIFTREL_THREADS` sets the number of threads (default: all CPUs); gzip input is also inflated by one extra thread.
- Other options with `--relatedness2`, BCF input and standard input run vcftools' original loop, at vcftools'
  speed. Other vcftools analyses are compiled in but untested; use upstream vcftools for them.
- Two deliberate differences: an allele index beyond a site's alleles, which makes vcftools write out of bounds,
  and corrupted gzip input, on which vcftools hangs, both make swiftrel exit with an error.

## Check it yourself

```
python3 prepare.py              # builds vcftools 0.1.17 (commit 1c53c3c), prepares the benchmark sets and records
                                # vcftools' outputs (needs gcc, g++, zlib, numpy, network)
bash solver/build.sh
python3 evaluate.py             # 300 random VCF files and five benchmark sets compared byte for byte, plus timing
python3 evaluate.py --holdout   # the five holdout sets
```

`data/` holds the seven 1000 Genomes sets used in the preprint and the 1000 Genomes pedigree file (see
`data/README.txt`).

## Licence

LGPL-3.0 (`LICENSE`), as a modified vcftools. Vendored code: ISA-L 2.32.1 (BSD-3-Clause, one file modified) and
vcftools' bundled MIT-licensed files; see `LICENSE` and `solver/LICENSE_NOTE.txt`.

## Citation

Christensen J. SwiftRel screens for relatives 470-fold faster than vcftools and 6-fold faster than PLINK 2, with
byte-identical output. Preprint, 2026. Please also cite vcftools (Danecek P et al., Bioinformatics 27:2156–2158,
2011). See `CITATION.cff`.
