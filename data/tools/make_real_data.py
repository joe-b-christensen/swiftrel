#!/usr/bin/env python3
"""Build the real-data VCFs shipped in data/ from 1000 Genomes phase 3 chr22 (GRCh37).

Run once on a machine with internet. Input: the head of the public chr22 VCF, streamed
by fetch (see data/README.txt), plus the sample panel. Each output keeps every record in its region (no site
filtering: --maf in the benchmark command does that inside the tool), the samples of one super-population
(or all), and optionally strips phasing ('|' -> '/') to make an unphased set for --geno-r2.
"""
import gzip
import sys

SETS = {
    # name: (superpop or ALL, start, end, unphase)
    "r_eur_hap":  ("EUR", 17_000_000, 18_000_000, False),
    "r_all_hap":  ("ALL", 18_000_000, 18_250_000, False),
    "r_afr_geno": ("AFR", 18_250_000, 18_750_000, True),
    "rh_eas_hap": ("EAS", 19_000_000, 19_500_000, False),
    "rh_sas_geno": ("SAS", 19_500_000, 19_800_000, True),
}


def main(src, panel, out_dir, only=None):
    pop = {}
    for line in open(panel):
        f = line.split()
        if f and f[0] != "sample":
            pop[f[0]] = f[2]
    header, samples = [], None
    outs = {}
    with gzip.open(src, "rt") as g:
        for line in g:
            if line.startswith("##"):
                header.append(line)
                continue
            if line.startswith("#CHROM"):
                cols = line.rstrip("\n").split("\t")
                samples = cols[9:]
                for name, (sp, a, b, unph) in SETS.items():
                    if only and name not in only:
                        continue
                    idx = [i for i, s in enumerate(samples) if sp == "ALL" or pop.get(s) == sp]
                    fh = gzip.open(f"{out_dir}/{name}.vcf.gz", "wt", compresslevel=9)
                    fh.write("".join(header))
                    fh.write("##SwiftX_subset=1000G phase3 chr22:%d-%d superpop=%s unphased=%s\n" % (a, b, sp, unph))
                    fh.write("\t".join(cols[:9] + [samples[i] for i in idx]) + "\n")
                    outs[name] = (fh, idx, a, b, unph, [0])
                continue
            parts = line.rstrip("\n").split("\t")
            pos = int(parts[1])
            for name, (fh, idx, a, b, unph, cnt) in outs.items():
                if a <= pos <= b:
                    gts = [parts[9 + i] for i in idx]
                    if unph:
                        gts = [x.replace("|", "/") for x in gts]
                    fh.write("\t".join(parts[:9] + gts) + "\n")
                    cnt[0] += 1
    for name, (fh, idx, a, b, unph, cnt) in outs.items():
        fh.close()
        print(name, "samples", len(idx), "records", cnt[0])


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], set(sys.argv[4].split(",")) if len(sys.argv) > 4 else None)
