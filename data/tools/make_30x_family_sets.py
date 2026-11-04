#!/usr/bin/env python3
"""Build the family-rich real VCFs shipped in data/ from the 1000 Genomes 30x (NYGC) phased chr22 panel.

Run once on a machine with internet, after fetch_30x_chr22_head.py:
    python make_30x_family_sets.py chr22_30x_head.vcf.gz 20130606_g1k_3202_samples_ped_population.txt OUT_DIR
Each output keeps every record in its region (no site filtering; --maf in the benchmark command does that inside the
tool) and all samples of three populations, including the trio children and their parents.
"""
import gzip
import sys

SETS = {
    # name: (populations, start, end)
    "r30x_fam_ceu_yri_chs": (("CEU", "YRI", "CHS"), 16_000_000, 16_700_000),
    "rh30x_fam_gwd_ibs_pjl": (("GWD", "IBS", "PJL"), 17_500_000, 18_000_000),
}


def main(src, ped, out_dir, only=None):
    pop, trio_children = {}, set()
    for line in open(ped):
        f = line.split()
        if f and f[0] != "FamilyID":
            pop[f[1]] = f[5]
            if f[2] != "0" or f[3] != "0":
                trio_children.add(f[1])
    header, outs = [], {}
    with gzip.open(src, "rt") as g:
        for line in g:
            if line.startswith("##"):
                header.append(line)
                continue
            if line.startswith("#CHROM"):
                cols = line.rstrip("\n").split("\t")
                samples = cols[9:]
                for name, (pops, a, b) in SETS.items():
                    if only and name not in only:
                        continue
                    idx = [i for i, s in enumerate(samples) if pop.get(s) in pops]
                    fh = gzip.open(f"{out_dir}/{name}.vcf.gz", "wt", compresslevel=9)
                    fh.write("".join(header))
                    fh.write("##SwiftX_subset=1000G 30x NYGC phased panel chr22:%d-%d populations=%s samples=%d "
                             "trio_children=%d\n" % (a, b, ",".join(pops), len(idx),
                                                     sum(samples[i] in trio_children for i in idx)))
                    fh.write("\t".join(cols[:9] + [samples[i] for i in idx]) + "\n")
                    outs[name] = (fh, idx, a, b, [0])
                continue
            parts = line.rstrip("\n").split("\t")
            pos = int(parts[1])
            for name, (fh, idx, a, b, cnt) in outs.items():
                if a <= pos <= b:
                    fh.write("\t".join(parts[:9] + [parts[9 + i] for i in idx]) + "\n")
                    cnt[0] += 1
    for name, (fh, idx, a, b, cnt) in outs.items():
        fh.close()
        print(name, "samples", len(idx), "records", cnt[0])


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], set(sys.argv[4].split(",")) if len(sys.argv) > 4 else None)
