#!/usr/bin/env python3
"""Stream the head of the 1000 Genomes 30x (NYGC, GRCh38) phased chr22 panel and keep records with POS <= END.

Run once on a machine with internet:
    python fetch_30x_chr22_head.py OUT_DIR [END]
Writes OUT_DIR/chr22_30x_head.vcf.gz (all 3,202 samples) and the pedigree/population file.
"""
import gzip
import os
import sys
import urllib.request

BASE = ("http://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/1000G_2504_high_coverage/")
VCF = ("working/20220422_3202_phased_SNV_INDEL_SV/"
       "1kGP_high_coverage_Illumina.chr22.filtered.SNV_INDEL_SV_phased_panel.vcf.gz")
PED = "20130606_g1k_3202_samples_ped_population.txt"


def main(out_dir, end):
    os.makedirs(out_dir, exist_ok=True)
    urllib.request.urlretrieve(BASE + PED, os.path.join(out_dir, PED))
    n, pos = 0, 0
    with urllib.request.urlopen(BASE + VCF, timeout=120) as resp, gzip.GzipFile(fileobj=resp) as g, \
            gzip.open(os.path.join(out_dir, "chr22_30x_head.vcf.gz"), "wb", compresslevel=3) as out:
        for line in g:
            if line[:1] != b"#":
                pos = int(line.split(b"\t", 2)[1])
                if pos > end:
                    break
                n += 1
            out.write(line)
    print("records", n, "last pos", pos)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 19_000_000)
