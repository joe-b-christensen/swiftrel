# Stream the 1000G phase 3 chr22 VCF from EBI and keep the header plus records with POS <= END.
import gzip, os, sys, urllib.request
BASE = "http://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502/"
VCF = "ALL.chr22.phase3_shapeit2_mvncall_integrated_v5b.20130502.genotypes.vcf.gz"
PANEL = "integrated_call_samples_v3.20130502.ALL.panel"
END = int(sys.argv[1]) if len(sys.argv) > 1 else 19_800_000
out_dir = os.path.expanduser("~/swiftx_ld_test/1kg"); os.makedirs(out_dir, exist_ok=True)
urllib.request.urlretrieve(BASE + PANEL, os.path.join(out_dir, PANEL))
n = 0
with urllib.request.urlopen(BASE + VCF, timeout=120) as resp, gzip.GzipFile(fileobj=resp) as g, \
        gzip.open(os.path.join(out_dir, "chr22_head.vcf.gz"), "wb", compresslevel=3) as out:
    for line in g:
        if line[:1] != b"#":
            pos = int(line.split(b"\t", 2)[1])
            if pos > END:
                break
            n += 1
        out.write(line)
print("records", n, "last pos", pos)
