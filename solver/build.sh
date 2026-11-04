#!/bin/bash
# Builds solver/swiftrel. evaluate.py runs it from the repository root.
# vcftools sources keep -O2; SwiftRel sources (swiftrel_*.cpp) use -O3 -march=native.
# -ffp-contract=off everywhere so no FMA contraction can change a double expression.
set -euo pipefail
cd "$(dirname "$0")"
CXXV="g++ -O2 -w -ffp-contract=off -pthread -DPACKAGE_VERSION=\"0.1.17\""
CXXS="g++ -O3 -march=native -w -ffp-contract=off -pthread -std=gnu++20 -DPACKAGE_VERSION=\"0.1.17\""
mkdir -p obj
rm -f obj/*.o
gcc -O2 -w -c bgzf.c -o obj/bgzf.o &
gcc -O2 -w -c knetfile.c -o obj/knetfile.o &
# ISA-L igzip (inflate) + crc, built with its own makefile (needs nasm); ~4 s on 2 cores
make -s -C isa-l -f Makefile.unx units=igzip lib >/dev/null &
wait
export CXXV CXXS
ls *.cpp | xargs -P "$(nproc)" -I{} bash -c 'f={}; o=obj/${f%.cpp}.o; case $f in swiftrel_*) $CXXS -c $f -o $o;; *) $CXXV -c $f -o $o;; esac'
# Linked fully statically and not position-independent: no dynamic loader and no start-up relocations
# (tiny input: 3.2 ms dynamic PIE, 2.4 ms with static libstdc++/libgcc, 2.1 ms static non-PIE). The getaddrinfo
# warning concerns knetfile's remote-URL support, which SwiftRel never uses.
g++ -pthread -static -no-pie -o swiftrel obj/*.o isa-l/bin/isa-l.a -lz
