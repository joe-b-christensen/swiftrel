// swiftrel_input.h -- input buffer with vcftools' line-reading semantics.
// Part of SwiftRel (LGPL-3.0, derived from vcftools 0.1.17).
//
// vcftools reads a VCF with getline() (plain) or gzgets() (gzip, via zlib's gzopen) and trims trailing
// " \t\n\r" from every line. Both readers report eof only after a read attempt reaches the end of the data:
// a final line without '\n' is returned with eof set, and a final '\n' produces one extra empty line.
// Input reproduces this on an in-memory copy of the (decompressed) file, so the header code of vcftools
// can run unchanged, and the relatedness fast path can parse the remaining bytes in place.
//
// Gzip input is inflated by a producer thread into a fixed virtual reservation (the buffer never moves) and the
// number of final bytes is published after every deflate block, so consumers can parse while it inflates.
#ifndef SWIFTREL_INPUT_H
#define SWIFTREL_INPUT_H

#include <atomic>
#include <cstddef>
#include <string>
#include <thread>
#include <vector>

namespace swiftrel {

class Input {
   public:
    Input() {}
    ~Input();
    // Load a file. gz=true follows zlib gzopen/gzread: gzip members are inflated (several members are
    // concatenated, trailing non-gzip bytes are ignored); a file without the gzip magic is read as is.
    // Returns false if the file cannot be opened.
    bool open(const std::string &path, bool gz);
    bool eof() const { return eof_; }
    void read_line(std::string &out);  // getline/gzgets semantics, without trimming

    // Bytes [0, available()) of data are final. Blocks until more than `have` bytes are final or the input is
    // complete; returns the number of final bytes. complete() is true once nothing more will arrive.
    size_t wait_beyond(size_t have);
    size_t available() const { return avail.load(std::memory_order_acquire); }
    bool complete() const { return done.load(std::memory_order_acquire); }
    bool streaming() const { return producer.joinable(); }
    void finish_producer();  // join the producer (if any); afterwards size is final

    char *data = nullptr;  // decompressed bytes (zero padding after the end)
    size_t size = 0;       // number of bytes (final once complete())
    size_t pos = 0;        // read position of read_line()
    bool eof_ = false;
    bool gz_mode = false;  // gzgets+strlen semantics: a line is cut at its first NUL byte

   private:
    size_t cap = 0;
    bool mapped = false;       // data is an mmap reservation (gzip output)
    bool data_is_file = false; // data points into the read-only file mapping (plain text)
    struct ByteView {
        const unsigned char *p = nullptr;
        size_t n = 0;
        size_t size() const { return n; }
        const unsigned char *data() const { return p; }
        unsigned char operator[](size_t i) const { return p[i]; }
    } raw;                                // the file's bytes (compressed for gzip)
    std::vector<unsigned char> raw_own;   // used when the file cannot be mapped (pipes, empty files)
    void *raw_map = nullptr;              // file mapping + one zero page
    size_t raw_map_len = 0;
    void *res_base = nullptr;             // gzip output reservation as mapped (data is its 2 MB-aligned start)
    size_t res_len = 0;
    size_t pf_front = 0, pf_cap = 0;      // output pre-faulted up to pf_front; never beyond pf_cap
    bool pf_ok = true;
    void prefault(size_t a);
    std::atomic<size_t> avail{0};
    std::atomic<bool> done{false};
    std::thread producer;
    void reserve(size_t n);
    void inflate_all();  // producer body
    void publish(size_t n);
};

double now_s();
bool profiling();

}  // namespace swiftrel

#endif
