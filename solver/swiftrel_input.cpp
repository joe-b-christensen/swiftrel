// swiftrel_input.cpp -- see swiftrel_input.h. Part of SwiftRel (LGPL-3.0, derived from vcftools 0.1.17).
#include "swiftrel_input.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#include <zlib.h>

#include "isa-l/include/igzip_lib.h"
#include "isa-l/include/crc.h"

extern "C" int swiftrel_isal_defer_crc;  // isa-l/igzip/igzip_inflate.c (SwiftRel modification)

#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <vector>

namespace swiftrel {

static const size_t kPad = 64;

double now_s() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

bool profiling() {
    static int p = -1;
    if (p < 0) {
        const char *s = getenv("SWIFTREL_PROFILE");
        p = (s && *s && *s != '0') ? 1 : 0;
    }
    return p == 1;
}

Input::~Input() {
    finish_producer();
    if (mapped)
        munmap(res_base, res_len);
    else if (!data_is_file)
        free(data);
    if (raw_map) munmap(raw_map, raw_map_len);
}

void Input::finish_producer() {
    if (producer.joinable()) producer.join();
}

void Input::reserve(size_t n) {
    if (n + kPad <= cap) return;
    size_t nc = cap ? cap : (1 << 20);
    while (nc < n + kPad) nc *= 2;
    const size_t HP = 2u << 20;
    nc = (nc + HP - 1) / HP * HP;
    char *d = (char *)aligned_alloc(HP, nc);
    if (!d) abort();
    madvise(d, nc, MADV_HUGEPAGE);
    if (data) {
        memcpy(d, data, size);
        free(data);
    }
    data = d;
    cap = nc;
}

static bool read_all(int fd, std::vector<unsigned char> &buf) {
    struct stat st;
    size_t hint = 0;
    if (fstat(fd, &st) == 0 && S_ISREG(st.st_mode)) hint = (size_t)st.st_size;
    buf.resize(hint + 1);
    size_t n = 0;
    for (;;) {
        if (n == buf.size()) buf.resize(buf.size() * 2 + 4096);
        ssize_t r = ::read(fd, buf.data() + n, buf.size() - n);
        if (r < 0) {
            if (errno == EINTR) continue;
            return false;
        }
        if (r == 0) break;
        n += (size_t)r;
    }
    buf.resize(n);
    return true;
}

static std::atomic<unsigned> g_gen{0};  // bumped on every publication (waiters sleep on it)

void Input::publish(size_t n) {
    avail.store(n, std::memory_order_release);
    g_gen.fetch_add(1, std::memory_order_release);
    g_gen.notify_all();
}

#ifndef MADV_POPULATE_WRITE
#define MADV_POPULATE_WRITE 23
#endif

// Pre-fault the inflate output ahead of the producer, from the consumer thread: the producer otherwise takes a page
// fault (and zeroes a 2 MB page) every 2 MB, on the critical path. MADV_POPULATE_WRITE maps pages writable without
// touching their contents, so it is safe while the producer writes nearby. Up to 8 MB ahead, and not beyond the
// size announced by the last gzip member's ISIZE (+4 MB). Disabled if the kernel refuses (before Linux 5.14).
void Input::prefault(size_t a) {
    const size_t HP = (size_t)2 << 20, ahead = (size_t)8 << 20;
    if (!pf_ok || !mapped) return;
    if (pf_front < a) pf_front = a & ~(HP - 1);
    while (pf_front < pf_cap && pf_front < a + ahead) {
        if (madvise(data + pf_front, HP, MADV_POPULATE_WRITE) != 0) {
            pf_ok = false;
            return;
        }
        pf_front += HP;
    }
}

size_t Input::wait_beyond(size_t have) {
    if (!done.load(std::memory_order_acquire)) prefault(avail.load(std::memory_order_acquire));
    for (;;) {
        unsigned g = g_gen.load(std::memory_order_acquire);
        size_t a = avail.load(std::memory_order_acquire);
        if (a > have) return a;
        if (done.load(std::memory_order_acquire)) return avail.load(std::memory_order_acquire);
        g_gen.wait(g, std::memory_order_acquire);
    }
}

// Producer: gzip members are inflated with ISA-L's igzip (streaming, 1 MB of output per call; progress is
// published after every call); a member igzip rejects (invalid data, bad checksum, truncated) and everything after
// it are redone with zlib's streaming inflate, which -- like gzread -- keeps whatever it decoded before the problem.
void Input::inflate_all() {
    double t_start = now_s();
    size_t in_pos = 0;
    size_t out = 0;
    const size_t limit = cap - kPad;
    bool use_zlib = false;
    struct inflate_state st;
    // The CRC32 of a member is computed only when another gzip member follows it. For the last member it would
    // change nothing: a bad CRC there made igzip fail at the very end, and the zlib redo below then keeps every
    // byte of the member (zlib reports the error only after writing all of its output), so the delivered data was
    // the same with or without the check. ISIZE is still verified by igzip. This takes the CRC (~5 ms for 80 MB)
    // off the inflate thread, which is the critical path for compressed input.
    swiftrel_isal_defer_crc = 1;
    while (!use_zlib && in_pos + 1 < raw.size() && raw[in_pos] == 31 && raw[in_pos + 1] == 139) {
        isal_inflate_init(&st);
        st.crc_flag = ISAL_GZIP;
        const size_t member_out = out;
        size_t fed = in_pos;  // input bytes handed to igzip so far
        bool finished = false;
        for (;;) {
            if (st.avail_in == 0 && fed < raw.size()) {
                size_t n = std::min<size_t>(raw.size() - fed, (size_t)1 << 30);
                st.next_in = const_cast<uint8_t *>(raw.data() + fed);
                st.avail_in = (uint32_t)n;
                fed += n;
            }
            size_t room = std::min<size_t>(limit - out, (size_t)1 << 20);
            if (room == 0) break;
            st.next_out = (uint8_t *)(data + out);
            st.avail_out = (uint32_t)room;
            int ret = isal_inflate(&st);
            out += room - st.avail_out;
            if (ret != ISAL_DECOMP_OK) break;
            if (st.block_state == ISAL_BLOCK_FINISH) {
                finished = true;
                break;
            }
            if (st.avail_in == 0 && fed >= raw.size() && st.avail_out != 0) break;  // truncated input
            publish(out);
        }
        if (finished) {
            const size_t next_pos = fed - st.avail_in;
            const bool more = next_pos + 1 < raw.size() && raw[next_pos] == 31 && raw[next_pos + 1] == 139;
            if (more && crc32_gzip_refl(0, (const unsigned char *)data + member_out, out - member_out) != st.crc)
                finished = false;  // bad CRC before another member: the zlib redo stops after this member
        }
        if (!finished) {
            out = member_out;  // zlib redoes this member (identical bytes for everything already published)
            use_zlib = true;
            break;
        }
        in_pos = fed - st.avail_in;
        publish(out);
    }
    if (use_zlib) {
        z_stream zs;
        memset(&zs, 0, sizeof zs);
        if (inflateInit2(&zs, 15 + 16) == Z_OK) {
            while (in_pos + 1 < raw.size() && raw[in_pos] == 31 && raw[in_pos + 1] == 139) {
                inflateReset(&zs);
                zs.next_in = (Bytef *)(raw.data() + in_pos);
                zs.avail_in = (uInt)(raw.size() - in_pos);
                int ret = Z_OK;
                for (;;) {
                    size_t room = std::min<size_t>(limit - out, (size_t)1 << 24);
                    if (room == 0) break;
                    zs.next_out = (Bytef *)(data + out);
                    zs.avail_out = (uInt)room;
                    ret = inflate(&zs, Z_NO_FLUSH);
                    out += room - zs.avail_out;
                    if (out > avail.load(std::memory_order_relaxed)) publish(out);
                    if (ret == Z_STREAM_END) break;
                    if (ret != Z_OK && ret != Z_BUF_ERROR) break;
                    if (ret == Z_BUF_ERROR && zs.avail_in == 0) break;  // truncated input
                }
                if (ret != Z_STREAM_END) break;  // error or truncated: keep what was decoded
                in_pos = raw.size() - zs.avail_in;
            }
            inflateEnd(&zs);
        }
    }
    // inflaters may write scratch bytes past the end: zero the padding before the final publication
    memset(data + out, 0, kPad);
    size = out;
    publish(out);
    done.store(true, std::memory_order_release);
    g_gen.fetch_add(1, std::memory_order_release);
    g_gen.notify_all();
    if (profiling()) fprintf(stderr, "[swiftrel] inflate %.3f s (%zu bytes)\n", now_s() - t_start, out);
}

// Map a regular file read-only, followed by one anonymous zero page, so that the bytes after the end of the file
// read as zero (the parsers rely on 64 bytes of zero padding). Returns false if the file cannot be mapped.
static bool map_file(int fd, size_t len, void *&base, size_t &tot) {
    if (len == 0) return false;
    const size_t pg = 4096;
    tot = (len + pg - 1) / pg * pg + pg;
    void *b = mmap(nullptr, tot, PROT_READ, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (b == MAP_FAILED) return false;
    void *f = mmap(b, len, PROT_READ, MAP_PRIVATE | MAP_FIXED | MAP_POPULATE, fd, 0);
    if (f == MAP_FAILED) {
        munmap(b, tot);
        return false;
    }
    base = b;
    return true;
}

bool Input::open(const std::string &path, bool gz) {
    double t_start = now_s();
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) return false;
    size = 0;
    pos = 0;
    eof_ = false;
    gz_mode = gz;
    struct stat st;
    const bool reg = fstat(fd, &st) == 0 && S_ISREG(st.st_mode);
    void *mb = nullptr;
    size_t mlen = 0;
    if (reg && map_file(fd, (size_t)st.st_size, mb, mlen)) {
        raw_map = mb;
        raw_map_len = mlen;
        raw.p = (const unsigned char *)mb;
        raw.n = (size_t)st.st_size;
    } else {
        if (!read_all(fd, raw_own)) {
            ::close(fd);
            return false;
        }
        raw.p = raw_own.data();
        raw.n = raw_own.size();
    }
    ::close(fd);
    if (gz && raw.size() >= 2 && raw[0] == 31 && raw[1] == 139) {
        // reserve address space once (pages are committed only when written), so the buffer never moves while
        // consumers read it; 1 TiB of virtual space is far beyond any decompressed VCF we could hold in RAM
        const size_t V = (size_t)1 << 40, HP = (size_t)2 << 20;
        void *m = mmap(nullptr, V + HP, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
        if (m != MAP_FAILED) {
            res_base = m;
            res_len = V + HP;
            data = (char *)(((uintptr_t)m + HP - 1) & ~(uintptr_t)(HP - 1));  // 2 MB aligned: whole huge pages
            madvise(data, V, MADV_HUGEPAGE);
            cap = V;
            mapped = true;
            if (raw.size() >= 18) {
                uint32_t isz;
                memcpy(&isz, raw.data() + raw.size() - 4, 4);
                pf_cap = (size_t)isz + ((size_t)4 << 20);
            }
            producer = std::thread(&Input::inflate_all, this);
            return true;
        }
        // no address space: inflate synchronously, doubling the buffer until the output fits
        size_t guess = std::max<size_t>(raw.size() * 4, 1 << 20);
        if (raw.size() >= 18) {
            uint32_t isz;
            memcpy(&isz, raw.data() + raw.size() - 4, 4);
            if ((size_t)isz + 1 > guess) guess = (size_t)isz + 1;
        }
        for (;;) {
            reserve(guess);
            done.store(false);
            avail.store(0);
            inflate_all();
            if (size + kPad < cap) break;  // did not run out of room
            guess = cap * 2;
        }
    } else if (raw_map) {
        // plain text (or gzopen's transparent reading of a non-gzip file): parse the mapping in place
        data = (char *)raw_map;
        data_is_file = true;
        size = raw.size();
        avail.store(size, std::memory_order_release);
        done.store(true, std::memory_order_release);
    } else {
        reserve(raw.size());
        if (raw.size()) memcpy(data, raw.data(), raw.size());
        size = raw.size();
        memset(data + size, 0, kPad);
        avail.store(size, std::memory_order_release);
        done.store(true, std::memory_order_release);
    }
    if (profiling()) fprintf(stderr, "[swiftrel] load %.3f s (%zu bytes)\n", now_s() - t_start, size);
    return true;
}

void Input::read_line(std::string &out) {
    out.clear();
    // wait until a complete line (or the end of the data) is available
    for (;;) {
        size_t a = available();
        if (pos < a && memchr(data + pos, '\n', a - pos)) break;
        if (complete()) break;
        wait_beyond(a);
    }
    const size_t lim = available();
    if (complete()) size = lim;
    if (pos >= lim) {
        eof_ = true;
        return;
    }
    const char *p = data + pos;
    const char *nl = (const char *)memchr(p, '\n', lim - pos);
    if (nl) {
        out.assign(p, (size_t)(nl - p));
        pos = (size_t)(nl - data) + 1;
    } else {
        out.assign(p, lim - pos);
        pos = lim;
        eof_ = true;
    }
    if (gz_mode) {
        size_t z = out.find('\0');
        if (z != std::string::npos) out.resize(z);
    }
}

}  // namespace swiftrel
