// probe-p2p.cu - P5.1 interconnect probe (task #26): P2P capability + measured copy bandwidth
// between the MI50s, to decide the P5.3 relay transport (peer-direct vs host-pinned bounce).
//
// Build (gfx906 is wave64 natively; one flag, no compat layer needed - this file uses hip* only):
//   /opt/rocm/bin/hipcc --offload-arch=gfx906 -O2 -o build-906/probe-p2p tools/gfx906/probe-p2p.cu
// Run via tools/gfx906/run-p5.1-probe.sh (engine-alive guard). Design input, not a speed claim:
// the numbers choose a transport and size a budget; they certify no model behavior.
//
// For every ordered GPU pair (i != j):
//   hipDeviceCanAccessPeer, hipDeviceEnablePeerAccess (actual result, incl. "already enabled"),
//   64 MB peer-direct copy MB/s (when enabled),
//   64 MB staged copy device i -> host-pinned (registered on i) -> device j MB/s,
//   64+64 MB bidirectional peer-direct MB/s (aggregate).
// Plus per-device properties and the sysfs link lines for the receipt.

#include <hip/hip_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CK(x)                                                              \
    do {                                                                   \
        hipError_t e_ = (x);                                               \
        if (e_ != hipSuccess) {                                            \
            std::fprintf(stderr, "FATAL %s -> %s (%s:%d)\n", #x,           \
                         hipGetErrorString(e_), __FILE__, __LINE__);       \
            std::exit(1);                                                  \
        }                                                                  \
    } while (0)

static const size_t MB = 1024ull * 1024ull;
static const size_t COPY_BYTES = 64ull * MB;
static const int ITERS = 5;

struct Dev {
    int ordinal;
    hipDevice_t h;
    hipDeviceProp_t prop;
    void* buf = nullptr;        // COPY_BYTES device buffer
    void* pin_i = nullptr;      // pinned buffer with device i as primary context (staged leg A)
    hipStream_t s2 = nullptr;   // helper stream for bidirectional
};

static float timed_copy(Dev& src, Dev& dst, void* dstbuf, hipStream_t stream) {
    // one warmup + ITERS timed device->device copies on `stream` with the dst device current
    CK(hipSetDevice(dst.ordinal));
    CK(hipMemcpyAsync(dstbuf, src.buf, COPY_BYTES, hipMemcpyDeviceToDevice, stream));
    hipEvent_t a, b;
    CK(hipEventCreate(&a));
    CK(hipEventCreate(&b));
    CK(hipEventRecord(a, stream));
    for (int it = 0; it < ITERS; ++it)
        CK(hipMemcpyAsync(dstbuf, src.buf, COPY_BYTES, hipMemcpyDeviceToDevice, stream));
    CK(hipEventRecord(b, stream));
    CK(hipEventSynchronize(b));
    float ms = 0.f;
    CK(hipEventElapsedTime(&ms, a, b));
    CK(hipEventDestroy(a));
    CK(hipEventDestroy(b));
    return (float)(COPY_BYTES * ITERS / (ms / 1000.0) / (1024.0 * 1024.0));  // MiB/s
}

static float staged_mbps(Dev& src, Dev& dst) {
    // src device -> pin registered on src -> dst device; the dst leg makes dst current.
    // The whole pipeline is timed wall-clock: this is what a bounce relay really costs.
    CK(hipSetDevice(src.ordinal));
    CK(hipMemcpyAsync(src.pin_i, src.buf, COPY_BYTES, hipMemcpyDeviceToHost, nullptr));
    CK(hipDeviceSynchronize());
    auto t0 = std::chrono::steady_clock::now();
    for (int it = 0; it < ITERS; ++it) {
        CK(hipSetDevice(src.ordinal));
        CK(hipMemcpyAsync(src.pin_i, src.buf, COPY_BYTES, hipMemcpyDeviceToHost, nullptr));
        CK(hipSetDevice(dst.ordinal));
        CK(hipMemcpyAsync(dst.buf, src.pin_i, COPY_BYTES, hipMemcpyHostToDevice, nullptr));
        CK(hipDeviceSynchronize());
        CK(hipSetDevice(src.ordinal));
        CK(hipDeviceSynchronize());
    }
    auto t1 = std::chrono::steady_clock::now();
    double s = std::chrono::duration<double>(t1 - t0).count();
    return (float)(COPY_BYTES * ITERS / (s * 1024.0 * 1024.0));
}

static float bidir_mbps(Dev& a, Dev& b) {
    // a->b on a's stream, b->a on b's helper stream, both devices current at issue time;
    // returns the AGGREGATE MiB/s of both directions.
    CK(hipSetDevice(a.ordinal));
    CK(hipMemcpyAsync(b.buf, a.buf, COPY_BYTES, hipMemcpyDeviceToDevice, nullptr));  // warm
    CK(hipSetDevice(b.ordinal));
    CK(hipMemcpyAsync(a.buf, b.buf, COPY_BYTES, hipMemcpyDeviceToDevice, b.s2));
    CK(hipSetDevice(a.ordinal));
    CK(hipDeviceSynchronize());
    CK(hipSetDevice(b.ordinal));
    CK(hipDeviceSynchronize());
    auto t0 = std::chrono::steady_clock::now();
    for (int it = 0; it < ITERS; ++it) {
        CK(hipSetDevice(a.ordinal));
        CK(hipMemcpyAsync(b.buf, a.buf, COPY_BYTES, hipMemcpyDeviceToDevice, nullptr));
        CK(hipSetDevice(b.ordinal));
        CK(hipMemcpyAsync(a.buf, b.buf, COPY_BYTES, hipMemcpyDeviceToDevice, b.s2));
    }
    CK(hipSetDevice(a.ordinal));
    CK(hipDeviceSynchronize());
    CK(hipSetDevice(b.ordinal));
    CK(hipDeviceSynchronize());
    auto t1 = std::chrono::steady_clock::now();
    double s = std::chrono::duration<double>(t1 - t0).count();
    return (float)(2.0 * COPY_BYTES * ITERS / (s * 1024.0 * 1024.0));
}

int main() {
    std::printf("probe-p2p: P5.1 interconnect probe (design input, not a speed claim)\n");
    int n = 0;
    CK(hipGetDeviceCount(&n));
    if (n < 2) { std::fprintf(stderr, "need >= 2 GPUs, found %d\n", n); return 1; }
    std::printf("devices: %d\n", n);

    std::vector<Dev> devs((size_t)n);
    for (int i = 0; i < n; ++i) {
        Dev& d = devs[(size_t)i];
        d.ordinal = i;
        CK(hipDeviceGet(&d.h, i));
        CK(hipGetDeviceProperties(&d.prop, i));
        size_t free_b = 0, total_b = 0;
        CK(hipSetDevice(i));
        CK(hipMemGetInfo(&free_b, &total_b));
        std::printf("dev %d: %s  vram %.1f/%.1f GiB free/total  pci %04x:%02x:%02x\n",
                    i, d.prop.name, free_b / 1073741824.0, total_b / 1073741824.0,
                    d.prop.pciDomainID, d.prop.pciBusID, d.prop.pciDeviceID);
        CK(hipMalloc(&d.buf, COPY_BYTES));
        CK(hipMemset(d.buf, 0x5a, COPY_BYTES));
        CK(hipHostMalloc(&d.pin_i, COPY_BYTES, hipHostMallocDefault));
        CK(hipStreamCreate(&d.s2));
    }

    std::printf("\n%-7s %-7s %-14s %-12s %-14s %-16s %-16s\n",
                "src", "dst", "canAccessPeer", "enablePeer", "peer MiB/s", "staged MiB/s", "bidir MiB/s");
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i == j) continue;
            int can = 0;
            CK(hipDeviceCanAccessPeer(&can, devs[(size_t)i].h, devs[(size_t)j].h));
            CK(hipSetDevice(j));   // EnablePeerAccess: target must NOT be current
            hipError_t en = hipDeviceEnablePeerAccess(devs[(size_t)i].h, 0);
            const char* en_s = en == hipSuccess ? "ok"
                             : en == hipErrorPeerAccessAlreadyEnabled ? "already" : "FAILED";
            float peer = 0.f, staged = 0.f, bidir = 0.f;
            if (can && (en == hipSuccess || en == hipErrorPeerAccessAlreadyEnabled))
                peer = timed_copy(devs[(size_t)i], devs[(size_t)j], devs[(size_t)j].buf, nullptr);
            staged = staged_mbps(devs[(size_t)i], devs[(size_t)j]);
            if (can && (en == hipSuccess || en == hipErrorPeerAccessAlreadyEnabled))
                bidir = bidir_mbps(devs[(size_t)i], devs[(size_t)j]);
            std::printf("%-7d %-7d %-14d %-12s %-14.0f %-16.0f %-16.0f\n",
                        i, j, can, en_s, peer, staged, bidir);
            std::fflush(stdout);
            if (en == hipSuccess) (void)hipDeviceDisablePeerAccess(devs[(size_t)i].h);  // leave state as found
        }
    }

    std::printf("\nstaged = device->host-pinned(primary ctx)->device, whole pipeline wall-clock;\n"
                "peer/bidir require EnablePeerAccess ok; 0 columns mean that transport is unavailable.\n"
                "Infinity Fabric, if present, would show as peer >> staged; all-PCIe shows them similar.\n");
    for (auto& d : devs) {
        (void)hipFree(d.buf);
        (void)hipHostFree(d.pin_i);
        (void)hipStreamDestroy(d.s2);
    }
    std::printf("probe-p2p: done, exit 0\n");
    return 0;
}
