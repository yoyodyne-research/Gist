// superfly_audio: the live audio host for superfly (https://github.com/yoyodyne-research/superfly).
//
// Owns the audio device. Each sample of the source (the interface's input, or a track played from memory)
// goes two ways:
//   - out through the filter chain (fly_filter.h) to the device, with no wait for anything else;
//   - through a lock-free ring to the analysis thread, which runs CARFAC causally and writes a frame
//     (plus that frame's source audio) to stdout every 256 samples.
// superfly (the parent process) runs the fly on those frames and sends filter parameters back on stdin;
// they take effect at the next audio block, ramped over one frame.
//
// Protocol (little endian): u32 length (of what follows), u8 type, payload.
//   host -> superfly
//     1 HELLO   JSON: the CARFAC feature description (as kitsune_test features writes) plus "host": {...}
//     2 FRAME   u64 k, float32 env_fast[b], env_slow[b], delta[b]   (state after samples [0, 256 (k+1)))
//     3 END
//     4 SOURCE  text key=value lines: at (sample), source (input|track|silence), pos (track sample at `at`),
//               playing (0|1), track (path)
//     5 AUDIO   u64 first sample, float32 x[256]: the source audio of the frame that follows
//     6 STATUS  text key=value lines: xruns, dropped, buffer, rate, lim_db, recording
//     7 LAG     u64 k, i64 samples from frame k's end to the block that applied its parameters
//   superfly -> host
//    16 PARAMS  u64 k, u8 kind, float32 cutoff, q, gain_db, vca
//    17 TRANSPORT  text: source=input|track, path=<wav> (tracks), play=0|1, pos=<seconds>, loop=<a>,<b> (s)
//    18 SETTINGS   text: volume_db, mix, limiter=0|1, agc=0|1, record=<path> (empty stops)
//    19 QUIT
//
// Drivers: jack (the real thing; run under pw-jack with PipeWire) and null (a timer instead of a device,
// for tests: --null-input WAV loops as the input, --null-output F32 captures the output, --fast skips pacing).
// `render IN.wav PARAMS.f32 OUT.f32 [--mix M]` runs only the filter chain offline (parity tests):
// PARAMS is float32 [frames][5] = kind, cutoff, q, gain_db, vca; frame k's values are set at sample 256 (k+1).

#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <unistd.h>
#if defined(__SSE__) || defined(__x86_64__)
#include <xmmintrin.h>
#endif

#include "carfac_cli.h"
#include "carfac_frontend.h"
#include "fly_filter.h"
#ifdef HAVE_JACK
#include <jack/jack.h>
#endif

namespace {

constexpr int kHop = 256;
enum : uint8_t { kHello = 1, kFrame = 2, kEnd = 3, kSource = 4, kAudio = 5, kStatus = 6, kLag = 7,
                 kParams = 16, kTransport = 17, kSettings = 18, kQuit = 19 };

// ---------- lock-free single-producer single-consumer ring ----------
template <typename T>
class Ring {
public:
    explicit Ring(size_t capacity) : buf_(capacity + 1) {}
    bool push(const T& v) {
        const size_t h = head_.load(std::memory_order_relaxed), n = next(h);
        if (n == tail_.load(std::memory_order_acquire)) return false;
        buf_[h] = v;
        head_.store(n, std::memory_order_release);
        return true;
    }
    size_t size() const {
        const size_t h = head_.load(std::memory_order_acquire), t = tail_.load(std::memory_order_acquire);
        return h >= t ? h - t : h + buf_.size() - t;
    }
    size_t capacity() const { return buf_.size() - 1; }
    bool pop(T& v) {
        const size_t t = tail_.load(std::memory_order_relaxed);
        if (t == head_.load(std::memory_order_acquire)) return false;
        v = buf_[t];
        tail_.store(next(t), std::memory_order_release);
        return true;
    }

private:
    size_t next(size_t i) const { return i + 1 == buf_.size() ? 0 : i + 1; }
    std::vector<T> buf_;
    std::atomic<size_t> head_{0}, tail_{0};
};

// ---------- output to superfly (analysis and control threads both write; one lock) ----------
std::mutex out_mu;
void writeMsg(uint8_t type, const void* a, uint32_t na, const void* b = nullptr, uint32_t nb = 0) {
    std::lock_guard<std::mutex> lock(out_mu);
    const uint32_t len = 1 + na + nb;
    std::fwrite(&len, 4, 1, stdout);
    std::fwrite(&type, 1, 1, stdout);
    if (na) std::fwrite(a, 1, na, stdout);
    if (nb) std::fwrite(b, 1, nb, stdout);
    std::fflush(stdout);
}
void writeText(uint8_t type, const std::string& s) { writeMsg(type, s.data(), static_cast<uint32_t>(s.size())); }

std::map<std::string, std::string> parseKv(const std::string& s) {
    std::map<std::string, std::string> kv;
    std::istringstream in(s);
    std::string line;
    while (std::getline(in, line)) {
        const size_t eq = line.find('=');
        if (eq != std::string::npos) kv[line.substr(0, eq)] = line.substr(eq + 1);
    }
    return kv;
}

bool loadWavMono(const std::string& path, std::vector<float>& out, int& rate);

// ---------- messages from the control thread to the audio thread ----------
struct Command {
    enum Kind { kParamsCmd, kTransportCmd, kMixCmd, kVolumeCmd, kLimiterCmd } kind;
    uint64_t k = 0;
    int filter = 0;
    fly::FilterParams p;
    double value = 0.0;
    // transport
    int source = 0;  // 0 input, 1 track
    const std::vector<float>* track = nullptr;
    const std::string* track_name = nullptr;  // map key: stable and never modified
    bool play = false;
    int64_t pos = 0, loop_a = -1, loop_b = -1;
};

// ---------- events from the audio thread to the analysis thread (stamped with the sample index) ----------
struct Event {
    enum Kind { kSourceEv, kLagEv } kind;
    uint64_t at = 0;  // global sample index
    int source = 0;   // 0 input, 1 track, 2 silence (a track that is paused or ended)
    int64_t pos = 0;
    bool playing = false;
    const std::string* track_name = nullptr;
    uint64_t k = 0;
    int64_t lag = 0;
};

struct Host {
    int rate = 44100;
    // audio thread state
    fly::FilterChain chain;
    int source = 0;
    const std::vector<float>* track = nullptr;
    const std::string* track_name = nullptr;
    int64_t pos = 0, loop_a = -1, loop_b = -1;
    bool playing = false;
    uint64_t n = 0;  // samples processed since start
    // cross-thread
    Ring<Command> commands{1024};
    Ring<float> analysis{1 << 18};  // ~6 s of audio at 44.1 kHz
    Ring<Event> events{4096};
    std::atomic<uint64_t> dropped{0}, xruns{0};
    std::atomic<int> buffer{0};
    std::atomic<bool> quit{false}, agc{true};
    std::atomic<double> lim_db{0.0};
    std::map<std::string, std::vector<float>> tracks;  // control thread only (never freed while running)
    std::mutex rec_mu;
    std::string rec_path;  // guarded by rec_mu; the analysis thread records

    void emitSource() {
        Event e;
        e.kind = Event::kSourceEv; e.at = n; e.source = source == 1 ? (playing ? 1 : 2) : 0;
        e.pos = pos; e.playing = playing; e.track_name = track_name;
        events.push(e);
    }

    // The real-time part. `in` may be null (no input), outputs may alias.
    void process(const float* in, float* out_l, float* out_r, int frames) {
        Command c;
        while (commands.pop(c)) {
            switch (c.kind) {
            case Command::kParamsCmd: {
                chain.setTarget(c.filter, c.p);
                Event e; e.kind = Event::kLagEv; e.at = n; e.k = c.k;
                e.lag = static_cast<int64_t>(n) - static_cast<int64_t>(kHop * (c.k + 1));
                events.push(e);
                break;
            }
            case Command::kMixCmd: chain.setMix(c.value); break;
            case Command::kVolumeCmd: chain.setVolume(c.value); break;
            case Command::kLimiterCmd: chain.setLimiter(c.value != 0.0); break;
            case Command::kTransportCmd:
                source = c.source; track = c.track; track_name = c.track_name;
                playing = c.play && c.track != nullptr;
                pos = c.pos; loop_a = c.loop_a; loop_b = c.loop_b;
                emitSource();
                break;
            }
        }
        for (int i = 0; i < frames; ++i, ++n) {
            float x = 0.0f;
            if (source == 0) {
                x = in ? in[i] : 0.0f;
            } else if (playing) {
                if (loop_b > loop_a && loop_a >= 0 && pos >= loop_b) { pos = loop_a; emitSource(); }
                if (pos < static_cast<int64_t>(track->size())) {
                    x = (*track)[pos++];
                } else {
                    playing = false;
                    emitSource();
                }
            }
            if (!analysis.push(x)) dropped.fetch_add(1, std::memory_order_relaxed);
            const float y = static_cast<float>(chain.process(x));
            out_l[i] = y;
            if (out_r != out_l) out_r[i] = y;
        }
        lim_db.store(chain.reduction_db(), std::memory_order_relaxed);
    }
};

// ---------- analysis thread: CARFAC on the ring, frames and audio to stdout, recording ----------
struct WavWriter {
    std::ofstream f;
    uint32_t samples = 0;
    bool open(const std::string& path, int rate) {
        f.open(path, std::ios::binary);
        if (!f) return false;
        samples = 0;
        header(rate);
        return true;
    }
    void header(int rate) {
        const uint32_t data = samples * 2, riff = 36 + data, fmt = 16, br = rate * 2;
        const uint16_t pcm = 1, ch = 1, align = 2, bits = 16;
        f.seekp(0);
        f.write("RIFF", 4); f.write(reinterpret_cast<const char*>(&riff), 4); f.write("WAVEfmt ", 8);
        f.write(reinterpret_cast<const char*>(&fmt), 4); f.write(reinterpret_cast<const char*>(&pcm), 2);
        f.write(reinterpret_cast<const char*>(&ch), 2); f.write(reinterpret_cast<const char*>(&rate), 4);
        f.write(reinterpret_cast<const char*>(&br), 4); f.write(reinterpret_cast<const char*>(&align), 2);
        f.write(reinterpret_cast<const char*>(&bits), 2); f.write("data", 4);
        f.write(reinterpret_cast<const char*>(&data), 4);
        f.seekp(0, std::ios::end);
    }
    void write(const float* x, int n) {
        for (int i = 0; i < n; ++i) {
            // the exact inverse of reading (s / 32768), so recorded 16-bit input round-trips unchanged
            const long v = std::lrint(static_cast<double>(x[i]) * 32768.0);
            const int16_t s = static_cast<int16_t>(std::max(-32768L, std::min(32767L, v)));
            f.write(reinterpret_cast<const char*>(&s), 2);
        }
        samples += n;
    }
    void close(int rate) { if (f.is_open()) { header(rate); f.close(); } }
};

void analysisLoop(Host& h, CarfacFrontend& carfac) {
    const int bands = carfac.numBands();
    std::vector<float> env_fast, env_slow, delta, frame(3 * bands), audio(kHop);
    std::vector<unsigned char> msg(8 + 3 * bands * sizeof(float));
    std::vector<unsigned char> amsg(8 + kHop * sizeof(float));
    uint64_t k = 0, s = 0;  // frames written, samples analysed
    int count = 0;
    WavWriter rec;
    std::string recording;
    Event pending;
    bool has_pending = false;
    auto flushEvents = [&](uint64_t upto) {
        for (;;) {
            if (!has_pending) {
                if (!h.events.pop(pending)) return;
                has_pending = true;
            }
            if (pending.kind == Event::kSourceEv && pending.at > upto) return;  // not reached yet
            if (pending.kind == Event::kSourceEv) {
                std::ostringstream t;
                t << "at=" << pending.at << "\nsource=" << (pending.source == 0 ? "input" : pending.source == 1 ? "track" : "silence")
                  << "\npos=" << pending.pos << "\nplaying=" << (pending.playing ? 1 : 0) << "\ntrack="
                  << (pending.track_name ? *pending.track_name : std::string()) << "\n";
                writeText(kSource, t.str());
            } else {
                unsigned char b[16];
                std::memcpy(b, &pending.k, 8);
                std::memcpy(b + 8, &pending.lag, 8);
                writeMsg(kLag, b, 16);
            }
            has_pending = false;
        }
    };
    while (!h.quit.load()) {
        float x;
        bool any = false;
        while (h.analysis.pop(x)) {
            any = true;
            flushEvents(s);
            audio[count] = x;
            carfac.processSample(x);
            ++s;
            if (++count == kHop) {
                count = 0;
                if (carfac.agc() != h.agc.load()) carfac.setAgc(h.agc.load());
                const uint64_t first = s - kHop;
                std::memcpy(amsg.data(), &first, 8);
                std::memcpy(amsg.data() + 8, audio.data(), kHop * sizeof(float));
                writeMsg(kAudio, amsg.data(), static_cast<uint32_t>(amsg.size()));
                carfac.publish(env_fast, env_slow, delta);
                std::memcpy(msg.data(), &k, 8);
                std::memcpy(msg.data() + 8, env_fast.data(), bands * sizeof(float));
                std::memcpy(msg.data() + 8 + bands * sizeof(float), env_slow.data(), bands * sizeof(float));
                std::memcpy(msg.data() + 8 + 2 * bands * sizeof(float), delta.data(), bands * sizeof(float));
                writeMsg(kFrame, msg.data(), static_cast<uint32_t>(msg.size()));
                ++k;
                {
                    std::lock_guard<std::mutex> lock(h.rec_mu);
                    if (h.rec_path != recording) {
                        rec.close(h.rate);
                        recording = h.rec_path;
                        if (!recording.empty() && !rec.open(recording, h.rate)) {
                            std::cerr << "superfly_audio: cannot record to " << recording << "\n";
                            recording.clear();
                            h.rec_path.clear();
                        }
                    }
                }
                if (!recording.empty()) rec.write(audio.data(), kHop);
            }
        }
        flushEvents(s);
        if (!any) std::this_thread::sleep_for(std::chrono::microseconds(500));
    }
    rec.close(h.rate);
}

// ---------- control thread: commands from superfly ----------
bool readExact(void* p, size_t n) {
    auto* c = static_cast<unsigned char*>(p);
    while (n) {
        const ssize_t r = ::read(STDIN_FILENO, c, n);
        if (r < 0 && errno == EINTR) continue;
        if (r <= 0) return false;
        c += r; n -= static_cast<size_t>(r);
    }
    return true;
}

void pushCommand(Host& h, const Command& c) {
    while (!h.commands.push(c) && !h.quit.load()) std::this_thread::sleep_for(std::chrono::microseconds(200));
}

void controlLoop(Host& h) {
    std::vector<unsigned char> buf;
    for (;;) {
        uint32_t len;
        uint8_t type;
        if (!readExact(&len, 4) || len < 1 || !readExact(&type, 1)) break;
        buf.resize(len - 1);
        if (len > 1 && !readExact(buf.data(), len - 1)) break;
        if (type == kQuit) break;
        if (type == kParams && buf.size() >= 8 + 1 + 16) {
            Command c;
            c.kind = Command::kParamsCmd;
            std::memcpy(&c.k, buf.data(), 8);
            c.filter = buf[8];
            float v[4];
            std::memcpy(v, buf.data() + 9, 16);
            c.p.cutoff = v[0]; c.p.q = v[1]; c.p.gain_db = v[2]; c.p.vca = v[3];
            pushCommand(h, c);
        } else if (type == kTransport) {
            auto kv = parseKv(std::string(buf.begin(), buf.end()));
            Command c;
            c.kind = Command::kTransportCmd;
            c.source = kv["source"] == "track" ? 1 : 0;
            if (c.source == 1) {
                const std::string path = kv["path"];
                auto it = h.tracks.find(path);
                if (it == h.tracks.end()) {
                    std::vector<float> x;
                    int rate = 0;
                    if (!loadWavMono(path, x, rate) || rate != h.rate) {
                        std::cerr << "superfly_audio: cannot play " << path << " (need mono 16-bit at " << h.rate << " Hz)\n";
                        continue;
                    }
                    it = h.tracks.emplace(path, std::move(x)).first;
                }
                c.track = &it->second;
                c.track_name = &it->first;
            }
            c.play = kv["play"] == "1";
            c.pos = static_cast<int64_t>(std::llround(std::atof(kv["pos"].c_str()) * h.rate));
            const std::string loop = kv["loop"];
            const size_t comma = loop.find(',');
            if (comma != std::string::npos) {
                c.loop_a = static_cast<int64_t>(std::llround(std::atof(loop.substr(0, comma).c_str()) * h.rate));
                c.loop_b = static_cast<int64_t>(std::llround(std::atof(loop.substr(comma + 1).c_str()) * h.rate));
            }
            pushCommand(h, c);
        } else if (type == kSettings) {
            auto kv = parseKv(std::string(buf.begin(), buf.end()));
            Command c;
            if (kv.count("volume_db")) { c.kind = Command::kVolumeCmd; c.value = std::pow(10.0, std::atof(kv["volume_db"].c_str()) / 20.0); pushCommand(h, c); }
            if (kv.count("mix")) { c.kind = Command::kMixCmd; c.value = std::atof(kv["mix"].c_str()); pushCommand(h, c); }
            if (kv.count("limiter")) { c.kind = Command::kLimiterCmd; c.value = kv["limiter"] == "1"; pushCommand(h, c); }
            if (kv.count("agc")) h.agc.store(kv["agc"] == "1");
            if (kv.count("record")) { std::lock_guard<std::mutex> lock(h.rec_mu); h.rec_path = kv["record"]; }
        }
    }
    h.quit.store(true);
}

void statusLoop(Host& h) {
    while (!h.quit.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        std::ostringstream t;
        std::string rec;
        {
            std::lock_guard<std::mutex> lock(h.rec_mu);
            rec = h.rec_path;
        }
        t << "xruns=" << h.xruns.load() << "\ndropped=" << h.dropped.load() << "\nbuffer=" << h.buffer.load()
          << "\nrate=" << h.rate << "\nlim_db=" << h.lim_db.load() << "\nrecording=" << rec << "\n";
        writeText(kStatus, t.str());
    }
}

// ---------- WAV reading (mono 16-bit; first channel of stereo) ----------
bool loadWavMono(const std::string& path, std::vector<float>& out, int& rate) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    char riff[4], wave[4];
    uint32_t size;
    f.read(riff, 4); f.read(reinterpret_cast<char*>(&size), 4); f.read(wave, 4);
    if (std::strncmp(riff, "RIFF", 4) || std::strncmp(wave, "WAVE", 4)) return false;
    uint16_t fmt_tag = 0, channels = 0, bits = 0;
    uint32_t sr = 0;
    char id[4];
    uint32_t n;
    while (f.read(id, 4) && f.read(reinterpret_cast<char*>(&n), 4)) {
        if (!std::strncmp(id, "fmt ", 4)) {
            std::vector<char> c(n);
            f.read(c.data(), n);
            std::memcpy(&fmt_tag, c.data(), 2); std::memcpy(&channels, c.data() + 2, 2);
            std::memcpy(&sr, c.data() + 4, 4); std::memcpy(&bits, c.data() + 14, 2);
        } else if (!std::strncmp(id, "data", 4)) {
            if (fmt_tag != 1 || bits != 16 || channels < 1) return false;
            std::vector<int16_t> s(n / 2);
            f.read(reinterpret_cast<char*>(s.data()), n);
            out.resize(s.size() / channels);
            for (size_t i = 0; i < out.size(); ++i) out[i] = s[i * channels] / 32768.0f;
            rate = static_cast<int>(sr);
            return true;
        } else {
            f.seekg(n + (n & 1), std::ios::cur);
        }
    }
    return false;
}

void denormalsOff() {
#if defined(__SSE__) || defined(__x86_64__)
    _mm_setcsr(_mm_getcsr() | 0x8040);  // flush-to-zero and denormals-are-zero
#endif
}

// ---------- drivers ----------
#ifdef HAVE_JACK
struct Jack {
    jack_client_t* client = nullptr;
    jack_port_t *in = nullptr, *out_l = nullptr, *out_r = nullptr;
    Host* host = nullptr;
    bool denormals = false;
};

int jackProcess(jack_nframes_t nf, void* arg) {
    auto* j = static_cast<Jack*>(arg);
    if (!j->denormals) { denormalsOff(); j->denormals = true; }
    auto* in = static_cast<float*>(jack_port_get_buffer(j->in, nf));
    auto* l = static_cast<float*>(jack_port_get_buffer(j->out_l, nf));
    auto* r = static_cast<float*>(jack_port_get_buffer(j->out_r, nf));
    j->host->process(in, l, r, static_cast<int>(nf));
    return 0;
}
int jackXrun(void* arg) { static_cast<Jack*>(arg)->host->xruns.fetch_add(1); return 0; }
int jackBuffer(jack_nframes_t nf, void* arg) { static_cast<Jack*>(arg)->host->buffer.store(nf); return 0; }
void jackShutdown(void* arg) {
    std::cerr << "superfly_audio: the audio server shut us down (device gone?)\n";
    static_cast<Jack*>(arg)->host->quit.store(true);
}

// Physical ports whose names contain `device`: capture ports are JACK outputs, playback ports inputs.
std::vector<std::string> findPorts(jack_client_t* c, const std::string& device, unsigned long flags) {
    std::vector<std::string> out;
    const char** ps = jack_get_ports(c, nullptr, JACK_DEFAULT_AUDIO_TYPE, flags | JackPortIsPhysical);
    for (int i = 0; ps && ps[i]; ++i) {
        const std::string name = ps[i];
        if (name.find(device) != std::string::npos && name.find("monitor") == std::string::npos) out.push_back(name);
    }
    if (ps) jack_free(ps);
    return out;
}
#endif

}  // namespace

// ---------- render: the filter chain alone, offline ----------
int cmdRender(int argc, char** argv) {
    if (argc < 5) { std::cerr << "usage: superfly_audio render IN.wav PARAMS.f32 OUT.f32 [--mix M] [--no-limiter]\n"; return 1; }
    std::vector<float> x;
    int rate = 0;
    if (!loadWavMono(argv[2], x, rate)) { std::cerr << "cannot read " << argv[2] << "\n"; return 1; }
    std::ifstream pf(argv[3], std::ios::binary);
    const std::vector<char> bytes((std::istreambuf_iterator<char>(pf)), std::istreambuf_iterator<char>());
    std::vector<float> pv(bytes.size() / sizeof(float));
    std::memcpy(pv.data(), bytes.data(), pv.size() * sizeof(float));
    double mix = 1.0;
    bool limiter = true;
    for (int i = 5; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--mix" && i + 1 < argc) mix = std::atof(argv[++i]);
        else if (a == "--no-limiter") limiter = false;
    }
    fly::FilterChain chain;
    chain.init(rate);
    chain.setMix(mix);
    chain.setLimiter(limiter);
    const size_t frames = pv.size() / 5;
    std::vector<float> y(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        if (i % kHop == 0 && i / kHop >= 1 && i / kHop - 1 < frames) {  // frame k's values at sample 256 (k+1)
            const float* p = &pv[(i / kHop - 1) * 5];
            fly::FilterParams fp;
            fp.cutoff = p[1]; fp.q = p[2]; fp.gain_db = p[3]; fp.vca = p[4];
            chain.setTarget(static_cast<int>(p[0]), fp);
        }
        y[i] = static_cast<float>(chain.process(x[i]));
    }
    std::ofstream of(argv[4], std::ios::binary);
    of.write(reinterpret_cast<const char*>(y.data()), y.size() * sizeof(float));
    return 0;
}

int main(int argc, char** argv) {
    if (argc >= 2 && std::string(argv[1]) == "render") return cmdRender(argc, argv);

    std::string driver = "jack", device = "Scarlett", null_input, null_output;
    int input = 1, block = 128;
    bool fast = false, list = false;
    std::vector<char*> carfac_args = {argv[0], argv[0]};  // parseCarfacOptions starts at index 2
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { std::cerr << "Missing value for " << a << "\n"; std::exit(1); }
            return argv[++i];
        };
        if (a == "--driver") driver = next();
        else if (a == "--device") device = next();
        else if (a == "--input") input = std::stoi(next());
        else if (a == "--block") block = std::stoi(next());
        else if (a == "--null-input") null_input = next();
        else if (a == "--null-output") null_output = next();
        else if (a == "--fast") fast = true;
        else if (a == "--list") list = true;
        else carfac_args.push_back(argv[i]);  // CARFAC options (--agc on|off, ...)
    }

    Host h;
#ifdef HAVE_JACK
    Jack jack;
    jack.host = &h;
    if (driver == "jack") {
        jack.client = jack_client_open("superfly-audio", JackNoStartServer, nullptr);
        if (!jack.client) { std::cerr << "superfly_audio: no JACK server (run under pw-jack)\n"; return 2; }
        if (list) {
            for (auto flags : {JackPortIsOutput, JackPortIsInput})
                for (const auto& p : findPorts(jack.client, "", flags))
                    std::cout << (flags == JackPortIsOutput ? "capture  " : "playback ") << p << "\n";
            jack_client_close(jack.client);
            return 0;
        }
        h.rate = static_cast<int>(jack_get_sample_rate(jack.client));
        h.buffer.store(static_cast<int>(jack_get_buffer_size(jack.client)));
    }
#else
    if (driver == "jack") { std::cerr << "superfly_audio: built without JACK\n"; return 2; }
#endif
    if (h.rate != 44100) {
        std::cerr << "superfly_audio: the audio graph runs at " << h.rate << " Hz; superfly needs 44100 "
                  << "(pw-metadata -n settings 0 clock.force-rate 44100)\n";
        return 3;
    }

    CarfacFrontend::Params params;
    if (!parseCarfacOptions(static_cast<int>(carfac_args.size()), carfac_args.data(), 2, h.rate, params)) return 1;
    CarfacFrontend carfac;
    if (!carfac.init(params)) { std::cerr << "Failed to initialize CARFAC\n"; return 1; }
    h.agc.store(params.enable_agc);
    h.chain.init(h.rate);

    std::string capture, play_l, play_r;
#ifdef HAVE_JACK
    if (driver == "jack") {
        const auto caps = findPorts(jack.client, device, JackPortIsOutput);
        const auto plays = findPorts(jack.client, device, JackPortIsInput);
        if (static_cast<int>(caps.size()) < input || plays.empty()) {
            std::cerr << "superfly_audio: no ports for device '" << device << "' (input " << input << "); see --list\n";
            return 2;
        }
        capture = caps[input - 1]; play_l = plays[0]; play_r = plays.size() > 1 ? plays[1] : plays[0];
    }
#endif
    std::ostringstream extra;
    extra << "  \"host\": {\"driver\": \"" << driver << "\", \"capture\": \"" << capture << "\", \"playback\": [\""
          << play_l << "\", \"" << play_r << "\"], \"buffer\": " << (driver == "jack" ? h.buffer.load() : block)
          << ", \"protocol\": 1},\n";
    std::ostringstream meta;
    writeCarfacMeta(meta, driver == "jack" ? capture : "null", 0, h.rate, carfac, params, extra.str());
    writeText(kHello, meta.str());

    h.emitSource();  // the starting state: the interface's input
    std::thread analysis(analysisLoop, std::ref(h), std::ref(carfac));
    std::thread status(statusLoop, std::ref(h));
    std::thread control(controlLoop, std::ref(h));
    control.detach();  // blocks on stdin; ends the process with QUIT or EOF

#ifdef HAVE_JACK
    if (driver == "jack") {
        jack_set_process_callback(jack.client, jackProcess, &jack);
        jack_set_xrun_callback(jack.client, jackXrun, &jack);
        jack_set_buffer_size_callback(jack.client, jackBuffer, &jack);
        jack_on_shutdown(jack.client, jackShutdown, &jack);
        jack.in = jack_port_register(jack.client, "in", JACK_DEFAULT_AUDIO_TYPE, JackPortIsInput, 0);
        jack.out_l = jack_port_register(jack.client, "out_l", JACK_DEFAULT_AUDIO_TYPE, JackPortIsOutput, 0);
        jack.out_r = jack_port_register(jack.client, "out_r", JACK_DEFAULT_AUDIO_TYPE, JackPortIsOutput, 0);
        if (jack_activate(jack.client) || jack_connect(jack.client, capture.c_str(), jack_port_name(jack.in)) ||
            jack_connect(jack.client, jack_port_name(jack.out_l), play_l.c_str()) ||
            jack_connect(jack.client, jack_port_name(jack.out_r), play_r.c_str())) {
            std::cerr << "superfly_audio: could not connect to " << capture << " / " << play_l << "\n";
            h.quit.store(true);
        }
        while (!h.quit.load()) std::this_thread::sleep_for(std::chrono::milliseconds(20));
        jack_deactivate(jack.client);
        jack_client_close(jack.client);
    }
#endif
    if (driver == "null") {
        denormalsOff();
        std::vector<float> in_audio;
        int rate = 0;
        if (!null_input.empty() && (!loadWavMono(null_input, in_audio, rate) || rate != h.rate)) {
            std::cerr << "superfly_audio: cannot use " << null_input << " as input\n";
            return 1;
        }
        std::ofstream out_f;
        if (!null_output.empty()) out_f.open(null_output, std::ios::binary);
        std::vector<float> in(block), out(block);
        size_t ip = 0;
        const auto period = std::chrono::duration<double>(static_cast<double>(block) / h.rate);
        auto next = std::chrono::steady_clock::now();
        while (!h.quit.load()) {
            for (int i = 0; i < block; ++i) {
                in[i] = in_audio.empty() ? 0.0f : in_audio[ip];
                if (!in_audio.empty()) ip = (ip + 1) % in_audio.size();
            }
            h.process(in.data(), out.data(), out.data(), block);
            if (out_f) out_f.write(reinterpret_cast<const char*>(out.data()), block * sizeof(float));
            while (fast && h.analysis.size() > h.analysis.capacity() / 2 && !h.quit.load())
                std::this_thread::sleep_for(std::chrono::microseconds(200));  // let analysis keep up
            if (!fast) {
                next += std::chrono::duration_cast<std::chrono::steady_clock::duration>(period);
                std::this_thread::sleep_until(next);
            }
        }
    }
    h.quit.store(true);
    analysis.join();
    status.join();
    writeMsg(kEnd, nullptr, 0);
    return 0;
}
