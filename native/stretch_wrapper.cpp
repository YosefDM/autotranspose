// A flat C interface over Signalsmith Stretch, so Python can drive it by ctypes.
//
// Signalsmith Stretch (MIT, github.com/Signalsmith-Audio/signalsmith-stretch) is
// a production-grade real-time pitch shifter. The hand-written numpy phase
// vocoder in shifter.py is honest but audibly worse, and parameter tuning turned
// out to be exhausted -- bigger windows measured no better. This is the
// algorithmic step up.
//
// The interface is deliberately minimal and allocation-free on the audio path:
// Python hands over planar float32 channel pointers and gets the same count of
// frames back, which is exactly what the engine's block loop already does.
//
// Build with native/build.bat (MSVC). If the DLL is missing, Python falls back
// to the numpy vocoder, so the project still works without a compiler.

#include <cstring>
#include <new>
#include <vector>

#include "signalsmith-stretch.h"

namespace {

struct Handle {
    signalsmith::stretch::SignalsmithStretch<float> stretch;
    int channels = 0;
    // Planar pointer arrays, reused every block so process() never allocates.
    std::vector<float *> inPtrs;
    std::vector<float *> outPtrs;
};

}  // namespace

extern "C" {

__declspec(dllexport) int ss_abi_version() { return 1; }

// cheaper!=0 selects the lighter preset (less CPU, slightly lower quality).
__declspec(dllexport) Handle *ss_create(int channels, float sampleRate, int cheaper) {
    if (channels <= 0 || sampleRate <= 0.0f) return nullptr;
    Handle *h = new (std::nothrow) Handle();
    if (!h) return nullptr;
    h->channels = channels;
    if (cheaper) {
        h->stretch.presetCheaper(channels, sampleRate);
    } else {
        h->stretch.presetDefault(channels, sampleRate);
    }
    h->inPtrs.resize(static_cast<size_t>(channels), nullptr);
    h->outPtrs.resize(static_cast<size_t>(channels), nullptr);
    return h;
}

__declspec(dllexport) void ss_destroy(Handle *h) { delete h; }

__declspec(dllexport) void ss_set_semitones(Handle *h, float semitones, float tonalityLimit) {
    if (h) h->stretch.setTransposeSemitones(semitones, tonalityLimit);
}

__declspec(dllexport) void ss_reset(Handle *h) {
    if (h) h->stretch.reset();
}

__declspec(dllexport) int ss_input_latency(Handle *h) {
    return h ? h->stretch.inputLatency() : 0;
}

__declspec(dllexport) int ss_output_latency(Handle *h) {
    return h ? h->stretch.outputLatency() : 0;
}

// `input` and `output` are planar: channels * frames floats each, channel-major.
// Returns 0 on success.
__declspec(dllexport) int ss_process(Handle *h, const float *input, float *output, int frames) {
    if (!h || !input || !output || frames <= 0) return -1;
    const int ch = h->channels;
    for (int c = 0; c < ch; ++c) {
        // The engine passes contiguous channel-major blocks; cast away const
        // because the library's Inputs concept takes non-const pointers but
        // only reads from them.
        h->inPtrs[static_cast<size_t>(c)] =
            const_cast<float *>(input) + static_cast<size_t>(c) * frames;
        h->outPtrs[static_cast<size_t>(c)] = output + static_cast<size_t>(c) * frames;
    }
    h->stretch.process(h->inPtrs.data(), frames, h->outPtrs.data(), frames);
    return 0;
}

}  // extern "C"
