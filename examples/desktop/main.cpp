// Kitsune Desktop Test Harness
//
// Tests the CARFAC frontend and latent layer pipeline without Bela hardware.
// Can process audio files (WAV) and compare output against Python reference.

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <cstring>
#include <cstdint>

#include "carfac_frontend.h"
#include "latent_layer.h"

// Simple WAV file reader (mono, 16-bit PCM only)
struct WavHeader {
    char riff[4];
    uint32_t file_size;
    char wave[4];
    char fmt[4];
    uint32_t fmt_size;
    uint16_t audio_format;
    uint16_t num_channels;
    uint32_t sample_rate;
    uint32_t byte_rate;
    uint16_t block_align;
    uint16_t bits_per_sample;
};

bool loadWav(const char* path, std::vector<float>& samples, int& sample_rate) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open WAV file: " << path << std::endl;
        return false;
    }

    WavHeader header;
    file.read(reinterpret_cast<char*>(&header), sizeof(header));

    if (std::strncmp(header.riff, "RIFF", 4) != 0 ||
        std::strncmp(header.wave, "WAVE", 4) != 0) {
        std::cerr << "Invalid WAV file format" << std::endl;
        return false;
    }

    // Skip to data chunk
    char chunk_id[4];
    uint32_t chunk_size;
    while (file.read(chunk_id, 4) && file.read(reinterpret_cast<char*>(&chunk_size), 4)) {
        if (std::strncmp(chunk_id, "data", 4) == 0) {
            break;
        }
        file.seekg(chunk_size, std::ios::cur);
    }

    sample_rate = header.sample_rate;
    int num_samples = chunk_size / (header.bits_per_sample / 8) / header.num_channels;
    samples.resize(num_samples);

    if (header.bits_per_sample == 16) {
        std::vector<int16_t> buffer(num_samples * header.num_channels);
        file.read(reinterpret_cast<char*>(buffer.data()),
                  num_samples * header.num_channels * sizeof(int16_t));

        for (int i = 0; i < num_samples; ++i) {
            // Take first channel if stereo
            samples[i] = buffer[i * header.num_channels] / 32768.0f;
        }
    } else {
        std::cerr << "Unsupported bit depth: " << header.bits_per_sample << std::endl;
        return false;
    }

    std::cout << "Loaded " << path << ": " << num_samples << " samples @ "
              << sample_rate << " Hz" << std::endl;
    return true;
}

// Simple NPZ-like binary format for reference comparison
struct ReferenceData {
    std::vector<std::vector<float>> X;  // Input features
    std::vector<std::vector<float>> Z;  // Expected PCA output
    std::vector<std::vector<float>> R;  // Expected GMM responsibilities

    bool load(const char* path) {
        // For simplicity, just load a CSV format:
        // First line: n_samples, input_dims, pca_dims, gmm_components
        // Then alternating lines: input, z, r
        std::ifstream file(path);
        if (!file.is_open()) return false;

        int n_samples, input_dims, pca_dims, gmm_components;
        file >> n_samples >> input_dims >> pca_dims >> gmm_components;

        X.resize(n_samples);
        Z.resize(n_samples);
        R.resize(n_samples);

        for (int i = 0; i < n_samples; ++i) {
            X[i].resize(input_dims);
            Z[i].resize(pca_dims);
            R[i].resize(gmm_components);

            for (int j = 0; j < input_dims; ++j) file >> X[i][j];
            for (int j = 0; j < pca_dims; ++j) file >> Z[i][j];
            for (int j = 0; j < gmm_components; ++j) file >> R[i][j];
        }

        return true;
    }
};

void printUsage(const char* prog) {
    std::cerr << "Usage: " << prog << " <command> [options]\n"
              << "\nCommands:\n"
              << "  process <audio.wav> <model.json> [output.csv]\n"
              << "      Process audio file through CARFAC + latent layer\n"
              << "      Outputs latent signals to stdout or file\n"
              << "\n"
              << "  validate <model.json> <reference.csv>\n"
              << "      Validate C++ latent layer against Python reference\n"
              << "      Reports max error for each output\n"
              << "\n"
              << "  info <model.json>\n"
              << "      Print model information\n";
}

int cmdProcess(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " process <audio.wav> <model.json> [output.csv]\n";
        return 1;
    }

    const char* audio_path = argv[2];
    const char* model_path = argv[3];
    const char* output_path = argc > 4 ? argv[4] : nullptr;

    // Load audio
    std::vector<float> audio;
    int sample_rate;
    if (!loadWav(audio_path, audio, sample_rate)) {
        return 1;
    }

    // Initialize CARFAC
    CarfacFrontend carfac;
    CarfacFrontend::Params params;
    params.sample_rate = sample_rate;
    params.carfac_rate = 0;  // Full rate for desktop testing
    params.enable_agc = true;
    params.publish_hop = 64;

    if (!carfac.init(params)) {
        std::cerr << "Failed to initialize CARFAC" << std::endl;
        return 1;
    }

    std::cout << "CARFAC initialized: " << carfac.numBands() << " bands" << std::endl;

    // Initialize latent layer
    LatentLayer latent;
    if (!latent.loadParams(model_path)) {
        std::cerr << "Failed to load latent model" << std::endl;
        return 1;
    }

    // Open output
    std::ostream* out = &std::cout;
    std::ofstream file_out;
    if (output_path) {
        file_out.open(output_path);
        out = &file_out;
    }

    // Process audio
    constexpr int kHopSize = 64;
    constexpr int kControlDecim = 3;  // Control rate = ~230 Hz at 44.1k

    int sample_count = 0;
    int hop_count = 0;
    int frame_count = 0;

    *out << "frame,";
    for (int i = 0; i < LatentOutput::kPcaDims; ++i) *out << "z" << i << ",";
    for (int i = 0; i < LatentOutput::kGmmComponents; ++i) *out << "r" << i << ",";
    *out << "mode_id,mode_strength,entropy,log_likelihood\n";

    for (size_t i = 0; i < audio.size(); ++i) {
        carfac.processSample(audio[i]);
        sample_count++;

        if (sample_count >= kHopSize) {
            sample_count = 0;
            hop_count++;

            if (hop_count >= kControlDecim) {
                hop_count = 0;

                // Get features
                LatentInput input;
                carfac.publishFeatures(input);

                // Process through latent layer
                LatentOutput output;
                latent.process(input, output);

                // Write output
                *out << frame_count << ",";
                for (int j = 0; j < LatentOutput::kPcaDims; ++j)
                    *out << output.z[j] << ",";
                for (int j = 0; j < LatentOutput::kGmmComponents; ++j)
                    *out << output.r[j] << ",";
                *out << output.mode_id << ","
                     << output.mode_strength << ","
                     << output.entropy << ","
                     << output.log_likelihood << "\n";

                frame_count++;
            }
        }
    }

    std::cerr << "Processed " << frame_count << " frames" << std::endl;
    return 0;
}

int cmdValidate(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " validate <model.json> <reference.csv>\n";
        return 1;
    }

    const char* model_path = argv[2];
    const char* ref_path = argv[3];

    // Load model
    LatentLayer latent;
    if (!latent.loadParams(model_path)) {
        std::cerr << "Failed to load latent model" << std::endl;
        return 1;
    }

    // Disable smoothing for validation (to match Python exactly)
    latent.setSmoothingMs(0.0f);

    // Load reference data
    ReferenceData ref;
    if (!ref.load(ref_path)) {
        std::cerr << "Failed to load reference data" << std::endl;
        return 1;
    }

    std::cout << "Validating against " << ref.X.size() << " reference samples\n";

    float max_z_error = 0.0f;
    float max_r_error = 0.0f;
    int max_z_idx = 0;
    int max_r_idx = 0;

    for (size_t i = 0; i < ref.X.size(); ++i) {
        // Construct input from reference features
        LatentInput input;
        for (int b = 0; b < LatentInput::kNumBands && b < 71; ++b) {
            input.env_fast[b] = ref.X[i][b];
            input.delta[b] = ref.X[i][71 + b];
        }
        input.total_energy = ref.X[i][142];
        input.spectral_centroid = ref.X[i][143];
        input.spectral_spread = ref.X[i][144];
        input.peak_band = ref.X[i][145];
        input.attack_breadth = ref.X[i][146];
        input.band_ratios[0] = ref.X[i][147];
        input.band_ratios[1] = ref.X[i][148];
        input.band_ratios[2] = ref.X[i][149];

        // Process
        LatentOutput output;
        latent.process(input, output);

        // Compare Z
        for (int j = 0; j < LatentOutput::kPcaDims; ++j) {
            float err = std::abs(output.z[j] - ref.Z[i][j]);
            if (err > max_z_error) {
                max_z_error = err;
                max_z_idx = i;
            }
        }

        // Compare R
        for (int j = 0; j < LatentOutput::kGmmComponents; ++j) {
            float err = std::abs(output.r[j] - ref.R[i][j]);
            if (err > max_r_error) {
                max_r_error = err;
                max_r_idx = i;
            }
        }
    }

    std::cout << "\nResults:\n";
    std::cout << "  Max Z error: " << max_z_error << " (sample " << max_z_idx << ")\n";
    std::cout << "  Max R error: " << max_r_error << " (sample " << max_r_idx << ")\n";

    // Pass/fail threshold
    constexpr float kTolerance = 1e-4f;
    bool pass = (max_z_error < kTolerance && max_r_error < kTolerance);

    std::cout << "\nStatus: " << (pass ? "PASS" : "FAIL") << "\n";
    return pass ? 0 : 1;
}

int cmdInfo(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " info <model.json>\n";
        return 1;
    }

    LatentLayer latent;
    if (!latent.loadParams(argv[2])) {
        return 1;
    }

    std::cout << "Model loaded successfully\n";
    std::cout << "  Input dims: " << LatentLayer::kInputDims << "\n";
    std::cout << "  PCA dims: " << LatentLayer::kPcaDims << "\n";
    std::cout << "  GMM components: " << LatentLayer::kGmmComponents << "\n";

    auto& ev = latent.getExplainedVariance();
    std::cout << "  Explained variance ratios:";
    for (int i = 0; i < LatentLayer::kPcaDims; ++i) {
        std::cout << " " << ev[i];
    }
    std::cout << "\n";

    return 0;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        printUsage(argv[0]);
        return 1;
    }

    std::string cmd = argv[1];

    if (cmd == "process") return cmdProcess(argc, argv);
    if (cmd == "validate") return cmdValidate(argc, argv);
    if (cmd == "info") return cmdInfo(argc, argv);

    std::cerr << "Unknown command: " << cmd << "\n";
    printUsage(argv[0]);
    return 1;
}
