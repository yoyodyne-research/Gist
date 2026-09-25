// CARFAC command-line options and the feature description, shared by kitsune_test and superfly_audio.
#pragma once

#include <cstdlib>
#include <iostream>
#include <string>

#include "carfac_frontend.h"

// CARFAC options shared by `features` and `stream`: high fidelity by default, --bela starts from the Bela
// budget settings, later options override either. Returns false (after printing why) on a bad option.
inline bool parseCarfacOptions(int argc, char** argv, int first, int sample_rate, CarfacFrontend::Params& params,
                               std::string* format = nullptr) {
    params = CarfacFrontend::Params::highFidelity(sample_rate);
    params.publish_hop = 256;
    for (int i = first; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { std::cerr << "Missing value for " << a << "\n"; std::exit(1); }
            return argv[++i];
        };
        if (a == "--bela") {
            CarfacFrontend::Params b;
            b.sample_rate = sample_rate;
            b.carfac_rate = 22050;
            b.publish_hop = params.publish_hop;
            params = b;
        }
        else if (a == "--hop") params.publish_hop = std::stoi(next());
        else if (a == "--carfac-rate") params.carfac_rate = std::stoi(next());
        else if (a == "--erb-per-step") params.erb_per_step = std::stof(next());
        else if (a == "--ihc") params.full_ihc = next() == "full";
        else if (a == "--env") params.env_from_ihc = next() == "ihc";
        else if (a == "--agc") params.enable_agc = next() == "on";
        else if (format && a == "--format") *format = next();
        else if (format && a == "--rate") { /* consumed by the caller */ next(); }
        else { std::cerr << "Unknown option " << a << "\n"; return false; }
    }
    return true;
}

// The feature description written as <prefix>.json by `features` and sent as HELLO by `stream`.
// `extra`: more top-level JSON fields, e.g. "\"host\": {...},\n" (each line ending in a comma).
inline void writeCarfacMeta(std::ostream& meta, const std::string& source, long frames, int sample_rate,
                            const CarfacFrontend& carfac, const CarfacFrontend::Params& params,
                            const std::string& extra = "") {
    const int bands = carfac.numBands();
    meta << "{\n  \"source\": \"" << source << "\",\n" << extra
         << "  \"layout\": [\"frames\", [\"env_fast\", \"env_slow\", \"delta\"], \"bands\"],\n"
         << "  \"frames\": " << frames << ",\n  \"bands\": " << bands << ",\n"
         << "  \"sample_rate\": " << sample_rate << ",\n  \"carfac_rate\": " << carfac.carfacRate() << ",\n"
         << "  \"frame_rate_hz\": " << static_cast<double>(sample_rate) / params.publish_hop << ",\n"
         << "  \"params\": {\"full_ihc\": " << (params.full_ihc ? "true" : "false")
         << ", \"env_from_ihc\": " << (params.env_from_ihc ? "true" : "false")
         << ", \"enable_agc\": " << (params.enable_agc ? "true" : "false")
         << ", \"erb_per_step\": " << params.erb_per_step
         << ", \"fast_attack\": " << params.fast_attack << ", \"fast_release\": " << params.fast_release
         << ", \"slow_attack\": " << params.slow_attack << ", \"slow_release\": " << params.slow_release << "},\n"
         << "  \"pole_hz\": [";
    const auto& poles = carfac.poleFrequencies();
    for (int b = 0; b < bands; ++b) meta << (b ? ", " : "") << poles(b);
    meta << "]\n}\n";
}

