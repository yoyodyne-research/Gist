// Minimal inference stub for a tiny recurrent controller on Bela.
// Note: This currently implements a 1-layer GRU fallback and FiLM on outputs.
// For CfC/LTC, wire in the corresponding closed-form cell equations and reuse the same interface.

#pragma once
#include <vector>
#include <string>
#include <cstdint>

struct NcpWeights {
    // Encoder: in_dim -> enc_dim
    std::vector<float> enc_W; // [enc_dim, in_dim]
    std::vector<float> enc_b; // [enc_dim]

    // GRU: enc_dim -> hid (1 layer)
    std::vector<float> gru_Wir, gru_Wiz, gru_Win; // [hid, enc_dim]
    std::vector<float> gru_Whr, gru_Whz, gru_Whn; // [hid, hid]
    std::vector<float> gru_br, gru_bz, gru_bn;     // [hid]

    // Head: hid -> out_dim
    std::vector<float> head_W; // [out_dim, hid]
    std::vector<float> head_b; // [out_dim]

    // FiLM on outputs: (macros+cond) -> out_dim
    std::vector<float> filmG_W; // [out_dim, cond_dim]
    std::vector<float> filmG_b; // [out_dim]
    std::vector<float> filmB_W; // [out_dim, cond_dim]
    std::vector<float> filmB_b; // [out_dim]

    int in_dim=0, enc_dim=0, hid=0, out_dim=0, cond_dim=0;
};

struct NcpState {
    std::vector<float> h; // [hid]
};

struct NcpRuntimeCfg {
    float out_min = 0.0f;
    float out_max = 1.0f;
    float slew_max = 0.1f; // max delta per tick
    float smooth_alpha = 0.0f; // 0..1 one-pole smoothing on outputs
};

bool load_weights_npz(const std::string& path, NcpWeights& W);

void ncp_init_state(const NcpWeights& W, NcpState& S);

// Step once: x[in_dim], cond[cond_dim] -> y[out_dim]
void ncp_step(const NcpWeights& W, NcpState& S, const float* x, const float* cond,
              float* y, const NcpRuntimeCfg* cfg=nullptr);

