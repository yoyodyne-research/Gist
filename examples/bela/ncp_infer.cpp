#include "ncp_infer.h"
#include <cmath>
#include <cstring>

static inline float clampf(float v, float lo, float hi){ return v < lo ? lo : (v > hi ? hi : v); }
static inline float fast_tanh(float x){ return std::tanh(x); }
static inline float fast_sigmoid(float x){ return 1.0f / (1.0f + std::exp(-x)); }

static void matvec(const float* W, const float* x, float* y, int rows, int cols){
    for(int r=0;r<rows;++r){
        float s=0.f;
        const float* Wr = W + r*cols;
        for(int c=0;c<cols;++c) s += Wr[c]*x[c];
        y[r]=s;
    }
}

bool load_weights_npz(const std::string&, NcpWeights&){
    // Placeholder: integrate CNpy or a minimal NPZ reader if desired.
    // For now, load via your own mechanism or bake weights in JSON and parse externally.
    return false;
}

void ncp_init_state(const NcpWeights& W, NcpState& S){ S.h.assign(W.hid, 0.0f); }

void ncp_step(const NcpWeights& W, NcpState& S, const float* x, const float* cond,
              float* y, const NcpRuntimeCfg* cfg){
    // Encode
    std::vector<float> z(W.enc_dim);
    matvec(W.enc_W.data(), x, z.data(), W.enc_dim, W.in_dim);
    for(int i=0;i<W.enc_dim;++i) z[i] = std::tanh(z[i] + W.enc_b[i]);

    // GRU cell
    const int H=W.hid, E=W.enc_dim;
    std::vector<float> r(H), uz(H), ntil(H);
    // r gate
    for(int h=0;h<H;++h){
        float s = W.gru_br[h];
        // Wir * z
        for(int e=0;e<E;++e) s += W.gru_Wir[h*E+e]*z[e];
        // Whr * h
        for(int hh=0;hh<H;++hh) s += W.gru_Whr[h*H+hh]*S.h[hh];
        r[h] = fast_sigmoid(s);
    }
    // z gate
    for(int h=0;h<H;++h){
        float s = W.gru_bz[h];
        for(int e=0;e<E;++e) s += W.gru_Wiz[h*E+e]*z[e];
        for(int hh=0;hh<H;++hh) s += W.gru_Whz[h*H+hh]*S.h[hh];
        uz[h] = fast_sigmoid(s);
    }
    // n~
    for(int h=0;h<H;++h){
        float s = W.gru_bn[h];
        for(int e=0;e<E;++e) s += W.gru_Win[h*E+e]*z[e];
        for(int hh=0;hh<H;++hh) s += W.gru_Whn[h*H+hh]*(r[hh]*S.h[hh]);
        ntil[h] = fast_tanh(s);
    }
    // h = (1-z)*n~ + z*h
    for(int h=0;h<H;++h){ S.h[h] = (1.0f - uz[h]) * ntil[h] + uz[h]*S.h[h]; }

    // Head
    std::vector<float> out(W.out_dim);
    matvec(W.head_W.data(), S.h.data(), out.data(), W.out_dim, W.hid);
    for(int o=0;o<W.out_dim;++o) out[o] += W.head_b[o];

    // FiLM on outputs from conditioning vector
    std::vector<float> gamma(W.out_dim), beta(W.out_dim);
    // gamma = 1 + 0.5*tanh(Wg*cond + bg)
    for(int o=0;o<W.out_dim;++o){
        float s = W.filmG_b[o];
        for(int c=0;c<W.cond_dim;++c) s += W.filmG_W[o*W.cond_dim+c]*cond[c];
        gamma[o] = 1.0f + 0.5f * std::tanh(s);
    }
    // beta = 0.5*tanh(Wb*cond + bb)
    for(int o=0;o<W.out_dim;++o){
        float s = W.filmB_b[o];
        for(int c=0;c<W.cond_dim;++c) s += W.filmB_W[o*W.cond_dim+c]*cond[c];
        beta[o] = 0.5f * std::tanh(s);
    }
    for(int o=0;o<W.out_dim;++o) out[o] = gamma[o]*out[o] + beta[o];

    // Post-processing
    static std::vector<float> y_prev; if(y_prev.size()!= (size_t)W.out_dim) y_prev.assign(W.out_dim, 0.0f);
    for(int o=0;o<W.out_dim;++o){
        float v = out[o];
        if(cfg){
            // slew
            float d = v - y_prev[o];
            if(std::fabs(d) > cfg->slew_max) v = y_prev[o] + (d>0? cfg->slew_max : -cfg->slew_max);
            // smooth
            if(cfg->smooth_alpha > 0.0f) v = cfg->smooth_alpha * y_prev[o] + (1.0f - cfg->smooth_alpha) * v;
            // clamp
            v = clampf(v, cfg->out_min, cfg->out_max);
        }
        y[o] = v; y_prev[o] = v;
    }
}

