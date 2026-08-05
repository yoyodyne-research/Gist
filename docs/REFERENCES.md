# References

This document collects key academic papers and technical references relevant to the **Gist Project**: a real-time C++ audio analysis library extended for embedded audio processing and neural control on Bela.

Each section includes a short note on *why the reference matters* to the current architecture and roadmap, with embedded URLs where available.

---

## 1. Real-Time Audio Feature Extraction

**Stark, A. M., Davies, M. E. P., & Plumbley, M. D.**
*Real-Time Audio Feature Extraction Using the Gist Descriptor*
[https://ieeexplore.ieee.org/document/4781132](https://ieeexplore.ieee.org/document/4781132)
→ Foundational reference for the original Gist feature set and real-time framing assumptions.

**Bello, J. P., Daudet, L., Abdallah, S., Duxbury, C., Davies, M., & Sandler, M.**
*A Tutorial on Onset Detection in Music Signals*
[https://ieeexplore.ieee.org/document/1541816](https://ieeexplore.ieee.org/document/1541816)
→ Canonical overview of energy-based, spectral-based, and phase-based onset detection methods.

**Peeters, G.**
*A Large Set of Audio Features for Sound Description*
[https://www.ircam.fr/media/uploads/documents/peeters_2004_cuidadofeatures.pdf](https://www.ircam.fr/media/uploads/documents/peeters_2004_cuidadofeatures.pdf)
→ Authoritative definitions and normalization strategies for spectral and temporal descriptors.

---

## 2. Pitch Detection (Accuracy vs. Latency)

**De Cheveigné, A., & Kawahara, H.**
*YIN, a Fundamental Frequency Estimator for Speech and Music*
[https://audition.ens.fr/adc/pdf/2002_JASA_YIN.pdf](https://audition.ens.fr/adc/pdf/2002_JASA_YIN.pdf)
→ High-accuracy pitch estimation; reference for YIN implementation and error handling.

**Tolonen, T., & Karjalainen, M.**
*A Computationally Efficient Multipitch Analysis Model*
[https://ieeexplore.ieee.org/document/747215](https://ieeexplore.ieee.org/document/747215)
→ Basis for bitstream autocorrelation approaches suitable for low-latency embedded systems.

---

## 3. Auditory & Cochlear Front Ends (CARFAC)

**Lyon, R. F.**
*A Computational Model of Filtering, Detection, and Compression in the Cochlea*
[https://authors.library.caltech.edu/records/6y9m3-8s822](https://authors.library.caltech.edu/records/6y9m3-8s822)
→ Original CARFAC model; justification for biologically inspired auditory preprocessing.

**Lyon, R. F., & Slaney, M.**
*Auditory Modeling Toolbox*
[https://engineering.purdue.edu/~malcolm/interval/1998-010/](https://engineering.purdue.edu/~malcolm/interval/1998-010/)
→ Practical reference connecting cochlear models to signal processing and machine listening.

---

## 4. Sample Rate Conversion & Anti-Aliasing

**Lyon, R. F.**
*The CARFAC v2 Cochlear Model in Matlab, NumPy, and JAX*
[https://arxiv.org/abs/2404.17490](https://arxiv.org/abs/2404.17490)
→ Updated CARFAC implementation; discusses sample rate selection (recommends 48 kHz for human hearing) and notes that CARFAC's nonlinearities generate distortion products that can alias at lower rates.

**Crochiere, R. E., & Rabiner, L. R.**
*Multirate Digital Signal Processing*
[https://ieeexplore.ieee.org/book/5265930](https://ieeexplore.ieee.org/book/5265930)
→ Canonical reference for decimation, interpolation, and polyphase filter structures; establishes efficiency of computing only retained samples.

**Smith, J. O.**
*Introduction to Digital Filters with Audio Applications*
[https://ccrma.stanford.edu/~jos/filters/](https://ccrma.stanford.edu/~jos/filters/)
→ Practical IIR/FIR filter design including Butterworth approximations; Section on downsampling covers anti-aliasing filter requirements.

**Vaidyanathan, P. P.**
*Multirate Systems and Filter Banks*
[https://ieeexplore.ieee.org/book/5263582](https://ieeexplore.ieee.org/book/5263582)
→ Theoretical foundation for polyphase decomposition; proves that M-fold decimation with polyphase FIR requires only 1/M of the multiply-accumulates.

**Wikipedia Contributors**
*Butterworth Filter*
[https://en.wikipedia.org/wiki/Butterworth_filter](https://en.wikipedia.org/wiki/Butterworth_filter)
→ Quick reference for maximally flat magnitude response; roll-off is 6n dB/octave for order n (first-order = 6 dB/oct, second-order = 12 dB/oct).

---

## 5. Embedded & Real-Time DSP Constraints

**McPherson, A., & Zappi, V.**
*An Environment for Submillisecond-Latency Audio and Sensor Processing*
[https://ieeexplore.ieee.org/document/6854959](https://ieeexplore.ieee.org/document/6854959)
→ Bela platform reference; supports design choices around hop size, buffering, and latency.

**Smith, J. O.**
*Physical Audio Signal Processing*
[https://ccrma.stanford.edu/~jos/pasp/](https://ccrma.stanford.edu/~jos/pasp/)
→ Core reference for real-time DSP stability, causality, and implementation tradeoffs.

---

## 6. Continuous-Time Neural Networks & Neural Controllers

**Hasani, R., Lechner, M., Amini, A., Rus, D., & Grosu, R.**
*Closed-form Continuous-time Neural Networks*
[https://arxiv.org/abs/2106.13898](https://arxiv.org/abs/2106.13898)
→ Primary reference for CfC formulation and closed-form forward inference.

**Hasani, R., Lechner, M., Rus, D., & Grosu, R.**
*Liquid Time-Constant Networks*
[https://arxiv.org/abs/2006.04439](https://arxiv.org/abs/2006.04439)
→ Explains robustness of continuous-time models under irregular sampling and real-time constraints.

**Nagumo, J., Arimoto, S., & Yoshizawa, S.**
*An Active Pulse Transmission Line Simulating Nerve Axon*
[https://ieeexplore.ieee.org/document/4038489](https://ieeexplore.ieee.org/document/4038489)
→ Early nonlinear neural dynamics; theoretical ancestry of continuous-time neurons.

---

## 7. Conditioning, Modulation & Control

**Perez, E., Strub, F., de Vries, H., Dumoulin, V., & Courville, A.**
*FiLM: Visual Reasoning with a General Conditioning Layer*
[https://arxiv.org/abs/1709.07871](https://arxiv.org/abs/1709.07871)
→ Canonical reference for feature-wise linear modulation used for metadata and macro conditioning.

**Ha, D., & Schmidhuber, J.**
*Recurrent World Models Facilitate Policy Evolution*
[https://arxiv.org/abs/1809.01999](https://arxiv.org/abs/1809.01999)
→ Neural controller framing for real-time, feedback-driven systems.

---

## 8. Online & Adaptive Learning

**Widrow, B., & Stearns, S. D.**
*Adaptive Signal Processing*
[https://ieeexplore.ieee.org/book/5266465](https://ieeexplore.ieee.org/book/5266465)
→ Classical grounding for stability, convergence, and online adaptation methods.

---

## Notes

* URLs favor open-access or canonical landing pages where possible.
* For long-term reproducibility, consider mirroring PDFs or storing DOIs alongside these links.
* Additional task-specific references (datasets, evaluation metrics) should live in adjacent docs as the project evolves.
