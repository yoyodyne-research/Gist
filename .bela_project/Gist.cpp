//=======================================================================
/** @file Gist.cpp
 *  @brief Implementation for all relevant parts of the 'Gist' audio analysis library
 *  @author Adam Stark
 *  @copyright Copyright (C) 2013  Adam Stark
 *
 * This file is part of the 'Gist' audio analysis library
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
//=======================================================================

#include "Gist.h"
#include <type_traits>
#include <cmath>
#include "NeonOps.h"
#include <assert.h>

//=======================================================================
template <class T>
Gist<T>::Gist (int audioFrameSize, int fs, WindowType windowType_)
 :  windowType (windowType_),
    fftConfigured (false),
    onsetDetectionFunction (audioFrameSize),
    yin (fs),
    mfcc (audioFrameSize, fs)
#ifdef USE_BITSTREAM_PITCH
    , bitstreamPitch(audioFrameSize, fs)
#endif
{
    samplingFrequency = fs;
    setAudioFrameSize (audioFrameSize);
}

//=======================================================================
template <class T>
Gist<T>::~Gist()
{
    if (fftConfigured)
    {
        freeFFT();
    }
}

//=======================================================================
template <class T>
void Gist<T>::setAudioFrameSize (int audioFrameSize)
{
    frameSize = audioFrameSize;
    
    audioFrame.resize (frameSize);
    
    windowFunction = WindowFunctions<T>::createWindow (audioFrameSize, windowType);
        
    fftReal.resize (frameSize);
    fftImag.resize (frameSize);
    magnitudeSpectrum.resize (frameSize / 2);
    
    configureFFT();
    
    onsetDetectionFunction.setFrameSize (frameSize);
    mfcc.setFrameSize (frameSize);
#ifdef USE_BITSTREAM_PITCH
    bitstreamPitch.setFrameSize(frameSize);
#endif
}

//=======================================================================
template <class T>
void Gist<T>::setSamplingFrequency (int fs)
{
    samplingFrequency = fs;
    yin.setSamplingFrequency (samplingFrequency);
    mfcc.setSamplingFrequency (samplingFrequency);
#ifdef USE_BITSTREAM_PITCH
    bitstreamPitch.setSampleRate(samplingFrequency);
#endif
}

#ifdef USE_BITSTREAM_PITCH
//=======================================================================
template <class T>
void Gist<T>::setBitstreamPitchRange(float lowestHz, float highestHz)
{
    bitstreamPitch.setFrequencyRange(lowestHz, highestHz);
}

//=======================================================================
template <class T>
void Gist<T>::setBitstreamPitchHysteresis(float hysteresisLinear)
{
    bitstreamPitch.setHysteresisLinear(hysteresisLinear);
}
#endif

//=======================================================================
template <class T>
int Gist<T>::getAudioFrameSize()
{
    return frameSize;
}

//=======================================================================
template <class T>
int Gist<T>::getSamplingFrequency()
{
    return samplingFrequency;
}

//=======================================================================
template <class T>
void Gist<T>::processAudioFrame (const std::vector<T>& a)
{
    // you are passing an audio frame of a different size to the
    // audio frame size setup in Gist
    assert (a.size() == audioFrame.size());
    
    std::copy (a.begin(), a.end(), audioFrame.begin());
    performFFT();
}

//=======================================================================
template <class T>
void Gist<T>::processAudioFrame (const T* frame, int numSamples)
{
    // you are passing an audio frame of a different size to the
    // audio frame size setup in Gist
    assert (static_cast<size_t> (numSamples) == audioFrame.size());
    
    for (size_t i = 0; i < audioFrame.size(); i++)
        audioFrame[i] = frame[i];
    
    performFFT();
}

//=======================================================================
template <class T>
const std::vector<T>& Gist<T>::getMagnitudeSpectrum()
{
    return magnitudeSpectrum;
}

//=======================================================================
template <class T>
T Gist<T>::rootMeanSquare()
{
    return coreTimeDomainFeatures.rootMeanSquare (audioFrame);
}

//=======================================================================
template <class T>
T Gist<T>::peakEnergy()
{
    return coreTimeDomainFeatures.peakEnergy (audioFrame);
}

//=======================================================================
template <class T>
T Gist<T>::zeroCrossingRate()
{
    return coreTimeDomainFeatures.zeroCrossingRate (audioFrame);
}

//=======================================================================
template <class T>
T Gist<T>::spectralCentroid()
{
    return coreFrequencyDomainFeatures.spectralCentroid (magnitudeSpectrum);
}

//=======================================================================
template <class T>
T Gist<T>::spectralCrest()
{
    return coreFrequencyDomainFeatures.spectralCrest (magnitudeSpectrum);
}

//=======================================================================
template <class T>
T Gist<T>::spectralFlatness()
{
    return coreFrequencyDomainFeatures.spectralFlatness (magnitudeSpectrum);
}

//=======================================================================
template <class T>
T Gist<T>::spectralRolloff()
{
    return coreFrequencyDomainFeatures.spectralRolloff (magnitudeSpectrum);
}

//=======================================================================
template <class T>
T Gist<T>::spectralKurtosis()
{
    return coreFrequencyDomainFeatures.spectralKurtosis (magnitudeSpectrum);
}

//=======================================================================
template <class T>
T Gist<T>::energyDifference()
{
    return onsetDetectionFunction.energyDifference (audioFrame);
}

//=======================================================================
template <class T>
T Gist<T>::spectralDifference()
{
    return onsetDetectionFunction.spectralDifference (magnitudeSpectrum);
}

//=======================================================================
template <class T>
T Gist<T>::spectralDifferenceHWR()
{
    return onsetDetectionFunction.spectralDifferenceHWR (magnitudeSpectrum);
}

//=======================================================================
template <class T>
T Gist<T>::complexSpectralDifference()
{
    return onsetDetectionFunction.complexSpectralDifference (fftReal, fftImag);
}

//=======================================================================
template <class T>
T Gist<T>::highFrequencyContent()
{
    return onsetDetectionFunction.highFrequencyContent (magnitudeSpectrum);
}

//=======================================================================
template <class T>
T Gist<T>::pitch()
{
    return yin.pitchYin (audioFrame);
}

//=======================================================================
template <class T>
T Gist<T>::pitchFast()
{
#ifdef USE_BITSTREAM_PITCH
    // Fast path supports float frames; otherwise fall back
    if (std::is_same<T, float>::value)
    {
        const float* fptr = reinterpret_cast<const float*>(audioFrame.data());
        const float hz = bitstreamPitch.processFrame(fptr, frameSize);
        return static_cast<T>(hz);
    }
    else
    {
        return pitch();
    }
#else
    return pitch();
#endif
}

//=======================================================================
template <class T>
const std::vector<T>& Gist<T>::getMelFrequencySpectrum()
{
    mfcc.calculateMelFrequencySpectrum (magnitudeSpectrum);
    return mfcc.melSpectrum;
}

//=======================================================================
template <class T>
const std::vector<T>& Gist<T>::getMelFrequencyCepstralCoefficients()
{
    mfcc.calculateMelFrequencyCepstralCoefficients (magnitudeSpectrum);
    return mfcc.MFCCs;
}

//=======================================================================
template <class T>
void Gist<T>::configureFFT()
{
    if (fftConfigured)
    {
        freeFFT();
    }
    
#ifdef USE_FFTW
    // ------------------------------------------------------
    // initialise the fft time and frequency domain audio frame arrays
    fftIn = (fftw_complex*)fftw_malloc (sizeof (fftw_complex) * frameSize);  // complex array to hold fft data
    fftOut = (fftw_complex*)fftw_malloc (sizeof (fftw_complex) * frameSize); // complex array to hold fft data
    
    // FFT plan initialisation
    p = fftw_plan_dft_1d (frameSize, fftIn, fftOut, FFTW_FORWARD, FFTW_ESTIMATE);
#endif /* END USE_FFTW */
    
#ifdef USE_KISS_FFT
    // ------------------------------------------------------
    // initialise the fft time and frequency domain audio frame arrays
    fftIn = new kiss_fft_cpx[frameSize];
    fftOut = new kiss_fft_cpx[frameSize];
    cfg = kiss_fft_alloc (frameSize, 0, 0, 0);
#endif /* END USE_KISS_FFT */
    
#ifdef USE_ACCELERATE_FFT
    accelerateFFT.setAudioFrameSize (frameSize);
#endif

#ifdef USE_PFFFT
    static_assert(std::is_same<T,float>::value, "PFFFT backend supports float only");
    pffftSetup = pffft_new_setup(frameSize, PFFFT_REAL);
    // aligned buffers
    pffftInput  = (float*)pffft_aligned_malloc(sizeof(float) * frameSize);
    pffftOutput = (float*)pffft_aligned_malloc(sizeof(float) * frameSize);
    pffftWork   = (float*)pffft_aligned_malloc(sizeof(float) * frameSize);
#endif

    fftConfigured = true;
}

//=======================================================================
template <class T>
void Gist<T>::freeFFT()
{
#ifdef USE_FFTW
    // destroy fft plan
    fftw_destroy_plan (p);
    
    fftw_free (fftIn);
    fftw_free (fftOut);
#endif
    
#ifdef USE_KISS_FFT
    // free the Kiss FFT configuration
    free (cfg);
    
    delete[] fftIn;
    delete[] fftOut;
#endif

#ifdef USE_PFFFT
    if (pffftSetup)
        pffft_destroy_setup(pffftSetup);
    pffftSetup = nullptr;
    if (pffftInput)  pffft_aligned_free(pffftInput);
    if (pffftOutput) pffft_aligned_free(pffftOutput);
    if (pffftWork)   pffft_aligned_free(pffftWork);
    pffftInput = pffftOutput = pffftWork = nullptr;
#endif
}

//=======================================================================
template <class T>
void Gist<T>::performFFT()
{
#ifdef USE_FFTW
    // copy samples from audio frame
    for (int i = 0; i < frameSize; i++)
    {
        fftIn[i][0] = (double)(audioFrame[i] * windowFunction[i]);
        fftIn[i][1] = (double)0.0;
    }
    
    // perform the FFT
    fftw_execute (p);
    
    // store real and imaginary parts of FFT
    for (int i = 0; i < frameSize; i++)
    {
        fftReal[i] = (T)fftOut[i][0];
        fftImag[i] = (T)fftOut[i][1];
    }
#endif
    
#ifdef USE_KISS_FFT
    for (int i = 0; i < frameSize; i++)
    {
        // Kiss FFT scalar is float by default; avoid unnecessary double casts
        fftIn[i].r = (kiss_fft_scalar)(audioFrame[i] * windowFunction[i]);
        fftIn[i].i = (kiss_fft_scalar)0;
    }
    
    // execute kiss fft
    kiss_fft (cfg, fftIn, fftOut);
    
    // store real and imaginary parts of FFT
    for (int i = 0; i < frameSize; i++)
    {
        fftReal[i] = (T)fftOut[i].r;
        fftImag[i] = (T)fftOut[i].i;
    }
#endif
    
#ifdef USE_ACCELERATE_FFT
    
    T inputFrame[frameSize];
    T outputReal[frameSize];
    T outputImag[frameSize];
    
    for (int i = 0; i < frameSize; i++)
    {
        inputFrame[i] = audioFrame[i] * windowFunction[i];
    }
    
    accelerateFFT.performFFT (inputFrame, outputReal, outputImag);
    
    for (int i = 0; i < frameSize; i++)
    {
        fftReal[i] = outputReal[i];
        fftImag[i] = outputImag[i];
    }
    
#endif

#ifdef USE_PFFFT
    // window and copy directly into aligned input
    for (int i = 0; i < frameSize; i++)
    {
        pffftInput[i] = static_cast<float>(audioFrame[i] * windowFunction[i]);
    }

    // ordered transform yields unpack-friendly layout
    pffft_transform_ordered(pffftSetup, pffftInput, pffftOutput, pffftWork, PFFFT_FORWARD);

    // unpack to fftReal/fftImag (first half only)
    // out[0] = Re(0), out[1] = Re(N/2), for k=1..N/2-1: out[2*k]=Re(k), out[2*k+1]=Im(k)
    const int nBinsP = frameSize / 2;
    // DC and Nyquist
    fftReal[0] = static_cast<T>(pffftOutput[0]);
    fftImag[0] = static_cast<T>(0);
    fftReal[nBinsP] = static_cast<T>(pffftOutput[1]);
    fftImag[nBinsP] = static_cast<T>(0);

    // Lower half bins [1..nBins-1]
    for (int k = 1; k < nBinsP; ++k)
    {
        const float re = pffftOutput[2 * k];
        const float im = pffftOutput[2 * k + 1];
        fftReal[k] = static_cast<T>(re);
        fftImag[k] = static_cast<T>(im);
    }

    // Reconstruct upper half by conjugate symmetry: X[N-k] = conj(X[k])
    for (int k = 1; k < nBinsP; ++k)
    {
        const int hk = frameSize - k;
        fftReal[hk] = fftReal[k];
        fftImag[hk] = -fftImag[k];
    }
#endif

    // calculate the magnitude spectrum (first half)
    const int nBins = frameSize / 2;

#if defined(USE_ARM_NEON) && (defined(__ARM_NEON) || defined(__ARM_NEON__))
    if (std::is_same<T, float>::value)
    {
    #if defined(__ARM_NEON) || defined(__ARM_NEON__)
        gist_neon::magnitude_f32(reinterpret_cast<const float*>(fftReal.data()),
                                 reinterpret_cast<const float*>(fftImag.data()),
                                 reinterpret_cast<float*>(magnitudeSpectrum.data()),
                                 static_cast<std::size_t>(nBins));
    #endif
    }
    else
#endif
    {
        for (int i = 0; i < nBins; i++)
        {
            // prefer type-appropriate sqrt
            const T rr = fftReal[i] * fftReal[i];
            const T ii = fftImag[i] * fftImag[i];
            magnitudeSpectrum[i] = static_cast<T>(std::sqrt(rr + ii));
        }
    }
}

//===========================================================
#ifdef USE_PFFFT
template class Gist<float>;
#else
template class Gist<float>;
template class Gist<double>;
#endif
template <class T>
void Gist<T>::processWindowedFrame (const T* frame, int numSamples)
{
    assert (static_cast<size_t> (numSamples) == static_cast<size_t> (frameSize));

#ifdef USE_FFTW
    for (int i = 0; i < frameSize; i++)
    {
        fftIn[i][0] = (double)frame[i];
        fftIn[i][1] = (double)0.0;
    }
    fftw_execute (p);
    for (int i = 0; i < frameSize; i++)
    {
        fftReal[i] = (T)fftOut[i][0];
        fftImag[i] = (T)fftOut[i][1];
    }
#endif

#ifdef USE_KISS_FFT
    for (int i = 0; i < frameSize; i++)
    {
        fftIn[i].r = (kiss_fft_scalar)frame[i];
        fftIn[i].i = (kiss_fft_scalar)0;
    }
    kiss_fft (cfg, fftIn, fftOut);
    for (int i = 0; i < frameSize; i++)
    {
        fftReal[i] = (T)fftOut[i].r;
        fftImag[i] = (T)fftOut[i].i;
    }
#endif

#ifdef USE_ACCELERATE_FFT
    T outputReal[frameSize];
    T outputImag[frameSize];
    accelerateFFT.performFFT (const_cast<T*>(frame), outputReal, outputImag);
    for (int i = 0; i < frameSize; i++)
    {
        fftReal[i] = outputReal[i];
        fftImag[i] = outputImag[i];
    }
#endif

#ifdef USE_PFFFT
    for (int i = 0; i < frameSize; i++)
    {
        pffftInput[i] = (float)frame[i];
    }
    pffft_transform_ordered(pffftSetup, pffftInput, pffftOutput, pffftWork, PFFFT_FORWARD);
    const int nBinsPW = frameSize / 2;
    // DC and Nyquist
    fftReal[0] = (T)pffftOutput[0];
    fftImag[0] = (T)0;
    fftReal[nBinsPW] = (T)pffftOutput[1];
    fftImag[nBinsPW] = (T)0;
    // Lower half
    for (int k = 1; k < nBinsPW; ++k)
    {
        fftReal[k] = (T)pffftOutput[2 * k];
        fftImag[k] = (T)pffftOutput[2 * k + 1];
    }
    // Upper half via conjugate symmetry
    for (int k = 1; k < nBinsPW; ++k)
    {
        const int hk = frameSize - k;
        fftReal[hk] = fftReal[k];
        fftImag[hk] = -fftImag[k];
    }
#endif

    // magnitude spectrum
    const int nBins = frameSize / 2;
#if defined(USE_ARM_NEON) && (defined(__ARM_NEON) || defined(__ARM_NEON__))
    if (std::is_same<T, float>::value)
    {
    #if defined(__ARM_NEON) || defined(__ARM_NEON__)
        gist_neon::magnitude_f32(reinterpret_cast<const float*>(fftReal.data()),
                                 reinterpret_cast<const float*>(fftImag.data()),
                                 reinterpret_cast<float*>(magnitudeSpectrum.data()),
                                 static_cast<std::size_t>(nBins));
    #endif
    }
    else
#endif
    {
        for (int i = 0; i < nBins; i++)
        {
            const T rr = fftReal[i] * fftReal[i];
            const T ii = fftImag[i] * fftImag[i];
            magnitudeSpectrum[i] = (T)std::sqrt(rr + ii);
        }
    }
}
