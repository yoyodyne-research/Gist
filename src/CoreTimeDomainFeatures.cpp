//=======================================================================
/** @file CoreTimeDomainFeatures.cpp
 *  @brief Implementations of common time domain audio features
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

#include <stdlib.h>  // for size_t
#include "CoreTimeDomainFeatures.h"
#include "NeonOps.h"

//===========================================================
template <class T>
CoreTimeDomainFeatures<T>::CoreTimeDomainFeatures()
{
}

//===========================================================
template <class T>
T CoreTimeDomainFeatures<T>::rootMeanSquare (const std::vector<T>& buffer)
{
    // create variable to hold the sum
    T sum = 0;

    // sum the squared samples
    for (size_t i = 0; i < buffer.size(); i++)
    {
        sum += buffer[i] * buffer[i];
    }

    // return the square root of the mean of squared samples
    return sqrt (sum / ((T)buffer.size()));
}

// Float specialization using NEON
template <>
float CoreTimeDomainFeatures<float>::rootMeanSquare (const std::vector<float>& buffer)
{
    float sum = gist_neon::sum_squares_f32(buffer.data(), buffer.size());
    return std::sqrt(sum / (float)buffer.size());
}

//===========================================================
template <class T>
T CoreTimeDomainFeatures<T>::peakEnergy (const std::vector<T>& buffer)
{
    // create variable with very small value to hold the peak value
    T peak = 0.0;

    // for each audio sample
    for (size_t i = 0; i < buffer.size(); i++)
    {
        // store the absolute value of the sample
        T absSample = fabs (buffer[i]);

        // if the absolute value is larger than the peak
        if (absSample > peak)
        {
            // the peak takes on the sample value
            peak = absSample;
        }
    }

    // return the peak value
    return peak;
}

// Float specialization using NEON
template <>
float CoreTimeDomainFeatures<float>::peakEnergy (const std::vector<float>& buffer)
{
    return gist_neon::max_abs_f32(buffer.data(), buffer.size());
}

//===========================================================
template <class T>
T CoreTimeDomainFeatures<T>::zeroCrossingRate (const std::vector<T>& buffer)
{
    // create a variable to hold the zero crossing rate
    T zcr = 0;

    // for each audio sample, starting from the second one
    for (size_t i = 1; i < buffer.size(); i++)
    {
        // initialise two booleans indicating whether or not
        // the current and previous sample are positive
        bool current = (buffer[i] > 0);
        bool previous = (buffer[i - 1] > 0);

        // if the sign is different
        if (current != previous)
        {
            // add one to the zero crossing rate
            zcr = zcr + 1.0;
        }
    }

    // return the zero crossing rate
    return zcr;
}

//===========================================================
template class CoreTimeDomainFeatures<float>;
template class CoreTimeDomainFeatures<double>;
