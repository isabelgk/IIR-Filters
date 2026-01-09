#ifndef FREQUENCY_RESPONSE_H
#define FREQUENCY_RESPONSE_H

#include "filters.hpp"
#include <complex>
#include <vector>

namespace demo {

struct FrequencyResponseData
{
    std::vector<double> frequencies;
    std::vector<double> magnitudeDb;
    std::vector<double> phase;
};

FrequencyResponseData computeAnalogFrequencyResponse(
    const std::vector<std::complex<double>>& zeros,
    const std::vector<std::complex<double>>& poles,
    double gain,
    double omegaMin,
    double omegaMax,
    size_t numPoints,
    bool logScale = true);

FrequencyResponseData computeDigitalFrequencyResponse(
    const std::vector<iirfilters::BiquadCoefficients>& sos,
    double sampleRate,
    size_t numPoints,
    bool logScale = true);

} // namespace demo

#endif
