#include "frequency_response.hpp"
#include <algorithm>
#include <cmath>

namespace demo {

FrequencyResponseData computeAnalogFrequencyResponse(
    const std::vector<std::complex<double>>& zeros,
    const std::vector<std::complex<double>>& poles,
    double gain,
    double omegaMin,
    double omegaMax,
    size_t numPoints,
    bool logScale)
{
    FrequencyResponseData result;
    result.frequencies.reserve(numPoints);
    result.magnitudeDb.reserve(numPoints);
    result.phase.reserve(numPoints);

    // Build frequency vector
    for (size_t i = 0; i < numPoints; ++i) {
        double t = static_cast<double>(i) / static_cast<double>(numPoints - 1);
        double omega;
        if (logScale) {
            double logMin = std::log10(omegaMin);
            double logMax = std::log10(omegaMax);
            omega = std::pow(10.0, logMin + t * (logMax - logMin));
        } else {
            omega = omegaMin + t * (omegaMax - omegaMin);
        }
        result.frequencies.push_back(omega);
    }

    // Compute frequency response
    iirfilters::Zpk zpk(zeros, poles, gain);
    auto H = iirfilters::freqsZpk(zpk, result.frequencies);

    // Extract magnitude and phase
    for (const auto& h : H) {
        double mag = std::abs(h);
        result.magnitudeDb.push_back(20.0 * std::log10(std::max(mag, 1e-20)));
        result.phase.push_back(std::arg(h));
    }

    return result;
}

FrequencyResponseData computeDigitalFrequencyResponse(
    const std::vector<iirfilters::BiquadCoefficients>& sos,
    double sampleRate,
    size_t numPoints,
    bool logScale)
{
    FrequencyResponseData result;
    result.frequencies.reserve(numPoints);
    result.magnitudeDb.reserve(numPoints);
    result.phase.reserve(numPoints);

    const double minFreq = 1.0;
    const double maxFreq = sampleRate / 2.0 * 0.999;

    // Build frequency vectors
    std::vector<double> normalizedFreqs;
    normalizedFreqs.reserve(numPoints);

    for (size_t i = 0; i < numPoints; ++i) {
        double t = static_cast<double>(i) / static_cast<double>(numPoints - 1);
        double freq;
        if (logScale) {
            double logMin = std::log10(minFreq);
            double logMax = std::log10(maxFreq);
            freq = std::pow(10.0, logMin + t * (logMax - logMin));
        } else {
            freq = minFreq + t * (maxFreq - minFreq);
        }
        result.frequencies.push_back(freq);
        normalizedFreqs.push_back(2.0 * M_PI * freq / sampleRate);
    }

    // Compute frequency response
    auto H = iirfilters::freqzSos(sos, normalizedFreqs);

    // Extract magnitude and phase
    for (const auto& h : H) {
        double mag = std::abs(h);
        result.magnitudeDb.push_back(20.0 * std::log10(std::max(mag, 1e-20)));
        result.phase.push_back(std::arg(h));
    }

    return result;
}

} // namespace demo
