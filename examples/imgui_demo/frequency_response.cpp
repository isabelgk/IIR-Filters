#include "frequency_response.hpp"
#include <algorithm>
#include <cmath>

namespace demo {

std::complex<double> evaluateAnalogTransferFunction(
    const std::vector<std::complex<double>>& zeros,
    const std::vector<std::complex<double>>& poles,
    double gain,
    double omega)
{
    std::complex<double> s(0.0, omega);

    std::complex<double> num(gain, 0.0);
    for (const auto& z : zeros) {
        num *= (s - z);
    }

    std::complex<double> den(1.0, 0.0);
    for (const auto& p : poles) {
        den *= (s - p);
    }

    return num / den;
}

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

        auto H = evaluateAnalogTransferFunction(zeros, poles, gain, omega);
        double mag = std::abs(H);
        double magDb = 20.0 * std::log10(std::max(mag, 1e-20));
        double ph = std::arg(H);

        result.frequencies.push_back(omega);
        result.magnitudeDb.push_back(magDb);
        result.phase.push_back(ph);
    }

    return result;
}

std::complex<double> evaluateBiquad(
    const iirfilters::BiquadCoefficients& coef,
    double w)
{
    std::complex<double> z = std::exp(std::complex<double>(0.0, w));
    std::complex<double> z_inv = 1.0 / z;
    std::complex<double> z_inv2 = z_inv * z_inv;

    std::complex<double> num = coef.a0 + coef.a1 * z_inv + coef.a2 * z_inv2;
    std::complex<double> den = 1.0 + coef.b1 * z_inv + coef.b2 * z_inv2;

    return num / den;
}

std::complex<double> evaluateCascade(
    const std::vector<iirfilters::BiquadCoefficients>& sos,
    double w)
{
    std::complex<double> H(1.0, 0.0);
    for (const auto& bq : sos) {
        H *= evaluateBiquad(bq, w);
    }
    return H;
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

        double w = 2.0 * M_PI * freq / sampleRate;

        auto H = evaluateCascade(sos, w);
        double mag = std::abs(H);
        double magDb = 20.0 * std::log10(std::max(mag, 1e-20));

        result.frequencies.push_back(freq);
        result.magnitudeDb.push_back(magDb);
        result.phase.push_back(std::arg(H));
    }

    return result;
}

} // namespace demo
