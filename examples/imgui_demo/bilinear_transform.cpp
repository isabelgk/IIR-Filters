#include "bilinear_transform.hpp"
#include <cmath>

namespace demo {

double prewarpFrequency(double digitalFreqHz, double sampleRate)
{
    double wd = 2.0 * M_PI * digitalFreqHz;
    return 2.0 * sampleRate * std::tan(wd / (2.0 * sampleRate));
}

iirfilters::Zpk bilinearTransform(const iirfilters::Zpk& analog, double sampleRate)
{
    const double fs2 = 2.0 * sampleRate;

    auto transformPoint = [fs2](std::complex<double> s) {
        return (1.0 + s / fs2) / (1.0 - s / fs2);
    };

    std::vector<std::complex<double>> digitalZeros;
    std::vector<std::complex<double>> digitalPoles;

    for (const auto& z : analog.getZeros()) {
        digitalZeros.push_back(transformPoint(z));
    }

    for (const auto& p : analog.getPoles()) {
        digitalPoles.push_back(transformPoint(p));
    }

    int degree = static_cast<int>(digitalPoles.size()) - static_cast<int>(digitalZeros.size());
    for (int i = 0; i < degree; ++i) {
        digitalZeros.emplace_back(-1.0, 0.0);
    }

    std::complex<double> analogDc(analog.getGain(), 0.0);
    for (const auto& z : analog.getZeros()) {
        analogDc *= -z;
    }
    for (const auto& p : analog.getPoles()) {
        analogDc /= -p;
    }

    std::complex<double> digitalDc(1.0, 0.0);
    for (const auto& z : digitalZeros) {
        digitalDc *= (1.0 - z);
    }
    for (const auto& p : digitalPoles) {
        digitalDc /= (1.0 - p);
    }

    double gain = std::abs(analogDc) / std::abs(digitalDc);

    return iirfilters::Zpk(digitalZeros, digitalPoles, gain);
}

} // namespace demo
