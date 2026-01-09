#ifndef FILTER_DESIGNER_H
#define FILTER_DESIGNER_H

#include "filters.hpp"
#include <string>
#include <vector>

namespace demo {

enum class FilterType
{
    Butterworth,
    Chebyshev1,
    Chebyshev2,
    Elliptic,
    Bessel,
    Legendre
};

enum class TransformType
{
    Lowpass,
    Highpass,
    Bandpass,
    Bandstop
};

struct FilterParameters
{
    FilterType type = FilterType::Butterworth;
    TransformType transform = TransformType::Lowpass;
    int order = 4;
    double cutoffFreq = 1000.0;
    double centerFreq = 1000.0;
    double bandwidth = 500.0;
    double passbandRipple = 1.0;
    double stopbandAttenuation = 40.0;
    double sampleRate = 48000.0;
};

struct DesignedFilter
{
    iirfilters::Zpk analogPrototype;
    iirfilters::Zpk analogTransformed;
    iirfilters::Zpk digitalZpk;
    std::vector<iirfilters::BiquadCoefficients> sos;
    bool valid = false;
};

DesignedFilter designFilter(const FilterParameters& params);

const char* filterTypeName(FilterType type);
const char* transformTypeName(TransformType type);

} // namespace demo

#endif
