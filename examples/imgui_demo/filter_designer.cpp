#include "filter_designer.hpp"
#include "bilinear_transform.hpp"

namespace demo {

DesignedFilter designFilter(const FilterParameters& params)
{
    DesignedFilter result;

    switch (params.type) {
        case FilterType::Butterworth:
            result.analogPrototype =
                iirfilters::butterworthPrototype(static_cast<size_t>(params.order));
            break;
        case FilterType::Chebyshev1:
            result.analogPrototype = iirfilters::chebyshev1Prototype(
                static_cast<size_t>(params.order), params.passbandRipple);
            break;
        case FilterType::Chebyshev2:
            result.analogPrototype = iirfilters::chebyshev2Prototype(
                static_cast<size_t>(params.order), params.stopbandAttenuation);
            break;
        case FilterType::Elliptic:
            result.analogPrototype = iirfilters::ellipticPrototype(
                static_cast<size_t>(params.order), params.passbandRipple,
                params.stopbandAttenuation);
            break;
        case FilterType::Bessel:
            result.analogPrototype = iirfilters::besselPrototype(params.order);
            break;
        case FilterType::Legendre:
            result.analogPrototype = iirfilters::legendrePrototype(params.order);
            break;
    }

    double wc = prewarpFrequency(params.cutoffFreq, params.sampleRate);
    double w0 = prewarpFrequency(params.centerFreq, params.sampleRate);
    double bwHigh = prewarpFrequency(params.centerFreq + params.bandwidth / 2.0,
                                     params.sampleRate);
    double bwLow = prewarpFrequency(params.centerFreq - params.bandwidth / 2.0,
                                    params.sampleRate);
    double bw = bwHigh - bwLow;

    switch (params.transform) {
        case TransformType::Lowpass:
            result.analogTransformed =
                iirfilters::lowpassToLowpass(result.analogPrototype, wc);
            break;
        case TransformType::Highpass:
            result.analogTransformed =
                iirfilters::lowpassToHighpass(result.analogPrototype, wc);
            break;
        case TransformType::Bandpass:
            result.analogTransformed =
                iirfilters::lowpassToBandpass(result.analogPrototype, w0, bw);
            break;
        case TransformType::Bandstop:
            result.analogTransformed =
                iirfilters::lowpassToBandstop(result.analogPrototype, w0, bw);
            break;
    }

    result.digitalZpk = bilinearTransform(result.analogTransformed, params.sampleRate);
    result.sos = result.digitalZpk.toSos();
    result.valid = true;

    return result;
}

const char* filterTypeName(FilterType type)
{
    switch (type) {
        case FilterType::Butterworth:
            return "Butterworth";
        case FilterType::Chebyshev1:
            return "Chebyshev I";
        case FilterType::Chebyshev2:
            return "Chebyshev II";
        case FilterType::Elliptic:
            return "Elliptic";
        case FilterType::Bessel:
            return "Bessel";
        case FilterType::Legendre:
            return "Legendre";
    }
    return "Unknown";
}

const char* transformTypeName(TransformType type)
{
    switch (type) {
        case TransformType::Lowpass:
            return "Lowpass";
        case TransformType::Highpass:
            return "Highpass";
        case TransformType::Bandpass:
            return "Bandpass";
        case TransformType::Bandstop:
            return "Bandstop";
    }
    return "Unknown";
}

} // namespace demo
