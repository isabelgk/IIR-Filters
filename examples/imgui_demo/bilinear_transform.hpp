#ifndef BILINEAR_TRANSFORM_H
#define BILINEAR_TRANSFORM_H

#include "filters.hpp"

namespace demo {

double prewarpFrequency(double digitalFreqHz, double sampleRate);

iirfilters::Zpk bilinearTransform(const iirfilters::Zpk& analog, double sampleRate);

} // namespace demo

#endif
