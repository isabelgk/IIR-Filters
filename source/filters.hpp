#ifndef FILTERS_H
#define FILTERS_H

#include <complex>
#include <vector>

namespace iirfilters {

struct BiquadCoefficients
{
    BiquadCoefficients() = default;

    BiquadCoefficients(std::complex<double> z1, std::complex<double> z2,
                       std::complex<double> p1, std::complex<double> p2);

    double a0;
    double a1;
    double a2;
    double b1;
    double b2;
};

class Cascade;

/**
 * Represents a filter in Zero-Pole-Gain (Zpk) form.
 */
class Zpk
{
  public:
    Zpk() = default;
    ~Zpk() = default;
    Zpk(const Zpk&) = default;
    Zpk& operator=(const Zpk&) = default;
    Zpk(Zpk&&) = default;

    /**
     * Constructs a filter in Zero-Pole-Gain (Zpk) form.
     *
     * @param z A vector of complex numbers representing the filter's zeros.
     * @param p A vector of complex numbers representing the filter's poles.
     * @param k A double representing the filter's gain.
     */
    Zpk(const std::vector<std::complex<double>>& z,
        const std::vector<std::complex<double>>& p,
        double k);

    std::vector<std::complex<double>> getZeros() const { return zeros; }
    std::vector<std::complex<double>> getPoles() const { return poles; }
    double getGain() const { return gain; }

    /**
     * Converts the Zero-Pole-Gain (Zpk) model to Second-Order Section (SOS) coefficients.
     *
     * This method translates the filter's transfer function, represented in
     * Zero-Pole-Gain form, to a representation in terms of second-order sections.
     * The resulting SOS coefficients can be directly used in cascaded biquad filters.
     *
     * @return A vector of BiquadCoefficients, where each set of coefficients
     * represents a second-order section of the filter.
     */
    std::vector<BiquadCoefficients> toSos();

    /**
     * Converts the Zero-Pole-Gain (Zpk) model to a cascading filter structure.
     *
     * @return A Cascade object representing the filter as a series of second-order sections.
     */
    Cascade toCascade();

  private:
    std::vector<std::complex<double>> zeros{};
    std::vector<std::complex<double>> poles{};
    double gain{};
};

/**
 * Biquad filter
 *
 * The transfer function is:
 * H(z) = a0 + a1*z^-1 + a2*z^-2 / 1 + b1*z^-1 + b2*z^-2
 */
class Biquad
{
  public:
    /**
     * Reset the internal state of the biquad filter.
     */
    void reset();

    /**
     * Process an input signal sample through the biquad filter.
     *
     * This method processes a single input sample using the Direct Form II Transposed implementation
     * of the biquad filter. The coefficients of the filter (a0, a1, a2, b1, b2)
     * and the internal state variables (s1, s2) are used to compute the output.
     *
     * @param in Input signal sample to be processed.
     * @return The output signal sample after processing through the biquad filter.
     */
    double process(double in);

    /**
     * Set the coefficients for the biquad filter.
     *
     * @param coefficients A BiquadCoefficients structure containing the filter's coefficients.
     */
    void setCoefficients(const BiquadCoefficients& coefficients);

  private:
    BiquadCoefficients coef{};

    // state
    double s1 = 0.0;
    double s2 = 0.0;
};

/**
 * Cascade filter
 */
class Cascade
{
  public:
    /**
     * Create a new Cascade object.
     *
     * @param sections number of biquad sections
     */
    explicit Cascade(size_t sections);

    /**
     * Construct a Cascade filter using a set of biquad coefficients.
     *
     * @param coefficients A vector of BiquadCoefficients, where each set of
     * coefficients defines a biquad section in the cascade.
     */
    explicit Cascade(const std::vector<BiquadCoefficients>& coefficients);

    /**
     * Reset the internal state of all biquad sections within the cascade filter.
     */
    void reset() const;

    /**
     * Process an input signal through the cascade filter.
     *
     * @param in Input signal sample to be processed.
     * @return The output signal sample after processing through the cascade filter.
     */
    double process(double in) const;

  private:
    std::vector<Biquad> biquads{};
};

/**
 * Generate the zero/pole/gain structure for the analog prototype of an
 * N-th order Butterworth filter. The filter has an angular cutoff frequency
 * of 1.
 *
 * Equivalent to the `buttap(N)` SciPy function.
 *
 * @param filterOrder The order of the Butterworth filter.
 * @return A Zpk object representing the Butterworth prototype filter
 */
Zpk butterworthPrototype(size_t filterOrder);

/**
 * Generate the zero/pole/gain structure for the analog prototype of an
 * N-th order Chebyshev Type I filter.
 *
 * Equivalent to the `cheb1ap(N, rp)` SciPy function.
 *
 * @param filterOrder The order of the Chebyshev Type I filter.
 * @param rp The passband ripple in decibels (dB).
 * @return A Zpk object representing the Chebyshev Type I prototype filter.
 */
Zpk chebyshev1Prototype(size_t filterOrder, double rp);

/**
 * Generate the zero/pole/gain structure for the analog prototype of an
 * N-th order Chebyshev Type II filter.
 *
 * Equivalent to the `cheb2ap(N, rs)` SciPy function.
 *
 * @param filterOrder The order of the Chebyshev Type II filter.
 * @param rs The stopband ripple in decibels (dB).
 * @return A Zpk object representing the Chebyshev Type II prototype filter.
 */
Zpk chebyshev2Prototype(size_t filterOrder, double rs);

/**
 * Generate the zero/pole/gain structure for the analog prototype of an
 * N-th order Elliptic filter.
 *
 * Equivalent to the `ellipap(N, rp, rs)` SciPy function.
 *
 * @param filterOrder The order of the Elliptic filter.
 * @param rp The maximum ripple allowed in the passband, expressed in decibels (dB).
 * @param rs The minimum attenuation required in the stopband, expressed in decibels (dB).
 * @return A Zpk object representing the Elliptic prototype filter.
 */
Zpk ellipticPrototype(size_t filterOrder, double rp, double rs);

/**
 * Generate the zero/pole/gain structure for the analog prototype of an
 * N-th order Bessel filter.
 *
 * Equivalent to the `besselap(N)` SciPy function.
 *
 * @param filterOrder The order of the Bessel filter.
 * @return A Zpk object representing the Bessel prototype filter.
 */
Zpk besselPrototype(int filterOrder);

/**
 * Transforms a lowpass filter in Zero-Pole-Gain (Zpk) form to another lowpass filter
 * with a different cutoff frequency.
 *
 * Equivalent to the SciPy `lp2lp_zpk` function.
 *
 * @param zpk A reference to a Zpk object representing the original lowpass filter.
 * @param wc The desired cutoff frequency (rad/s) for the resulting lowpass filter.
 * @return A Zpk object representing the transformed lowpass filter with the specified cutoff frequency.
 */
Zpk lowpassToLowpass(const Zpk& zpk, double wc);

} // namespace iirfilters

#endif
