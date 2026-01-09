#include "filters.hpp"
#include "math_supplement.hpp"

#include <cassert>
#include <limits>

namespace iirfilters {

BiquadCoefficients::BiquadCoefficients(std::complex<double> z1, std::complex<double> z2, std::complex<double> p1, std::complex<double> p2)
{
    if (!math::isReal(p1)) {
        b1 = -2.0 * p1.real();
        b2 = std::norm(p1);
    }
    else {
        b1 = -(p1.real() + p2.real());
        b2 = p1.real() * p2.real();
    }

    if (!math::isReal(z1)) {
        a0 = 1.0;
        a1 = -2.0 * z1.real();
        a2 = std::norm(z1);
    }
    else {
        a0 = 1.0;
        a1 = -(z1.real() + z2.real());
        a2 = z1.real() * z2.real();
    }
}

Zpk::Zpk(const std::vector<std::complex<double>>& z, const std::vector<std::complex<double>>& p, double k)
    : zeros(z)
    , poles(p)
    , gain(k)
{
}

std::vector<BiquadCoefficients> Zpk::toSos() const
{
    // Port of SciPy's zpk2sos with 'nearest' pairing algorithm
    // https://github.com/scipy/scipy/blob/v1.16.2/scipy/signal/_filter_design.py#L1475-L1795

    using CDouble = std::complex<double>;

    std::vector<CDouble> z = zeros;
    std::vector<CDouble> p = poles;

    if (z.empty() && p.empty()) {
        std::vector<BiquadCoefficients> result;
        BiquadCoefficients bc{};
        bc.a0 = gain;
        result.push_back(bc);
        return result;
    }

    // Ensure we have the same number of poles and zeros and make copies
    while (z.size() < p.size()) {
        z.emplace_back(0.0, 0.0);
    }
    while (p.size() < z.size()) {
        p.emplace_back(0.0, 0.0);
    }

    if (p.size() % 2 == 1) {
        p.emplace_back(0.0, 0.0);
        z.emplace_back(0.0, 0.0);
    }

    const auto nsections = p.size() / 2;

    // Ensure we have complex conjugate pairs
    p = math::sortComplexNumbers(p);
    z = math::sortComplexNumbers(z);

    std::vector<BiquadCoefficients> sos(nsections);

    auto idxWorst = [](const std::vector<CDouble>& vals) -> size_t {
        // Find the index of the pole/zero closest to the unit circle
        double minDist = std::numeric_limits<double>::infinity();
        size_t min_idx = 0;
        for (size_t i = 0; i < vals.size(); ++i) {
            double dist = std::abs(1.0 - std::abs(vals[i]));
            if (dist < minDist) {
                minDist = dist;
                min_idx = i;
            }
        }
        return min_idx;
    };

    auto nearestZeroIdx = [](const std::vector<CDouble>& zeros_remaining,
                             const CDouble& pole) -> size_t {
        // Find the index of the zero nearest to a given pole
        double min_dist = std::numeric_limits<double>::infinity();
        size_t min_idx = 0;
        for (size_t i = 0; i < zeros_remaining.size(); ++i) {
            double dist = std::abs(pole - zeros_remaining[i]);
            if (dist < min_dist) {
                min_dist = dist;
                min_idx = i;
            }
        }
        return min_idx;
    };

    // Construct the system, reversing order so the "worst" are last
    for (int si = static_cast<int>(nsections) - 1; si >= 0; --si) {
        // Select the next "worst" pole (closest to unit circle)
        const auto p1Idx = idxWorst(p);
        CDouble p1 = p[p1Idx];
        p.erase(p.begin() + static_cast<long>(p1Idx));

        // Pair that pole with a zero
        CDouble p2, z1, z2;

        if (math::isReal(p1)) {
            // Pole is real, need to find another real pole to pair it with
            size_t p2Idx = 0;
            for (size_t i = 0; i < p.size(); ++i) {
                if (math::isReal(p[i])) {
                    // Find the worst among remaining real poles
                    double dist = std::abs(1.0 - std::abs(p[i]));
                    if (i == 0 || dist < std::abs(1.0 - std::abs(p[p2Idx]))) {
                        p2Idx = i;
                    }
                }
            }
            p2 = p[p2Idx];
            p.erase(p.begin() + static_cast<long>(p2Idx));
        }
        else {
            p2 = std::conj(p1);
        }

        // Find nearest zero to p1
        if (!z.empty()) {
            size_t z1Idx = nearestZeroIdx(z, p1);
            z1 = z[z1Idx];
            z.erase(z.begin() + static_cast<long>(z1Idx));

            if (!math::isReal(z1)) {
                // Complex zero
                z2 = std::conj(z1);
            }
            else {
                // Real zero, need another zero
                if (!z.empty()) {
                    size_t z2_idx = nearestZeroIdx(z, p1);
                    z2 = z[z2_idx];
                    z.erase(z.begin() + static_cast<long>(z2_idx));
                }
                else {
                    // No more zeros, this will be (z+1) form
                    z2 = std::complex<double>(0.0, 0.0);
                }
            }
        }
        else {
            // No zeros left
            z1 = std::complex<double>(0.0, 0.0);
            z2 = std::complex<double>(0.0, 0.0);
        }

        sos[si] = BiquadCoefficients(z1, z2, p1, p2);
    }

    // Put gain in the first SOS
    sos[0].a0 *= gain;
    sos[0].a1 *= gain;
    sos[0].a2 *= gain;

    return sos;
}

Cascade Zpk::toCascade() const
{
    return Cascade(toSos());
}

void Biquad::reset()
{
    s1 = 0.0;
    s2 = 0.0;
}

double Biquad::process(const double in)
{
    // Direct Form II Transposed implementation
    // H(z) = (a0 + a1*z^-1 + a2*z^-2) / (1 + b1*z^-1 + b2*z^-2)

    // See:
    // https://ccrma.stanford.edu/~jos/fp/Transposed_Direct_Forms.html
    // Note that a and b are flipped

    const double out = s1 + coef.a0 * in;
    s1 = s2 + coef.a1 * in - coef.b1 * out;
    s2 = coef.a2 * in - coef.b2 * out;

    return out;
}

void Biquad::setCoefficients(const BiquadCoefficients& coefficients)
{
    coef = coefficients;
}

Cascade::Cascade(const size_t sections)
{
    biquads.resize(sections);
}

Cascade::Cascade(const std::vector<BiquadCoefficients>& coefficients)
{
    biquads.resize(coefficients.size());
    for (auto i = 0; i < coefficients.size(); i++) {
        biquads[i].setCoefficients(coefficients[i]);
    }
}

void Cascade::reset() const
{
    for (auto b : biquads) {
        b.reset();
    }
}

double Cascade::process(const double in) const
{
    double out = in;
    for (auto b : biquads) {
        out = b.process(out);
    }

    return out;
}

Zpk butterworthPrototype(const size_t filterOrder)
{
    // Evenly distribute poles around the unit circle in the left half-plane
    auto poles = std::vector<std::complex<double>>();

    // This indexing ensures that if filterOrder is odd, the middle value will
    // be 0 to ensure exactly one real pole
    for (int n = -static_cast<int>(filterOrder) + 1; n < static_cast<int>(filterOrder); n += 2) {
        poles.push_back(std::polar(1.0, M_PI + M_PI * n / (2.0 * static_cast<double>(filterOrder))));
    }

    const auto zeros = std::vector<std::complex<double>>(); // no zeros
    constexpr double k = 1;
    return { zeros, poles, k };
}

Zpk chebyshev1Prototype(size_t filterOrder, double rp)
{
    const auto zeros = std::vector<std::complex<double>>(); // no zeros

    // Ripple factor (epsilon)
    const double eps = std::sqrt(std::pow(10.0, 0.1 * rp) - 1.0);
    const double mu = 1.0 / static_cast<double>(filterOrder) * std::asinh(1.0 / eps);

    // Arrange poles in an ellipse on the left half of the S-plane
    auto poles = std::vector<std::complex<double>>();

    for (int n = -static_cast<int>(filterOrder) + 1; n < static_cast<int>(filterOrder); n += 2) {
        const auto theta = M_PI * n / (2.0 * static_cast<double>(filterOrder));
        poles.push_back(-std::sinh(std::complex<double>(mu, theta)));
    }

    std::complex<double> product(1.0, 0.0);
    for (const auto& val : poles) {
        product *= -val;
    }
    double k = product.real();

    if (filterOrder % 2 == 0) {
        k = k / std::sqrt(1 + eps * eps);
    }

    return { zeros, poles, k };
}

Zpk chebyshev2Prototype(size_t filterOrder, double rs)
{
    const auto N = static_cast<int>(filterOrder);

    // Ripple factor
    const double de = 1.0 / std::sqrt(std::pow(10.0, 0.1 * rs) - 1.0);
    const double mu = std::asinh(1.0 / de) / static_cast<double>(N);

    std::vector<int> iterIndices{};
    if (N % 2) {
        for (int i = -N + 1; i < 0; i += 2) {
            iterIndices.push_back(i);
        }
        for (int i = 2; i < N; i += 2) {
            iterIndices.push_back(i);
        }
    }
    else {
        for (int i = -N + 1; i < N; i += 2) {
            iterIndices.push_back(i);
        }
    }

    auto zeros = std::vector<std::complex<double>>();

    for (const int m : iterIndices) {
        std::complex<double> val(0.0, std::sin(static_cast<double>(m) * M_PI / (2.0 * static_cast<double>(N))));
        zeros.push_back(std::conj(1.0 / val));
    }

    // Create poles like with Butterworth
    auto poles = std::vector<std::complex<double>>();
    for (int n = -static_cast<int>(filterOrder) + 1; n < static_cast<int>(filterOrder); n += 2) {
        poles.push_back(-std::exp(std::complex<double>(0.0, M_PI * n / (2.0 * static_cast<double>(filterOrder)))));
    }

    // Warp into Chebyshev II
    for (auto& i : poles) {
        auto pole = i;
        std::complex<double> newPole(std::sinh(mu) * pole.real(), std::cosh(mu) * pole.imag());
        i = 1.0 / newPole;
    }

    std::complex<double> prod_p = 1.0;
    std::complex<double> prod_z = 1.0;

    for (auto pole : poles) {
        prod_p *= -pole;
    }
    for (auto zero : zeros) {
        prod_z *= -zero;
    }

    double k = (prod_p / prod_z).real();

    return { zeros, poles, k };
}

Zpk ellipticPrototype(size_t filterOrder, double rp, double rs)
{
    if (filterOrder == 0) {
        return { {}, {}, std::pow(10, -rp / 20.0) };
    }
    if (filterOrder == 1) {
        const auto k = std::sqrt(1.0 / math::pow10m1(0.1 * rp));
        return { {}, { -k }, k };
    }

    const int N = static_cast<int>(filterOrder);

    const double epsSq = math::pow10m1(0.1 * rp);
    const double eps = std::sqrt(epsSq);
    const double ck1sq = epsSq / math::pow10m1(0.1 * rs);
    if (std::abs(ck1sq) < std::numeric_limits<double>::epsilon()) {
        // Cannot design a filter with the given rp and rs
        return {};
    }

    const double val0 = math::ellipk(ck1sq);
    const double m = math::solveDegreeEquation(N, ck1sq);
    const double capk = math::ellipk(m);
    const double r = math::arcjacsc1(1.0 / eps, ck1sq);
    const double v0 = (capk * r) / (N * val0);
    double sv, cv, dv, phi_v;
    math::ellipj(v0, 1.0 - m, sv, cv, dv, phi_v);

    std::vector<std::complex<double>> zeros;
    std::vector<std::complex<double>> poles;

    // j = np.arange(1 - N % 2, N, 2)
    for (int idx = (1 - N % 2); idx < N; idx += 2) {
        double u = idx * capk / N;
        double s, c, d, phi;
        math::ellipj(u, m, s, c, d, phi);

        // snew = np.compress(abs(s) > EPSILON, s, ...)
        if (std::abs(s) > std::numeric_limits<double>::epsilon()) {
            std::complex<double> zElem(0.0, 1.0 / (std::sqrt(m) * s));
            zeros.push_back(zElem);
            zeros.push_back(std::conj(zElem));
        }

        std::complex<double> pNum(-(c * d * sv * cv), -(s * dv));
        double pDen = 1.0 - std::pow(d * sv, 2.0);
        std::complex<double> pVal = pNum / pDen;
        if (std::abs(pVal.imag()) < std::numeric_limits<double>::epsilon()) {
            poles.push_back(pVal);
        }
        else {
            poles.push_back(pVal);
            poles.push_back(std::conj(pVal));
        }
    }

    std::complex<double> polesProd(1.0, 0.0);
    for (const auto& p : poles) {
        polesProd *= -p;
    }
    std::complex<double> zerosProd(1.0, 0.0);
    for (const auto& z : zeros) {
        zerosProd *= -z;
    }

    double gain = (polesProd / zerosProd).real();

    if (N % 2 == 0) {
        gain /= std::sqrt(1.0 + epsSq);
    }

    return { zeros, poles, gain };
}

Zpk besselPrototype(int filterOrder)
{
    std::vector<std::complex<double>> zeros{};
    constexpr double gain = 1.0;

    const auto coef = math::reverseBesselPolynomial(filterOrder);
    auto poles = math::findPolynomialRoots(coef);

    // Apply phase normalization
    const double factor = std::pow(coef[0], -1.0 / filterOrder);
    for (auto& p : poles) {
        p *= factor;
    }

    return { zeros, poles, gain };
}

Zpk legendrePrototype(int filterOrder)
{
    if (filterOrder <= 0) {
        return { {}, {}, 1.0 };
    }

    std::vector<std::complex<double>> zeros{};
    constexpr double gain = 1.0;

    // Get Legendre "Optimum-L" polynomial coefficients
    const auto w = math::legendreOptimumLCoefficients(filterOrder);

    // Form the characteristic polynomial: 1 + w[0] + w[1]*s^2 + w[2]*s^4 + ...
    // This alternates coefficients for even powers of s
    const int degree = filterOrder * 2;
    std::vector<double> polyCoef(degree + 1, 0.0);

    polyCoef[0] = 1.0 + w[0];
    for (int i = 1; i <= filterOrder; i++) {
        // w[i] corresponds to s^(2i), with alternating sign
        polyCoef[2 * i] = w[i] * ((i & 1) ? -1.0 : 1.0);
    }

    // Find roots of the polynomial
    auto allRoots = math::findPolynomialRoots(polyCoef);

    // Select only roots in the left half-plane (stable poles)
    std::vector<std::complex<double>> poles;
    for (const auto& root : allRoots) {
        if (root.real() <= 0.0) {
            poles.push_back(root);
        }
    }

    // Sort by descending imaginary part and keep only filterOrder poles
    std::sort(poles.begin(), poles.end(), [](const auto& a, const auto& b) {
        return std::abs(a.imag()) > std::abs(b.imag());
    });

    poles.resize(filterOrder);

    return { zeros, poles, gain };
}

Zpk lowpassToLowpass(const Zpk& zpk, double wc)
{
    auto zeros = zpk.getZeros();
    auto poles = zpk.getPoles();
    const auto gain = zpk.getGain();

    const int degree = static_cast<int>(poles.size()) - static_cast<int>(zeros.size());
    if (degree < 0) {
        // Improper transfer function
        return {};
    }

    // Scale all points radially from origin to shift cutoff frequency
    for (auto& z : zeros) {
        z *= wc;
    }

    for (auto& p : poles) {
        p *= wc;
    }

    // Each shifted pole decreases gain by wc, each shifted zero increases it.
    // Cancel out the net change to keep overall gain the same
    const double k = gain * std::pow(wc, degree);

    return { zeros, poles, k };
}

Zpk lowpassToHighpass(const Zpk& zpk, double wc)
{
    // https://github.com/scipy/scipy/blob/b1296b9b4393e251511fe8fdd3e58c22a1124899/scipy/signal/_filter_design.py#L3067

    auto zeros = zpk.getZeros();
    auto poles = zpk.getPoles();
    const auto gain = zpk.getGain();

    const int degree = static_cast<int>(poles.size()) - static_cast<int>(zeros.size());
    if (degree < 0) {
        // Improper transfer function
        return {};
    }

    // Calculate gain first
    // Cancel out gain change caused by inversion
    // k_hp = k * real(prod(-z) / prod(-p))
    std::complex<double> zerosProd(1.0, 0.0);
    for (const auto& z : zeros) {
        zerosProd *= -z;
    }

    std::complex<double> polesProd(1.0, 0.0);
    for (const auto& p : poles) {
        polesProd *= -p;
    }
    std::complex<double> ratio = zerosProd / polesProd;
    // Invert positions radially about unit circle to convert LPF to HPF
    // Scale all points radially from origin to shift cutoff frequency
    for (auto& z : zeros) {
        z = wc / z;
    }

    for (auto& p : poles) {
        p = wc / p;
    }

    // If lowpass had zeros at infinity, inverting moves them to origin.
    if (degree > 0) {
        for (int i = 0; i < degree; ++i) {
            zeros.emplace_back(0, 0);
        }
    }

    return { zeros, poles, gain * ratio.real() };
}

Zpk lowpassToBandpass(const Zpk& zpk, double w0, double bw)
{
    // https://github.com/scipy/scipy/blob/b1296b9b4393e251511fe8fdd3e58c22a1124899/scipy/signal/_filter_design.py#L3152

    const auto zeros = zpk.getZeros();
    const auto poles = zpk.getPoles();
    const double gain = zpk.getGain();

    const int degree = static_cast<int>(poles.size()) - static_cast<int>(zeros.size());
    if (degree < 0) {
        // Improper transfer function
        return {};
    }

    const double k = gain * std::pow(bw, degree);

    std::vector<std::complex<double>> newZeros;
    std::vector<std::complex<double>> newPoles;

    // The same transform is done on the zeros and poles
    auto transform = [&](const std::complex<double>& s) {
        const auto term = s * (bw / 2.0);
        const auto root = std::sqrt(term * term - w0 * w0);
        return std::make_pair(term + root, term - root);
    };

    for (const auto& z : zeros) {
        auto [fst, snd] = transform(z);
        newZeros.push_back(fst);
        newZeros.push_back(snd);
    }

    for (const auto& p : poles) {
        auto [fst, snd] = transform(p);
        newPoles.push_back(fst);
        newPoles.push_back(snd);
    }

    // Move any zeros that were at infinity to the center of the stopband
    for (int i = 0; i < degree; ++i) {
        newZeros.emplace_back(0, 0);
    }

    return { newZeros, newPoles, k };
}

Zpk lowpassToBandstop(const Zpk& zpk, double w0, double bw)
{
https: // github.com/scipy/scipy/blob/b1296b9b4393e251511fe8fdd3e58c22a1124899/scipy/signal/_filter_design.py#L3253

    const auto zeros = zpk.getZeros();
    const auto poles = zpk.getPoles();
    const auto gain = zpk.getGain();

    const int degree = static_cast<int>(poles.size()) - static_cast<int>(zeros.size());
    if (degree < 0) {
        // Improper transfer function
        return {};
    }

    std::complex<double> prodZ(1.0, 0.0);
    for (const auto& z : zeros) {
        prodZ *= -z;
    }

    std::complex<double> prodP(1.0, 0.0);
    for (const auto& p : poles) {
        prodP *= -p;
    }

    double k = gain * (prodZ / prodP).real();

    std::vector<std::complex<double>> newZeros;
    std::vector<std::complex<double>> newPoles;

    // The same transform is done on the zeros and poles
    auto transform = [&](const std::complex<double>& s) {
        // Scale to desired bandwidth
        const auto s_hp = bw / 2.0 / s;
        // Duplicate and shift both +w0 and -w0
        const auto a = std::sqrt(s_hp * s_hp - w0 * w0);
        return std::make_pair(s_hp + a, s_hp - a);
    };

    for (const auto& z : zeros) {
        const auto [fst, snd] = transform(z);
        newZeros.push_back(fst);
        newZeros.push_back(snd);
    }

    for (const auto& p : poles) {
        const auto [fst, snd] = transform(p);
        newPoles.push_back(fst);
        newPoles.push_back(snd);
    }

    // Move degree zeros to origin, leaving degree zeros at infinity for BPF
    std::complex<double> center_pos(0, w0);
    std::complex<double> center_neg(0, -w0);

    for (int i = 0; i < degree; ++i) {
        newZeros.push_back(center_pos);
        newZeros.push_back(center_neg);
    }

    return { newZeros, newPoles, k };
}

double prewarpFrequency(const double digitalFreqHz, const double sampleRate)
{
    const double wd = 2.0 * M_PI * digitalFreqHz;
    return 2.0 * sampleRate * std::tan(wd / (2.0 * sampleRate));
}

Zpk bilinearTransform(const Zpk& analog, double sampleRate)
{
    // https://github.com/scipy/scipy/blob/v1.16.2/scipy/signal/_filter_design.py#L2904

    const double fs2 = 2.0 * sampleRate;

    auto transformPoint = [fs2](const std::complex<double> s) {
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

    // Any zeros that were at infinity get moved to the Nyquist frequency
    const int degree = static_cast<int>(digitalPoles.size()) - static_cast<int>(digitalZeros.size());
    for (int i = 0; i < degree; ++i) {
        digitalZeros.emplace_back(-1.0, 0.0);
    }

    // Compensate for gain change
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

    return { digitalZeros, digitalPoles, gain };
}

std::complex<double> freqsZpk(const Zpk& zpk, double omega)
{
    // https://github.com/scipy/scipy/blob/v1.16.2/scipy/signal/_filter_design.py#L638
    std::complex<double> s(0.0, omega);

    std::complex<double> num(zpk.getGain(), 0.0);
    for (const auto& z : zpk.getZeros()) {
        num *= (s - z);
    }

    std::complex<double> den(1.0, 0.0);
    for (const auto& p : zpk.getPoles()) {
        den *= (s - p);
    }

    return num / den;
}

std::complex<double> freqzSos(const BiquadCoefficients& coef, double w)
{
    // https://github.com/scipy/scipy/blob/v1.16.2/scipy/signal/_filter_design.py#L341
    std::complex<double> z = std::exp(std::complex<double>(0.0, w));
    std::complex<double> z_inv = 1.0 / z;
    std::complex<double> z_inv2 = z_inv * z_inv;

    std::complex<double> num = coef.a0 + coef.a1 * z_inv + coef.a2 * z_inv2;
    std::complex<double> den = 1.0 + coef.b1 * z_inv + coef.b2 * z_inv2;

    return num / den;
}

std::complex<double> freqzSos(const std::vector<BiquadCoefficients>& sos, double w)
{
    std::complex<double> H(1.0, 0.0);
    for (const auto& bq : sos) {
        H *= freqzSos(bq, w);
    }
    return H;
}

std::vector<std::complex<double>> freqsZpk(const Zpk& zpk, const std::vector<double>& omega)
{
    std::vector<std::complex<double>> result;
    result.reserve(omega.size());
    for (const auto& w : omega) {
        result.push_back(freqsZpk(zpk, w));
    }
    return result;
}

std::vector<std::complex<double>> freqzSos(const std::vector<BiquadCoefficients>& sos, const std::vector<double>& w)
{
    std::vector<std::complex<double>> result;
    result.reserve(w.size());
    for (const auto& freq : w) {
        result.push_back(freqzSos(sos, freq));
    }
    return result;
}

} // namespace iirfilters
