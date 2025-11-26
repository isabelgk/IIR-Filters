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

std::vector<BiquadCoefficients> Zpk::toSos()
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
        p.erase(p.begin() + p1Idx);

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
            p.erase(p.begin() + p2Idx);
        }
        else {
            p2 = std::conj(p1);
        }

        // Find nearest zero to p1
        if (!z.empty()) {
            size_t z1Idx = nearestZeroIdx(z, p1);
            z1 = z[z1Idx];
            z.erase(z.begin() + z1Idx);

            if (!math::isReal(z1)) {
                // Complex zero
                z2 = std::conj(z1);
            }
            else {
                // Real zero, need another zero
                if (!z.empty()) {
                    size_t z2_idx = nearestZeroIdx(z, p1);
                    z2 = z[z2_idx];
                    z.erase(z.begin() + z2_idx);
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

Cascade Zpk::toCascade()
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
    for (int n = -filterOrder + 1; n < static_cast<int>(filterOrder); n += 2) {
        poles.push_back(std::polar(1.0, M_PI + M_PI * n / (2.0 * filterOrder)));
    }

    const auto zeros = std::vector<std::complex<double>>(); // no zeros
    constexpr double k = 1;
    return Zpk(zeros, poles, k);
}

Zpk chebyshev1Prototype(size_t filterOrder, double rp)
{
    const auto zeros = std::vector<std::complex<double>>(); // no zeros

    // Ripple factor (epsilon)
    const double eps = std::sqrt(std::pow(10.0, 0.1 * rp) - 1.0);
    const double mu = 1.0 / static_cast<double>(filterOrder) * std::asinh(1.0 / eps);

    // Arrange poles in an ellipse on the left half of the S-plane
    auto poles = std::vector<std::complex<double>>();

    for (int n = -filterOrder + 1; n < static_cast<int>(filterOrder); n += 2) {
        const auto theta = M_PI * n / (2.0 * filterOrder);
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

    return Zpk(zeros, poles, k);
}

Zpk chebyshev2Prototype(size_t filterOrder, double rs)
{
    const int N = filterOrder;

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

    for (int i = 0; i < iterIndices.size(); i++) {
        const int m = iterIndices[i];
        std::complex<double> val(0.0, std::sin(static_cast<double>(m) * M_PI / (2.0 * static_cast<double>(N))));
        zeros.push_back(std::conj(1.0 / val));
    }

    // Create poles like with Butterworth
    auto poles = std::vector<std::complex<double>>();
    for (int n = -filterOrder + 1; n < static_cast<int>(filterOrder); n += 2) {
        poles.push_back(-std::exp(std::complex<double>(0.0, M_PI * n / (2.0 * filterOrder))));
    }

    // Warp into Chebyshev II
    for (int i = 0; i < poles.size(); i++) {
        auto pole = poles[i];
        std::complex<double> newPole(std::sinh(mu) * pole.real(), std::cosh(mu) * pole.imag());
        poles[i] = 1.0 / newPole;
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

    return Zpk(zeros, poles, k);
}

Zpk ellipticPrototype(size_t filterOrder, double rp, double rs)
{
    if (filterOrder == 0) {
        return Zpk({}, {}, std::pow(10, -rp / 20.0));
    }
    if (filterOrder == 1) {
        const auto k = std::sqrt(1.0 / math::pow10m1(0.1 * rp));
        return Zpk({}, { -k }, k);
    }

    const int N = filterOrder;

    const double epsSq = math::pow10m1(0.1 * rp);
    const double eps = std::sqrt(epsSq);
    const double ck1sq = epsSq / math::pow10m1(0.1 * rs);
    if (std::abs(ck1sq) < std::numeric_limits<double>::epsilon()) {
        // Cannot design a filter with the given rp and rs
        return {};
    }

    const double val0 = math::ellipk(ck1sq);
    const double val1 = math::ellipkm1(ck1sq);
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

    return Zpk(zeros, poles, gain);
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

    return Zpk(zeros, poles, gain);
}

} // namespace iirfilters
