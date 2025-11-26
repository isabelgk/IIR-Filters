#include "math_supplement.hpp"
#include <algorithm>

namespace iirfilters::math {

bool isReal(const std::complex<double>& z)
{
    constexpr double tol = 100 * std::numeric_limits<double>::epsilon();
    return std::abs(z.imag()) < tol;
}

std::vector<std::complex<double>> sortComplexNumbers(const std::vector<std::complex<double>>& z)
{
    // Port of SciPy _cplxreal(z)
    // https://github.com/scipy/scipy/blob/b1296b9b4393e251511fe8fdd3e58c22a1124899/scipy/signal/_filter_design.py#L980
    if (z.empty()) {
        return {};
    }

    constexpr double tol = 100.0 * std::numeric_limits<double>::epsilon();

    // Sort by real part then by the magnitude of the imaginary part to speed up further sorting
    std::vector<std::complex<double>> zSorted = z;
    std::sort(zSorted.begin(), zSorted.end(),
              [](const std::complex<double>& a, const std::complex<double>& b) {
                  if (a.real() != b.real()) {
                      return a.real() < b.real();
                  }
                  return std::abs(a.imag()) < std::abs(b.imag());
              });

    // Split off reals from complex conjugates
    std::vector<std::complex<double>> zReal;
    std::vector<std::complex<double>> zComplex;
    for (const auto& val : zSorted) {
        if (std::abs(val.imag()) <= tol * std::abs(val)) {
            zReal.push_back(val.real());
        }
        else {
            zComplex.push_back(val);
        }
    }

    if (zComplex.empty()) {
        // Input is entirely real and we're done
        return zReal;
    }

    // Split positive and negative imaginary parts
    std::vector<std::complex<double>> zPosImag;
    std::vector<std::complex<double>> zNegImag;
    for (const auto& val : zComplex) {
        if (val.imag() > 0) {
            zPosImag.push_back(val);
        }
        else if (val.imag() < 0) {
            zNegImag.push_back(val);
        }
    }

    if (zPosImag.size() != zNegImag.size()) {
        // Cannot match conjugates
        return {};
    }

    auto sameRealPart = [tol](std::complex<double> a, std::complex<double> b) -> bool {
        return std::abs(a.real() - b.real()) <= tol * std::abs(b);
    };

    // Find runs of approximately the same real part and sort by imaginary magnitude
    size_t index = 0;
    while (index < zPosImag.size()) {
        size_t runStart = index;

        // Find the end of this run (same real part within tolerance)
        while (index + 1 < zPosImag.size() && sameRealPart(zPosImag[index + 1], zPosImag[index])) {
            index++;
        }
        size_t runEnd = index + 1;

        // Sort this run by imaginary magnitude
        std::sort(zPosImag.begin() + runStart, zPosImag.begin() + runEnd,
                  [](const std::complex<double>& a, const std::complex<double>& b) {
                      return std::abs(a.imag()) < std::abs(b.imag());
                  });
        std::sort(zNegImag.begin() + runStart, zNegImag.begin() + runEnd,
                  [](const std::complex<double>& a, const std::complex<double>& b) {
                      return std::abs(a.imag()) < std::abs(b.imag());
                  });

        index++;
    }

    // Check that negatives match positives (are conjugates)
    for (size_t i = 0; i < zPosImag.size(); i++) {
        if (std::abs(zPosImag[i] - std::conj(zNegImag[i])) > tol * std::abs(zNegImag[i])) {
            // Array contains a complex value with no matching conjugate
            return {};
        }
    }

    // Average out numerical inaccuracy in real vs. imag parts of pairs
    std::vector<std::complex<double>> averagedComplexPairs;
    averagedComplexPairs.reserve(zPosImag.size());
    for (size_t i = 0; i < zPosImag.size(); i++) {
        averagedComplexPairs.push_back((zPosImag[i] + std::conj(zNegImag[i])) / 2.0);
    }

    // Concatenate
    std::vector<std::complex<double>> result;
    result.reserve(z.size());
    for (const auto& val : averagedComplexPairs) {
        result.push_back(std::conj(val));
    }

    for (const auto& real_val : zReal) {
        result.push_back(real_val);
    }

    return result;
}

double pow10m1(double x)
{
    // This works because:
    // 1) expm1(x) = e^(x) - 1
    // 2) 10^x - 1 = e^(x * ln(10)) - 1
    return std::expm1(std::log(10) * x);
}

double ellipk(double m)
{
    if (m == 1.0) {
        return std::numeric_limits<double>::infinity();
    }
    if (m > 1.0) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    double a = 1.0;
    double b = std::sqrt(1.0 - m);

    while (std::abs(a - b) > std::numeric_limits<double>::epsilon()) {
        const double a_next = 0.5 * (a + b);
        const double b_next = std::sqrt(a * b);

        a = a_next;
        b = b_next;
    }

    // K(m) = pi / (2 * AGM(1, sqrt(1-m)))
    return M_PI / (a + a);
}

double ellipkm1(double k)
{
    // P and Q are reversed from the original cephes code so it can be used
    // with our polynomial evaluation code.
    // They are also translated from the Hex data to doubles.
    static constexpr std::array<double, 11> P = {
        1.3862943611198906e+00,
        9.6573590281169013e-02,
        3.0885146524671200e-02,
        1.4938044891680525e-02,
        8.7907827395274377e-03,
        6.1890103363768761e-03,
        6.8748968744994988e-03,
        9.8582137902122601e-03,
        7.9740401322041518e-03,
        2.2802572400587557e-03,
        1.3798286460627324e-04
    };

    static constexpr std::array<double, 11> Q = {
        4.9999999999999992e-01,
        1.2499999999987082e-01,
        7.0312499696395747e-02,
        4.8828034757099824e-02,
        3.7377431417382323e-02,
        3.0120471522760405e-02,
        2.3908960271592489e-02,
        1.5485051664976240e-02,
        5.9405830375316779e-03,
        9.1418472386591723e-04,
        2.9407895504859851e-05
    };

    static constexpr double C1 = 1.3862943611198906188;

    if (k < 0.0 || k > 1.0) {
        return 0.0;
    }

    if (k > 1.0e-5) {
        // For larger k, the regular ellipk method is more accurate
        return ellipk(1.0 - k);
    }

    if (k > std::numeric_limits<double>::epsilon()) {
        const double polyP = evaluatePolynomial(P, k);
        const double polyQ = evaluatePolynomial(Q, k);
        return polyP - std::log(k) * polyQ;
    }

    if (k == 0.0) {
        return std::numeric_limits<double>::max();
    }

    return C1 - 0.5 * std::log(k);
}

int ellipj(double u, double m, double& sn, double& cn, double& dn, double& ph)
{
    constexpr double MACHEP = std::numeric_limits<double>::epsilon();
    double ai, b, phi, t, twon;
    double a[9], c[9];
    int i;

    if (m < 0.0 || m > 1.0) {
        sn = 0.0;
        cn = 0.0;
        ph = 0.0;
        dn = 0.0;
        return -1;
    }

    if (m < 1.0e-9) {
        t = std::sin(u);
        b = std::cos(u);
        ai = 0.25 * m * (u - t * b);
        sn = t - ai * b;
        cn = b + ai * t;
        ph = u - ai;
        dn = 1.0 - 0.5 * m * t * t;
        return 0;
    }

    if (m >= 0.9999999999) {
        constexpr double PIO2 = 1.57079632679489661923;
        ai = 0.25 * (1.0 - m);
        b = std::cosh(u);
        t = std::tanh(u);
        phi = 1.0 / b;
        twon = b * std::sinh(u);
        sn = t + ai * (twon - u) / (b * b);
        ph = 2.0 * std::atan(exp(u)) - PIO2 + ai * (twon - u) / b;
        ai *= t * phi;
        cn = phi - ai * (twon - u);
        dn = phi + ai * (twon + u);
        return 0;
    }

    // A.G.M. scale
    a[0] = 1.0;
    b = std::sqrt(1.0 - m);
    c[0] = std::sqrt(m);
    twon = 1.0;
    i = 0;

    while (std::abs(c[i] / a[i]) > MACHEP) {
        if (i > 7) {
            // Overflow
            goto done;
        }
        ai = a[i];
        ++i;
        c[i] = (ai - b) / 2.0;
        t = std::sqrt(ai * b);
        a[i] = (ai + b) / 2.0;
        b = t;
        twon *= 2.0;
    }

done:
    // backward recurrence
    phi = twon * a[i] * u;
    do {
        t = c[i] * std::sin(phi) / a[i];
        b = phi;
        phi = (std::asin(t) + phi) / 2.0;
    } while (--i);

    t = std::sin(phi);
    sn = t;
    cn = std::cos(phi);
    dn = std::sqrt(1.0 - m * t * t);
    ph = phi;
    return 0;
}

double arcjacsc1(double w, double m)
{
    const auto wStart = std::complex<double>(0, w);
    auto zComplex = wStart;
    constexpr int maxIter = 10;

    auto complement = [](auto kx) {
        // sqrt(1 - k^2)
        // works for small kx
        return std::sqrt((1.0 - kx) * (1.0 + kx));
    };

    const double kValue = std::sqrt(m);
    if (kValue > 1) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    if (std::abs(kValue - 1) <= std::numeric_limits<double>::epsilon()) {
        zComplex = std::atanh(wStart);
    }
    else {
        std::vector<std::complex<double>> ks;
        ks.reserve(maxIter + 1);
        ks.push_back({ kValue, 0.0 });

        int niter = 0;
        while (true) {
            const std::complex<double> kCurrent = ks.back();
            if (std::abs(kCurrent) < std::numeric_limits<double>::min()) {
                break;
            }

            if (niter >= maxIter) {
                // Landen transformation not converging
                return std::numeric_limits<double>::quiet_NaN();
            }

            const auto kp = complement(kCurrent);
            ks.push_back((1.0 - kp) / (1.0 + kp));
            ++niter;
        }

        // Elliptic Integral K
        // K = np.prod(1 + np.array(ks[1:])) * np.pi/2
        std::complex<double> K;
        if (ks.size() <= 1) {
            K = M_PI / 2.0;
        }
        else {
            K = 1.0;
            for (auto i = 1; i < ks.size(); ++i) {
                K *= 1.0 + ks[i];
            }
            K *= M_PI / 2.0;
        }

        // Python: zip(ks[:-1], ks[1:])
        // Iterate from 0 to (size -2)
        const int loop_end = static_cast<int>(ks.size()) - 1;

        auto wCurrent = wStart;
        for (int i = 0; i < loop_end; ++i) {
            auto kn = ks[i];
            auto knext = ks[i + 1];

            // wnext = 2 * wn / ((1 + knext) * (1 + complement(kn * wn)))
            auto denom = (1.0 + knext) * (1.0 + complement(kn * wCurrent));

            if (std::abs(denom) == 0.0) {
                return std::numeric_limits<double>::quiet_NaN();
            }

            wCurrent = 2.0 * wCurrent / denom;
        }

        const auto u = 2.0 / M_PI * std::asin(wCurrent);
        zComplex = K * u;
    }

    if (std::abs(zComplex.real()) > 1e-14) {
        // ValueError
        return std::numeric_limits<double>::quiet_NaN();
    }

    return zComplex.imag();
}

double solveDegreeEquation(int n, double m1)
{
    // https://github.com/scipy/scipy/blob/b1296b9b4393e251511fe8fdd3e58c22a1124899/scipy/signal/_filter_design.py#L4697
    constexpr int ellipDegMax = 7;

    const double k1 = ellipk(m1);
    const double k1p = ellipkm1(m1);

    if (k1 == 0.0) {
        // don't divide by zero
        return 0.0;
    }

    const double q1 = std::exp(-M_PI * k1p / k1);
    const double q = std::pow(q1, 1.0 / static_cast<double>(n));

    double num_sum = 0.0;
    for (int i = 0; i <= ellipDegMax; ++i) {
        double p = i * (i + 1);
        num_sum += std::pow(q, p);
    }

    double den_sum = 1.0;
    for (int i = 1; i <= ellipDegMax + 1; ++i) {
        double p = i * i;
        den_sum += 2.0 * std::pow(q, p);
    }

    const double ratio = num_sum / den_sum;
    const double ratio4 = ratio * ratio * ratio * ratio;
    return 16.0 * q * ratio4;
}

std::vector<double> reverseBesselPolynomial(int order)
{
    // Use the recursion relation
    // B_n(s) = (2n - 1) * B_{n-1}(s) + s^2 * B_{n-2}(s)
    // https://en.wikipedia.org/wiki/Bessel_polynomials#Recursion
    if (order == 0) {
        return { 1.0 };
    }
    if (order == 1) {
        return { 1.0, 1.0 };
    }

    std::vector<double> b2 = { 1.0 }; // B_{n-2}
    std::vector<double> b1 = { 1.0, 1.0 }; // B_{n-1}
    std::vector<double> b;

    for (int n = 2; n <= order; n++) {
        // Initialize with 0.0 at n + 1 (since order is n)
        b.assign(n + 1, 0.0);

        // Term: (2n - 1) * B_{n-1}(s)
        double c = (2.0 * n - 1);
        for (auto i = 0; i < b1.size(); ++i) {
            b[i] += c * b1[i];
        }

        // Term: s^2 * B_{n-2}(s)
        for (auto i = 0; i < b2.size(); ++i) {
            // b[i + 2] because of s^2
            b[i + 2] += b2[i];
        }

        b2 = b1;
        b1 = b;
    }

    return b;
}

std::vector<std::complex<double>> findPolynomialRoots(const std::vector<double>& coef)
{
    // https://www.johndcook.com/blog/2022/11/14/simultaneous-root-finding/
    // https://numbersandshapes.net/posts/weierstrass-durand-kerner/

    constexpr int MAX_ITERATIONS = 1e6;
    constexpr double TOL = 10e-12;
    const int degree = coef.size() - 1;
    std::vector<std::complex<double>> roots(degree);

    // Initial guesses should not be real, so we put them on the unit circle and avoid 1 + 0i by
    // a small angle offset
    for (int i = 0; i < degree; ++i) {
        double angle = 2.0 * M_PI * i / static_cast<double>(degree) + 0.1;
        roots[i] = std::polar(1.0, angle);
    }

    // To simplify iterations, normalize the polynomial
    std::vector<double> monicCoef = coef;
    const double highestDegreeCoef = coef.back();
    for (auto& c : monicCoef) {
        c = c / highestDegreeCoef;
    }

    for (int iter = 0; iter < MAX_ITERATIONS; ++iter) {
        bool converged = true;
        std::vector<std::complex<double>> nextRoots = roots;

        for (int i = 0; i < degree; ++i) {
            std::complex<double> currentRoot = roots[i];
            auto numerator = evaluatePolynomial(monicCoef, currentRoot);

            std::complex<double> denominator = 1.0;
            for (int j = 0; j < degree; ++j) {
                if (i != j) {
                    denominator = denominator * (currentRoot - roots[j]);
                }
            }

            const auto delta = numerator / denominator;
            nextRoots[i] = currentRoot - delta;

            if (std::abs(delta) > TOL) {
                converged = false;
            }
        }

        roots = nextRoots;
        if (converged) {
            break;
        }
    }

    return roots;
}

} // namespace iirfilters::math