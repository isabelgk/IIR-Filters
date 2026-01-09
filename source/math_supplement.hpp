#ifndef MATH_SUPPLEMENT_H
#define MATH_SUPPLEMENT_H

#include <cmath>
#include <complex>
#include <vector>

namespace iirfilters::math {

/**
 * Evaluate a polynomial at a given value using Horner's method
 *
 * @param coef the container holding the coefficients of the polynomial,
 *        where coef[i] represents the coefficient for the term with power i
 * @param x the value at which the polynomial is evaluated
 * @return the computed value of the polynomial at the specified point
 */
template <typename Container, typename T>
T evaluatePolynomial(const Container& coef, T x)
{
    T result = 0.0;

    // Iterate from highest to lowest power
    for (int i = coef.size() - 1; i >= 0; --i) {
        result = result * x + coef[i];
    }
    return result;
}

/**
 * Determine if a complex number is approximately real
 *
 * A complex number is considered real if the magnitude of its imaginary part
 * is less than a defined tolerance threshold.
 *
 * @param z the complex number to check
 * @return true if the complex number is approximately real, false otherwise
 */
bool isReal(const std::complex<double>& z);

/**
 * Sort a vector of complex numbers into conjugate pairs followed by real numbers
 *
 * The complex elements of `z` are expected to come as conjugate pairs. In the output,
 * each pair is represented by a single value having a positive imaginary part, sorted
 * first by real part and then by magnitude of the imaginary part. The pairs are
 * averaged when combined to reduce error. The real elements follow and are sorted by
 * value.
 *
 * This is a port of scipy.signal._cplxreal.
 *
 * @param z input vector of complex numbers
 * @return sorted input
 */
std::vector<std::complex<double>> sortComplexNumbers(const std::vector<std::complex<double>>& z);

/**
 * Computes 10^x - 1 for x near 0
 */
double pow10m1(double x);

/**
 * Computes the Complete Elliptic Integral of the First Kind
 *
 * Uses the Arithmetic-Geometric Mean (AGM) method.
 * Formula: K(m) = int(0, pi/2) [1 / sqrt(1 - m * sin^2(theta))] d_theta
 *
 * @param m The parameter m = k^2. Must be less than or equal to 1.
 * @return The value of the integral. Returns infinity if m = 1, NaN if m > 1
 */
double ellipk(double m);

/**
 * Complete elliptic integral of the first kind around m = 1.
 *
 * From: http://www.netlib.org/cephes/
 * https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.ellipkm1.html#scipy.special.ellipkm1
 */
double ellipkm1(double p);

/**
 * Jacobian elliptic functions
 *
 * From: http://www.netlib.org/cephes/
 * https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.ellipj.html
 *
 * @param u Argument
 * @param m Parameter
 * @param sn sn(u|m) output parameter
 * @param cn cn(u|m) output parameter
 * @param dn dn(u|m) output parameter
 * @param ph The value ph is such that if u = ellipkinc(ph, m), then sn(u|m) = sin(ph) and cn(u|m) = cos(ph)
 * @return 0 if success, -1 if domain error
 */
int ellipj(double u, double m, double& sn, double& cn, double& dn, double& ph);

/**
 * Complete elliptic integral of the first kind around m = 1.
 *
 * https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.ellipkm1.html#scipy.special.ellipkm1
 */
double ellipkm1(double k);

/**
 * Compute the Real inverse Jacobian sc with complementary modulus
 *
 * Solve for z in w = sc(z, 1-m)
 *
 * https://github.com/scipy/scipy/blob/b1296b9b4393e251511fe8fdd3e58c22a1124899/scipy/signal/_filter_design.py#L4793
 *
 * @param w Real scalar
 * @param m Modulus (in interval [0, 1])
 * @return Real inverse Jacobian sc
 */
double arcjacsc1(double w, double m);

/**
 * Solve degree equation using nomes
 *
 * Given n, m1, solve the following for m:
 *    n * K(m) / K'(m) = K1(m1) / K1'(m1)
 *
 * https://github.com/scipy/scipy/blob/b1296b9b4393e251511fe8fdd3e58c22a1124899/scipy/signal/_filter_design.py#L4697
 * https://web.archive.org/web/20220626012950/https://www.ece.rutgers.edu/~orfanidi/ece521/notes.pdf
 */
double solveDegreeEquation(int n, double m1);

/**
 * Compute the coefficients of the reverse Bessel polynomial
 *
 * https://en.wikipedia.org/wiki/Bessel_polynomials
 *
 * @param order the order of the Bessel polynomial
 * @return a vector containing the coefficients of the reverse Bessel polynomial
 *
 * @note the coefficients are ordered from the lowest power coefficient to the highest
 */
std::vector<double> reverseBesselPolynomial(int order);

/**
 * Find the roots of a polynomial
 *
 * Uses the Durand-Kerner (Weierstrass) method which is slightly slower than Laguerre's method
 * but finds all roots simultaneously and reduces floating point error accumulation.
 *
 * @param coef a vector containing the polynomial coefficients,
 *             where coefficients[i] corresponds to the coefficient
 *             of the x^i term
 * @return roots of the polynomial
 */
std::vector<std::complex<double>> findPolynomialRoots(const std::vector<double>& coef);

/**
 * Compute Legendre polynomial of the 1st kind using recursion relation
 *
 * Computes the coefficients of the Legendre polynomial P_n(x) using the recursion:
 * (n+1)P_{n+1} = (2n+1)xP_n - nP_{n-1}
 *
 * See also https://github.com/vinniefalco/DSPFilters/blob/acc49170e79a94fcb9c04b8a2116e9f8dffd1c7d/shared/DSPFilters/source/Legendre.cpp#L71
 *
 * @param p Output vector that will contain the polynomial coefficients,
 *          where p[i] is the coefficient of x^i
 * @param n The order of the Legendre polynomial
 */
void legendrePolynomial(std::vector<double>& p, int n);

/**
 * Compute coefficients for Legendre "Optimum-L" filter polynomial
 *
 * Implements the Papoulis algorithm for computing the characteristic polynomial
 * coefficients for an "Optimum-L" (Legendre) filter. This algorithm produces
 * filters with monotonic response in both passband and stopband.
 *
 * See also https://github.com/vinniefalco/DSPFilters/blob/acc49170e79a94fcb9c04b8a2116e9f8dffd1c7d/shared/DSPFilters/source/Legendre.cpp#L126
 *
 * Based on algorithm by C. Bond from Kuo "Network Analysis and Synthesis"
 * and Papoulis "On Monotonic Response Filters", Proc. IRE, 47, Feb. 1959.
 *
 * @param n The filter order
 * @return Vector of polynomial coefficients w[i] corresponding to s^(2i) terms
 */
std::vector<double> legendreOptimumLCoefficients(int n);

} // namespace iirfilters::math

#endif
