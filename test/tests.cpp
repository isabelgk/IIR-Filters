#include "filters.hpp"
#include "math_supplement.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

using namespace iirfilters;

namespace {

constexpr double TOLERANCE = 1e-7;

bool approxEqual(const std::complex<double>& a, const std::complex<double>& b)
{
    return a.real() == Catch::Approx(b.real()).margin(TOLERANCE)
        && a.imag() == Catch::Approx(b.imag()).margin(TOLERANCE);
}

bool contains(const std::vector<std::complex<double>>& x, const std::complex<double>& element)
{
    for (const auto& el : x) {
        if (approxEqual(el, element)) {
            return true;
        }
    }
    return false;
}

void requireApproxEqual(const std::vector<double>& a, const std::vector<double>& b)
{
    REQUIRE(a.size() == b.size());
    for (size_t i = 0; i < a.size(); i++) {
        INFO("Index " << i << ": expected " << b[i] << ", got " << a[i]);
        REQUIRE(a[i] == Catch::Approx(b[i]).margin(TOLERANCE));
    }
}

void requireApproxEqual(const std::vector<std::complex<double>>& a, const std::vector<std::complex<double>>& b)
{
    REQUIRE(a.size() == b.size());
    for (size_t i = 0; i < a.size(); i++) {
        INFO("Index " << i << ": " << a[i] << " not found in expected values");
        REQUIRE(contains(b, a[i]));
    }
}

} // namespace

TEST_CASE("toSos - Fourth Order", "[toSos]")
{
    // >>> from scipy import signal
    // >>> z, p, k = signal.butter(4, 0.87, 'low', output='zpk')
    // >>> sos = signal.zpk2sos(z, p, k)
    std::vector<std::complex<double>> zeros = {
        { -1., 0 },
        { -1., 0 },
        { -1., 0 },
        { -1., 0 },
    };
    std::vector<std::complex<double>> poles = {
        { -0.79667451, 0.31850917 },
        { -0.67140489, 0.11118593 },
        { -0.67140489, -0.11118593 },
        { -0.79667451, -0.31850917 },
    };
    double gain = 0.5838998191275759;

    Zpk zpk(zeros, poles, gain);
    auto result = zpk.toSos();
    REQUIRE(result.size() == 2);

    REQUIRE(result[0].a0 == Catch::Approx(0.58389982));
    REQUIRE(result[0].a1 == Catch::Approx(1.16779964));
    REQUIRE(result[0].a2 == Catch::Approx(0.58389982));
    REQUIRE(result[0].b1 == Catch::Approx(1.34280978));
    REQUIRE(result[0].b2 == Catch::Approx(0.46314683));

    REQUIRE(result[1].a0 == Catch::Approx(1.0));
    REQUIRE(result[1].a1 == Catch::Approx(2.0));
    REQUIRE(result[1].a2 == Catch::Approx(1.0));
    REQUIRE(result[1].b1 == Catch::Approx(1.59334901));
    REQUIRE(result[1].b2 == Catch::Approx(0.73613836));
}

TEST_CASE("toSos - First Order", "[toSos]")
{
    // >>> from scipy import signal
    // >>> z, p, k = signal.butter(1, 0.5, 'low', output='zpk')
    // >>> sos = signal.zpk2sos(z, p, k)
    std::vector<std::complex<double>> zeros = {
        { -1., 0 },
    };
    std::vector<std::complex<double>> poles = {
        { 5.55111512e-17, 0.0 },
    };
    double gain = 0.49999999999999994;

    Zpk zpk(zeros, poles, gain);
    auto result = zpk.toSos();
    REQUIRE(result.size() == 1);

    REQUIRE(result[0].a0 == Catch::Approx(5.00000000e-01));
    REQUIRE(result[0].a1 == Catch::Approx(5.00000000e-01));
    REQUIRE(result[0].a2 == Catch::Approx(0.0));
    REQUIRE(result[0].b1 == Catch::Approx(-5.55111512e-17));
    REQUIRE(result[0].b2 == Catch::Approx(0.0));
}

TEST_CASE("butterworthPrototype - Sixth Order", "[butterworthPrototype]")
{
    // >>> from scipy import signal
    // >>> signal.buttap(6)
    const auto result = butterworthPrototype(6);
    const auto poles = result.getPoles();
    REQUIRE(result.getZeros().size() == 0);
    REQUIRE(poles.size() == 6);
    REQUIRE(result.getGain() == Catch::Approx(1.0));

    const std::vector<std::complex<double>> expectedPoles = {
        { -0.25881905, 0.96592583 },
        { -0.70710678, 0.70710678 },
        { -0.96592583, 0.25881905 },
        { -0.96592583, -0.25881905 },
        { -0.70710678, -0.70710678 },
        { -0.25881905, -0.96592583 },
    };
    requireApproxEqual(poles, expectedPoles);
}

TEST_CASE("butterworthPrototype - Seventh Order", "[butterworthPrototype]")
{
    // >>> from scipy import signal
    // >>> signal.buttap(7)
    const auto result = butterworthPrototype(7);
    const auto poles = result.getPoles();
    REQUIRE(result.getZeros().size() == 0);
    REQUIRE(poles.size() == 7);
    REQUIRE(result.getGain() == Catch::Approx(1.0));

    const std::vector<std::complex<double>> expectedPoles = {
        { -0.22252093, 0.97492791 },
        { -0.6234898, 0.78183148 },
        { -0.90096887, 0.43388374 },
        { -1., 0. },
        { -0.90096887, -0.43388374 },
        { -0.6234898, -0.78183148 },
        { -0.22252093, -0.97492791 },
    };
    requireApproxEqual(poles, expectedPoles);
}

TEST_CASE("chebyshev1Prototype - Sixth Order", "[chebyshev1Prototype]")
{
    // >>> from scipy import signal
    // >>> signal.cheb1ap(6, 0.71)
    const auto result = chebyshev1Prototype(6, 0.71);
    const auto poles = result.getPoles();
    REQUIRE(result.getZeros().size() == 0);
    REQUIRE(poles.size() == 6);
    REQUIRE(result.getGain() == Catch::Approx(0.07415172139359721));

    const std::vector<std::complex<double>> expectedPoles = {
        { -0.06979227, 1.00042813 },
        { -0.19067604, 0.73236422 },
        { -0.26046831, 0.26806391 },
        { -0.26046831, -0.26806391 },
        { -0.19067604, -0.73236422 },
        { -0.06979227, -1.00042813 },
    };
    requireApproxEqual(poles, expectedPoles);
}

TEST_CASE("chebyshev1Prototype - Seventh Order", "[chebyshev1Prototype]")
{
    // >>> from scipy import signal
    // >>> signal.cheb1ap(7, 0.71)
    const auto result = chebyshev1Prototype(7, 0.71);
    const auto poles = result.getPoles();
    REQUIRE(result.getZeros().size() == 0);
    REQUIRE(poles.size() == 7);
    REQUIRE(result.getGain() == Catch::Approx(0.037075860696798635));

    const std::vector<std::complex<double>> expectedPoles = {
        { -0.05127161, 1.00047268 },
        { -0.14365987, 0.8023168 },
        { -0.20759452, 0.44525223 },
        { -0.23041254, 0.0 },
        { -0.20759452, -0.44525223 },
        { -0.14365987, -0.8023168 },
        { -0.05127161, -1.00047268 },
    };
    requireApproxEqual(poles, expectedPoles);
}

TEST_CASE("chebyshev2Prototype - Sixth Order", "[chebyshev2Prototype]")
{
    // >>> from scipy import signal
    // >>> signal.cheb2ap(6, 0.15)
    const auto result = chebyshev2Prototype(6, 0.15);
    const auto zeros = result.getZeros();
    const auto poles = result.getPoles();
    REQUIRE(zeros.size() == 6);
    REQUIRE(poles.size() == 6);

    const std::vector<std::complex<double>> expectedZeros = {
        { 0., -1.03527618 },
        { 0., -1.41421356 },
        { 0., -3.86370331 },
        { 0., 3.86370331 },
        { 0., 1.41421356 },
        { 0., 1.03527618 },
    };
    requireApproxEqual(zeros, expectedZeros);

    const std::vector<std::complex<double>> expectedPoles = {
        { -0.00860956, -1.03470525 },
        { -0.04385294, -1.41216973 },
        { -0.44163078, -3.81065775 },
        { -0.44163078, 3.81065775 },
        { -0.04385294, 1.41216973 },
        { -0.00860956, 1.03470525 }
    };
    requireApproxEqual(poles, expectedPoles);

    REQUIRE(result.getGain() == Catch::Approx(0.9828788730000327));
}

TEST_CASE("chebyshev2Prototype - Seventh Order", "[chebyshev2Prototype]")
{
    // >>> from scipy import signal
    // >>> signal.cheb2ap(7, 0.25)
    const auto result = chebyshev2Prototype(7, 0.25);
    const auto zeros = result.getZeros();
    const auto poles = result.getPoles();
    REQUIRE(zeros.size() == 6);
    REQUIRE(poles.size() == 7);

    const std::vector<std::complex<double>> expectedZeros = {
        { 0., -1.02571686 },
        { 0., -1.27904801 },
        { 0., -2.30476487 },
        { 0., 2.30476487 },
        { 0., 1.27904801 },
        { 0., 1.02571686 },
    };
    requireApproxEqual(zeros, expectedZeros);

    const std::vector<std::complex<double>> expectedPoles = {
        { -8.05435931e-03, -1.02504557 },
        { -3.50677401e-02, -1.27732709 },
        { -1.63825399e-01, -2.29168735 },
        { -2.90304011e+01, -0. },
        { -1.63825399e-01, 2.29168735 },
        { -3.50677401e-02, 1.27732709 },
        { -8.05435931e-03, 1.02504557 }
    };
    requireApproxEqual(poles, expectedPoles);
    REQUIRE(result.getGain() == Catch::Approx(28.756777064046997));
}

TEST_CASE("ellipk - typical value", "[ellipk]")
{
    // m = 0.6
    REQUIRE(math::ellipk(0.6) == Catch::Approx(1.9495677498060258));
}

TEST_CASE("ellipkm1 - midrange value", "[ellipkm1]")
{
    // m1 = 0.62
    REQUIRE(math::ellipkm1(0.62) == Catch::Approx(1.7638983888837312));
}

TEST_CASE("ellipkm1 - close to 1", "[ellipkm1]")
{
    REQUIRE(math::ellipkm1(0.99) == Catch::Approx(1.5747455615173558));
}

TEST_CASE("ellipkm1 - close to 0", "[ellipkm1]")
{
    REQUIRE(math::ellipkm1(0.01) == Catch::Approx(3.6956373629898747));
}

TEST_CASE("ellipj - all Jacobi functions", "[ellipj]")
{
    // u = 4, m = 0.2
    double sn, cn, dn, ph;
    math::ellipj(4, 0.2, sn, cn, dn, ph);
    REQUIRE(sn == Catch::Approx(-0.6219243502072862));
    REQUIRE(cn == Catch::Approx(-0.7830773286331593));
    REQUIRE(dn == Catch::Approx(0.9605425657012028));
    REQUIRE(ph == Catch::Approx(3.8127903862602888));
}

TEST_CASE("solveDegreeEquation - low order", "[solveDegreeEquation]")
{
    // n = 2, k = 0.5
    REQUIRE(math::solveDegreeEquation(2, 0.5) == Catch::Approx(0.9705627484771397));
}

TEST_CASE("solveDegreeEquation - high order", "[solveDegreeEquation]")
{
    // n = 10, k = 0.9
    REQUIRE(math::solveDegreeEquation(10, 0.9) == Catch::Approx(0.9999988431568166));
}

TEST_CASE("ellipticPrototype - Sixth Order", "[ellipticPrototype]")
{
    // >>> from scipy import signal
    // >>> signal.ellipap(6, 0.1, 0.3)
    const auto result = ellipticPrototype(6, 0.1, 0.3);
    REQUIRE(result.getGain() == Catch::Approx(0.9660508789983091));

    const auto zeros = result.getZeros();
    const std::vector<std::complex<double>> expectedZeros = {
        { 0., 1.1497539283693863 },
        { 0., 1.0006767599608006 },
        { 0., 1.0000037537313895 },
        { 0., -1.1497539283693863 },
        { 0., -1.0006767599608006 },
        { 0., -1.0000037537313895 },
    };
    requireApproxEqual(zeros, expectedZeros);

    const auto poles = result.getPoles();
    REQUIRE(zeros.size() == 6);
    REQUIRE(poles.size() == 6);

    const std::vector<std::complex<double>> expectedPoles = {
        { -0.06270876717485811, -1.1349243616564826 },
        { -0.00026698989352419325, -1.0006218078938955 },
        { -1.2885162093489116e-06, -1.0000034861217522 },
        { -0.06270876717485811, 1.1349243616564826 },
        { -0.00026698989352419325, 1.0006218078938955 },
        { -1.2885162093489116e-06, 1.0000034861217522 },
    };
    requireApproxEqual(poles, expectedPoles);
}

TEST_CASE("ellipticPrototype - Seventh Order", "[ellipticPrototype]")
{
    // >>> from scipy import signal
    // >>> signal.ellipap(7, 0.36, 0.8)
    const auto result = ellipticPrototype(7, 0.36, 0.8);
    REQUIRE(result.getGain() == Catch::Approx(2.719419411789406));

    const auto zeros = result.getZeros();
    const std::vector<std::complex<double>> expectedZeros = {
        { 0.0, 1.00562207846308 },
        { 0.0, 1.0000157201681765 },
        { 0.0, 1.0000000488408904 },
        { 0.0, -1.00562207846308 },
        { 0.0, -1.0000157201681765 },
        { 0.0, -1.0000000488408904 },
    };
    requireApproxEqual(zeros, expectedZeros);

    const auto poles = result.getPoles();
    REQUIRE(zeros.size() == 6);
    REQUIRE(poles.size() == 7);

    const std::vector<std::complex<double>> expectedPoles = {
        { -2.7266795554772374, 0.0 },
        { -0.003640204065346889, -1.0042795409540897 },
        { -1.0160623077295934e-05, -1.0000119937371659 },
        { -2.8401644848815185e-08, -1.0000000383661272 },
        { -0.003640204065346889, 1.0042795409540897 },
        { -1.0160623077295934e-05, 1.0000119937371659 },
        { -2.8401644848815185e-08, 1.0000000383661272 },
    };
    requireApproxEqual(poles, expectedPoles);
}

TEST_CASE("reverseBesselPolynomial - fifth order", "[reverseBesselPolynomial]")
{
    const auto result = math::reverseBesselPolynomial(5);
    const std::vector<double> expected = {
        945.0,
        945.0,
        420.0,
        105.0,
        15.0,
        1.0
    };

    requireApproxEqual(result, expected);
}

TEST_CASE("evaluatePolynomial - quadratic with complex argument", "[evaluatePolynomial]")
{
    std::vector<double> coef = {
        3.5, 1.0, 2.0
    };
    std::complex<double> x(1.0, 0.5);
    const auto result = math::evaluatePolynomial(coef, x);
    REQUIRE(result.imag() == Catch::Approx(2.5));
    REQUIRE(result.real() == Catch::Approx(6.0));
}

TEST_CASE("evaluatePolynomial - sixth degree with complex argument", "[evaluatePolynomial]")
{
    std::vector<double> coef = {
        0.9, 10.2, -0.3, -5.9, 2.1, 15.3, 0.5
    };
    std::complex<double> x(1.3, -0.4);
    const auto result = math::evaluatePolynomial(coef, x);
    REQUIRE(std::abs(result.real() - 11.9445) < 10e-3);
    REQUIRE(std::abs(result.imag() + 72.9477) < 10e-3);
}

TEST_CASE("besselPrototype - Sixth Order", "[besselPrototype]")
{
    // >>> from scipy import signal
    // >>> signal.besselap(6)
    const auto result = besselPrototype(6);
    REQUIRE(result.getGain() == Catch::Approx(1.0));

    const auto zeros = result.getZeros();
    const auto poles = result.getPoles();
    REQUIRE(zeros.size() == 0);
    REQUIRE(poles.size() == 6);

    const std::vector<std::complex<double>> expectedPoles = {
        { -0.53855268, 0.96168769 },
        { -0.79965419, 0.56217173 },
        { -0.90939068, 0.18569644 },
        { -0.90939068, -0.18569644 },
        { -0.79965419, -0.56217173 },
        { -0.53855268, -0.96168769 },
    };
    requireApproxEqual(poles, expectedPoles);
}

TEST_CASE("besselPrototype - Seventh Order", "[besselPrototype]")
{
    // >>> from scipy import signal
    // >>> signal.besselap(7)
    const auto result = besselPrototype(7);
    REQUIRE(result.getGain() == Catch::Approx(1.0));

    const auto zeros = result.getZeros();
    const auto poles = result.getPoles();
    REQUIRE(zeros.size() == 0);
    REQUIRE(poles.size() == 7);

    const std::vector<std::complex<double>> expectedPoles = {
        { -0.49669173, 1.00250851 },
        { -0.75273554, 0.65046963 },
        { -0.88000293, 0.32166528 },
        { -0.91948716, -0. },
        { -0.88000293, -0.32166528 },
        { -0.75273554, -0.65046963 },
        { -0.49669173, -1.00250851 },
    };
    requireApproxEqual(poles, expectedPoles);
}

TEST_CASE("legendrePrototype - Third Order", "[legendrePrototype]")
{
    // Legendre "Optimum-L" filter poles
    const auto result = legendrePrototype(3);
    REQUIRE(result.getGain() == Catch::Approx(1.0));

    const auto zeros = result.getZeros();
    const auto poles = result.getPoles();
    REQUIRE(zeros.size() == 0);
    REQUIRE(poles.size() == 3);

    const std::vector<std::complex<double>> expectedPoles = {
        { -0.345185619, 0.9008656355 },
        { -0.345185619, -0.9008656355 },
        { -0.6203318171, 0.0 },
    };
    requireApproxEqual(poles, expectedPoles);
}

TEST_CASE("legendrePrototype - Fourth Order", "[legendrePrototype]")
{
    // Legendre "Optimum-L" filter poles
    const auto result = legendrePrototype(4);
    REQUIRE(result.getGain() == Catch::Approx(1.0));

    const auto zeros = result.getZeros();
    const auto poles = result.getPoles();
    REQUIRE(zeros.size() == 0);
    REQUIRE(poles.size() == 4);

    const std::vector<std::complex<double>> expectedPoles = {
        { -0.2316887227, -0.9455106639 },
        { -0.2316887227, 0.9455106639 },
        { -0.5497434238, 0.3585718162 },
        { -0.5497434238, -0.3585718162 },
    };
    requireApproxEqual(poles, expectedPoles);
}

TEST_CASE("legendrePrototype - Fifth Order", "[legendrePrototype]")
{
    // Legendre "Optimum-L" filter poles
    const auto result = legendrePrototype(5);
    REQUIRE(result.getGain() == Catch::Approx(1.0));

    const auto zeros = result.getZeros();
    const auto poles = result.getPoles();
    REQUIRE(zeros.size() == 0);
    REQUIRE(poles.size() == 5);

    const std::vector<std::complex<double>> expectedPoles = {
        { -0.1535867376, 0.9681464078 },
        { -0.1535867376, -0.9681464078 },
        { -0.3881398518, 0.5886323381 },
        { -0.3881398518, -0.5886323381 },
        { -0.4680898756, 0.0 },
    };
    requireApproxEqual(poles, expectedPoles);
}

TEST_CASE("lowpassToLowpass - scales cutoff frequency", "[lowpassToLowpass]")
{
    // >>> from scipy.signal import lp2lp_zpk
    // >>> z = [ 7, 2 ]
    // >>> p = [ 5, 13 ]
    // >>> k = 0.8
    // >>> wo = 0.4
    // >>> lp2lp_zpk(z, p, k, wo)
    const auto input = Zpk({ 7, 2 }, { 5, 13 }, 0.8);
    const auto result = lowpassToLowpass(input, 0.4);
    const auto zeros = result.getZeros();
    const auto poles = result.getPoles();

    const std::vector<std::complex<double>> expectedZeros = {
        { 2.8 },
        { 0.8 },
    };

    const std::vector<std::complex<double>> expectedPoles = {
        { 2.0 },
        { 5.2 },
    };

    requireApproxEqual(zeros, expectedZeros);
    requireApproxEqual(poles, expectedPoles);
    REQUIRE(result.getGain() == Catch::Approx(0.8));
}

TEST_CASE("lowpassToHighpass - complex zeros and real poles", "[lowpassToHighpass]")
{
    // >>> from scipy.signal import lp2hp_zpk
    // >>> z   = [ -2 + 3j ,  -0.5 - 0.8j ]
    // >>> p   = [ -1      ,  -4          ]
    // >>> k   = 10
    // >>> wo  = 0.6
    // >>> lp2hp_zpk(z, p, k, wo)
    const auto input = Zpk(
        { { -2, 3 }, { -0.5, -0.8 } },
        { -1, -4 },
        10);
    const auto result = lowpassToHighpass(input, 0.6);
    const auto zeros = result.getZeros();
    const auto poles = result.getPoles();

    const std::vector<std::complex<double>> expectedZeros = {
        { -0.0923076923076923, -0.13846153846153847 },
        { -0.33707865168539325, 0.5393258426966292 },
    };

    const std::vector<std::complex<double>> expectedPoles = {
        { -0.6 },
        { -0.15 },
    };

    requireApproxEqual(zeros, expectedZeros);
    requireApproxEqual(poles, expectedPoles);
    REQUIRE(result.getGain() == Catch::Approx(8.5));
}

TEST_CASE("lowpassToBandpass - conjugate pair doubles poles and zeros", "[lowpassToBandpass]")
{
    // >>> from scipy.signal import lp2bp_zpk
    // >>> z   = [ 5 + 2j ,  5 - 2j ]
    // >>> p   = [ 7      ,  -16    ]
    // >>> k   = 0.8
    // >>> wo  = 0.62
    // >>> bw  = 15
    // >>> lp2bp_zpk(z, p, k, wo, bw)
    const auto input = Zpk(
        { { 5.0, 2.0 }, { 5.0, -2.0 } },
        { 7, -16 },
        0.8);
    const auto result = lowpassToBandpass(input, 0.62, 15);
    const auto zeros = result.getZeros();
    const auto poles = result.getPoles();

    const std::vector<std::complex<double>> expectedZeros = {
        { 74.9955814925219, 30.00176761126334 },
        { 74.9955814925219, -30.00176761126334 },
        { 0.004418507478114009, 0.001767611263341351 },
        { 0.004418507478114009, -0.001767611263341351 },
    };

    const std::vector<std::complex<double>> expectedPoles = {
        { 104.9963389199666 },
        { -0.0016016773557083752 },
        { 0.003661080033396047 },
        { -239.99839832264428 },
    };

    requireApproxEqual(zeros, expectedZeros);
    requireApproxEqual(poles, expectedPoles);
    REQUIRE(result.getGain() == Catch::Approx(0.8));
}

TEST_CASE("lowpassToBandstop - all-pole filter adds notch zeros", "[lowpassToBandstop]")
{
    // >>> from scipy.signal import lp2bs_zpk
    // >>> z   = [             ]
    // >>> p   = [ 0.7 ,    -1 ]
    // >>> k   = 9
    // >>> wo  = 0.5
    // >>> bw  = 10
    // >>> lp2bs_zpk(z, p, k, wo, bw)
    const auto input = Zpk(
        {},
        { 0.7, -1 },
        9);
    const auto result = lowpassToBandstop(input, 0.5, 10);
    const auto zeros = result.getZeros();
    const auto poles = result.getPoles();

    const std::vector<std::complex<double>> expectedZeros = {
        { 0.0, 0.5 },
        { 0.0, 0.5 },
        { 0.0, -0.5 },
        { 0.0, -0.5 },
    };

    const std::vector<std::complex<double>> expectedPoles = {
        { 14.268192795531009 },
        { -0.025062814466900285 },
        { 0.01752149018327742 },
        { -9.9749371855331 },
    };

    requireApproxEqual(zeros, expectedZeros);
    requireApproxEqual(poles, expectedPoles);
    REQUIRE(result.getGain() == Catch::Approx(-12.857142857142858));
}

TEST_CASE("prewarpFrequency - 1kHz at 48kHz", "[prewarpFrequency]")
{
    // >>> fs = 48000; fc = 1000
    // >>> 2 * fs * np.tan(np.pi * fc / fs)
    REQUIRE(prewarpFrequency(1000, 48000) == Catch::Approx(6292.172430262869));
}

TEST_CASE("prewarpFrequency - 5kHz at 44.1kHz", "[prewarpFrequency]")
{
    // >>> fs = 44100; fc = 5000
    // >>> 2 * fs * np.tan(np.pi * fc / fs)
    REQUIRE(prewarpFrequency(5000, 44100) == Catch::Approx(32815.591119406854));
}

TEST_CASE("prewarpFrequency - 100Hz at 96kHz", "[prewarpFrequency]")
{
    // >>> fs = 96000; fc = 100
    // >>> 2 * fs * np.tan(np.pi * fc / fs)
    REQUIRE(prewarpFrequency(100, 96000) == Catch::Approx(628.3207736584609));
}

TEST_CASE("bilinearTransform - Butterworth 4th order lowpass", "[bilinearTransform]")
{
    // >>> from scipy import signal
    // >>> z, p, k = signal.buttap(4)
    // >>> fs = 48000; fc = 1000
    // >>> wc = 2 * fs * np.tan(np.pi * fc / fs)
    // >>> z_s, p_s, k_s = signal.lp2lp_zpk(z, p, k, wc)
    // >>> signal.bilinear_zpk(z_s, p_s, k_s, fs)
    const std::vector<std::complex<double>> analogZeros = {};
    const std::vector<std::complex<double>> analogPoles = {
        { -2407.91014265, 5813.20932335 },
        { -5813.20932335, 2407.91014265 },
        { -5813.20932335, -2407.91014265 },
        { -2407.91014265, -5813.20932335 },
    };
    const double analogGain = 1567481637637286.5;

    const Zpk analog(analogZeros, analogPoles, analogGain);
    const auto result = bilinearTransform(analog, 48000);

    const std::vector<std::complex<double>> expectedZeros = {
        { -1.0, 0.0 },
        { -1.0, 0.0 },
        { -1.0, 0.0 },
        { -1.0, 0.0 },
    };
    const std::vector<std::complex<double>> expectedPoles = {
        { 0.94427798, 0.11485352 },
        { 0.88475217, 0.0445749 },
        { 0.88475217, -0.0445749 },
        { 0.94427798, -0.11485352 },
    };

    requireApproxEqual(result.getZeros(), expectedZeros);
    requireApproxEqual(result.getPoles(), expectedPoles);
    REQUIRE(result.getGain() == Catch::Approx(1.555172178089176e-05));
}

TEST_CASE("bilinearTransform - Chebyshev II 3rd order with zeros", "[bilinearTransform]")
{
    // >>> from scipy import signal
    // >>> z, p, k = signal.cheb2ap(3, 40)
    // >>> fs = 44100; fc = 2000
    // >>> wc = 2 * fs * np.tan(np.pi * fc / fs)
    // >>> z_s, p_s, k_s = signal.lp2lp_zpk(z, p, k, wc)
    // >>> signal.bilinear_zpk(z_s, p_s, k_s, fs)
    const std::vector<std::complex<double>> analogZeros = {
        { 0.0, -14609.38270555 },
        { 0.0, 14609.38270555 },
    };
    const std::vector<std::complex<double>> analogPoles = {
        { -2038.87277885, -3744.17480025 },
        { -4457.32743398, 0.0 },
        { -2038.87277885, 3744.17480025 },
    };
    const double analogGain = 379.58187626671537;

    const Zpk analog(analogZeros, analogPoles, analogGain);
    const auto result = bilinearTransform(analog, 44100);

    const std::vector<std::complex<double>> expectedZeros = {
        { 0.94659258, -0.32243215 },
        { 0.94659258, 0.32243215 },
        { -1.0, 0.0 },
    };
    const std::vector<std::complex<double>> expectedPoles = {
        { 0.95145209, -0.08096929 },
        { 0.90378899, 0.0 },
        { 0.95145209, 0.08096929 },
    };

    requireApproxEqual(result.getZeros(), expectedZeros);
    requireApproxEqual(result.getPoles(), expectedPoles);
    REQUIRE(result.getGain() == Catch::Approx(0.00401405623566463));
}

TEST_CASE("bilinearTransform - First order Butterworth", "[bilinearTransform]")
{
    // >>> from scipy import signal
    // >>> z, p, k = signal.buttap(1)
    // >>> fs = 48000; fc = 5000
    // >>> wc = 2 * fs * np.tan(np.pi * fc / fs)
    // >>> z_s, p_s, k_s = signal.lp2lp_zpk(z, p, k, wc)
    // >>> signal.bilinear_zpk(z_s, p_s, k_s, fs)
    const std::vector<std::complex<double>> analogZeros = {};
    const std::vector<std::complex<double>> analogPoles = {
        { -32587.60885088, 0.0 },
    };
    const double analogGain = 32587.60885088408;

    const Zpk analog(analogZeros, analogPoles, analogGain);
    const auto result = bilinearTransform(analog, 48000);

    const std::vector<std::complex<double>> expectedZeros = {
        { -1.0, 0.0 },
    };
    const std::vector<std::complex<double>> expectedPoles = {
        { 0.49314543, 0.0 },
    };

    requireApproxEqual(result.getZeros(), expectedZeros);
    requireApproxEqual(result.getPoles(), expectedPoles);
    REQUIRE(result.getGain() == Catch::Approx(0.25342728698434797));
}

TEST_CASE("freqsZpk - Butterworth 2nd order prototype", "[freqsZpk]")
{
    // >>> from scipy import signal
    // >>> z, p, k = signal.buttap(2)
    // >>> signal.freqs_zpk(z, p, k, [0.1, 1.0, 10.0])
    const Zpk zpk(
        {},
        { { -0.70710678, 0.70710678 }, { -0.70710678, -0.70710678 } },
        1.0);

    auto h1 = freqsZpk(zpk, 0.1);
    REQUIRE(h1.real() == Catch::Approx(0.9899010098990102));
    REQUIRE(h1.imag() == Catch::Approx(-0.1414072155157579));

    auto h2 = freqsZpk(zpk, 1.0);
    REQUIRE(h2.real() == Catch::Approx(0.0).margin(TOLERANCE));
    REQUIRE(h2.imag() == Catch::Approx(-0.7071067811865475));

    auto h3 = freqsZpk(zpk, 10.0);
    REQUIRE(h3.real() == Catch::Approx(-0.0098990100989901));
    REQUIRE(h3.imag() == Catch::Approx(-0.0014140721551575794));
}

TEST_CASE("freqsZpk - Chebyshev II 3rd order with zeros", "[freqsZpk]")
{
    // >>> from scipy import signal
    // >>> z, p, k = signal.cheb2ap(3, 40)
    // >>> signal.freqs_zpk(z, p, k, [0.5, 1.0, 2.0])
    const Zpk zpk(
        { { 0.0, -1.15470054 }, { 0.0, 1.15470054 } },
        { { -0.16114901, -0.29593315 }, { -0.35229951, 0.0 }, { -0.16114901, 0.29593315 } },
        0.03000150011250936);

    auto h1 = freqsZpk(zpk, 0.5);
    REQUIRE(h1.real() == Catch::Approx(-0.2506540250421875));
    REQUIRE(h1.imag() == Catch::Approx(0.022317855233752404));

    auto h2 = freqsZpk(zpk, 1.0);
    REQUIRE(h2.real() == Catch::Approx(-0.006345637917933374));
    REQUIRE(h2.imag() == Catch::Approx(0.007728704898913293));

    auto h3 = freqsZpk(zpk, 2.0);
    REQUIRE(h3.real() == Catch::Approx(0.003322818958966683));
    REQUIRE(h3.imag() == Catch::Approx(-0.009431801215352847));
}

TEST_CASE("freqzSos - single biquad section", "[freqzSos]")
{
    // >>> from scipy import signal
    // >>> sos = np.array([[0.25, 0.5, 0.25, 1.0, -0.5, 0.1]])
    // >>> signal.sosfreqz(sos, worN=[0.0, 0.5, 1.0, 2.0, np.pi])
    BiquadCoefficients coef;
    coef.a0 = 0.25;
    coef.a1 = 0.5;
    coef.a2 = 0.25;
    coef.b1 = -0.5;
    coef.b2 = 0.1;

    auto h1 = freqzSos(coef, 0.0);
    REQUIRE(h1.real() == Catch::Approx(1.6666666666666667));
    REQUIRE(h1.imag() == Catch::Approx(0.0).margin(TOLERANCE));

    auto h2 = freqzSos(coef, 0.5);
    REQUIRE(h2.real() == Catch::Approx(1.0847692934080337));
    REQUIRE(h2.imag() == Catch::Approx(-1.005842329520787));

    auto h3 = freqzSos(coef, 1.0);
    REQUIRE(h3.real() == Catch::Approx(0.12473482105570932));
    REQUIRE(h3.imag() == Catch::Approx(-1.0014006088698293));

    auto h4 = freqzSos(coef, 2.0);
    REQUIRE(h4.real() == Catch::Approx(-0.1761753426905888));
    REQUIRE(h4.imag() == Catch::Approx(-0.15053455809169503));

    auto h5 = freqzSos(coef, M_PI);
    REQUIRE(h5.real() == Catch::Approx(0.0).margin(TOLERANCE));
    REQUIRE(h5.imag() == Catch::Approx(0.0).margin(TOLERANCE));
}

TEST_CASE("freqzSos - cascade of biquads", "[freqzSos]")
{
    // >>> from scipy import signal
    // >>> sos = signal.butter(4, 0.2, output='sos')
    // >>> signal.sosfreqz(sos, worN=[0.0, 0.1*np.pi, 0.2*np.pi, 0.5*np.pi])
    std::vector<BiquadCoefficients> sos(2);

    sos[0].a0 = 0.0048243434;
    sos[0].a1 = 0.0096486867;
    sos[0].a2 = 0.0048243434;
    sos[0].b1 = -1.0485995764;
    sos[0].b2 = 0.2961403576;

    sos[1].a0 = 1.0;
    sos[1].a1 = 2.0;
    sos[1].a2 = 1.0;
    sos[1].b1 = -1.3209134308;
    sos[1].b2 = 0.6327387929;

    auto h1 = freqzSos(sos, 0.0);
    REQUIRE(h1.real() == Catch::Approx(1.0));
    REQUIRE(h1.imag() == Catch::Approx(0.0).margin(TOLERANCE));

    auto h2 = freqzSos(sos, 0.1 * M_PI);
    REQUIRE(h2.real() == Catch::Approx(0.2444148353123437));
    REQUIRE(h2.imag() == Catch::Approx(-0.9680308428253678));

    auto h3 = freqzSos(sos, 0.2 * M_PI);
    REQUIRE(h3.real() == Catch::Approx(-0.7071067811865477));
    REQUIRE(h3.imag() == Catch::Approx(0.0).margin(TOLERANCE));

    auto h4 = freqzSos(sos, 0.5 * M_PI);
    REQUIRE(h4.real() == Catch::Approx(0.007251524968304738));
    REQUIRE(h4.imag() == Catch::Approx(0.008463141045463913));
}

TEST_CASE("freqsZpk - vector of frequencies", "[freqsZpk]")
{
    // Same as single-frequency test but using vector overload
    const Zpk zpk(
        {},
        { { -0.70710678, 0.70710678 }, { -0.70710678, -0.70710678 } },
        1.0);

    std::vector<double> omega = { 0.1, 1.0, 10.0 };
    auto H = freqsZpk(zpk, omega);

    REQUIRE(H.size() == 3);
    REQUIRE(H[0].real() == Catch::Approx(0.9899010098990102));
    REQUIRE(H[0].imag() == Catch::Approx(-0.1414072155157579));
    REQUIRE(H[1].real() == Catch::Approx(0.0).margin(TOLERANCE));
    REQUIRE(H[1].imag() == Catch::Approx(-0.7071067811865475));
    REQUIRE(H[2].real() == Catch::Approx(-0.0098990100989901));
    REQUIRE(H[2].imag() == Catch::Approx(-0.0014140721551575794));
}

TEST_CASE("freqzSos - vector of frequencies", "[freqzSos]")
{
    // Same as cascade test but using vector overload
    std::vector<BiquadCoefficients> sos(2);

    sos[0].a0 = 0.0048243434;
    sos[0].a1 = 0.0096486867;
    sos[0].a2 = 0.0048243434;
    sos[0].b1 = -1.0485995764;
    sos[0].b2 = 0.2961403576;

    sos[1].a0 = 1.0;
    sos[1].a1 = 2.0;
    sos[1].a2 = 1.0;
    sos[1].b1 = -1.3209134308;
    sos[1].b2 = 0.6327387929;

    std::vector<double> w = { 0.0, 0.1 * M_PI, 0.2 * M_PI, 0.5 * M_PI };
    auto H = freqzSos(sos, w);

    REQUIRE(H.size() == 4);
    REQUIRE(H[0].real() == Catch::Approx(1.0));
    REQUIRE(H[0].imag() == Catch::Approx(0.0).margin(TOLERANCE));
    REQUIRE(H[1].real() == Catch::Approx(0.2444148353123437));
    REQUIRE(H[1].imag() == Catch::Approx(-0.9680308428253678));
    REQUIRE(H[2].real() == Catch::Approx(-0.7071067811865477));
    REQUIRE(H[2].imag() == Catch::Approx(0.0).margin(TOLERANCE));
    REQUIRE(H[3].real() == Catch::Approx(0.007251524968304738));
    REQUIRE(H[3].imag() == Catch::Approx(0.008463141045463913));
}
