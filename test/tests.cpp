#include "filters.hpp"
#include "math_supplement.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

using namespace iirfilters;

namespace {

constexpr double TOLERANCE = 1e-7;

bool contains(const std::vector<std::complex<double>>& x, const std::complex<double>& element)
{
    for (auto& el : x) {
        if (std::abs(el.real() - element.real()) < TOLERANCE
            && std::abs(el.imag() - element.imag()) < TOLERANCE) {
            return true;
        }
    }
    return false;
}

void requireApproxEqual(const std::vector<double>& a, const std::vector<double>& b)
{
    REQUIRE(a.size() == b.size());
    for (auto i = 0; i < a.size(); i++) {
        REQUIRE(std::abs(a[i] - b[i]) < TOLERANCE);
    }
}

void requireApproxEqual(const std::vector<std::complex<double>>& a, const std::vector<std::complex<double>>& b)
{
    REQUIRE(a.size() == b.size());
    for (auto& el : a) {
        REQUIRE(contains(b, el));
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
    // >>> signal.butter(7, 0.71)
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

TEST_CASE("ellipk - 0.6", "[ellipk]")
{
    REQUIRE(math::ellipk(0.6) == Catch::Approx(1.9495677498060258));
}

TEST_CASE("ellipkm1 - basic test", "[ellipkm1]")
{
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

TEST_CASE("ellipj - u = 4, m = 0.2", "[ellipj]")
{
    double sn, cn, dn, ph;
    math::ellipj(4, 0.2, sn, cn, dn, ph);
    REQUIRE(sn == Catch::Approx(-0.6219243502072862));
    REQUIRE(cn == Catch::Approx(-0.7830773286331593));
    REQUIRE(dn == Catch::Approx(0.9605425657012028));
    REQUIRE(ph == Catch::Approx(3.8127903862602888));
}

TEST_CASE("ellipdeg - basic test 1", "[ellipdeg]")
{
    REQUIRE(math::solveDegreeEquation(2, 0.5) == Catch::Approx(0.9705627484771397));
}

TEST_CASE("ellipdeg - basic test 2", "[ellipdeg]")
{
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

TEST_CASE("reverseBesselPolynomial", "[reverseBesselPolynomial]")
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

TEST_CASE("evaluatePolynomial - degree 2", "[evaluatePolynomial]")
{
    std::vector<double> coef = {
        3.5, 1.0, 2.0
    };
    std::complex<double> x(1.0, 0.5);
    const auto result = math::evaluatePolynomial(coef, x);
    REQUIRE(result.imag() == Catch::Approx(2.5));
    REQUIRE(result.real() == Catch::Approx(6.0));
}

TEST_CASE("evaluatePolynomial - degree 6", "[evaluatePolynomial]")
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

TEST_CASE("lowpassToLowpass - basic test", "[lowpassToLowpass]")
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

TEST_CASE("lowpassToHighpass - basic test", "[lowpassToHighpass]")
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

TEST_CASE("lowpassToBandpass - basic test", "[lowpassToBandpass]")
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

TEST_CASE("lowpassToBandstop - basic test", "[lowpassToBandstop]")
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
