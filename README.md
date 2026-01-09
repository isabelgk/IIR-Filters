# IIR-Filters

A C++17 library implementing Infinite Impulse Response (IIR) digital filters.

This is primarily a personal learning experiment for understanding IIR digital filters.
This project focuses on accuracy and compatibility with reference implementations from SciPy and Vinnie
Falco's [DSPFilters](https://github.com/vinniefalco/DSPFilters) project.
Correctness and readability are primary goals, while being the fastest implementation is not.

Aside from dependencies for tests and examples, this project uses only the C++ standard library. Dependencies are
managed using CMake `FetchContent`.

## Features

- **Prototype filters**: Butterworth, Chebyshev Type I/II, Elliptic, Bessel, and Legendre (Optimum-L)
- **Frequency transformations**: Lowpass, highpass, bandpass, and bandstop
- **Filter representations**: Zero-Pole-Gain (Zpk), second-order sections (SOS), and biquad cascades
- **Mathematical utilities**: Elliptic integrals, polynomial root finding, and complex number handling

## Building

Currently only MacOS has been tested, but Windows and Linux should also be supported.

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -G Ninja
cmake --build build
```

Run tests:

```bash
./build/test/tests
```

Build and run UI example:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_DEMO=ON
cmake --build build
./build/examples/imgui_demo/imgui_demo
```

## Example Usage

```cpp
#include "filters.hpp"

using namespace iirfilters;

// Design a 4th-order Butterworth lowpass filter at 1kHz (48kHz sample rate)
auto zpk = butterworthPrototype(4);
zpk = lowpassToLowpass(zpk, 1000.0, 48000.0);
auto sos = zpk.toSos();
Cascade filter(sos);

// Process samples
double output = filter.process(inputSample);
```

## See also

The following codebases were essential resources in creating this library.

- [DSPFilters](https://github.com/vinniefalco/DSPFilters)
- [Netlib Cephes](http://www.netlib.org/cephes/) - Elliptic integral implementations
- [scipy.signal filter design](https://github.com/scipy/scipy/blob/v1.16.2/scipy/signal/_filter_design.py) - Primary
  reference and model for filter algorithms
