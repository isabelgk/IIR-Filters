#ifndef APP_H
#define APP_H

#include "filter_designer.hpp"
#include "frequency_response.hpp"

namespace demo {

class App
{
  public:
    App();
    void render();

  private:
    void renderControlPanel();
    void renderFrequencyResponsePlot();
    void renderPoleZeroPlot();
    void renderFilterInfo();
    void updateFilter();

    FilterParameters m_params;
    DesignedFilter m_filter;
    FrequencyResponseData m_analogResponse;
    FrequencyResponseData m_digitalResponse;
    bool m_needsUpdate = true;

    bool m_showAnalog = true;
    bool m_showDigital = true;
    bool m_logFreqScale = true;
    int m_responsePoints = 1000;
};

} // namespace demo

#endif
