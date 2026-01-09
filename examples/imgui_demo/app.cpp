#include "app.hpp"
#include "imgui.h"
#include "implot.h"
#include <algorithm>
#include <cmath>

namespace demo {

App::App()
{
    updateFilter();
}

void App::render()
{
    ImGuiViewport* viewport = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(viewport->WorkPos);
    ImGui::SetNextWindowSize(viewport->WorkSize);

    ImGuiWindowFlags flags = ImGuiWindowFlags_NoDecoration |
                             ImGuiWindowFlags_NoMove |
                             ImGuiWindowFlags_NoSavedSettings;

    ImGui::Begin("IIR Filter Visualization", nullptr, flags);

    ImGui::BeginChild("ControlPanel", ImVec2(300, 0), ImGuiChildFlags_Borders);
    renderControlPanel();
    ImGui::EndChild();

    ImGui::SameLine();

    ImGui::BeginChild("PlotPanel", ImVec2(0, 0), ImGuiChildFlags_None);

    float plotHeight = (ImGui::GetContentRegionAvail().y - 10) / 2;
    ImGui::BeginChild("FreqResponse", ImVec2(0, plotHeight), ImGuiChildFlags_Borders);
    renderFrequencyResponsePlot();
    ImGui::EndChild();

    ImGui::BeginChild("PoleZero", ImVec2(0, 0), ImGuiChildFlags_Borders);
    renderPoleZeroPlot();
    ImGui::EndChild();

    ImGui::EndChild();

    ImGui::End();

    if (m_needsUpdate) {
        updateFilter();
        m_needsUpdate = false;
    }
}

void App::renderControlPanel()
{
    ImGui::Text("Filter Parameters");
    ImGui::Separator();

    const char* filterTypes[] = {"Butterworth", "Chebyshev I", "Chebyshev II",
                                 "Elliptic", "Bessel", "Legendre"};
    int currentType = static_cast<int>(m_params.type);
    if (ImGui::Combo("Filter Type", &currentType, filterTypes, 6)) {
        m_params.type = static_cast<FilterType>(currentType);
        m_needsUpdate = true;
    }

    const char* transformTypes[] = {"Lowpass", "Highpass", "Bandpass", "Bandstop"};
    int currentTransform = static_cast<int>(m_params.transform);
    if (ImGui::Combo("Transform", &currentTransform, transformTypes, 4)) {
        m_params.transform = static_cast<TransformType>(currentTransform);
        m_needsUpdate = true;
    }

    if (ImGui::SliderInt("Order", &m_params.order, 1, 12)) {
        m_needsUpdate = true;
    }

    float sampleRate = static_cast<float>(m_params.sampleRate);
    if (ImGui::InputFloat("Sample Rate (Hz)", &sampleRate, 1000, 10000, "%.0f")) {
        m_params.sampleRate = std::max(1000.0, static_cast<double>(sampleRate));
        m_needsUpdate = true;
    }

    ImGui::Separator();
    ImGui::Text("Frequency Settings");

    bool isBandFilter = (m_params.transform == TransformType::Bandpass ||
                         m_params.transform == TransformType::Bandstop);

    if (!isBandFilter) {
        float maxCutoff = static_cast<float>(m_params.sampleRate / 2.0 * 0.99);
        float cutoff = static_cast<float>(m_params.cutoffFreq);
        if (ImGui::SliderFloat("Cutoff (Hz)", &cutoff, 10.0f, maxCutoff, "%.1f",
                               ImGuiSliderFlags_Logarithmic)) {
            m_params.cutoffFreq = cutoff;
            m_needsUpdate = true;
        }
    } else {
        float maxFreq = static_cast<float>(m_params.sampleRate / 2.0 * 0.99);
        float center = static_cast<float>(m_params.centerFreq);
        float bw = static_cast<float>(m_params.bandwidth);

        if (ImGui::SliderFloat("Center (Hz)", &center, 10.0f, maxFreq, "%.1f",
                               ImGuiSliderFlags_Logarithmic)) {
            m_params.centerFreq = center;
            m_needsUpdate = true;
        }
        if (ImGui::SliderFloat("Bandwidth (Hz)", &bw, 10.0f, maxFreq, "%.1f",
                               ImGuiSliderFlags_Logarithmic)) {
            m_params.bandwidth = bw;
            m_needsUpdate = true;
        }
    }

    ImGui::Separator();
    ImGui::Text("Ripple Parameters");

    bool needsPassbandRipple =
        (m_params.type == FilterType::Chebyshev1 ||
         m_params.type == FilterType::Elliptic);
    bool needsStopbandAtten =
        (m_params.type == FilterType::Chebyshev2 ||
         m_params.type == FilterType::Elliptic);

    if (needsPassbandRipple) {
        float ripple = static_cast<float>(m_params.passbandRipple);
        if (ImGui::SliderFloat("Passband Ripple (dB)", &ripple, 0.01f, 6.0f, "%.2f")) {
            m_params.passbandRipple = ripple;
            m_needsUpdate = true;
        }
    }

    if (needsStopbandAtten) {
        float atten = static_cast<float>(m_params.stopbandAttenuation);
        if (ImGui::SliderFloat("Stopband Atten (dB)", &atten, 6.0f, 120.0f, "%.1f")) {
            m_params.stopbandAttenuation = atten;
            m_needsUpdate = true;
        }
    }

    ImGui::Separator();
    ImGui::Text("Display Options");

    ImGui::Checkbox("Show Analog Response", &m_showAnalog);
    ImGui::Checkbox("Show Digital Response", &m_showDigital);
    if (ImGui::Checkbox("Logarithmic Frequency", &m_logFreqScale)) {
        m_needsUpdate = true;
    }

    ImGui::Separator();
    renderFilterInfo();
}

void App::renderFrequencyResponsePlot()
{
    ImGui::Text("Frequency Response");

    if (ImPlot::BeginPlot("##FreqResponse", ImVec2(-1, -1))) {
        ImPlot::SetupAxes("Frequency (Hz)", "Magnitude (dB)");
        if (m_logFreqScale) {
            ImPlot::SetupAxisScale(ImAxis_X1, ImPlotScale_Log10);
        }
        ImPlot::SetupAxisLimits(ImAxis_Y1, -100, 10, ImPlotCond_Once);

        if (m_showDigital && !m_digitalResponse.frequencies.empty()) {
            ImPlot::PlotLine("Digital", m_digitalResponse.frequencies.data(),
                             m_digitalResponse.magnitudeDb.data(),
                             static_cast<int>(m_digitalResponse.frequencies.size()));
        }

        if (m_showAnalog && !m_analogResponse.frequencies.empty()) {
            std::vector<double> freqHz(m_analogResponse.frequencies.size());
            for (size_t i = 0; i < freqHz.size(); ++i) {
                freqHz[i] = m_analogResponse.frequencies[i] / (2.0 * M_PI);
            }
            ImPlot::PlotLine("Analog", freqHz.data(),
                             m_analogResponse.magnitudeDb.data(),
                             static_cast<int>(freqHz.size()));
        }

        ImPlot::EndPlot();
    }
}

void App::renderPoleZeroPlot()
{
    ImGui::Text("Pole-Zero Plot");

    float width = (ImGui::GetContentRegionAvail().x - 10) / 2;

    ImGui::BeginChild("AnalogPZ", ImVec2(width, -1));
    ImGui::Text("S-Plane (Analog)");

    if (ImPlot::BeginPlot("##SPlane", ImVec2(-1, -1), ImPlotFlags_Equal)) {
        ImPlot::SetupAxes("Real", "Imag");

        auto& zpk = m_filter.analogTransformed;
        auto zeros = zpk.getZeros();
        auto poles = zpk.getPoles();

        if (!zeros.empty()) {
            std::vector<double> zRe, zIm;
            for (const auto& z : zeros) {
                zRe.push_back(z.real());
                zIm.push_back(z.imag());
            }
            ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle, 8);
            ImPlot::PlotScatter("Zeros", zRe.data(), zIm.data(),
                                static_cast<int>(zRe.size()));
        }

        if (!poles.empty()) {
            std::vector<double> pRe, pIm;
            for (const auto& p : poles) {
                pRe.push_back(p.real());
                pIm.push_back(p.imag());
            }
            ImPlot::SetNextMarkerStyle(ImPlotMarker_Cross, 8);
            ImPlot::PlotScatter("Poles", pRe.data(), pIm.data(),
                                static_cast<int>(pRe.size()));
        }

        double yline[] = {-100000, 100000};
        double xline[] = {0, 0};
        ImPlot::PlotLine("##jw", xline, yline, 2);

        ImPlot::EndPlot();
    }
    ImGui::EndChild();

    ImGui::SameLine();

    ImGui::BeginChild("DigitalPZ", ImVec2(0, -1));
    ImGui::Text("Z-Plane (Digital)");

    if (ImPlot::BeginPlot("##ZPlane", ImVec2(-1, -1), ImPlotFlags_Equal)) {
        ImPlot::SetupAxes("Real", "Imag");
        ImPlot::SetupAxisLimits(ImAxis_X1, -2, 2, ImPlotCond_Once);
        ImPlot::SetupAxisLimits(ImAxis_Y1, -2, 2, ImPlotCond_Once);

        std::vector<double> circleX, circleY;
        for (int i = 0; i <= 100; ++i) {
            double theta = 2.0 * M_PI * i / 100.0;
            circleX.push_back(std::cos(theta));
            circleY.push_back(std::sin(theta));
        }
        ImPlot::PlotLine("##UnitCircle", circleX.data(), circleY.data(),
                         static_cast<int>(circleX.size()));

        auto& zpk = m_filter.digitalZpk;
        auto zeros = zpk.getZeros();
        auto poles = zpk.getPoles();

        if (!zeros.empty()) {
            std::vector<double> zRe, zIm;
            for (const auto& z : zeros) {
                zRe.push_back(z.real());
                zIm.push_back(z.imag());
            }
            ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle, 8);
            ImPlot::PlotScatter("Zeros", zRe.data(), zIm.data(),
                                static_cast<int>(zRe.size()));
        }

        if (!poles.empty()) {
            std::vector<double> pRe, pIm;
            for (const auto& p : poles) {
                pRe.push_back(p.real());
                pIm.push_back(p.imag());
            }
            ImPlot::SetNextMarkerStyle(ImPlotMarker_Cross, 8);
            ImPlot::PlotScatter("Poles", pRe.data(), pIm.data(),
                                static_cast<int>(pRe.size()));
        }

        ImPlot::EndPlot();
    }
    ImGui::EndChild();
}

void App::renderFilterInfo()
{
    ImGui::Text("Filter Info");

    auto& zpk = m_filter.digitalZpk;
    ImGui::Text("Digital Poles: %zu", zpk.getPoles().size());
    ImGui::Text("Digital Zeros: %zu", zpk.getZeros().size());
    ImGui::Text("SOS Sections: %zu", m_filter.sos.size());

    bool stable = true;
    for (const auto& p : zpk.getPoles()) {
        if (std::abs(p) >= 1.0) {
            stable = false;
            break;
        }
    }

    if (stable) {
        ImGui::TextColored(ImVec4(0, 1, 0, 1), "Filter is STABLE");
    } else {
        ImGui::TextColored(ImVec4(1, 0, 0, 1), "Filter is UNSTABLE!");
    }
}

void App::updateFilter()
{
    m_filter = designFilter(m_params);

    if (!m_filter.valid) {
        return;
    }

    double nyquist = m_params.sampleRate / 2.0;
    double omegaMin = 2.0 * M_PI * 1.0;
    double omegaMax = 2.0 * M_PI * nyquist;

    m_analogResponse = computeAnalogFrequencyResponse(
        m_filter.analogTransformed.getZeros(),
        m_filter.analogTransformed.getPoles(),
        m_filter.analogTransformed.getGain(), omegaMin, omegaMax,
        static_cast<size_t>(m_responsePoints), m_logFreqScale);

    m_digitalResponse = computeDigitalFrequencyResponse(
        m_filter.sos, m_params.sampleRate,
        static_cast<size_t>(m_responsePoints), m_logFreqScale);
}

} // namespace demo
