#include "DataStorage.h"
#include <fstream>
#include <iostream>
#include <chrono>
#include <ctime>

void writeToCsv(float sma200, float ema80, float stochasticCalculation, bool buySignal) {
    auto now = std::chrono::system_clock::now();
    std::time_t currentTime = std::chrono::system_clock::to_time_t(now);
    std::tm localTime;
    localtime_s(&localTime, &currentTime);

    char dateBuffer[11];
    std::strftime(dateBuffer, sizeof(dateBuffer), "%Y-%m-%d", &localTime);

    std::ofstream file("tradingOutput.txt", std::ios::app);

    if (file.is_open()) {
        std::ifstream checkFile("tradingOutput.txt");
        if (checkFile.peek() == std::ifstream::traits_type::eof()) {
            file << "CurrentDate\t200SMA\t80EMA\tStochasticOscillation\tBuySignal\n";
        }
        checkFile.close();

        file << dateBuffer << "\t" << sma200 << "\t" << ema80 << "\t" << stochasticCalculation << "\t\t\t" << (buySignal ? "BUY" : "NO BUY") << "\n";
    }

    file.close();
}
