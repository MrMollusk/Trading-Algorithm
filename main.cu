#include <iostream>
#include <vector>
#include <fstream>
#include <cassert>
#include <limits>

#include "DataStorage.h"
#include "API.h"
#include "indicatorsHEADER.h"

// d_ = on GPU, h_ = on CPU

int main() {
    std::string url = "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=IBM&outputsize=full&apikey=P18CNKW6MGYB66PX";
    std::string response = fetchApiData(url);

    // Gets first 200 close prices from API JSON
    std::vector<float> h_close = jsonNumberExtractor(jsonExtraction(response, "4. close"), 200);
    std::vector<float> h_high = jsonNumberExtractor(jsonExtraction(response, "2. high"), 200);
    std::vector<float> h_low = jsonNumberExtractor(jsonExtraction(response, "3. low"), 200);

    float sma200 = SMAKernelLaunch(h_close, 200);
    float ema80 = EMAKernelLaunch(h_close, 80);
    std::vector<float> stochasticValues = stochasticCalculation(h_high.data(), h_low.data(), h_close.data(), 1, 200, 14, 5);

    float currentPrice = h_close[0];

    float currentStochastic = stochasticValues.back();
    float previousStochastic = stochasticValues[stochasticValues.size() - 2];

    std::cout << "200 SMA is " << sma200 << "\n";
    std::cout << "80 EMA is " << ema80 << "\n";
    std::cout << "Current Stochastic Oscillation is " << currentStochastic << "\n";

    bool buySignal = false;
    if (currentPrice > sma200 && currentPrice > ema80 && currentStochastic < 20 && currentStochastic > previousStochastic) {
        std::cout << "BOT SAYS BUY" << "\n";
        buySignal = true;
    }
    else {
        std::cout << "No buy signal" << "\n";
    }

    //Write the results to a CSV file, including the buy signal
    writeToCsv(sma200, ema80, currentStochastic, buySignal);

    return 0;
}