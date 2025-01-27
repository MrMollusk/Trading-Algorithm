# Trading Algorithm with GPU Acceleration

This program is designed to run on Windows machines and utilizes the Windows API for fetching stock data. It leverages Nvidia CUDA to accelerate the calculation of several technical indicators, including the 200 Simple Moving Average (SMA), 80 Exponential Moving Average (EMA), and the Stochastic Oscillator.

## Prerequisites

- **Platform**: Windows OS.
- **C++ Compiler:** A modern C++ compiler (e.g., GCC, Clang, or MSVC).
- **CUDA**: Nvidia GPU with CUDA support for hardware acceleration.
- **Libraries**:
    - [nlohmann/json](https://github.com/nlohmann/json) for JSON parsing.
    - CUDA Toolkit for GPU operations.

Ensure you have CUDA set up and nlohmann/json installed via Git:

```
git clone https://github.com/nlohmann/json.git
```

## Features
- **GPU Acceleration:** Computes technical indicators on the GPU for faster performance.
- **Windows API Integration:** Fetches stock market data directly using the Windows API and Alpha Vantage https://www.alphavantage.co/.
- **Automated Buy Signals:** Generates buy signals based on a combination of SMA, EMA, and Stochastic Oscillator thresholds.
- **Data Logging:** Outputs calculations and buy signals to a CSV file for record-keeping and analysis.

## Uses
1. The program fetches the daily time series data for the stock "IBM" by default. If you wish to analyze other stocks, change the "IBM" symbol to another company that AlphaVantage Supports.

2. The data store file [TradingOutput.txt](./tradingOutput.txt) will contain:
- **CurrentDate:** The date of the calculation.
- **200SMA:** The 200-day Simple Moving Average.
- **80EMA:** The 80-day Exponential Moving Average.
- **StochasticOscillation:** The Stochastic Oscillator value.
- **BuySignal:** Indicates whether the algorithm recommends buying (BUY) or not (NO BUY).

The data store file is currently empty, as upon initialisation of the program, it will create the headers if there are none present.

## Core Algorithms
- **200 SMA Calculation:** Uses the GPU to calculate the average of the last 200 closing prices.
- **80 EMA Calculation:** Applies an exponentially weighted smoothing factor to recent prices.
- **Stochastic Oscillator:** Measures the current price relative to its price range over a period.

## Buy Signal Logic
A buy signal is generated when:
1. The current price is above both the 200 SMA and 80 EMA.
2. The Stochastic Oscillator is below 20 (oversold condition).
3. The Stochastic Oscillator is trending upward.



