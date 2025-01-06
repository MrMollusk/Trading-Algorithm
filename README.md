# Trading Algorithm with GPU Acceleration

This program is designed to run on Windows machines and utilizes the Windows API for fetching stock data. It leverages Nvidia CUDA to accelerate the calculation of several technical indicators, including the 200Simple Moving Average (SMA), 80 Exponential Moving Average (EMA), and the Stochastic Oscillator.

## Prerequisites

- **Platform**: Windows OS
- **CUDA**: Nvidia GPU with CUDA support for hardware acceleration
- **Libraries**:
    - [nlohmann/json](https://github.com/nlohmann/json) for JSON parsing
    - CUDA Toolkit for GPU operations

Ensure you have CUDA set up and nlohmann/json installed via Git:

```bash
git clone https://github.com/nlohmann/json.git
