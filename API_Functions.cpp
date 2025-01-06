#include "API.h"
#include <iostream>
#include <vector>
#include <algorithm>

//Only Used for API access in windows only
#include <windows.h>
#include <wininet.h>

//For JSON manipulation
#include <json.hpp>
using json = nlohmann::json;

//API fetching
std::string fetchApiData(const std::string& url) {
    //Array of characters called buffer, 4096 storage capacity (4kb)
    char buffer[4096];

    //Tracks the amount of bytes that are actually read from each InternetReadFile call
    //(bytesRead > 0) is used as a condition so that the program keeps reading
    DWORD bytesRead;

    //Api data will be stored in this
    std::string response;

    //Initialises internet session
    //Parameters are (UserString, AccessType, ProxyName, ProxyBypass, Flags)
    HINTERNET hInternet = InternetOpen("STOCKTEST", INTERNET_OPEN_TYPE_DIRECT, NULL, NULL, 0);

    //Checks if InternetOpen fails
    if (hInternet == NULL) {
        std::cout << "InternetOpen Failed" << GetLastError();
        return "";
    }

    //Opens API url
    //Parameters are (Url, CustomHeader, HeaderLength, Flags, AsyncRequests)
    HINTERNET hConnect = InternetOpenUrl(hInternet, url.c_str(), NULL, 0, INTERNET_FLAG_RELOAD, 0);

    //Checks if InternetOpenUrl fails
    if (hConnect == NULL) {
        std::cout << "InternetOpenUrl Failed" << GetLastError();
        InternetCloseHandle(hInternet);
        return "";
    }

    //Reads data in chunks up to 4096 bytes.
    //Loop continues until bytesRead is 0
    while (InternetReadFile(hConnect, buffer, sizeof(buffer), &bytesRead) && bytesRead > 0) {
        response.append(buffer, bytesRead);
    }

    //Clean up
    InternetCloseHandle(hInternet);
    InternetCloseHandle(hConnect);

    return response;

}

//Putting extractor values into vector
std::vector <float> jsonExtraction(const std::string& response, const std::string extractor) {

    std::vector<float> result;
    json j = json::parse(response);

    json timeSeries = j["Time Series (Daily)"];

    for (const auto& entry : timeSeries.items()) {
         result.emplace_back(std::stof(entry.value()[extractor].get<std::string>()));
    }

    //The oldest entries appeared at the start in the JSON, reversed to fix it
    std::reverse(result.begin(), result.end());

    return result;
}