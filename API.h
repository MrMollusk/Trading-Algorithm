#ifndef HEADER_H
#define HEADER_H
#include <string>
#include <vector>

std::string fetchApiData(const std::string& url);
std::vector <float> jsonExtraction(const std::string& response, const std::string extractor);

#endif