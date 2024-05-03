#pragma once 

#include <iostream>
#include <fstream>
#include <ctime>

enum class LogLevel { INFO, WARNING, ERROR };

class Logger {
public:
    Logger(const std::string& filename);

    void log(LogLevel level, const std::string& message, ...);

private:
    std::ofstream ofs;

    std::string levelToString(LogLevel level);
};
