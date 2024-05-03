#include "logger.h"
#include <cstdarg>
#include <cstdio>

Logger::Logger(const std::string& filename) : ofs(filename, std::ofstream::out | std::ofstream::app) {}

void Logger::log(LogLevel level, const std::string& message, ...) {
    std::time_t now = std::time(nullptr);
    char timeString[100];
    std::strftime(timeString, sizeof(timeString), "%Y-%m-%d %H:%M:%S", std::localtime(&now));

    std::va_list args;
    va_start(args, message);
    char formattedMessage[1024]; // Adjust buffer size as needed
    std::vsprintf(formattedMessage, message.c_str(), args);
    va_end(args);

    std::cout << timeString << " [" << levelToString(level) << "] " << formattedMessage << std::endl;
    ofs << timeString << " [" << levelToString(level) << "] " << message << std::endl;
}

std::string Logger::levelToString(LogLevel level) {
    switch (level) {
        case LogLevel::INFO:
            return "INFO";
        case LogLevel::WARNING:
            return "WARNING";
        case LogLevel::ERROR:
            return "ERROR";
    }
    return "";
}
