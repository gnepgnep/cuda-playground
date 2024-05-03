#include "../base/logger.h"

int main() {
    Logger logger("logfile.txt");

    logger.log(LogLevel::INFO, "This is an informational message.");
    logger.log(LogLevel::WARNING, "This is a warning message.");
    logger.log(LogLevel::ERROR, "This is an error message.");

    return 0;
}
