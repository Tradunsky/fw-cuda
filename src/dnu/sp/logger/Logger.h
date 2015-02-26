/*
 * Logger.h
 *
 *  Created on: 22 февр. 2015
 *      Author: o111o1oo
 */

#ifndef LOGGER_H_
#define LOGGER_H_
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <iostream>
#include <time.h>

using namespace std;

#define LOGGING(level, message, args) {											\
	Logger::LOG(level, __FILE__, __LINE__, message, args);					\
}
#define INFO(message, args) {											    \
	Logger::LOG(Level::INFO, __FILE__, __LINE__, message, args);			\
}

#define INFO2(message, args, args2) {											    \
	Logger::LOG(Level::INFO, __FILE__, __LINE__, message, args, args2);			\
}

#define DEBUG(message, args) {											    \
	Logger::LOG(Level::DEBUG, __FILE__, __LINE__, message, args);			\
}

#define ERROR(message, args) {											    \
	Logger::LOG(Level::ERROR, __FILE__, __LINE__, message, args);			\
}

struct Level {
	static const int DEBUG = 0;
	static const int INFO = 1;
	static const int ERROR = 2;
};

class Logger {
private:
	static const bool isDebug= true;
	static const bool isInfo = true;
	static const bool isError = true;
//	static void printFormat(_IO_FILE* file, const char* format, const char* message, va_list args);
	static void printFormat(_IO_FILE* file, const char* message, va_list args, const char* levelFmt,...);
public:
//	Logger();
	static void LOG(int level, const char* message, ...);
	static void LOG(int level, const char* file, int line, const char* message, ...);
	static bool isLevelEnabled(int level);
//	virtual ~Logger();
};

#endif /* LOGGER_H_ */
