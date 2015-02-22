/*
 * Logger.cpp
 *
 *  Created on: 22 февр. 2015
 *      Author: o111o1oo
 */

#include "Logger.h"

void Logger::LOG(int level, const char* message,...) {
	va_list args;
	va_start(args, message);
	switch (level) {
	case Level::DEBUG:
		if (isDebug)
			printFormat(stdout, message, args, "DEBUG :: ", 0);
		break;
	case Level::INFO:
		if (isInfo)
			printFormat(stdout, message, args, "INFO :: ", 0);
		break;
	case Level::ERROR:
		if (isError)
			printFormat(stderr, message, args, "ERROR :: ", 0);
		break;
	default:
		printFormat(stderr, message, args, "Level %i is not supported. Message :: ", level);
	}
	va_end(args);
}

void Logger::LOG(int level,const char* file, int line, const char* message,...) {
	va_list args;
	va_start(args, message);
	switch (level) {
	case Level::DEBUG:
		if (isDebug)
			printFormat(stdout, message, args, "DEBUG %s(%d) :: ", file, line);
		break;
	case Level::INFO:
		if (isInfo)
			printFormat(stdout, message, args, "INFO %s(%d) :: ", file, line);
		break;
	case Level::ERROR:
		if (isError)
			printFormat(stderr, message, args, "ERROR %s(%d) :: ", file, line);
		break;
	default:
		fprintf(stderr, "%s(%d). Level %i is not supported. Message :: %s",
				file, line, level, message);
	}
	va_end(args);
}


void Logger::printFormat(_IO_FILE* file, const char* message, va_list args, const char* levelFmt,...) {
	time_t now;
	char buffer[20];
	va_list lvlArgs;
	va_start(lvlArgs, levelFmt);
	time(&now);
	strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", gmtime(&now));
	fprintf(file, "[%s] ", buffer);
	//__FILE__, __LINE__
	vfprintf(file, levelFmt, lvlArgs);
	vfprintf(file, message, args);
	fputc('\n', file);
	va_end(lvlArgs);
}

