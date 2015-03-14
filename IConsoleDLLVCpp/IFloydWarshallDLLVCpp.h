#pragma once

#include "stdafx.h"
#include "IFloydWarshall.h"

#ifdef ICONSOLEDLLVCPP_EXPORTS
#define ICONSOLEDLLVCPP_API __declspec(dllexport) __stdcall
#else
#define ICONSOLEDLLVCPP_API __declspec(dllimport) __stdcall
#endif

extern "C" HRESULT ICONSOLEDLLVCPP_API CreateFloydWarshall(LPFLOYDWARSHALL *console);
