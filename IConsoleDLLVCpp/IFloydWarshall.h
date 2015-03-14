#pragma once

#include "stdafx.h"
#include <string>

struct __declspec(uuid("{9BBDA1A4-21E7-4D11-8F1C-E2AD13D2779C}"))
    IFloydWarshall : public IUnknown
{
public:    
	virtual HRESULT __stdcall csvShortnessPathGpu(const char* csvGraph, BSTR* csvWeightMatrix) = 0;
	virtual HRESULT __stdcall uniShortnessPathGpu(const char* filePathOrCsv, BSTR* csvWeightMatrix) = 0;
};

typedef IFloydWarshall * LPFLOYDWARSHALL;

