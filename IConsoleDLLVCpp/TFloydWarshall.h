#pragma once
#include "stdafx.h"
#include "IFloydWarshall.h"
#include "shortness_path.h"
#include <stdio.h>
#include <string>


class TFloydWarshall : public IFloydWarshall
{
public:

    TFloydWarshall() : FRefCount(0) { }

    ~TFloydWarshall() { }	

	HRESULT __stdcall csvShortnessPathGpu(const char* csvGraph, BSTR* csvWeightMatrix){
		if (!csvWeightMatrix) return E_POINTER;					
		string csvShortnessWeightMatrix = fWarshall.csvFwGpu(csvGraph);		
		wstring w(csvShortnessWeightMatrix.begin(), csvShortnessWeightMatrix.end());
		*csvWeightMatrix = SysAllocStringLen(w.c_str(), w.length());
		return (*csvWeightMatrix) ? S_OK : E_OUTOFMEMORY;
	}

	HRESULT __stdcall uniShortnessPathGpu(const char* filePathOrCsv, BSTR* csvWeightMatrix){
		if (!csvWeightMatrix) return E_POINTER;
		string csvShortnessWeightMatrix = fWarshall.uniFwGpu(filePathOrCsv);
		wstring w(csvShortnessWeightMatrix.begin(), csvShortnessWeightMatrix.end());
		*csvWeightMatrix = SysAllocStringLen(w.c_str(), w.length());
		return (*csvWeightMatrix) ? S_OK : E_OUTOFMEMORY;
	}

	int stringToWString(std::wstring &ws, const std::string &s)
	{
		std::wstring wsTmp(s.begin(), s.end());

		ws = wsTmp;

		return 0;
	}

	// Methods of IUnknown

    HRESULT __stdcall QueryInterface(REFIID riid, void **ppvObject)
    {
        if (IsEqualGUID(riid, __uuidof(IFloydWarshall)))
        {
            *ppvObject = (void *)this;
            return S_OK;
        }
        else
        {
            *ppvObject = NULL;
            return E_NOINTERFACE;
        }
    }

    ULONG __stdcall AddRef(void)
    {
        return InterlockedIncrement(&FRefCount);
    }


    ULONG __stdcall Release(void)
    {
        ULONG result = InterlockedDecrement(&FRefCount);
        if (!result)
            delete this;
        return result;
    }
    
private:
	FWarshall fWarshall;
    ULONG FRefCount;
};

