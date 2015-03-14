#include "stdafx.h"
#define ICONSOLEDLLVCPP_EXPORTS 1
#include "IFloydWarshallDLLVCpp.h"
#include "TFloydWarshall.h"

HRESULT ICONSOLEDLLVCPP_API CreateFloydWarshall(LPFLOYDWARSHALL *floydWarshall)
{
    *floydWarshall = new TFloydWarshall();
    if (*floydWarshall)
    {
        (*floydWarshall)->AddRef();
        return S_OK;
    }
    else
        return E_NOINTERFACE;
}
