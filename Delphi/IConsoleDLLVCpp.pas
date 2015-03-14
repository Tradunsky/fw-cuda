unit IConsoleDLLVCpp;

interface

type
  IFloydWarshall = interface
  ['{9BBDA1A4-21E7-4D11-8F1C-E2AD13D2779C}']
    function csvShortnessPathGpu(csvGraph: PChar; var csvWeightMatrix: WideString): HRESULT; stdcall;
    function uniShortnessPathGpu(filePathOrCsv: PChar; var csvWeightMatrix: WideString): HRESULT; stdcall;
  end;

function CreateFloydWarshall: IFloydWarshall; safecall;

implementation

function CreateFloydWarshall; external '..\Target\IFloydWarshallDLLVCpp.dll ' name 'CreateFloydWarshall';

end.