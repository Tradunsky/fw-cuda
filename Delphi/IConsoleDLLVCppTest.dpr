program IConsoleDLLVCppTest;

{$APPTYPE CONSOLE}

{$R *.res}

uses
  SysUtils,
  DateUtils,
  IConsoleDLLVCpp in 'IConsoleDLLVCpp.pas';

procedure Test;
var
  FW: IFloydWarshall;
  graphCsv: PChar;
  weightMatrixCsv: WideString;
  start, stop: TDateTime;
  execution: Integer;
  i: Integer;
begin
    FW := CreateFloydWarshall;
    graphCsv := 'A;B;4;A;D;5;A;E;5;B;C;7;B;D;3;C;F;4;D;A;7;D;C;3;D;E;4;D;F;3;E;A;2;E;D;6;F;D;2;F;E;1';
  for i:= 1 to 1 do
  Begin
    start := Now;
    FW.csvShortnessPathGpu(graphCsv, weightMatrixCsv);
    stop := Now;
    Writeln;
    Write('Execution time: ');
    Writeln(IntToStr(MillisecondsBetween(stop, start)));
    //execution := execution+MillisecondsBetween(stop, start);
  end;
  Write('Weight matrix: ');
  writeln(weightMatrixCsv);
  Readln;
end;

begin
  try
    Test;
  except
    on E: Exception do
    begin
      Writeln('Error ', E.ClassName, ': ', E.Message);
      Readln;
    end;
  end;
end.



