echo off
setlocal
for %%I in (python.exe) do if exist %%~$path:I set f=%%~$path:I
if exist %f% do (
  %f:python.exe=%omniidl.exe -bpython -I"%RTM_ROOT%rtm\idl" -I"/home/rsdlab/OpenRTM-aist/MY_IDL" idl/Img.idl idl/Img.idl idl/ObjectRecognition.idl 
) else do (
  echo "python.exe" can not be found.
  echo Please modify PATH environmental variable for python command.
)
endlocal
