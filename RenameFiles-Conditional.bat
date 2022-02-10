::	https://stackoverflow.com/questions/53450747/rename-files-with-certain-characters-with-cmd
@echo off
setlocal EnableExtensions EnableDelayedExpansion
for %%I in ("* NA -*") do (
    set "FileName=%%~nI"
    ren "%%I" "!FileName:NA -=-!%%~xI"
)
endlocal