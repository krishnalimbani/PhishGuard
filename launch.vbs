Dim scriptDir
scriptDir = Left(WScript.ScriptFullName, InStrRev(WScript.ScriptFullName, "\"))

Dim WshShell
Set WshShell = CreateObject("WScript.Shell")
WshShell.Run "cmd /c """ & scriptDir & "launch.bat""", 0, False
