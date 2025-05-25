#Requires AutoHotkey v2.0
#SingleInstance Force

; --- Press F5 to toggle pause/resume for the whole script ---
F5:: {
    Pause
    if (A_IsPaused)
        ToolTip "Script paused"
    else
        ToolTip "Script resumed"
    Sleep 1000
    ToolTip
}

EEGWindowTitle := "Biopac Student Lab"
SharedFolder   := "\\LAPTOP-PQA7946A\AAA\"
TimerInterval  := 20000 ; in milliseconds
FileExtension  := ".txt"

; Exit if the Biopac window does not exist
if !WinExist(EEGWindowTitle)
    return

WinActivate
Send "^{Space}"  ; Start measurement
Sleep 100
SetTimer(CaptureEEG, -TimerInterval)

CaptureEEG(*) {
    global EEGWindowTitle, SharedFolder, FileExtension

    if !WinExist(EEGWindowTitle)
        return

    WinActivate

    ; --- Stop recording ---
    Send "^{Space}"
    Sleep 100

    ; --- Select all and copy data ---
    Clipboard := ""
    Send "^a"
    Sleep 100
    Send "^l"
    Sleep 100
    if !ClipWait(2)
        return

    ; --- Save to local file ---
    LocalFile := A_ScriptDir . "\1.txt"
    Run "notepad.exe " . LocalFile
    WinWaitActive "1.txt - Notepad", , 2
    Send "^a"
    Send "^v"
    Sleep 100
    Send "^s"
    Send "!{F4}" ; Close Notepad

    ; --- Copy file to remote shared folder ---
    RemoteFile := SharedFolder . "1.txt"
    cmd := 'powershell -command "Copy-Item -Path ' . '"' . LocalFile . '" -Destination "' . RemoteFile . '" -Force"'
    RunWait cmd

    ; --- Clear and restart Biopac recording ---
    Send "^x"
    Sleep 5100
    Send "^{Space}"

    ; Schedule next capture
    SetTimer(CaptureEEG, -TimerInterval)
}
