[Setup]
AppName=NMR 13C Predictor
AppVersion=1.0
AppPublisher=Open Source Chemistry
DefaultDirName={autopf}\NMR_Predictor
DefaultGroupName=NMR Predictor
OutputBaseFilename=NMR_App_Setup
Compression=lzma2
SolidCompression=yes
ArchitecturesInstallIn64BitMode=x64
; This tells the installer to ask for admin privileges
PrivilegesRequired=admin

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
; The base python code and resources
Source: "nmr_app.py"; DestDir: "{app}"; Flags: ignoreversion
Source: "Start_NMR_App.bat"; DestDir: "{app}"; Flags: ignoreversion
; Exclude git and dev artifacts
Source: "models\*"; DestDir: "{app}\models"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "codes\*"; DestDir: "{app}\codes"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "ketcher\*"; DestDir: "{app}\ketcher"; Flags: ignoreversion recursesubdirs createallsubdirs
; The portable python environment
Source: "portable_python\*"; DestDir: "{app}\portable_python"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\NMR 13C Predictor"; Filename: "{app}\Start_NMR_App.bat"
Name: "{commondesktop}\NMR 13C Predictor"; Filename: "{app}\Start_NMR_App.bat"; Tasks: desktopicon

[Run]
Filename: "{app}\Start_NMR_App.bat"; Description: "Launch NMR 13C Predictor now"; Flags: postinstall shellexec skipifsilent
