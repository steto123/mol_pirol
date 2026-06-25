[Setup]
AppName=NMR 13C Predictor
AppVersion=0.5
AppPublisher=Open Source Chemistry
DefaultDirName={autopf}\NMR_Predictor
DefaultGroupName=NMR Predictor
OutputBaseFilename=NMR_App_Setup_v0.5
Compression=lzma2
SolidCompression=yes
ArchitecturesInstallIn64BitMode=x64
; Kein Admin-Recht nötig (portable in AppData/Programme)
PrivilegesRequired=lowest
; Installer-Icon
SetupIconFile=app_icon.ico
UninstallDisplayIcon={app}\app_icon.ico


[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked


[Files]
; ---- Hauptanwendung ----
Source: "nmr_app.py";           DestDir: "{app}"; Flags: ignoreversion
Source: "symmetry_ranking.py";  DestDir: "{app}"; Flags: ignoreversion
Source: "symmetry_tester.py";   DestDir: "{app}"; Flags: ignoreversion
Source: "Start_NMR_App.bat";    DestDir: "{app}"; Flags: ignoreversion
Source: "Start_Symmetry_Tester.bat"; DestDir: "{app}"; Flags: ignoreversion

; ---- Icons ----
Source: "app_icon.png";         DestDir: "{app}"; Flags: ignoreversion
Source: "app_icon.ico";         DestDir: "{app}"; Flags: ignoreversion

; ---- Dokumentation ----
Source: "README.md";            DestDir: "{app}"; Flags: ignoreversion
Source: "Documentation.md";     DestDir: "{app}"; Flags: ignoreversion
Source: "RELEASE_NOTES.md";     DestDir: "{app}"; Flags: ignoreversion
Source: "INSTALL_DE_EN.md";     DestDir: "{app}"; Flags: ignoreversion

; ---- Modelle & Daten ----
; Exclude git and dev artifacts
Source: "models\*";   DestDir: "{app}\models";  Flags: ignoreversion recursesubdirs createallsubdirs
Source: "codes\*";    DestDir: "{app}\codes";   Flags: ignoreversion recursesubdirs createallsubdirs
Source: "ketcher\*";  DestDir: "{app}\ketcher"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "dcode\*";    DestDir: "{app}\dcode";   Flags: ignoreversion recursesubdirs createallsubdirs

; ---- Portable Python-Umgebung ----
Source: "NMR_App_Portable\python\*"; DestDir: "{app}\portable_python"; Flags: ignoreversion recursesubdirs createallsubdirs


[Dirs]
; Stelle sicher, dass das templates-Verzeichnis existiert
Name: "{app}\models\templates_est_nmr"


[Icons]
; Startmenü
Name: "{group}\NMR 13C Predictor";        Filename: "{app}\Start_NMR_App.bat";         IconFilename: "{app}\app_icon.ico"
Name: "{group}\Symmetry Tester";          Filename: "{app}\Start_Symmetry_Tester.bat"; IconFilename: "{app}\app_icon.ico"
Name: "{group}\Deinstallieren";           Filename: "{uninstallexe}"
; Desktop (optional, Nutzer muss Checkbox aktivieren)
Name: "{commondesktop}\NMR 13C Predictor"; Filename: "{app}\Start_NMR_App.bat"; Tasks: desktopicon; IconFilename: "{app}\app_icon.ico"


[Run]
; Nach der Installation starten (optional)
Filename: "{app}\Start_NMR_App.bat"; Description: "NMR 13C Predictor jetzt starten"; Flags: postinstall shellexec skipifsilent
