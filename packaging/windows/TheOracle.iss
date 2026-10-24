; Unsigned EXPERIMENTAL installer recipe. Windows support is not qualified.
#define AppName "The Oracle (EXPERIMENTAL)"
#define AppPublisher "The Oracle"
#ifndef ProductVersion
  #define ProductVersion "1.5.0"
#endif
#ifndef SourceDir
  #define SourceDir "dist\\TheOracle"
#endif

[Setup]
AppId={{C8E0EF20-4C09-4CF8-9F6F-150000000001}
AppName={#AppName}
AppVersion={#ProductVersion}
AppPublisher={#AppPublisher}
DefaultDirName={autopf}\The Oracle
DefaultGroupName=The Oracle (EXPERIMENTAL)
OutputBaseFilename=the-oracle-{#ProductVersion}-windows-experimental-unsigned
OutputDir=release_artifacts\windows-experimental
ArchitecturesInstallIn64BitMode=x64
PrivilegesRequired=lowest
Compression=lzma
SolidCompression=yes
UninstallDisplayName=The Oracle (EXPERIMENTAL)
Uninstallable=yes

[Files]
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\The Oracle (EXPERIMENTAL)"; Filename: "{app}\TheOracle.exe"
Name: "{commondesktop}\The Oracle (EXPERIMENTAL)"; Filename: "{app}\TheOracle.exe"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; Flags: unchecked

[UninstallDelete]
Type: filesandordirs; Name: "{app}"

[Code]
function InitializeSetup(): Boolean;
begin
  MsgBox('This unsigned installer is EXPERIMENTAL and Windows is not a supported release target.', mbInformation, MB_OK);
  Result := True;
end;
