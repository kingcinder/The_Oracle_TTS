# PyInstaller one-folder specification for the unsigned Linux rehearsal build.
# Models, caches, signing material, and user data are intentionally absent.

from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files

repo = Path(SPECPATH).parents[2]
datas = collect_data_files("the_oracle")
datas.extend(
    [
        (str(repo / "LICENSE"), "."),
        (str(repo / "PRIVACY.md"), "."),
        (str(repo / "EULA-DRAFT.md"), "."),
        (str(repo / "THIRD_PARTY_NOTICES.md"), "."),
    ]
)

a = Analysis(
    [str(repo / "packaging" / "linux" / "entrypoint.py")],
    pathex=[str(repo / "src")],
    binaries=[],
    datas=datas,
    hiddenimports=[],
    excludes=["tests", "pytest"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, a.binaries, a.datas, name="the-oracle", console=True)
coll = COLLECT(exe, a.binaries, a.datas, name="the-oracle")
