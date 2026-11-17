# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all

datas = [
    ("yolov8n.pt", "."),
    ("people_counter_bytetrack.yaml", "."),
]
binaries = []
hiddenimports = []

for package in ("ultralytics", "torch", "torchvision", "cv2", "PIL"):
    package_datas, package_binaries, package_hiddenimports = collect_all(package)
    datas += package_datas
    binaries += package_binaries
    hiddenimports += package_hiddenimports

a = Analysis(
    ["people_counter_dual_gate.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="PeopleCounter",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="PeopleCounter",
)

app = BUNDLE(
    coll,
    name="PeopleCounter.app",
    bundle_identifier="com.peoplecounter.dualgate",
    info_plist={
        "NSCameraUsageDescription": "PeopleCounter uses the selected camera as a video source.",
    },
)
