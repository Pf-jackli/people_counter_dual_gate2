# Running PeopleCounter on an Intel Mac

Windows `.exe` files cannot run natively on macOS. Build this project on the
Intel Mac that will run it; PyInstaller does not cross-build a macOS app from
Windows.

## Requirements

- An Intel Mac (`uname -m` should print `x86_64`)
- Python 3 and an internet connection for installing Python packages
- The project files, including `yolov8n.pt` and
  `people_counter_bytetrack.yaml`

## Install and test from source

Open Terminal, change to the project folder, and run:

```sh
python3 -m venv .venv-macos
source .venv-macos/bin/activate
python -m pip install --upgrade pip
python -m pip install ultralytics opencv-python pillow pyinstaller
python people_counter_dual_gate.py --source "/path/to/video-or-folder"
```

For two video sources, pass `--source` twice. Add one `--source-lines LEFT,RIGHT`
for each source if line positions should be preset; otherwise the application
will prompt for line setup.

For harder-to-detect people, add `--gate-assist-source 1` to enable an
additional enlarged gate-area inference for the first source. Repeat the
option with other 1-based source indices as needed. This is disabled by
default and increases processing time; keep the full video frame as well, and
verify the selected finite gate segments and counting results on your video.

## Build the native Mac app

After the source version starts successfully:

```sh
python -m PyInstaller --clean --noconfirm PeopleCounter_macos.spec
```

The build produces `dist/PeopleCounter.app` and the command-line bundle at
`dist/PeopleCounter/PeopleCounter`. Run the latter from Terminal to see
diagnostic output and pass video paths, for example:

```sh
./dist/PeopleCounter/PeopleCounter --source "/path/to/video-or-folder"
```

The bundle includes the YOLO weights and tracker configuration. For testing on
another Intel Mac, copy the complete `dist/PeopleCounter` folder or the
`dist/PeopleCounter.app` bundle, not just the executable. This build targets
the Mac's native architecture; build separately on Apple Silicon for a native
Apple Silicon package. An unsigned app may require approval in macOS Privacy &
Security settings. Public distribution requires signing and notarization.
