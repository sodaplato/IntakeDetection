# Build on Windows with Python >=3.9: pyinstaller intake_submission.spec
import sys
from pathlib import Path
from PyInstaller.utils.hooks import collect_submodules

root = Path(SPECPATH)
sys.path.insert(0, str(root/'Actionformer'))
a = Analysis(
    [str(root/'submission.py')],
    pathex=[str(root), str(root/'Actionformer')],
    binaries=[],
    datas=[(str(root/'00 TaskRequirments'/'input.py'), '00 TaskRequirments'),
           (str(root/'00 TaskRequirments'/'output.py'), '00 TaskRequirments'),
           (str(root/'TriDet-master'/'libs'), 'TriDet-master/libs')],
    hiddenimports=collect_submodules('libs') + collect_submodules('intake'),
    hookspath=[], hooksconfig={}, runtime_hooks=[], excludes=['tensorboard'], noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name='IntakeDetection',
          debug=False, bootloader_ignore_signals=False, strip=False, upx=False, console=True)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name='IntakeDetection')
