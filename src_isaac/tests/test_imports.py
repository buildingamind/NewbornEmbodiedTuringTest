from __future__ import annotations

import subprocess
import sys


def test_top_level_import_does_not_load_optional_wrappers():
    code = (
        "import sys; import nett_skrl; "
        "raise SystemExit(int('cv2' in sys.modules or 'torchvision' in sys.modules))"
    )
    result = subprocess.run([sys.executable, "-c", code], check=False)
    assert result.returncode == 0
