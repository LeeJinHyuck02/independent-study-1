"""``python -m mfin.app [streamlit 옵션]``: 프로젝트 루트에서 Streamlit 앱을 띄운다.

루트로 이동해서 실행하므로 어느 폴더에서 부르든 루트의 .streamlit/config.toml이 적용된다.
추가 인자는 그대로 streamlit에 넘긴다 (예: ``--server.port 8502``, ``--server.headless true``).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def main() -> None:
    try:
        from streamlit.web import cli
    except ImportError:
        sys.exit('streamlit is not installed: pip install -e ".[app]"')
    from mfin.paths import ROOT

    os.chdir(ROOT)
    sys.argv = ["streamlit", "run", str(Path(__file__).with_name("main.py")), *sys.argv[1:]]
    sys.exit(cli.main())


if __name__ == "__main__":
    main()
