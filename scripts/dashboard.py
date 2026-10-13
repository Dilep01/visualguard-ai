"""Launch Streamlit with its user-config writes redirected inside this project."""

import sys
from pathlib import Path

from visualguard.settings import ROOT, local_caches, project_path

if __name__ == "__main__":
    local_caches()
    # Streamlit has no public configuration-root option. Isolate this compatibility
    # override to the launcher; never modify global HOME or the installed package.
    import streamlit.file_util

    def local_streamlit_path(*parts: str) -> str:
        path = project_path(Path(".cache/streamlit") / Path(*parts))
        path.parent.mkdir(parents=True, exist_ok=True)
        return str(path)

    streamlit.file_util.get_streamlit_file_path = local_streamlit_path
    from streamlit.web.cli import main

    sys.argv = [
        "streamlit",
        "run",
        str(ROOT / "dashboard/app.py"),
        "--server.headless=true",
        "--browser.gatherUsageStats=false",
        *sys.argv[1:],
    ]
    main()
