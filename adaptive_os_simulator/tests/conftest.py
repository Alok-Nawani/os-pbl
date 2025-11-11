import os
import sys


def pytest_sessionstart(session):
    # Ensure project root is on sys.path so 'backend' can be imported
    here = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(here, os.pardir))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)


