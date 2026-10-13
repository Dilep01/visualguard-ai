import pytest

from visualguard.settings import Settings, local_caches

local_caches()


@pytest.fixture
def settings(tmp_path):
    return Settings(storage=str(tmp_path / "runtime"), _env_file=None)
