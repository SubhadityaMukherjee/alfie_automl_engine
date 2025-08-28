import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--full",
        action="store_true",
        default=False,
        help="Run full test suite including long-running tests",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--full"):
        return

    skip_full = pytest.mark.skip(
        reason="skipping long-running tests; use --full to run"
    )
    for item in items:
        if "full" in item.keywords:
            item.add_marker(skip_full)


