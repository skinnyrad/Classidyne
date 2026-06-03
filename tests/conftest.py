import os
import httpx
import pytest


@pytest.fixture
def client():
    port = os.getenv("CLASSIDYNE_PORT", "5000")
    base_url = f"http://localhost:{port}"
    with httpx.Client(base_url=base_url, timeout=30) as c:
        yield c
