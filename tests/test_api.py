import argparse

import httpx

DEFAULT_HOST = "localhost"
DEFAULT_PORT = 5000
TEST_IMAGE = "tests/lora.png"


def parse_args():
    parser = argparse.ArgumentParser(description="Run Classidyne API tests")
    parser.add_argument("--host", default=DEFAULT_HOST, help="Classidyne host/IP")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Classidyne port")
    return parser.parse_args()


def build_base_url(host, port):
    if host.startswith(("http://", "https://")):
        return f"{host}:{port}"
    return f"http://{host}:{port}"


def test_stats(client):
    r = client.get("/api/stats")
    data = r.json()
    assert data["success"]
    assert data["embedding_status"] == "Idle"
    print(f"  waterfall_size={data['waterfall_size']}, fft_size={data['fft_size']}")


def test_waterfall_types(client):
    r = client.get("/api/waterfall_types")
    data = r.json()
    assert data["success"]
    assert "lora" in data["types"]
    print(f"  {len(data['types'])} waterfall types found")


def test_fft_types(client):
    r = client.get("/api/fft_types")
    data = r.json()
    assert data["success"]
    print(f"  {len(data['types'])} fft types found")


def test_classify_waterfall(client):
    with open(TEST_IMAGE, "rb") as f:
        r = client.post(
            "/api/classify",
            files={"query_image": ("lora.png", f, "image/png")},
            data={"collection": "waterfall", "similarity_threshold": "0.5"},
        )
    data = r.json()
    assert data["success"]
    assert len(data["class_scores"]) > 0
    top = data["class_scores"][0]
    assert top["class"] == "lora"
    print(f"  top class: {top['class']} ({top['confidence']}%), freq: {top['frequency_range']}")


def test_classify_fft(client):
    with open(TEST_IMAGE, "rb") as f:
        r = client.post(
            "/api/classify",
            files={"query_image": ("lora.png", f, "image/png")},
            data={"collection": "fft", "similarity_threshold": "0.5"},
        )
    data = r.json()
    assert data["success"]
    assert len(data["class_scores"]) == 0
    print(f"  no matches (expected)")


def test_find_image(client):
    r = client.get("/api/find_image", params={"identifier": "lora"})
    data = r.json()
    assert not data["success"]
    assert "matches" in data
    assert len(data["matches"]) > 0
    print(f"  multiple matches returned (status {r.status_code})")


def test_identify_frequency(client):
    r = client.get("/api/identify_frequency", params={"freq": 915000000})
    data = r.json()
    assert data["success"]
    assert "lora" in data["signals"]
    print(f"  signals: {data['signals']}")


def test_type_collage(client):
    r = client.get(
        "/api/type_collage", params={"type": "lora", "collection": "waterfall"}
    )
    data = r.json()
    assert data["success"]
    assert "collage_base64" in data
    print(f"  collage returned ({len(data['collage_base64'])} chars)")


if __name__ == "__main__":
    args = parse_args()
    base_url = build_base_url(args.host, args.port)

    tests = [
        ("GET  /api/stats", test_stats),
        ("GET  /api/waterfall_types", test_waterfall_types),
        ("GET  /api/fft_types", test_fft_types),
        ("POST /api/classify (waterfall)", test_classify_waterfall),
        ("POST /api/classify (fft)", test_classify_fft),
        ("GET  /api/find_image", test_find_image),
        ("GET  /api/identify_frequency", test_identify_frequency),
        ("GET  /api/type_collage", test_type_collage),
    ]

    passed = 0
    failed = 0
    with httpx.Client(base_url=base_url, timeout=30) as client:
        for label, fn in tests:
            try:
                print(f"\n[TEST] {label}")
                fn(client)
                print(f"  PASS")
                passed += 1
            except Exception as e:
                print(f"  FAIL: {e}")
                failed += 1

    print(f"\n{'='*40}")
    print(f"Results: {passed} passed, {failed} failed")
