import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(tiny_bundle_path, monkeypatch):
    monkeypatch.setenv("LYDIA_MODEL_PATH", str(tiny_bundle_path))
    from api.main import app

    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "model_loaded": True, "model_version": "test-0"}


def test_model_info(client):
    r = client.get("/v1/model")
    assert r.status_code == 200 and r.json()["model_name"] == "merchant_lookup"


def test_predict_single(client):
    body = {"transaction_id": 10003553, "title": "CARREFOURMARKET", "amount_value": -6.83,
            "merchant_pan_entry_method": "CHIP_CONTACTLESS", "created_at": "2026-03-16T19:51:25Z", "member_id": 51617680}
    r = client.post("/v1/predict", json=body)
    assert r.status_code == 200
    out = r.json()
    assert out["minor_category"] == "supermarkets" and out["major_category"] == "groceries"
    assert out["transaction_id"] == "10003553" and len(out["top_k"]) == 3


def test_predict_validation_error(client):
    r = client.post("/v1/predict", json={"amount_value": -1})
    assert r.status_code == 422


def test_predict_batch(client):
    r = client.post("/v1/predict/batch", json=[{"title": "NETFLIX.COM", "amount_value": -13.49},
                                              {"title": "RETRAIT DAB", "amount_value": -40}])
    assert r.status_code == 200 and len(r.json()) == 2
    for item in r.json():
        assert item["minor_category"] and item["model_version"] == "test-0"
        assert len(item["top_k"]) == 3 and 0 < item["confidence"] <= 1
        assert item["top_k"][0]["label"] == item["minor_category"]


def test_batch_too_large_gives_422(client):
    item = {"title": "X", "amount_value": -1}
    assert client.post("/v1/predict/batch", json=[item] * 1000).status_code == 200
    assert client.post("/v1/predict/batch", json=[item] * 1001).status_code == 422


def test_missing_model_gives_503(tmp_path, monkeypatch):
    monkeypatch.setenv("LYDIA_MODEL_PATH", str(tmp_path / "nope.joblib"))
    from api.main import app

    with TestClient(app) as c:
        r = c.get("/health")
        assert r.status_code == 503
        assert r.json() == {"status": "unavailable", "model_loaded": False, "model_version": None}
        assert c.post("/v1/predict", json={"title": "X", "amount_value": -1}).status_code == 503


def test_all_model_endpoints_503_without_model(tmp_path, monkeypatch):
    monkeypatch.setenv("LYDIA_MODEL_PATH", str(tmp_path / "nope.joblib"))
    from api.main import app

    item = {"title": "X", "amount_value": -1}
    with TestClient(app) as c:
        assert c.get("/v1/model").status_code == 503
        assert c.post("/v1/predict", json=item).status_code == 503
        assert c.post("/v1/predict/batch", json=[item]).status_code == 503


RAW = {"content-type": "application/json"}


@pytest.mark.parametrize("url,body", [
    ("/v1/predict", b'{"title":"X","amount_value":1e309}'),
    ("/v1/predict/batch", b'[{"title":"X","amount_value":1e309}]'),
])
def test_non_finite_amount_gives_422(client, url, body):
    r = client.post(url, content=body, headers=RAW)
    assert r.status_code == 422
    assert r.json()["detail"][0]["loc"][-1] == "amount_value"


@pytest.mark.parametrize("extra", [
    {"fees": 9223372036854775808}, {"fees": 10**400}, {"fees": -1}, {"amount_value": 1e12},
])
def test_out_of_range_gives_422(client, extra):
    body = {"title": "X", "amount_value": -1, **extra}
    assert client.post("/v1/predict", json=body).status_code == 422


def test_index_page_serves_form(client):
    r = client.get("/")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/html")
    body = r.text
    assert 'id="predict-form"' in body and "/v1/predict" in body
    for field in ("title", "amount_value", "created_at", "merchant_pan_entry_method",
                  "digitalization_type", "pfm_recurring_frequency", "status", "fees"):
        assert f'id="{field}"' in body
    # the page must render offline: no external scripts, styles or fonts
    assert "http://" not in body and "https://" not in body
    assert "<script src" not in body and "<link" not in body
    assert "/" not in client.get("/openapi.json").json()["paths"]
    for text in ("CARD_ON_FILE — stored card charged by the merchant", "UBER — token held by Uber",
                 'id="csv-text"', 'id="csv-predict-btn"', "/v1/predict/batch"):
        assert text in body


def test_index_page_without_model(tmp_path, monkeypatch):
    monkeypatch.setenv("LYDIA_MODEL_PATH", str(tmp_path / "nope.joblib"))
    from api.main import app

    with TestClient(app) as c:
        assert c.get("/").status_code == 200
        assert c.post("/v1/predict", json={"title": "X", "amount_value": -1}).status_code == 503


def test_422_detail_is_a_list(client):
    r = client.post("/v1/predict", json={"amount_value": -1})
    assert r.status_code == 422 and isinstance(r.json()["detail"], list)


def test_predictor_value_error_becomes_422(client, monkeypatch):
    def boom(records):
        raise ValueError("bad record")

    monkeypatch.setattr(client.app.state.predictor, "predict", boom)
    r = client.post("/v1/predict", json={"title": "X", "amount_value": -1})
    expected = [{"loc": ["body"], "msg": "bad record", "type": "value_error"}]
    assert r.status_code == 422 and r.json()["detail"] == expected
    r = client.post("/v1/predict/batch", json=[{"title": "X", "amount_value": -1}])
    assert r.status_code == 422 and r.json()["detail"] == expected


def test_other_predictor_exceptions_are_not_translated(tiny_bundle_path, monkeypatch):
    monkeypatch.setenv("LYDIA_MODEL_PATH", str(tiny_bundle_path))
    from api.main import app

    def boom(record):
        raise RuntimeError("secret internals")

    with TestClient(app, raise_server_exceptions=False) as c:
        monkeypatch.setattr(c.app.state.predictor, "predict_one", boom)
        r = c.post("/v1/predict", json={"title": "X", "amount_value": -1})
    assert r.status_code == 500 and "secret" not in r.text


def test_corrupt_model_fails_fast(tmp_path, monkeypatch):
    bad = tmp_path / "bad.joblib"
    bad.write_bytes(b"\x00garbage not a joblib file")
    monkeypatch.setenv("LYDIA_MODEL_PATH", str(bad))
    from api.main import app

    with pytest.raises(Exception):
        with TestClient(app):
            pass


def test_model_path_directory_fails_fast(tmp_path, monkeypatch):
    monkeypatch.setenv("LYDIA_MODEL_PATH", str(tmp_path))
    from api.main import app

    with pytest.raises(Exception):
        with TestClient(app):
            pass


def test_validation_errors_do_not_echo_payload(client):
    item = {"title": "X", "amount_value": -1}
    r = client.post("/v1/predict/batch", json=[item] * 1001)
    assert r.status_code == 422 and len(r.text) < 2000 and '"input"' not in r.text
    r = client.post("/v1/predict", json={"title": "X", "amount_value": True})
    assert r.status_code == 422 and '"input"' not in r.text


def test_numeric_string_created_at_rejected_date_only_accepted(client):
    assert client.post("/v1/predict", json=_body(created_at="1777376721")).status_code == 422
    assert client.post("/v1/predict", json=_body(created_at="2026-03-16")).status_code == 200


def _body(**kw):
    return {"title": "X", "amount_value": -1, **kw}


@pytest.mark.parametrize("extra", [
    {"amount_value": True}, {"amount_value": "1_0"}, {"amount_value": "abc"}, {"amount_value": "1e999"},
    {"fees": True}, {"created_at": 1777376721}, {"created_at": 1777376721.5},
])
def test_strict_inputs_give_422(client, extra):
    r = client.post("/v1/predict", json=_body(**extra))
    assert r.status_code == 422 and isinstance(r.json()["detail"], list)


def test_numeric_string_amount_and_iso_date_accepted(client):
    assert client.post("/v1/predict", json=_body(amount_value="12.5")).status_code == 200
    assert client.post("/v1/predict", json=_body(created_at="2026-03-16T19:51:25Z")).status_code == 200


@pytest.mark.parametrize("title", [False, 123, 1.5, ["x"]])
def test_non_string_title_falls_back_to_merchant_name(client, title):
    base = {"merchant_name": "CARREFOUR CITY", "amount_value": "-12.5"}
    expected = client.post("/v1/predict", json={"title": "CARREFOUR CITY", **base}).json()
    r = client.post("/v1/predict", json={"title": title, **base})
    assert r.status_code == 200
    assert r.json()["minor_category"] == expected["minor_category"] == "supermarkets"
    assert r.json()["top_k"] == expected["top_k"]


@pytest.mark.parametrize("body", [
    {"title": 123, "amount_value": -1},
    {"title": False, "amount_value": -1},
    {"title": 123, "merchant_name": 5, "amount_value": -1},
])
def test_non_string_text_alone_gives_422(client, body):
    r = client.post("/v1/predict", json=body)
    assert r.status_code == 422
    assert "non-blank title or merchant_name" in str(r.json()["detail"])


def test_batch_keeps_input_order(client):
    a = {"title": "NETFLIX.COM", "amount_value": -13.49}
    b = {"title": "CARREFOURMARKET", "amount_value": -6.83}
    ab = client.post("/v1/predict/batch", json=[{**a, "transaction_id": "a"}, {**b, "transaction_id": "b"}]).json()
    ba = client.post("/v1/predict/batch", json=[{**b, "transaction_id": "b"}, {**a, "transaction_id": "a"}]).json()
    assert [x["transaction_id"] for x in ab] == ["a", "b"] and [x["transaction_id"] for x in ba] == ["b", "a"]
    assert ab[0]["minor_category"] == ba[1]["minor_category"] == "multimedia"
    assert ab[1]["minor_category"] == ba[0]["minor_category"] == "supermarkets"


def test_batch_with_one_invalid_item_is_rejected_whole(client, monkeypatch):
    calls = []
    real = client.app.state.predictor.predict
    monkeypatch.setattr(client.app.state.predictor, "predict", lambda r: calls.append(r) or real(r))
    r = client.post("/v1/predict/batch", json=[_body(), _body(amount_value=True), _body()])
    assert r.status_code == 422
    assert any(e["loc"][:2] == ["body", 1] for e in r.json()["detail"])
    assert calls == []


def test_unknown_categorical_values(client):
    body = {"title": "ZZZ UNSEEN SHOP 123", "amount_value": -9.99, "status": "weird_status",
            "merchant_pan_entry_method": "NEW_METHOD", "digitalization_type": "new_wallet",
            "pfm_recurring_frequency": "every_leap_year"}
    r = client.post("/v1/predict", json=body)
    assert r.status_code == 200 and r.json()["minor_category"]
