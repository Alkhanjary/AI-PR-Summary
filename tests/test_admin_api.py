import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
import pytest


@pytest.fixture()
def client(tmp_path, monkeypatch):
    # Point users.json at a throwaway dir so tests never touch the real
    # AppData store, and so runs don't leak accounts into each other.
    monkeypatch.setenv("AI_PR_SUMMARY_DATA_DIR", str(tmp_path))
    import importlib
    import server as server_module
    importlib.reload(server_module)
    server_module.app.config["TESTING"] = True
    with server_module.app.test_client() as c:
        yield c, server_module


def login(client, username, password):
    return client.post("/api/auth/login", json={"username": username, "password": password})


def test_list_users_requires_login(client):
    c, _ = client
    resp = c.get("/api/admin/users")
    assert resp.status_code == 401


def test_list_users_rejects_non_admin(client):
    c, _ = client
    login(c, "user", "user123")
    resp = c.get("/api/admin/users")
    assert resp.status_code == 403


def test_list_users_allows_admin_and_hides_password_hash(client):
    c, _ = client
    login(c, "admin", "admin123")
    resp = c.get("/api/admin/users")
    assert resp.status_code == 200
    body = resp.get_json()
    usernames = {u["username"] for u in body["users"]}
    assert {"admin", "user"} <= usernames
    for u in body["users"]:
        assert "password" not in u
        assert set(u.keys()) == {"username", "role"}


def test_reset_password_requires_login(client):
    c, _ = client
    resp = c.post("/api/admin/reset-password", json={"username": "user", "new_password": "newpass1"})
    assert resp.status_code == 401


def test_reset_password_rejects_non_admin(client):
    c, _ = client
    login(c, "user", "user123")
    resp = c.post("/api/admin/reset-password", json={"username": "admin", "new_password": "newpass1"})
    assert resp.status_code == 403


def test_reset_password_rejects_missing_csrf_token(client):
    c, _ = client
    login(c, "admin", "admin123")
    resp = c.post("/api/admin/reset-password", json={"username": "user", "new_password": "newpass1"})
    assert resp.status_code == 403


def test_reset_password_rejects_unknown_account(client):
    c, server_module = client
    login(c, "admin", "admin123")
    token = server_module._get_csrf_token()
    resp = c.post("/api/admin/reset-password", json={
        "username": "nobody", "new_password": "newpass1", "csrf_token": token,
    })
    assert resp.status_code == 404


def test_reset_password_rejects_short_password(client):
    c, server_module = client
    login(c, "admin", "admin123")
    token = server_module._get_csrf_token()
    resp = c.post("/api/admin/reset-password", json={
        "username": "user", "new_password": "abc", "csrf_token": token,
    })
    assert resp.status_code == 400


def test_reset_password_succeeds_and_new_password_logs_in(client):
    c, server_module = client
    login(c, "admin", "admin123")
    token = server_module._get_csrf_token()
    resp = c.post("/api/admin/reset-password", json={
        "username": "user", "new_password": "brandnewpass1", "csrf_token": token,
    })
    assert resp.status_code == 200

    c.post("/api/auth/logout")
    ok = login(c, "user", "brandnewpass1")
    assert ok.status_code == 200

    # old password no longer works
    c.post("/api/auth/logout")
    stale = login(c, "user", "user123")
    assert stale.status_code == 401
