from urirun_connector_supervisor import core


def test_bindings():
    b = core.urirun_bindings()["bindings"]
    assert any("host/query/scan" in k for k in b)
    assert any("host/query/detect" in k for k in b)


def test_classify_window_families_and_protected():
    assert core.classify_window("PyCharm", "if-uri – core.py")["family"] == "jetbrains"
    assert core.classify_window("VSCodium", "foo")["family"] == "vscodium"
    cur = core.classify_window("Cursor", "proj")
    assert cur["ide"] is True and cur["protected"] is False
    # okno hostujące agenta → protected (sprzężenie zwrotne)
    prot = core.classify_window("Terminal", "claude — if-uri")
    assert prot["protected"] is True and prot["matched"] == "claude"
    # nie-IDE okno
    assert core.classify_window("Firefox", "news")["ide"] is False


def test_detect_windows_deterministic_no_llm():
    wins = [
        {"app": "PyCharm", "title": "core.py", "monitor": 1},
        {"app": "Cursor", "title": "proj", "monitor": 2},
        {"app": "Firefox", "title": "news", "monitor": 1},
        {"app": "kitty", "title": "claude-code run", "monitor": 3},
    ]
    r = core.detect_windows(lister=lambda: wins)
    families = {w["family"] for w in r["ide_windows"]}
    assert families == {"jetbrains", "cursor", "claude-code"}  # Firefox odrzucony
    assert r["total_windows"] == 4
    assert "claude-code" in r["protected_families"]
    assert "claude-code" in r["protected_tokens"]


def test_scan_grounds_protected_from_detector(monkeypatch):
    # detektor widzi realne okno claude-code na hoście — jego token chroni okno,
    # które vision omyłkowo zaraportowało jako zwykły VSCodium waiting_approval.
    monkeypatch.setattr(core, "detect_windows",
                        lambda *a, **k: {"ide_windows": [], "protected_tokens": ["claude-code"],
                                          "protected_families": ["claude-code"], "total_windows": 1})
    monkeypatch.setattr(core, "_monitors", lambda: [1])
    monkeypatch.setattr(core, "_capture_host", lambda mon=0: ("ZZZ", 1600, 1000))
    monkeypatch.setattr(core, "_vision", lambda b, p: {"windows": [
        {"app": "VSCodium claude-code", "agent_state": "waiting_approval",
         "push_action": "klik Accept", "click_xy": [800, 500]}]})
    r = core.scan()
    assert r["windows"][0]["protected"] is True
    assert r["actionable"] == []  # grounding zablokował push na okno hostujące agenta


def test_scan_marks_protected_and_actionable(monkeypatch):
    monkeypatch.setattr(core, "detect_windows",
                        lambda *a, **k: {"ide_windows": [], "protected_tokens": [],
                                          "protected_families": [], "total_windows": 0})
    monkeypatch.setattr(core, "_monitors", lambda: [1])
    monkeypatch.setattr(core, "_capture_host", lambda mon=0: ("ZZZ", 1600, 1000))
    monkeypatch.setattr(core, "_vision", lambda b, p: {"windows": [
        {"app": "JetBrains Junie", "agent_state": "working", "push_action": "nic"},
        {"app": "VSCodium", "agent_state": "waiting_approval", "push_action": "klik Accept", "click_xy": [800, 500]},
        {"app": "Cursor", "agent_state": "done", "push_action": "wpisz next prompt"}]})
    r = core.scan()
    prot = next(w for w in r["windows"] if "Junie" in w["app"])
    assert prot["protected"] is True
    assert len(r["actionable"]) == 2  # VSCodium + Cursor, nie Junie


def test_push_requires_confirm():
    r = core.host_command_push(x=100, y=100)
    assert r["ok"] is False and "confirm" in r["error"]


# ── IFURI-137: executor/reframe/verify chain ──────────────────────────────────

class _FakeKvm:
    def __init__(self, click_ok=True):
        self.click_ok = click_ok
        self.calls = []
    def click_abs(self, **kw):
        self.calls.append(("click", kw))
        return {"ok": self.click_ok}
    def task_run(self, **kw):
        self.calls.append(("task", kw))
        return {"ok": True}


def test_push_window_host_input_primary():
    kvm = _FakeKvm(click_ok=True)
    win = {"app": "PyCharm", "click_xy": [100, 200], "agent_state": "waiting_approval", "push_action": "kliknij"}
    r = core.push_window(win, push=True, kvm=kvm)
    assert r["executor"] == "host-input" and r["ok"] and not r["escalate"]
    assert kvm.calls[0][0] == "click"


def test_push_window_reframes_to_agent_when_host_input_fails(monkeypatch):
    kvm = _FakeKvm(click_ok=False)  # host-input dispatched but no effect
    monkeypatch.setattr(core, "_push_via_agent", lambda w: {"executor": "agent://", "ok": True, "detail": "claude"})
    win = {"app": "VSCodium", "click_xy": [10, 20], "agent_state": "done", "push_action": "wpisz kontynuuj"}
    r = core.push_window(win, push=True, kvm=kvm)
    assert r["executor"] == "agent://" and r["ok"] and not r["abort_retry"]


def test_push_window_escalates_and_aborts_when_no_executor(monkeypatch):
    kvm = _FakeKvm(click_ok=False)
    monkeypatch.setattr(core, "_push_via_agent", lambda w: {"executor": "agent://", "ok": False, "error": "no agent"})
    win = {"app": "VSCodium", "click_xy": [10, 20], "agent_state": "done", "push_action": "x"}
    r = core.push_window(win, push=True, kvm=kvm)
    assert r["escalate"] is True and r["abort_retry"] is True and r["ok"] is False


def test_push_window_dry_run_no_action():
    r = core.push_window({"app": "X", "click_xy": [1, 2]}, push=False)
    assert r["executor"] == "dry" and not r["ok"]


def test_vision_with_oqllm_docker(monkeypatch):
    """Verify supervisor vision extraction through local OQLLM Docker instance."""
    import urllib.request
    try:
        with urllib.request.urlopen("http://127.0.0.1:11437/health", timeout=1.0) as resp:
            if resp.status != 200:
                return
    except Exception:
        return

    monkeypatch.setenv("SUPERVISOR_LLM_API_BASE", "http://127.0.0.1:11437/v1")
    monkeypatch.setenv("SUPERVISOR_VISION_MODEL", "oqlos/system-one-router")
    monkeypatch.setenv("SUPERVISOR_LLM_API_KEY", "oqllm_live_loopback_root")

    b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
    prompt = "proc://supervisor/host/query/detect"
    res = core._vision(b64, prompt)
    assert isinstance(res, dict)
    assert res.get("selected_uri") == "proc://supervisor/host/query/detect"


class _Resp:
    def __init__(self, content):
        msg = type("M", (), {"content": content})()
        self.choices = [type("C", (), {"message": msg})()]

    def __getitem__(self, key):  # dict-style access used by some connectors
        return {"choices": [{"message": {"content": self.choices[0].message.content}}]}[key]


def _capture(monkeypatch):
    import litellm
    seen = {}
    def fake(**kw):
        seen.update(kw)
        return _Resp('{"ok": true}')
    monkeypatch.setattr(litellm, "completion", fake)
    for var in ("URIRUN_LLM_API_BASE", "URIRUN_LLM_API_KEY", "OQLLM_API_KEY", "OPENAI_API_BASE", "SUPERVISOR_LLM_API_BASE", "SUPERVISOR_LLM_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-must-not-leak")
    return seen


def test_default_call_does_not_forward_foreign_credentials(monkeypatch):
    seen = _capture(monkeypatch)
    core._vision("ZZZ", "prompt")
    assert "api_key" not in seen and "api_base" not in seen
    assert seen["model"].startswith("openrouter/")


def test_explicit_gateway_gets_only_gateway_key(monkeypatch):
    seen = _capture(monkeypatch)
    monkeypatch.setenv("SUPERVISOR_LLM_API_BASE", "http://127.0.0.1:11437/v1")
    monkeypatch.setenv("OQLLM_API_KEY", "oqllm-key")
    core._vision("ZZZ", "prompt")
    assert seen["api_base"] == "http://127.0.0.1:11437/v1"
    assert seen["api_key"] == "oqllm-key"
