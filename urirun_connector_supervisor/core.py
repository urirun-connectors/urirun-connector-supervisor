# Author: Tom Sapletta · Part of the ifURI solution.
"""urirun-connector-supervisor — NADZORCA IDE: patrzy na okna IDE hosta, popycha agenty do przodu.

Wiele okien IDE (JetBrains/Junie, VSCodium, Cursor) prowadzi agentów AI. Ten connector:
  1. OBSERWUJE host (KVM mutter-screencast — patrz [[host-self-capture-mutter]]);
  2. ROZUMIE per-okno stan agenta (working / done / waiting-approval / idle) — vision-LLM;
  3. POPYCHA: klik continue/accept lub next-prompt dla okien done/waiting.

GUARD krytyczny: NIE działa na WŁASNEJ sesji (okno które hostuje TEGO agenta — sprzężenie zwrotne).
Perception (scan) bezpieczne. Push (klik/type) za bramką dry_run + protected-window.
"""
from __future__ import annotations

import json
import os
from typing import Any

import urirun

CONNECTOR_ID = "supervisor"
conn = urirun.connector(CONNECTOR_ID, scheme="supervisor")

_VISION_MODEL = os.environ.get("SUPERVISOR_VISION_MODEL", "openrouter/google/gemini-2.5-flash")
_IMG_W = 1600
# okna które NIGDY nie ruszamy (sprzężenie zwrotne — hostują tego agenta)
_PROTECTED = ("junie", "claude", "claude-code", "anthropic")

# Deterministyczny detektor/klasyfikator okien IDE (grounding PRZED vision-LLM).
# Sygnatury dopasowywane po tokenach z app+title okna hosta (kvm window_list —
# realny inwentarz WM, nie zgadywanie modelu). Kolejność = priorytet: pierwszy
# trafiony family wygrywa. protected=True → okno hostuje agenta (nie ruszać).
_IDE_SIGNATURES: tuple[tuple[str, str, bool], ...] = (
    # token,           family,       protected
    ("claude-code",    "claude-code", True),
    ("claude",         "claude",      True),
    ("junie",          "junie",       True),
    ("anthropic",      "anthropic",   True),
    ("cursor",         "cursor",      False),
    ("vscodium",       "vscodium",    False),
    ("code - oss",     "vscode",      False),
    ("visual studio code", "vscode",  False),
    ("vscode",         "vscode",      False),
    ("intellij",       "jetbrains",   False),
    ("pycharm",        "jetbrains",   False),
    ("webstorm",       "jetbrains",   False),
    ("goland",         "jetbrains",   False),
    ("rider",          "jetbrains",   False),
    ("clion",          "jetbrains",   False),
    ("phpstorm",       "jetbrains",   False),
    ("jetbrains",      "jetbrains",   False),
    ("windsurf",       "windsurf",    False),
    ("zed",            "zed",         False),
)


def _ok(**kw: Any) -> dict[str, Any]:
    return urirun.ok(connector=CONNECTOR_ID, **kw)


def _fail(msg: str, action: str, **extra: Any) -> dict[str, Any]:
    return urirun.fail(msg, connector=CONNECTOR_ID, action=action, **extra)


def classify_window(app: str = "", title: str = "") -> dict[str, Any]:
    """Deterministycznie klasyfikuj JEDNO okno hosta po app+title (bez LLM/zrzutu).

    Zwraca ``{ide, family, protected, matched}``: ``ide`` czy to okno IDE/agenta,
    ``family`` rodzina IDE (jetbrains/vscode/cursor/…), ``protected`` czy hostuje
    agenta (sprzężenie zwrotne — nie ruszać), ``matched`` token który trafił.
    """
    hay = f"{app} {title}".lower()
    for token, family, protected in _IDE_SIGNATURES:
        if token in hay:
            return {"ide": True, "family": family, "protected": protected, "matched": token}
    return {"ide": False, "family": "", "protected": False, "matched": ""}


def _list_windows() -> list[dict[str, Any]]:
    """Realny inwentarz okien hosta przez kvm window_list (WM, nie vision). Best-effort."""
    from urirun_connector_kvm import core as kvm
    r = kvm.window_list()
    if not r.get("ok", True):
        return []
    return list(r.get("windows") or [])


def detect_windows(lister: Any = None) -> dict[str, Any]:
    """DETEKTOR: enumeruj okna hosta i sklasyfikuj IDE/agentów — deterministycznie,
    bez vision-LLM i bez zrzutu. Grounding dla scan() i tanie źródło ``protected``.

    Zwraca ``{ide_windows, protected_families, protected_tokens, total_windows}``.
    ``protected_families``/``protected_tokens`` = runtime-odkryte sygnatury okien
    hostujących agenta — rozszerzają statyczne ``_PROTECTED`` o to co realnie działa.
    """
    raw = (lister or _list_windows)()
    ide_wins: list[dict[str, Any]] = []
    prot_families: set[str] = set()
    prot_tokens: set[str] = set()
    for win in raw:
        cls = classify_window(str(win.get("app", "")), str(win.get("title", "")))
        if not cls["ide"]:
            continue
        entry = {**{k: win.get(k) for k in ("app", "title", "monitor", "id") if k in win}, **cls}
        ide_wins.append(entry)
        if cls["protected"]:
            prot_families.add(cls["family"])
            prot_tokens.add(cls["matched"])
    return {
        "ide_windows": ide_wins,
        "protected_families": sorted(prot_families),
        "protected_tokens": sorted(prot_tokens),
        "total_windows": len(raw),
    }


@conn.handler("host/query/detect", isolated=False,
              meta={"label": "Detektor IDE: deterministyczny inwentarz okien IDE/agentów hosta (bez LLM/zrzutu)"})
def host_query_detect() -> dict[str, Any]:
    try:
        return _ok(action="ide-detect", **detect_windows())
    except Exception as exc:  # noqa: BLE001
        return _fail(f"{type(exc).__name__}: {exc}", "ide-detect")


def _monitors() -> list[int]:
    """Indeksy monitorów hosta (multi-monitor: agenci IDE są na różnych ekranach)."""
    try:
        from urirun_connector_kvm import core as kvm
        d = kvm.display_info()
        mons = d.get("monitors") or []
        return [m.get("index", i + 1) for i, m in enumerate(mons)] or [0]
    except Exception:  # noqa: BLE001
        return [0]


def _capture_host(monitor: int = 0) -> tuple[str, int, int]:
    """Zrzut JEDNEGO monitora hosta (mutter-screencast — full canvas ucina). Zwraca (base64, w, h)."""
    from urirun_connector_kvm import core as kvm
    r = kvm.capture(base64=True, max_width=_IMG_W, monitor=monitor)
    return r.get("pngBase64") or "", r.get("width") or 0, r.get("height") or 0


def _vision(b64: str, prompt: str) -> dict:
    import litellm
    resp = litellm.completion(model=_VISION_MODEL, temperature=0.1, messages=[{"role": "user", "content": [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}]}])
    t = resp.choices[0].message.content or ""
    if "```" in t:
        t = t.split("```")[1].removeprefix("json").strip()
    try:
        return json.loads(t[t.index("{"):t.rindex("}") + 1])
    except Exception:  # noqa: BLE001
        return {"_raw": t[:400]}


_SCAN_PROMPT = (
    "To zrzut pulpitu z WIELOMA oknami IDE prowadzącymi agentów AI (JetBrains/Junie, VSCodium, Cursor). "
    "Obraz {w}px szeroki. Dla KAŻDEGO widocznego okna IDE/agenta zwróć wpis. CZYSTY JSON: {{\"windows\":["
    "{{\"app\": nazwa IDE, \"agent_state\": working|done|waiting_approval|idle, "
    "\"evidence\": co widzisz (np. 'przycisk Accept', 'auto mode on', 'pole prompt puste', 'Running...'), "
    "\"protected\": true jeśli to sesja Claude/Junie/Claude-Code (NIE ruszać), "
    "\"push_action\": KONKRETNIE co zrobić, "
    "\"click_xy\": [x,y] w pikselach obrazu}}]}}. "
    "ZASADY: agent_state=waiting_approval gdy widać przycisk Accept/Continue/Yes/Run/Allow LUB dialog decyzji "
    "— WTEDY push_action='klik <nazwa przycisku>' i click_xy = ŚRODEK tego przycisku w pikselach (ZAWSZE podaj, nie null). "
    "agent_state=done gdy skończył (pole prompt puste, brak Running) — push_action='wpisz kontynuuj' i click_xy=środek pola promptu. "
    "agent_state=working/idle → push_action='nic', click_xy=null. NIE zwracaj 'nic' dla waiting_approval/done."
)


def _is_protected(app: str, extra_tokens: tuple[str, ...] = ()) -> bool:
    """Czy nazwa okna wskazuje na sesję hostującą agenta (statyczne + runtime tokeny)."""
    low = str(app).lower()
    return any(tok and tok in low for tok in (_PROTECTED + tuple(extra_tokens)))


def scan(dry: bool = True, all_monitors: bool = True) -> dict[str, Any]:
    """OBSERWUJ+ZROZUM: zrzut KAŻDEGO monitora hosta → per-okno IDE stan agenta + push. Read-only.

    Grounding: przed vision uruchamia deterministyczny detektor (detect_windows) i jego
    runtime-odkryte ``protected_tokens`` rozszerzają statyczną listę ``_PROTECTED`` —
    okno realnie hostujące agenta nie trafi do ``actionable`` nawet gdy model je przeoczy.
    """
    try:
        detected = detect_windows()
    except Exception:  # noqa: BLE001 — detektor best-effort, brak groundingu nie blokuje scanu
        detected = {"ide_windows": [], "protected_tokens": [], "protected_families": [], "total_windows": 0}
    prot_tokens = tuple(detected.get("protected_tokens") or ())
    mons = _monitors() if all_monitors else [0]
    wins: list[dict] = []
    for mon in mons:
        b64, w, h = _capture_host(mon)
        if not b64:
            continue
        scene = _vision(b64, _SCAN_PROMPT.format(w=_IMG_W))
        for win in (scene.get("windows") or []):
            win["monitor"] = mon
            app = str(win.get("app", ""))
            if _is_protected(app, prot_tokens):
                win["protected"] = True
            wins.append(win)
    actionable = [win for win in wins if win.get("agent_state") in ("done", "waiting_approval")
                  and not win.get("protected") and win.get("push_action") not in (None, "", "nic")]
    return {"windows": wins, "actionable": actionable, "monitors_scanned": mons, "img_w": _IMG_W,
            "detected": detected}


@conn.handler("host/query/scan", isolated=False,
              meta={"label": "Nadzorca IDE: per-okno stan agenta (working/done/waiting) + rekomendowany push"})
def host_query_scan(dry: bool = True) -> dict[str, Any]:
    try:
        return _ok(action="ide-scan", **scan(dry))
    except Exception as exc:  # noqa: BLE001
        return _fail(f"{type(exc).__name__}: {exc}", "ide-scan")


def _push_via_agent(win: dict) -> dict[str, Any]:
    """Executor REFRAME: delegate the window's next step to a headless agent (agent://) when
    host-input can't do it. Envelope never raises."""
    try:
        from urirun_connector_agents import core as ag  # noqa: PLC0415
        prompt = (f"IDE window '{win.get('app')}' is in state {win.get('agent_state')}; "
                  f"perform this next step: {win.get('push_action')}")
        r = ag.task_run(prompt=prompt)
        return {"executor": "agent://", "ok": bool(r.get("ok")), "detail": r.get("agent")}
    except Exception as exc:  # noqa: BLE001
        return {"executor": "agent://", "ok": False, "error": str(exc)[:120]}


def push_window(win: dict, *, push: bool = False, kvm: Any = None) -> dict[str, Any]:
    """IFURI-137 — resilient executor/reframe chain for ONE actionable window:
      1. host-input (kvm abs-click / type) — primary
      2. agent:// — REFRAME fallback when host-input is unavailable or fails
      3. escalate-human + ``abort_retry`` — no viable executor, so STOP the idle loop
    Returns ``{executor, ok, escalate, abort_retry, ...}``. Verify is the loop's next scan:
    a truly pushed window leaves ``actionable`` on the following cycle."""
    if not push:
        return {"executor": "dry", "ok": False, "escalate": False, "abort_retry": False}
    if kvm is None:
        from urirun_connector_kvm import core as kvm  # noqa: PLC0415
    xy = win.get("click_xy") or []
    if len(xy) == 2:
        try:
            res = kvm.click_abs(x=int(xy[0]), y=int(xy[1]), sw=_IMG_W, sh=int(_IMG_W * 10 / 16), do_click=True)
            if res.get("ok"):
                if str(win.get("agent_state")) == "done" and "wpisz" in str(win.get("push_action", "")):
                    kvm.task_run(steps=[{"op": "type", "text": "kontynuuj", "after": 0.4}, {"op": "key", "keys": "return"}])
                return {"executor": "host-input", "ok": True, "escalate": False, "abort_retry": False}
        except Exception:  # noqa: BLE001 - host-input failed → reframe to agent
            pass
    reframed = _push_via_agent(win)
    if reframed.get("ok"):
        return {"executor": "agent://", "ok": True, "escalate": False, "abort_retry": False,
                "detail": reframed.get("detail")}
    return {"executor": "human", "ok": False, "escalate": True, "abort_retry": True,
            "reason": "brak wykonawcy (host-input + agent:// padły) → operator; przerwij pętlę"}


def supervise_loop(cycles: int = 0, interval: float = 30.0, push: bool = False, log_path: str = "") -> dict[str, Any]:
    """CIĄGŁA pętla nadzoru: scan → dla actionable (nie-protected done/waiting) PUSH → sleep → powtórz.
    cycles=0 → nieskończona (daemon). push=False → tylko obserwuj+loguj (bezpieczne). Guard: protected pomijane."""
    import time
    from urirun_connector_kvm import core as kvm
    lp = log_path or os.path.expanduser("~/.urirun/host-dashboard/supervisor.log")
    os.makedirs(os.path.dirname(lp), exist_ok=True)

    def _log(msg: str):
        try:
            with open(lp, "a", encoding="utf-8") as f:
                f.write(msg + "\n")
        except OSError:
            pass

    done_cycles, pushed_total, escalated = 0, 0, False
    while cycles == 0 or done_cycles < cycles:
        r = scan()
        act = r.get("actionable", [])
        _log(f"cycle {done_cycles}: {len(r.get('windows', []))} okien, {len(act)} actionable, push={push}")
        escalated = False
        for a in act:
            if a.get("protected"):
                continue
            res = push_window(a, push=push, kvm=kvm)
            if res.get("ok"):
                pushed_total += 1
                _log(f"  PUSH[{res['executor']}] {a.get('app')} mon{a.get('monitor')} → {a.get('push_action')} @ {a.get('click_xy')}")
            elif res.get("escalate"):
                escalated = True
                _log(f"  ESCALATE {a.get('app')}: {res.get('reason')}")
            else:
                _log(f"  [dry] would push {a.get('app')} → {a.get('push_action')} @ {a.get('click_xy')}")
        done_cycles += 1
        if escalated:  # no executor could drive it → stop idle looping, hand to operator
            _log("  abort_retry: brak wykonawcy — przerwano pętlę, eskalacja do operatora")
            break
        if cycles and done_cycles >= cycles:
            break
        time.sleep(interval)
    return {"cycles": done_cycles, "pushed": pushed_total, "escalated": escalated, "log": lp}


@conn.handler("host/command/supervise", isolated=True,
              meta={"label": "Ciągła pętla nadzorcy IDE (scan→push→sleep); push=False bezpieczne obserwowanie"})
def host_command_supervise(cycles: int = 3, interval: float = 30.0, push: bool = False) -> dict[str, Any]:
    try:
        return _ok(action="supervise", **supervise_loop(cycles, interval, push))
    except Exception as exc:  # noqa: BLE001
        return _fail(f"{type(exc).__name__}: {exc}", "supervise")


@conn.handler("host/command/push", isolated=True,
              meta={"label": "Popchnij agenta w oknie IDE (klik/type) — GUARD: nie rusza protected/własnej sesji"})
def host_command_push(x: int = 0, y: int = 0, sw: int = 0, sh: int = 0, text: str = "",
                      confirm: bool = False) -> dict[str, Any]:
    """Wykonaj push: abs-click (x,y w przestrzeni zrzutu sw,sh) i/lub wpisz tekst. confirm=True wymagane."""
    if not confirm:
        return _fail("confirm=True wymagane (host-input rusza prawdziwy kursor)", "ide-push")
    try:
        from urirun_connector_kvm import core as kvm
        acted = []
        if x and y:
            kvm.click_abs(x=x, y=y, sw=sw or _IMG_W, sh=sh or int(_IMG_W * 10 / 16), do_click=True)
            acted.append(f"klik({x},{y})")
        if text:
            kvm.task_run(steps=[{"op": "type", "text": text, "after": 0.4}, {"op": "key", "keys": "return"}])
            acted.append(f"type+enter '{text[:30]}'")
        return _ok(action="ide-push", acted=acted)
    except Exception as exc:  # noqa: BLE001
        return _fail(f"{type(exc).__name__}: {exc}", "ide-push")


def urirun_bindings() -> dict[str, Any]:
    return conn.bindings()
