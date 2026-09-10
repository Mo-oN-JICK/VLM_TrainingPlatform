"""편집기 로컬 서버 — 얇은 HTTP 껍데기.

로직은 전부 `ui/api.py`에 있고 여기는 요청을 그리로 넘기기만 한다. 프레임워크를 쓰지 않는다.
127.0.0.1에만 바인딩한다 — 편집 도구이지 서비스가 아니다.
"""

from __future__ import annotations

import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Optional, Sequence, Tuple

from .api import Editor

ROUTES = ("/api/state", "/api/library", "/api/connect", "/api/disconnect",
          "/api/param", "/api/add", "/api/remove", "/api/save",
          "/api/undo", "/api/redo", "/api/rewind",
          "/api/recipe/select", "/api/recipe/add-path", "/api/recipe/drop-path",
          "/api/recipe/store", "/api/recipe/delete", "/api/recipe/active",
          "/api/run", "/api/run/state", "/api/run/stop")


def handle(editor: Editor, path: str, body: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
    """라우팅. HTTP를 모르는 순수 함수라 테스트가 소켓 없이 돈다."""
    if path == "/api/state":
        return 200, editor.state()
    if path == "/api/library":
        return 200, {"ok": True, "nodes": editor.library()}
    if path == "/api/run/state":
        return 200, editor.run_state()
    if path == "/api/connect":
        res = editor.connect(body["from"], body["to"])
    elif path == "/api/disconnect":
        res = editor.disconnect(body["to"])
    elif path == "/api/param":
        res = editor.set_param(body["node"], body["param"], body["value"])
    elif path == "/api/add":
        res = editor.add_node(body["type"], body.get("id", ""))
    elif path == "/api/remove":
        res = editor.remove_node(body["node"])
    elif path == "/api/save":
        res = editor.save()
    elif path == "/api/undo":
        res = editor.undo()
    elif path == "/api/redo":
        res = editor.redo()
    elif path == "/api/rewind":
        res = editor.rewind(body["index"])
    elif path == "/api/recipe/select":
        res = editor.recipe_select(body.get("id"))
    elif path == "/api/recipe/add-path":
        res = editor.recipe_add_path(body["path"])
    elif path == "/api/recipe/drop-path":
        res = editor.recipe_drop_path(body["path"])
    elif path == "/api/recipe/store":
        res = editor.recipe_store(body.get("id"), body.get("name", ""), body.get("note", ""))
    elif path == "/api/recipe/delete":
        res = editor.recipe_delete(body["id"])
    elif path == "/api/recipe/active":
        res = editor.recipe_set_active(body.get("id"))
    elif path == "/api/run":
        res = editor.run_start(body.get("limit", 8), bool(body.get("debug_output")))
    elif path == "/api/run/stop":
        res = editor.run_stop()
    else:
        return 404, {"ok": False, "reason": f"알 수 없는 경로 {path}"}

    if res.get("ok"):
        res["state"] = editor.state()
    return (200 if res.get("ok") else 409), res


def make_handler(editor: Editor, page: Any):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args: Any) -> None:  # 조용히
            return

        def _send(self, code: int, payload: Any, ctype: str = "application/json") -> None:
            data = payload if isinstance(payload, bytes) else json.dumps(payload, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", f"{ctype}; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:  # noqa: N802
            if self.path in ("/", "/index.html"):
                return self._send(200, page(editor).encode("utf-8"), "text/html")
            if self.path.startswith("/preview/"):
                data = editor.preview_file(self.path.split("?")[0][len("/preview/"):])
                if data is None:
                    return self._send(404, {"ok": False, "reason": "그 미리보기가 없다"})
                return self._send(200, data, "image/png")
            if self.path.startswith("/api/"):
                code, payload = handle(editor, self.path.split("?")[0], {})
                return self._send(code, payload)
            self._send(404, {"ok": False, "reason": "not found"})

        def do_POST(self) -> None:  # noqa: N802
            n = int(self.headers.get("Content-Length") or 0)
            try:
                body = json.loads(self.rfile.read(n) or b"{}")
            except json.JSONDecodeError:
                return self._send(400, {"ok": False, "reason": "본문이 JSON이 아니다"})
            code, payload = handle(editor, self.path.split("?")[0], body)
            self._send(code, payload)

    return Handler


def serve(project_path: str, port: int = 8770, open_browser: bool = False,
          extra_modules: Tuple[str, ...] = ()) -> None:
    from . import render as render_mod

    editor = Editor.open(project_path)
    # 편집기를 띄울 때 준 --nodes 를 하위 프로세스에도 그대로 넘긴다
    editor.extra_modules = tuple(extra_modules)
    page = lambda ed: render_mod.render_editor(ed)  # noqa: E731
    httpd = ThreadingHTTPServer(("127.0.0.1", port), make_handler(editor, page))
    url = f"http://127.0.0.1:{port}/"
    print(f"편집기: {url}")
    print(f"  프로젝트: {editor.path}")
    print("  저장을 눌러야 project.yaml에 쓰인다. Ctrl+C로 종료.")
    if open_browser:
        import webbrowser

        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n종료.")
    finally:
        httpd.server_close()
