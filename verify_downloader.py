"""Verify range download recovery with a local deliberately truncated response."""
import json
import re
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

import run_experiments as workflow


PAYLOAD = json.dumps({"example": "test " * 50000}).encode()
state = {"interrupted": False, "resume_seen": False}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        match = re.fullmatch(r"bytes=(\d+)-(\d+)", self.headers.get("Range", ""))
        start, end = map(int, match.groups())
        end = min(end, len(PAYLOAD) - 1)
        content = PAYLOAD[start:end + 1]
        self.send_response(206)
        self.send_header("Content-Range", f"bytes {start}-{end}/{len(PAYLOAD)}")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        if len(content) > 1 and not state["interrupted"]:
            self.wfile.write(content[:1024])
            self.wfile.flush()
            state["interrupted"] = True
            self.close_connection = True
        else:
            if start == 1024:
                state["resume_seen"] = True
            self.wfile.write(content)


def main():
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with tempfile.TemporaryDirectory(prefix="download_verify_", dir=workflow.ROOT) as temp:
            target = Path(temp) / "test.json"
            with patch.object(workflow, "SOURCE", f"http://127.0.0.1:{server.server_port}/"):
                workflow.download(target)
            assert target.read_bytes() == PAYLOAD
            assert state["interrupted"] and state["resume_seen"]
            assert not target.with_suffix(".download").exists()
        print("PASS: truncated transfer resumed at byte 1024; exact content and valid JSON verified.")
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


if __name__ == "__main__":
    main()
