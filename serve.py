#!/usr/bin/env python3
from http.server import HTTPServer, SimpleHTTPRequestHandler
import socketserver
import sys


class CORSRequestHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory="./dist", **kwargs)

    def end_headers(self):
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        SimpleHTTPRequestHandler.end_headers(self)


Handler = CORSRequestHandler
# Handler.extensions_map = {
#     ".manifest": "text/cache-manifest",
#     ".html": "text/html",
#     ".png": "image/png",
#     ".jpg": "image/jpg",
#     ".svg": "image/svg+xml",
#     ".css": "text/css",
#     ".js": "application/x-javascript",
#     ".wasm": "application/wasm",
#     "": "application/octet-stream",  # Default
# }

PORT = 8000
with socketserver.TCPServer(("", PORT), Handler) as httpd:
    print("serving at port", PORT)
    httpd.serve_forever()
