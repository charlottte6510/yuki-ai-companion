"""
Desktop launcher for Yuki.

This is the actual entry point for the packaged app: it starts her
websocket backend (server.py) in a background thread, then opens her in a
real native window via pywebview -- no terminal, no separate browser tab,
no manually running two different things. Double-click, and she's there.

To run from source (before packaging):
    pip install pywebview
    py app.py

To package as a real .exe, see the PyInstaller instructions that came with
this file.
"""

import asyncio
import os
import sys
import threading

import webview

import server  # reuses everything from server.py exactly as it runs today


def resource_path(relative_path: str) -> str:
    """Resolves a bundled READ-ONLY asset (index.html, yuki.vrm) correctly
    whether running from source or as a PyInstaller-frozen .exe. This is
    deliberately separate from server.py's _app_dir(), which resolves
    WRITABLE user data (settings/memory) instead -- the two must not be
    confused, since PyInstaller's bundle folder isn't reliably writable."""
    base_path = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)


def start_backend():
    """Runs Yuki's websocket brain on its own asyncio event loop, in a
    background thread -- separate from pywebview's main-thread GUI loop,
    which needs the main thread to itself on Windows/macOS."""
    asyncio.run(server.main())


if __name__ == "__main__":
    threading.Thread(target=start_backend, daemon=True).start()

    webview.create_window(
        "Yuki",
        resource_path("index.html"),
        width=420,
        height=760,
        resizable=False,
    )
    webview.start()
