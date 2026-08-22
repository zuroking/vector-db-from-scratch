"""CLI layer (Phase 5): Typer application wrapping the public library API.

Deliberately does NOT re-export ``app``: an attribute named ``app`` on this
package shadows the ``vectordb.cli.app`` submodule for ``import ... as``
statements (CPython resolves the name through the parent package first).
Import from ``vectordb.cli.app`` directly.
"""
