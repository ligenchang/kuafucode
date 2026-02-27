"""
Code intelligence — symbol extraction, import graph traversal, dependency graph, and symbol resolution.

Strategy:
  Python  → built-in `ast` module  (zero extra deps, fully accurate)
  JS/TS   → regex extraction  (covers functions, classes, exports, imports)
  Go      → regex extraction  (func, type, interface, import)
  Rust    → regex extraction  (fn, struct, enum, trait, impl, use)
  Java/C# → regex extraction  (class, method signatures)
  C/C++   → regex extraction  (function declarations, struct/class)

Public API
----------
  extract_symbols(path)                          → SymbolIndex
  resolve_imports(path, workspace)               → list[Path]  (backward compat shim)
  build_symbol_context(paths, ws)                → str  (LLM-ready block)
  get_dependency_graph(workspace)                → DependencyGraph  (cached singleton)
  find_definition(name, workspace, hint_paths)   → list[DefinitionSite]
  find_references(name, workspace, hint_paths)   → list[ReferenceSite]
"""

from __future__ import annotations

import ast
import json
import os
import re
import sys
import threading
import time
from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Data model
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Symbol:
    kind: str          # "function" | "class" | "method" | "import" | "type" | "const"
    name: str
    signature: str     # one-line human-readable signature
    line: int = 0
    docstring: str = ""

    def __str__(self) -> str:
        doc = f"  # {self.docstring[:80]}" if self.docstring else ""
        return f"{self.signature}{doc}"


@dataclass
class SymbolIndex:
    path: Path
    language: str
    symbols: list[Symbol] = field(default_factory=list)
    imports: list[str] = field(default_factory=list)   # raw import strings

    def functions(self) -> list[Symbol]:
        return [s for s in self.symbols if s.kind in ("function", "method")]

    def classes(self) -> list[Symbol]:
        return [s for s in self.symbols if s.kind == "class"]

    def types(self) -> list[Symbol]:
        return [s for s in self.symbols if s.kind == "type"]

    def is_empty(self) -> bool:
        return len(self.symbols) == 0 and len(self.imports) == 0

    def render(self, max_symbols: int = 60) -> str:
        """Render as a compact text block suitable for LLM context."""
        lines: list[str] = []
        shown = 0
        for sym in self.symbols:
            if shown >= max_symbols:
                remaining = len(self.symbols) - shown
                lines.append(f"  ... ({remaining} more symbols)")
                break
            lines.append(f"  {sym}")
            shown += 1
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Language detection
# ─────────────────────────────────────────────────────────────────────────────

_EXT_LANG: dict[str, str] = {
    ".py":   "python",
    ".pyi":  "python",
    ".js":   "javascript",
    ".mjs":  "javascript",
    ".cjs":  "javascript",
    ".jsx":  "javascript",
    ".ts":   "typescript",
    ".tsx":  "typescript",
    ".go":   "go",
    ".rs":   "rust",
    ".java": "java",
    ".cs":   "csharp",
    ".c":    "c",
    ".h":    "c",
    ".cpp":  "cpp",
    ".cc":   "cpp",
    ".hpp":  "cpp",
    ".rb":   "ruby",
    ".php":  "php",
}


def _detect_language(path: Path) -> str:
    return _EXT_LANG.get(path.suffix.lower(), "unknown")


# ─────────────────────────────────────────────────────────────────────────────
# Python extractor  (ast-based)
# ─────────────────────────────────────────────────────────────────────────────

def _py_ann(node: ast.expr | None) -> str:
    """Unparse a type annotation node to string."""
    if node is None:
        return ""
    try:
        return ast.unparse(node)
    except Exception:
        return "?"


def _py_arg_str(arg: ast.arg) -> str:
    ann = f": {_py_ann(arg.annotation)}" if arg.annotation else ""
    return f"{arg.arg}{ann}"


def _py_func_sig(node: ast.FunctionDef | ast.AsyncFunctionDef, prefix: str = "") -> str:
    """Build a one-line function signature."""
    args = node.args
    parts: list[str] = []

    # positional-only args (before /)
    for i, a in enumerate(args.posonlyargs):
        parts.append(_py_arg_str(a))
    if args.posonlyargs:
        parts.append("/")

    # regular args
    num_defaults = len(args.defaults)
    num_args = len(args.args)
    for i, a in enumerate(args.args):
        default_idx = i - (num_args - num_defaults)
        arg_s = _py_arg_str(a)
        if default_idx >= 0:
            try:
                default_val = ast.unparse(args.defaults[default_idx])
                arg_s += f"={default_val}"
            except Exception:
                arg_s += "=..."
        parts.append(arg_s)

    if args.vararg:
        parts.append(f"*{_py_arg_str(args.vararg)}")
    elif args.kwonlyargs:
        parts.append("*")

    for i, a in enumerate(args.kwonlyargs):
        kw_s = _py_arg_str(a)
        if i < len(args.kw_defaults) and args.kw_defaults[i] is not None:
            try:
                kw_s += f"={ast.unparse(args.kw_defaults[i])}"
            except Exception:
                kw_s += "=..."
        parts.append(kw_s)

    if args.kwarg:
        parts.append(f"**{_py_arg_str(args.kwarg)}")

    ret = f" -> {_py_ann(node.returns)}" if node.returns else ""
    async_prefix = "async " if isinstance(node, ast.AsyncFunctionDef) else ""
    dec_prefix = ""
    # Include first decorator if it's a simple name/attr (e.g. @property, @staticmethod)
    if node.decorator_list:
        d = node.decorator_list[0]
        if isinstance(d, (ast.Name, ast.Attribute)):
            try:
                dec_prefix = f"@{ast.unparse(d)}\n    "
            except Exception:
                pass
    return f"{prefix}{dec_prefix}{async_prefix}def {node.name}({', '.join(parts)}){ret}"


def _py_get_docstring(node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) -> str:
    try:
        doc = ast.get_docstring(node, clean=True)
        if doc:
            return doc.splitlines()[0][:120]
    except Exception:
        pass
    return ""


def _py_is_dataclass(node: ast.ClassDef) -> bool:
    """Return True if the class is decorated with @dataclass or @dataclasses.dataclass."""
    for dec in node.decorator_list:
        func = dec
        if isinstance(dec, ast.Call):  # @dataclass(frozen=True)
            func = dec.func
        if isinstance(func, ast.Name) and func.id == "dataclass":
            return True
        if isinstance(func, ast.Attribute) and func.attr == "dataclass":
            return True
    return False


def _py_base_names(node: ast.ClassDef) -> set[str]:
    """Return the simple (unqualified) names of all base classes."""
    names: set[str] = set()
    for base in node.bases:
        if isinstance(base, ast.Name):
            names.add(base.id)
        elif isinstance(base, ast.Attribute):
            names.add(base.attr)
        elif isinstance(base, ast.Subscript) and isinstance(base.value, ast.Name):
            names.add(base.value.id)  # Generic[T] etc.
    return names


def _py_class_decorators(node: ast.ClassDef) -> str:
    """Build a decorator prefix string for a class (e.g. '@dataclass\n')."""
    lines: list[str] = []
    for dec in node.decorator_list:
        try:
            lines.append(f"@{ast.unparse(dec)}")
        except Exception:
            pass
    return ("\n".join(lines) + "\n") if lines else ""


def _extract_python(path: Path, source: str) -> SymbolIndex:
    idx = SymbolIndex(path=path, language="python")
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError:
        return idx

    for node in ast.walk(tree):
        # Top-level imports
        if isinstance(node, ast.Import):
            for alias in node.names:
                idx.imports.append(f"import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            dots = "." * (node.level or 0)   # preserve relative import dots
            mod = node.module or ""
            names = ", ".join(
                (a.asname if a.asname else a.name) for a in node.names
            )
            idx.imports.append(f"from {dots}{mod} import {names}")

    # Walk top level + class bodies for functions/classes
    def _walk_body(
        body: list[ast.stmt],
        class_name: str = "",
        is_dataclass_ctx: bool = False,
        is_typeddict_ctx: bool = False,
        is_protocol_ctx: bool = False,
    ) -> None:
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                prefix = "    " if class_name else ""
                sig = _py_func_sig(node, prefix)
                if class_name:
                    kind = "protocol_method" if is_protocol_ctx else "method"
                else:
                    kind = "function"
                doc = _py_get_docstring(node)
                # Skip __init__ bodies for dataclasses — fields are captured as annassign
                if is_dataclass_ctx and node.name == "__init__":
                    continue
                idx.symbols.append(Symbol(
                    kind=kind, name=node.name,
                    signature=sig, line=node.lineno, docstring=doc,
                ))

            elif isinstance(node, ast.ClassDef):
                bases = ", ".join(_py_ann(b) for b in node.bases) if node.bases else ""
                base_str = f"({bases})" if bases else ""
                doc = _py_get_docstring(node)
                is_dc  = _py_is_dataclass(node)
                bnames = _py_base_names(node)
                is_td  = "TypedDict" in bnames
                is_pt  = "Protocol" in bnames
                # Build class signature: include decorator(s) and tag special kinds
                dec_str = _py_class_decorators(node)
                tag = ""
                if is_td:  tag = "  # TypedDict"
                elif is_pt: tag = "  # Protocol"
                sig = f"{dec_str}class {node.name}{base_str}:{tag}"
                idx.symbols.append(Symbol(
                    kind="class", name=node.name,
                    signature=sig, line=node.lineno, docstring=doc,
                ))
                _walk_body(
                    node.body,
                    class_name=node.name,
                    is_dataclass_ctx=is_dc,
                    is_typeddict_ctx=is_td,
                    is_protocol_ctx=is_pt,
                )

            elif isinstance(node, ast.Assign):
                if class_name:
                    continue
                for target in node.targets:
                    if not isinstance(target, ast.Name):
                        continue
                    # __all__ = [...] — public export list
                    if target.id == "__all__":
                        try:
                            if isinstance(node.value, (ast.List, ast.Tuple)):
                                names_list = [
                                    ast.unparse(elt)
                                    for elt in node.value.elts
                                    if isinstance(elt, ast.Constant)
                                ]
                                preview = ", ".join(names_list[:8])
                                if len(names_list) > 8:
                                    preview += f", ... ({len(names_list)} total)"
                                idx.symbols.append(Symbol(
                                    kind="export", name="__all__",
                                    signature=f"__all__ = [{preview}]",
                                    line=node.lineno,
                                ))
                        except Exception:
                            pass
                    # UPPER_CASE top-level constants
                    elif target.id.isupper():
                        try:
                            val = ast.unparse(node.value)
                            if len(val) > 60:
                                val = val[:60] + "..."
                            idx.symbols.append(Symbol(
                                kind="const", name=target.id,
                                signature=f"{target.id} = {val}",
                                line=node.lineno,
                            ))
                        except Exception:
                            pass

            elif isinstance(node, ast.AnnAssign):
                if not isinstance(node.target, ast.Name):
                    continue
                try:
                    ann = _py_ann(node.annotation)
                    name = node.target.id
                    # Skip ClassVar / private fields in normal classes
                    if class_name and not (is_dataclass_ctx or is_typeddict_ctx):
                        continue
                    if class_name:
                        # Dataclass field or TypedDict key
                        default_part = ""
                        if node.value is not None:
                            try:
                                dv = ast.unparse(node.value)
                                if len(dv) > 40:
                                    dv = dv[:40] + "..."
                                default_part = f" = {dv}"
                            except Exception:
                                pass
                        kind = "field"
                        idx.symbols.append(Symbol(
                            kind=kind, name=name,
                            signature=f"    {name}: {ann}{default_part}",
                            line=node.lineno,
                        ))
                    else:
                        # Top-level annotated name (module-level type alias / var)
                        idx.symbols.append(Symbol(
                            kind="type", name=name,
                            signature=f"{name}: {ann}",
                            line=node.lineno,
                        ))
                except Exception:
                    pass

    _walk_body(tree.body)
    return idx


# ─────────────────────────────────────────────────────────────────────────────
# JavaScript / TypeScript extractor  (regex-based)
# ─────────────────────────────────────────────────────────────────────────────

_JS_IMPORT    = re.compile(r'^(?:import\s+.*?from\s+[\'"]([^\'"]+)[\'"]|require\([\'"]([^\'"]+)[\'"]\))', re.MULTILINE)
_JS_FUNCTION  = re.compile(
    r'^(?:export\s+)?(?:export\s+default\s+)?(?:async\s+)?function\s*\*?\s*(\w+)\s*(<[^>]*>)?\s*(\([^)]*\))\s*(?::\s*[\w<>\[\]|&\s,\.]+)?',
    re.MULTILINE
)
_JS_ARROW     = re.compile(
    r'^(?:export\s+)?(?:const|let|var)\s+(\w+)\s*(?::\s*[^=]+)?\s*=\s*(?:async\s+)?(?:<[^>]*>)?\s*(\([^)]*\)|[\w]+)\s*(?::\s*[\w<>\[\]|&\s,\.]+)?\s*=>',
    re.MULTILINE
)
_JS_CLASS     = re.compile(r'^(?:export\s+)?(?:abstract\s+)?class\s+(\w+)(?:\s+extends\s+([\w.<>, ]+))?(?:\s+implements\s+([\w.<>, ]+))?', re.MULTILINE)
_JS_METHOD    = re.compile(r'^\s+(?:(?:public|private|protected|static|async|readonly|abstract|override)\s+)*(\w+)\s*(?:<[^>]*>)?\s*(\([^)]*\))\s*(?::\s*[\w<>\[\]|&\s,\.]+)?\s*\{', re.MULTILINE)
_TS_INTERFACE = re.compile(r'^(?:export\s+)?interface\s+(\w+)(?:\s+extends\s+([\w,\s]+))?', re.MULTILINE)
_TS_TYPE      = re.compile(r'^(?:export\s+)?type\s+(\w+)(?:<[^>]*>)?\s*=', re.MULTILINE)


def _extract_js(path: Path, source: str, lang: str) -> SymbolIndex:
    idx = SymbolIndex(path=path, language=lang)

    for m in _JS_IMPORT.finditer(source):
        mod = m.group(1) or m.group(2) or ""
        idx.imports.append(f"import '{mod}'")

    for m in _JS_FUNCTION.finditer(source):
        sig = f"function {m.group(1)}{m.group(3) or '()'}"
        idx.symbols.append(Symbol(kind="function", name=m.group(1), signature=sig, line=source[:m.start()].count("\n") + 1))

    for m in _JS_ARROW.finditer(source):
        sig = f"const {m.group(1)} = ({m.group(2)}) => ..."
        idx.symbols.append(Symbol(kind="function", name=m.group(1), signature=sig, line=source[:m.start()].count("\n") + 1))

    for m in _JS_CLASS.finditer(source):
        extends = f" extends {m.group(2)}" if m.group(2) else ""
        sig = f"class {m.group(1)}{extends}"
        idx.symbols.append(Symbol(kind="class", name=m.group(1), signature=sig, line=source[:m.start()].count("\n") + 1))

    if lang == "typescript":
        for m in _TS_INTERFACE.finditer(source):
            extends = f" extends {m.group(2)}" if m.group(2) else ""
            sig = f"interface {m.group(1)}{extends}"
            idx.symbols.append(Symbol(kind="type", name=m.group(1), signature=sig, line=source[:m.start()].count("\n") + 1))

        for m in _TS_TYPE.finditer(source):
            sig = f"type {m.group(1)} = ..."
            idx.symbols.append(Symbol(kind="type", name=m.group(1), signature=sig, line=source[:m.start()].count("\n") + 1))

    return idx


# ─────────────────────────────────────────────────────────────────────────────
# Go extractor  (regex-based)
# ─────────────────────────────────────────────────────────────────────────────

_GO_IMPORT   = re.compile(r'"([\w./\-]+)"')
_GO_FUNC     = re.compile(r'^func\s+(?:\(\s*\w+\s+[\*]?(\w+)\s*\)\s+)?(\w+)\s*(\([^)]*(?:\([^)]*\)[^)]*)*\))\s*([\(\w\*\[\], ]*)', re.MULTILINE)
_GO_TYPE     = re.compile(r'^type\s+(\w+)\s+(struct|interface|[\w\[\]]+)', re.MULTILINE)


def _extract_go(path: Path, source: str) -> SymbolIndex:
    idx = SymbolIndex(path=path, language="go")

    import_block = re.search(r'import\s*\(([^)]+)\)', source, re.DOTALL)
    if import_block:
        for m in _GO_IMPORT.finditer(import_block.group(1)):
            idx.imports.append(f'import "{m.group(1)}"')
    single = re.finditer(r'^import\s+"([\w./\-]+)"', source, re.MULTILINE)
    for m in single:
        idx.imports.append(f'import "{m.group(1)}"')

    for m in _GO_FUNC.finditer(source):
        receiver = f"({m.group(1)}) " if m.group(1) else ""
        returns = f" {m.group(4).strip()}" if m.group(4) and m.group(4).strip() else ""
        sig = f"func {receiver}{m.group(2)}{m.group(3)}{returns}"
        idx.symbols.append(Symbol(kind="function", name=m.group(2), signature=sig, line=source[:m.start()].count("\n") + 1))

    for m in _GO_TYPE.finditer(source):
        sig = f"type {m.group(1)} {m.group(2)}"
        kind = "class" if m.group(2) in ("struct", "interface") else "type"
        idx.symbols.append(Symbol(kind=kind, name=m.group(1), signature=sig, line=source[:m.start()].count("\n") + 1))

    return idx


# ─────────────────────────────────────────────────────────────────────────────
# Rust extractor  (regex-based)
# ─────────────────────────────────────────────────────────────────────────────

_RS_USE    = re.compile(r'^use\s+([\w::{}, ]+);', re.MULTILINE)
_RS_FN     = re.compile(r'^(?:pub(?:\s*\([^)]*\))?\s+)?(?:async\s+)?fn\s+(\w+)(?:<[^>]*>)?\s*(\([^)]*\))\s*(?:->\s*([\w<>\[\]:&\s,\']+))?', re.MULTILINE)
_RS_STRUCT = re.compile(r'^(?:pub(?:\s*\([^)]*\))?\s+)?struct\s+(\w+)(?:<[^>]*>)?', re.MULTILINE)
_RS_ENUM   = re.compile(r'^(?:pub(?:\s*\([^)]*\))?\s+)?enum\s+(\w+)(?:<[^>]*>)?', re.MULTILINE)
_RS_TRAIT  = re.compile(r'^(?:pub(?:\s*\([^)]*\))?\s+)?trait\s+(\w+)(?:<[^>]*>)?', re.MULTILINE)
_RS_IMPL   = re.compile(r'^impl(?:<[^>]*>)?\s+(?:(\w+)\s+for\s+)?(\w+)(?:<[^>]*>)?', re.MULTILINE)


def _extract_rust(path: Path, source: str) -> SymbolIndex:
    idx = SymbolIndex(path=path, language="rust")

    for m in _RS_USE.finditer(source):
        idx.imports.append(f"use {m.group(1)};")

    for m in _RS_FN.finditer(source):
        ret = f" -> {m.group(3).strip()}" if m.group(3) else ""
        sig = f"fn {m.group(1)}{m.group(2)}{ret}"
        idx.symbols.append(Symbol(kind="function", name=m.group(1), signature=sig, line=source[:m.start()].count("\n") + 1))

    for m in _RS_STRUCT.finditer(source):
        idx.symbols.append(Symbol(kind="class", name=m.group(1), signature=f"struct {m.group(1)}", line=source[:m.start()].count("\n") + 1))

    for m in _RS_ENUM.finditer(source):
        idx.symbols.append(Symbol(kind="type", name=m.group(1), signature=f"enum {m.group(1)}", line=source[:m.start()].count("\n") + 1))

    for m in _RS_TRAIT.finditer(source):
        idx.symbols.append(Symbol(kind="type", name=m.group(1), signature=f"trait {m.group(1)}", line=source[:m.start()].count("\n") + 1))

    for m in _RS_IMPL.finditer(source):
        target = m.group(2)
        trait  = f"{m.group(1)} for " if m.group(1) else ""
        idx.symbols.append(Symbol(kind="class", name=target, signature=f"impl {trait}{target}", line=source[:m.start()].count("\n") + 1))

    return idx


# ─────────────────────────────────────────────────────────────────────────────
# Java / C# extractor  (regex-based)
# ─────────────────────────────────────────────────────────────────────────────

_JAVA_IMPORT  = re.compile(r'^import\s+([\w.]+);', re.MULTILINE)
_JAVA_CLASS   = re.compile(r'^(?:public\s+|private\s+|protected\s+)?(?:abstract\s+|final\s+)?(?:class|interface|enum|record)\s+(\w+)(?:<[^>]*>)?(?:\s+extends\s+([\w, ]+))?(?:\s+implements\s+([\w, ]+))?', re.MULTILINE)
_JAVA_METHOD  = re.compile(r'^\s+(?:(?:public|private|protected|static|final|abstract|synchronized|native|default|override)\s+)*(?:[\w<>\[\]]+\s+)+(\w+)\s*(\([^)]*\))\s*(?:throws\s+[\w, ]+)?\s*\{', re.MULTILINE)


def _extract_java(path: Path, source: str) -> SymbolIndex:
    idx = SymbolIndex(path=path, language="java")

    for m in _JAVA_IMPORT.finditer(source):
        idx.imports.append(f"import {m.group(1)};")

    for m in _JAVA_CLASS.finditer(source):
        extends = f" extends {m.group(2)}" if m.group(2) else ""
        sig = f"class {m.group(1)}{extends}"
        idx.symbols.append(Symbol(kind="class", name=m.group(1), signature=sig, line=source[:m.start()].count("\n") + 1))

    for m in _JAVA_METHOD.finditer(source):
        parts = m.group(0).strip().rstrip("{").strip()
        idx.symbols.append(Symbol(kind="method", name=m.group(1), signature=parts, line=source[:m.start()].count("\n") + 1))

    return idx


# ─────────────────────────────────────────────────────────────────────────────
# C / C++ extractor  (regex-based)
# ─────────────────────────────────────────────────────────────────────────────

_C_INCLUDE = re.compile(r'^#include\s+[<"]([\w./]+)[>"]', re.MULTILINE)
_C_FUNC    = re.compile(r'^(?:static\s+|inline\s+|extern\s+)?(?:[\w:*&<>]+\s+)+(\w+)\s*(\([^)]+\))\s*(?:const\s*)?\{', re.MULTILINE)
_C_STRUCT  = re.compile(r'^(?:typedef\s+)?(?:struct|class|union)\s+(\w+)', re.MULTILINE)


def _extract_c(path: Path, source: str, lang: str) -> SymbolIndex:
    idx = SymbolIndex(path=path, language=lang)

    for m in _C_INCLUDE.finditer(source):
        idx.imports.append(f"#include <{m.group(1)}>")

    for m in _C_FUNC.finditer(source):
        name = m.group(1)
        if name in ("if", "for", "while", "switch", "return"):
            continue
        sig = m.group(0).rstrip("{").strip()
        if len(sig) > 120:
            sig = sig[:120] + "..."
        idx.symbols.append(Symbol(kind="function", name=name, signature=sig, line=source[:m.start()].count("\n") + 1))

    for m in _C_STRUCT.finditer(source):
        idx.symbols.append(Symbol(kind="class", name=m.group(1), signature=f"struct {m.group(1)}", line=source[:m.start()].count("\n") + 1))

    return idx


# ─────────────────────────────────────────────────────────────────────────────
# Public: extract_symbols
# ─────────────────────────────────────────────────────────────────────────────

_MAX_FILE_BYTES = 512 * 1024   # skip files larger than 512KB


def extract_symbols(path: Path) -> SymbolIndex:
    """
    Parse *path* and return a SymbolIndex.
    Returns an empty index on any error.
    """
    lang = _detect_language(path)
    if lang == "unknown":
        return SymbolIndex(path=path, language="unknown")

    try:
        if path.stat().st_size > _MAX_FILE_BYTES:
            return SymbolIndex(path=path, language=lang)
        source = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return SymbolIndex(path=path, language=lang)

    if lang == "python":
        return _extract_python(path, source)
    elif lang in ("javascript", "typescript"):
        return _extract_js(path, source, lang)
    elif lang == "go":
        return _extract_go(path, source)
    elif lang == "rust":
        return _extract_rust(path, source)
    elif lang in ("java", "csharp"):
        return _extract_java(path, source)
    elif lang in ("c", "cpp"):
        return _extract_c(path, source, lang)
    else:
        return SymbolIndex(path=path, language=lang)


# ─────────────────────────────────────────────────────────────────────────────
# Dependency graph  (replaces old resolve_imports + _import_to_paths)
# ─────────────────────────────────────────────────────────────────────────────

# Python stdlib module names (3.10+ ships sys.stdlib_module_names; fallback set for older)
try:
    _PY_STDLIB: frozenset[str] = frozenset(sys.stdlib_module_names)  # type: ignore[attr-defined]
except AttributeError:
    _PY_STDLIB = frozenset({
        "abc", "ast", "asyncio", "base64", "builtins", "collections", "concurrent",
        "contextlib", "copy", "dataclasses", "datetime", "decimal", "difflib",
        "email", "enum", "fnmatch", "functools", "gc", "glob", "hashlib", "heapq",
        "http", "importlib", "inspect", "io", "itertools", "json", "logging",
        "math", "multiprocessing", "operator", "os", "pathlib", "pickle", "platform",
        "pprint", "queue", "random", "re", "shutil", "signal", "socket", "sqlite3",
        "ssl", "stat", "string", "struct", "subprocess", "sys", "tempfile", "textwrap",
        "threading", "time", "traceback", "types", "typing", "unittest", "urllib",
        "uuid", "warnings", "weakref", "xml", "zipfile",
    })


# Pre-compiled regexes for Python import resolution in DependencyGraph.
# Matches both "import foo.bar" and "from foo.bar import X" — group(1) is the
# module path in both cases (stops at whitespace before 'import').
_PY_ABS_IMPORT_RE = re.compile(r'^(?:from\s+|import\s+)([\w.]+)')
# Extracts the names list from "from foo.bar import X, Y, Z" — group(1) = "X, Y, Z"
_PY_FROM_IMPORT_RE = re.compile(r'^from\s+[\w.]+\s+import\s+(.+)$')


@dataclass
class DependencyNode:
    """All import information extracted from one source file."""
    path: Path                         # resolved absolute path
    language: str
    dep_files: list[Path]              # workspace-local files this file imports
    external_pkgs: list[str]           # third-party / stdlib package names
    raw_imports: list[str]             # raw import strings (for display / debug)
    mtime: float = 0.0                 # mtime at parse time (staleness check)


class DependencyGraph:
    """
    An in-memory dependency graph for a workspace.

    Features
    --------
    - Forward edges  : file → [files it imports]
    - Reverse edges  : file → [files that import it]  ("dependents")
    - External pkg tracking : which third-party packages each file uses
    - Mtime-based staleness : each node stores the mtime at parse time;
      call invalidate(path) or refresh(path) when a file changes
    - tsconfig.paths alias resolution for TypeScript projects
    - __init__.py re-export chasing for Python packages
    """

    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace.resolve()
        self._nodes:   dict[str, DependencyNode]  = {}   # resolved str → node
        self._reverse: dict[str, set[str]]         = {}   # path_str → set(importers)
        self._lock  = threading.Lock()
        self._path_aliases:  dict[str, str] = {}    # @alias → base-dir (from tsconfig)
        self._aliases_loaded = False
        # Can be overridden to share a symbol cache with WorkspaceIndex.
        # Default: call extract_symbols() directly (no shared cache).
        self._symbol_fetcher: Callable[[Path], SymbolIndex] = extract_symbols

    # ── Cache helpers ────────────────────────────────────────────────────────

    def _key(self, path: Path) -> str:
        return str(path.resolve())

    def _is_stale(self, node: DependencyNode) -> bool:
        try:
            return node.path.stat().st_mtime != node.mtime
        except Exception:
            return True

    def _add_to_reverse(self, importer_key: str, imported_key: str) -> None:
        if imported_key not in self._reverse:
            self._reverse[imported_key] = set()
        self._reverse[imported_key].add(importer_key)

    def _remove_from_reverse(self, importer_key: str) -> None:
        """Remove an importer from all reverse edges it appears in."""
        for deps in self._reverse.values():
            deps.discard(importer_key)

    # ── tsconfig.paths loader ────────────────────────────────────────────────

    def _load_tsconfig_paths(self) -> None:
        if self._aliases_loaded:
            return
        self._aliases_loaded = True
        for tsconfig_name in ("tsconfig.json", "tsconfig.base.json", "jsconfig.json"):
            ts = self.workspace / tsconfig_name
            if not ts.exists():
                continue
            try:
                data  = json.loads(ts.read_text(encoding="utf-8"))
                paths = data.get("compilerOptions", {}).get("paths", {})
                base  = data.get("compilerOptions", {}).get("baseUrl", ".")
                base_dir = (self.workspace / base).resolve()
                for alias_pattern, targets in paths.items():
                    # alias_pattern is like "@app/*" or "@utils"
                    # targets is like ["src/*"] or ["src/utils"]
                    if not targets:
                        continue
                    # Strip trailing /* for prefix matching
                    key = alias_pattern.rstrip("/*")
                    target_rel = targets[0].rstrip("/*")
                    self._path_aliases[key] = str((base_dir / target_rel).resolve())
            except Exception:
                pass

    # ── Language-specific import resolvers ──────────────────────────────────

    def _resolve_python_import(
        self, import_str: str, from_file: Path,
    ) -> tuple[list[Path], list[str]]:
        """
        Returns (local_files, external_names).
        Handles:
          - absolute imports: `import pkg.sub` / `from pkg.sub import X`
          - relative imports: `from .sister import Y`
          - __init__.py re-export chaining for `from pkg import X`
        """
        local: list[Path]  = []
        ext:   list[str]   = []

        # Relative imports: `from . import X` / `from .mod import Y`
        rel_m = re.match(r'from ([\.]+)(\w[\w.]*)?\s+import\s+(.+)', import_str)
        if rel_m:
            dots  = rel_m.group(1)
            mod   = rel_m.group(2) or ""
            names = [n.strip().split(" as ")[0] for n in rel_m.group(3).split(",")]
            base  = from_file.parent
            for _ in range(len(dots) - 1):   # one dot = same dir, two dots = parent
                base = base.parent
            if mod:
                sub = base / Path(mod.replace(".", "/"))
                for candidate in (sub.with_suffix(".py"), sub / "__init__.py"):
                    if candidate.exists():
                        rp = candidate.resolve()
                        local.append(rp)
                        # Re-export chaining: if it's an __init__.py, follow re-exports
                        if candidate.name == "__init__.py":
                            reexports = _resolve_reexports(candidate, self.workspace)
                            for name in names:
                                if name in reexports and reexports[name] not in local:
                                    local.append(reexports[name])
                        break
            else:
                # `from . import name` — each name is a sibling module
                for name in names:
                    for candidate in (base / f"{name}.py", base / name / "__init__.py"):
                        if candidate.exists():
                            local.append(candidate.resolve())
                            break
            return local, ext

        # Absolute imports
        m = _PY_ABS_IMPORT_RE.match(import_str)
        if not m:
            return local, ext

        mod_name  = m.group(1)
        top_level = mod_name.split(".")[0]

        # Stdlib → external (informational)
        if top_level in _PY_STDLIB:
            ext.append(top_level)
            return local, ext

        # Try to resolve in workspace
        rel = Path(mod_name.replace(".", "/"))
        resolved_in_ws = False
        for base in (from_file.parent, self.workspace):
            for candidate in (base / (str(rel) + ".py"), base / str(rel) / "__init__.py"):
                if candidate.exists():
                    rp = candidate.resolve()
                    local.append(rp)
                    resolved_in_ws = True
                    # Re-export chaining for `from pkg import X`
                    if candidate.name == "__init__.py":
                        rest = _PY_FROM_IMPORT_RE.match(import_str)
                        if rest:
                            names = [n.strip().split(" as ")[0] for n in rest.group(1).split(",")]
                            reexports = _resolve_reexports(candidate, self.workspace)
                            for name in names:
                                if name in reexports and reexports[name] not in local:
                                    local.append(reexports[name])
                    break
            if resolved_in_ws:
                break

        if not resolved_in_ws:
            # Third-party package
            ext.append(top_level)

        return local, ext

    def _resolve_js_import(
        self, import_str: str, from_file: Path,
    ) -> tuple[list[Path], list[str]]:
        """
        Resolve a JS/TS import string.
        Handles: relative paths, @alias paths (tsconfig.paths), and bare specifiers.
        """
        self._load_tsconfig_paths()
        local: list[Path] = []
        ext:   list[str]  = []

        m = re.match(r"import '([^']+)'", import_str)
        if not m:
            return local, ext
        raw = m.group(1)

        # Relative import
        if raw.startswith("."):
            base = from_file.parent / raw
            for ext_sfx in (".ts", ".tsx", ".js", ".jsx", "/index.ts", "/index.tsx", "/index.js"):
                candidate = Path(str(base) + ext_sfx) if not raw.endswith(ext_sfx.lstrip("/")) else Path(str(base))
                # Cleaner approach: strip and try extensions
                stripped = (from_file.parent / raw)
                if ext_sfx.startswith("/"):
                    candidate = stripped / ext_sfx.lstrip("/")
                else:
                    candidate = stripped.with_suffix(ext_sfx) if not stripped.suffix else stripped
                    if not candidate.suffix:
                        candidate = Path(str(stripped) + ext_sfx)
                if candidate.exists():
                    local.append(candidate.resolve())
                    return local, ext
            return local, ext

        # @alias import (tsconfig.paths)
        for alias_prefix, alias_dir in self._path_aliases.items():
            if raw == alias_prefix or raw.startswith(alias_prefix + "/"):
                remainder = raw[len(alias_prefix):].lstrip("/")
                base = Path(alias_dir) / remainder
                for ext_sfx in (".ts", ".tsx", ".js", ".jsx", "/index.ts", "/index.js"):
                    if ext_sfx.startswith("/"):
                        candidate = base / ext_sfx.lstrip("/")
                    else:
                        candidate = Path(str(base) + ext_sfx)
                    if candidate.exists():
                        local.append(candidate.resolve())
                        return local, ext
                break

        # Bare specifier — external npm package
        if not raw.startswith(".") and not raw.startswith("/"):
            top = raw.split("/")[0]
            # Handle scoped packages: @scope/pkg
            if raw.startswith("@") and "/" in raw:
                top = "/".join(raw.split("/")[:2])
            ext.append(top)

        return local, ext

    def _resolve_go_import(
        self, import_str: str, from_file: Path,
    ) -> tuple[list[Path], list[str]]:
        local: list[Path] = []
        ext:   list[str]  = []
        m = re.match(r'import "([^"]+)"', import_str)
        if not m:
            return local, ext
        pkg_path = m.group(1)
        pkg_last = pkg_path.split("/")[-1]

        # Check if it looks like a workspace-local package (contains module path prefix)
        gomod = self.workspace / "go.mod"
        module_prefix = ""
        if gomod.exists():
            try:
                first_line = gomod.read_text().splitlines()[0]
                module_prefix = first_line.replace("module ", "").strip()
            except Exception:
                pass

        if module_prefix and pkg_path.startswith(module_prefix):
            rel_pkg = pkg_path[len(module_prefix):].lstrip("/")
            candidate_dir = self.workspace / rel_pkg
            if candidate_dir.is_dir():
                for go_file in sorted(candidate_dir.glob("*.go")):
                    local.append(go_file.resolve())
                    break
        else:
            # Try last-component heuristic for workspace-local
            found = False
            for p in self.workspace.rglob(f"{pkg_last}/*.go"):
                local.append(p.resolve())
                found = True
                break
            if not found:
                ext.append(pkg_path)

        return local, ext

    def _resolve_rust_import(
        self, import_str: str, from_file: Path,
    ) -> tuple[list[Path], list[str]]:
        local: list[Path] = []
        ext:   list[str]  = []
        # `use crate::module::Type` → workspace-local
        m = re.match(r'use\s+(?:crate|self|super)::(\S+)', import_str)
        if m:
            parts = m.group(1).split("::")[0]   # first path segment
            for candidate in (
                from_file.parent / f"{parts}.rs",
                from_file.parent / parts / "mod.rs",
            ):
                if candidate.exists():
                    local.append(candidate.resolve())
                    break
        else:
            # External crate
            m2 = re.match(r'use\s+([\w]+)', import_str)
            if m2 and m2.group(1) not in ("std", "core", "alloc"):
                ext.append(m2.group(1))
        return local, ext

    def _resolve_java_import(
        self, import_str: str, from_file: Path,
    ) -> tuple[list[Path], list[str]]:
        local: list[Path] = []
        ext:   list[str]  = []
        m = re.match(r'import\s+([\w.]+);', import_str)
        if not m:
            return local, ext
        fqn     = m.group(1)
        rel_path = fqn.replace(".", "/") + ".java"
        for candidate in self.workspace.rglob(Path(rel_path).name):
            if str(candidate).replace("\\", "/").endswith(rel_path.replace("\\", "/")):
                local.append(candidate.resolve())
                break
        if not local:
            top = fqn.split(".")[0]
            if top not in ("java", "javax", "com", "org", "net"):
                ext.append(top)
            else:
                ext.append(".".join(fqn.split(".")[:2]))
        return local, ext

    # ── Core build method ───────────────────────────────────────────────────────

    def build_file(self, path: Path) -> DependencyNode:
        """
        Parse *path*, extract its imports, resolve them, and cache the result.
        Returns the (possibly freshly-built) DependencyNode.
        Idempotent: returns the cached node if the file hasn't changed.
        """
        rpath = path.resolve()
        key   = str(rpath)

        with self._lock:
            existing = self._nodes.get(key)
            if existing and not self._is_stale(existing):
                return existing

        idx = self._symbol_fetcher(rpath)
        lang = idx.language

        dep_files:    list[Path] = []
        ext_pkgs:     list[str]  = []
        seen_local:   set[str]   = set()
        seen_ext:     set[str]   = set()

        for imp_str in idx.imports:
            if lang == "python":
                lf, ef = self._resolve_python_import(imp_str, rpath)
            elif lang in ("javascript", "typescript"):
                lf, ef = self._resolve_js_import(imp_str, rpath)
            elif lang == "go":
                lf, ef = self._resolve_go_import(imp_str, rpath)
            elif lang == "rust":
                lf, ef = self._resolve_rust_import(imp_str, rpath)
            elif lang in ("java", "csharp"):
                lf, ef = self._resolve_java_import(imp_str, rpath)
            else:
                lf, ef = [], []

            for p in lf:
                ps = str(p)
                if ps not in seen_local and ps != key:
                    seen_local.add(ps)
                    dep_files.append(p)

            for e in ef:
                if e not in seen_ext:
                    seen_ext.add(e)
                    ext_pkgs.append(e)

        try:
            mtime = rpath.stat().st_mtime
        except Exception:
            mtime = 0.0

        node = DependencyNode(
            path=rpath,
            language=lang,
            dep_files=dep_files,
            external_pkgs=ext_pkgs,
            raw_imports=list(idx.imports),
            mtime=mtime,
        )

        with self._lock:
            # Remove stale reverse-edge entries for this file
            self._remove_from_reverse(key)
            self._nodes[key] = node
            for dep in dep_files:
                self._add_to_reverse(key, str(dep))

        return node

    def build_workspace(
        self,
        max_files: int = 2000,
        progress_callback: Optional[object] = None,
    ) -> int:
        """
        Parse every source file in the workspace and cache nodes.
        Returns the number of files indexed.
        Does not re-parse files whose mtime hasn't changed.
        """
        count = 0
        for fpath in _iter_workspace_files(self.workspace):
            if count >= max_files:
                break
            try:
                self.build_file(fpath)
                count += 1
            except Exception:
                pass
        return count

    def invalidate(self, path: Path) -> None:
        """Remove a file from the cache. Next access will re-parse it."""
        key = str(path.resolve())
        with self._lock:
            if key in self._nodes:
                self._remove_from_reverse(key)
                del self._nodes[key]

    # ── Graph query API ────────────────────────────────────────────────────────

    def dependencies(self, path: Path) -> list[Path]:
        """Files that *path* directly imports (first-level only)."""
        node = self.build_file(path)
        return list(node.dep_files)

    def dependents(self, path: Path) -> list[Path]:
        """Files that directly import *path* (reverse lookup)."""
        key = str(path.resolve())
        with self._lock:
            importer_keys = list(self._reverse.get(key, set()))
        return [Path(k) for k in importer_keys]

    def transitive_dependencies(self, path: Path, max_depth: int = 4) -> list[Path]:
        """
        All files reachable from *path* through the import graph, up to *max_depth*.
        Returns a deduplicated list in BFS order.
        """
        visited: set[str] = set()
        queue:   deque[tuple[Path, int]] = deque([(path.resolve(), 0)])
        result:  list[Path] = []

        while queue:
            current, depth = queue.popleft()
            key = str(current)
            if key in visited or depth > max_depth:
                continue
            visited.add(key)
            if depth > 0:   # skip the root itself
                result.append(current)
            try:
                node = self.build_file(current)
                for dep in node.dep_files:
                    if str(dep) not in visited:
                        queue.append((dep, depth + 1))
            except Exception:
                pass

        return result

    def transitive_dependents(self, path: Path, max_depth: int = 4) -> list[Path]:
        """All files that (transitively) import *path*."""
        visited: set[str] = set()
        queue:   deque[tuple[Path, int]] = deque([(path.resolve(), 0)])
        result:  list[Path] = []

        while queue:
            current, depth = queue.popleft()
            key = str(current)
            if key in visited or depth > max_depth:
                continue
            visited.add(key)
            if depth > 0:
                result.append(current)
            for imp_path in self.dependents(current):
                if str(imp_path) not in visited:
                    queue.append((imp_path, depth + 1))

        return result

    def external_packages(self, path: Path) -> list[str]:
        """External (third-party) package names directly imported by *path*."""
        try:
            node = self.build_file(path)
            return list(node.external_pkgs)
        except Exception:
            return []

    def all_external_packages(self) -> dict[str, list[str]]:
        """
        Returns {package_name: [relative_file_paths]} for all cached nodes.
        Useful for auditing third-party dependencies.
        """
        result: dict[str, list[str]] = {}
        with self._lock:
            nodes = list(self._nodes.values())
        for node in nodes:
            for pkg in node.external_pkgs:
                try:
                    rel = str(node.path.relative_to(self.workspace))
                except ValueError:
                    rel = str(node.path)
                result.setdefault(pkg, []).append(rel)
        return result

    def detect_cycles(self) -> list[list[str]]:
        """
        Detect circular imports using iterative DFS.
        Returns a list of cycles, each cycle is a list of relative file paths.
        """
        with self._lock:
            keys = list(self._nodes.keys())

        cycles: list[list[str]] = []
        visited:    set[str] = set()
        in_stack:   set[str] = set()

        def _dfs(key: str, stack: list[str]) -> None:
            if key in in_stack:
                # Found a cycle — extract the loop portion
                start = stack.index(key)
                cycle_keys = stack[start:] + [key]
                cycle_rels: list[str] = []
                for k in cycle_keys:
                    try:
                        cycle_rels.append(str(Path(k).relative_to(self.workspace)))
                    except ValueError:
                        cycle_rels.append(k)
                cycles.append(cycle_rels)
                return
            if key in visited:
                return
            visited.add(key)
            in_stack.add(key)
            stack.append(key)

            with self._lock:
                node = self._nodes.get(key)
            if node:
                for dep in node.dep_files:
                    _dfs(str(dep), stack)

            stack.pop()
            in_stack.discard(key)

        for k in keys:
            if k not in visited:
                _dfs(k, [])

        return cycles

    # ── Render helpers (for LLM context / tool output) ─────────────────────

    def render_tree(
        self, path: Path, max_depth: int = 3, _depth: int = 0, _seen: Optional[set] = None
    ) -> str:
        """Render a dependency tree rooted at *path* as a text block."""
        if _seen is None:
            _seen = set()
        key = str(path.resolve())
        try:
            rel = str(path.resolve().relative_to(self.workspace))
        except ValueError:
            rel = str(path)
        if key in _seen or _depth > max_depth:
            return ("  " * _depth) + f"{rel}  (cycle or max depth)"
        _seen.add(key)
        lines: list[str] = ["  " * _depth + rel]
        if _depth < max_depth:
            for dep in self.dependencies(path):
                sub = self.render_tree(dep, max_depth, _depth + 1, _seen)
                lines.append(sub)
        return "\n".join(lines)

    def render_dependents(self, path: Path, max_depth: int = 3) -> str:
        """Render the reverse-dependency tree (files that import *path*)."""
        try:
            rel = str(path.resolve().relative_to(self.workspace))
        except ValueError:
            rel = str(path)
        lines = [f"{rel}  is imported by:"]
        for dep in self.transitive_dependents(path, max_depth=max_depth):
            try:
                dr = str(dep.relative_to(self.workspace))
            except ValueError:
                dr = str(dep)
            lines.append(f"  {dr}")
        return "\n".join(lines) if len(lines) > 1 else f"{rel}  (no dependents found)"

    def stats(self) -> str:
        """Return a summary string of graph statistics."""
        with self._lock:
            n_nodes = len(self._nodes)
            n_edges = sum(len(node.dep_files) for node in self._nodes.values())
            all_ext: set[str] = set()
            for node in self._nodes.values():
                all_ext.update(node.external_pkgs)
        return (
            f"{n_nodes} files indexed, "
            f"{n_edges} import edges, "
            f"{len(all_ext)} distinct external packages"
        )


# ── Module-level cache  (one graph per workspace path) ───────────────────────

_GRAPHS: dict[str, DependencyGraph] = {}
_GRAPHS_LOCK = threading.Lock()


def get_dependency_graph(workspace: Path) -> DependencyGraph:
    """
    Return the singleton DependencyGraph for *workspace*.
    Creates it on first call (does NOT build the full index — nodes are built
    lazily on first access). Call graph.build_workspace() for a full upfront index.
    """
    key = str(workspace.resolve())
    with _GRAPHS_LOCK:
        if key not in _GRAPHS:
            _GRAPHS[key] = DependencyGraph(workspace)
        return _GRAPHS[key]


# ── Backwards-compatible shim ────────────────────────────────────────────────

def resolve_imports(path: Path, workspace: Path, max_depth: int = 2) -> list[Path]:
    """
    Backward-compatible shim over DependencyGraph.
    Returns flat list of workspace-local files reachable from *path*.
    """
    graph = get_dependency_graph(workspace)
    return graph.transitive_dependencies(path, max_depth=max_depth)


# ─────────────────────────────────────────────────────────────────────────────
# Public: build_symbol_context
# ─────────────────────────────────────────────────────────────────────────────

def build_symbol_context(
    active_paths: list[Path],
    workspace: Path,
    include_imports: bool = False,
    max_symbols_per_file: int = 50,
    max_total_chars: int = 8000,
    symbol_fetcher: Optional[Callable[[Path], "SymbolIndex"]] = None,
) -> str:
    """
    Build a `## Symbols` LLM context block for the given active files.

    Also resolves their first-level import graph and injects signatures for
    any workspace-local dependencies that aren't already in active_paths.
    """
    if not active_paths:
        return ""

    # Expand with import-graph dependencies (signatures only, no full content)
    all_paths: list[Path] = list(active_paths)
    active_set = {p.resolve() for p in active_paths}

    for ap in list(active_paths):
        for dep in resolve_imports(ap, workspace, max_depth=1):
            if dep.resolve() not in active_set:
                all_paths.append(dep)
                active_set.add(dep.resolve())

    parts:  list[str] = []
    total:  int       = 0

    for fpath in all_paths:
        if total >= max_total_chars:
            remaining = len(all_paths) - len(parts)
            if remaining > 0:
                parts.append(f"... ({remaining} more files — symbol limit reached)")
            break

        _fetcher = symbol_fetcher if symbol_fetcher is not None else extract_symbols
        idx = _fetcher(fpath)
        if idx.is_empty():
            continue

        try:
            rel = fpath.relative_to(workspace)
        except ValueError:
            rel = fpath

        header = f"### {rel}  [{idx.language}]"
        body_lines: list[str] = []

        if include_imports and idx.imports:
            body_lines.append("  # imports: " + ", ".join(idx.imports[:8])
                               + (" ..." if len(idx.imports) > 8 else ""))

        shown = 0
        for sym in idx.symbols:
            if shown >= max_symbols_per_file:
                body_lines.append(f"  ... ({len(idx.symbols) - shown} more)")
                break
            body_lines.append(f"  {sym}")
            shown += 1

        if not body_lines:
            continue

        block = header + "\n" + "\n".join(body_lines)
        total += len(block)
        parts.append(block)

    if not parts:
        return ""
    return "\n## Symbol Index (signatures of active files + imports)\n" + "\n\n".join(parts) + "\n"


# ─────────────────────────────────────────────────────────────────────────────
# Symbol resolution — find_definition / find_references
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DefinitionSite:
    """A location where a symbol is defined."""
    file: Path
    line: int
    signature: str
    kind: str           # "function"|"class"|"method"|"field"|"const"|"type"|"export"
    via_reexport: bool = False   # found through __all__ re-export chain

    def render(self, workspace: Optional[Path] = None) -> str:
        try:
            ws_r = workspace.resolve() if workspace else None
            rel = self.file.relative_to(ws_r) if ws_r else self.file
        except ValueError:
            rel = self.file
        tag = "  (via re-export)" if self.via_reexport else ""
        return f"{rel}:{self.line}  [{self.kind}]  {self.signature.strip()}{tag}"


@dataclass
class ReferenceSite:
    """A location where a symbol is referenced (used)."""
    file: Path
    line: int
    col: int
    context: str        # the full source line
    ref_kind: str       # "call"|"import"|"assign"|"type_hint"|"definition"|"unknown"

    def render(self, workspace: Optional[Path] = None) -> str:
        try:
            ws_r = workspace.resolve() if workspace else None
            rel = self.file.relative_to(ws_r) if ws_r else self.file
        except ValueError:
            rel = self.file
        ctx = self.context.strip()
        if len(ctx) > 100:
            ctx = ctx[:100] + "..."
        return f"{rel}:{self.line}  [{self.ref_kind}]  {ctx}"


# ── Reference classifier ──────────────────────────────────────────────────────

def _classify_reference(line_text: str, name: str, col: int) -> str:
    """
    Classify a reference to *name* appearing at column *col* in *line_text*.
    Returns one of: "definition", "call", "import", "assign", "type_hint", "unknown"
    """
    # Definition: def name / class name
    if re.match(r'^\s*(?:async\s+)?def\s+' + re.escape(name) + r'\s*[\(:]', line_text) or \
       re.match(r'^\s*class\s+' + re.escape(name) + r'\s*[:(]', line_text):
        return "definition"

    # Import statement
    if re.search(r'\bimport\b.*\b' + re.escape(name) + r'\b', line_text):
        return "import"

    # What immediately follows the name at its position
    after = line_text[col + len(name):].lstrip()
    if after.startswith("("):
        return "call"

    # Assignment target: `name =` / `name +=` etc.
    if re.match(r'^\s*' + re.escape(name) + r'\s*[+\-*/|&^]?=', line_text):
        return "assign"

    # Type hint context: `: name` / `-> name` / `[name` / `name,` inside annotation
    before = line_text[:col]
    if re.search(r'(?::\s*|->\s*|[\[|,\s]\s*)$', before):
        return "type_hint"

    return "unknown"


# ── Workspace file iterator ──────────────────────────────────────────────────

_SOURCE_EXTS = {
    ".py", ".pyi", ".ts", ".tsx", ".js", ".jsx",
    ".go", ".rs", ".java", ".cs", ".c", ".cpp", ".h", ".hpp",
}

_IGNORE_DIRS = {
    "__pycache__", ".venv", "venv", "node_modules",
    ".git", "build", "dist", ".tox", ".mypy_cache",
    ".pytest_cache", "target", ".next", "out",
}


def _iter_workspace_files(workspace: Path) -> Iterator[Path]:
    """Yield all source files in workspace, skipping common ignore dirs.

    Uses os.scandir instead of os.walk — DirEntry.is_dir/is_file reuse cached
    stat data from the OS, avoiding an extra syscall per entry on Linux/macOS.
    """
    stack: list[Path] = [workspace]
    while stack:
        current = stack.pop()
        try:
            with os.scandir(current) as it:
                for entry in it:
                    if entry.is_dir(follow_symlinks=False):
                        if entry.name not in _IGNORE_DIRS and not entry.name.startswith("."):
                            stack.append(Path(entry.path))
                    elif entry.is_file(follow_symlinks=False):
                        p = Path(entry.path)
                        if p.suffix in _SOURCE_EXTS:
                            yield p
        except PermissionError:
            pass


# ── __all__ re-export resolver ───────────────────────────────────────────────

def _resolve_reexports(init_path: Path, workspace: Path) -> dict[str, Path]:
    """
    Parse an __init__.py and return a map: exported_name → source_file.
    Handles:
      - `from .submodule import Foo, Bar`
      - `from . import submodule` (maps submodule → submodule.py)
      - `from .submodule import *`  (follows submodule.__all__)
    """
    reexports: dict[str, Path] = {}
    if not init_path.exists():
        return reexports

    try:
        source = init_path.read_text(encoding="utf-8", errors="replace")
        tree   = ast.parse(source, filename=str(init_path))
    except Exception:
        return reexports

    # Collect __all__ names to constrain (optional)
    all_names: Optional[set[str]] = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == "__all__":
                    if isinstance(node.value, (ast.List, ast.Tuple)):
                        all_names = {
                            elt.value
                            for elt in node.value.elts
                            if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
                        }

    pkg_dir = init_path.parent

    for node in ast.walk(tree):
        if not (isinstance(node, ast.ImportFrom) and node.level and node.level > 0):
            continue

        mod = node.module or ""
        # Resolve the submodule path relative to the package
        candidate_base = pkg_dir
        for _ in range(node.level - 1):
            candidate_base = candidate_base.parent

        if not mod:
            # `from . import name` — each alias is a sibling module or name
            for alias in node.names:
                n = alias.asname or alias.name
                if all_names is not None and n not in all_names:
                    continue
                for candidate in (
                    candidate_base / f"{alias.name}.py",
                    candidate_base / alias.name / "__init__.py",
                ):
                    if candidate.exists():
                        reexports[n] = candidate.resolve()
                        break
            continue

        # `from .submodule import X` — locate submodule file
        parts = mod.split(".")
        sub = candidate_base
        for part in parts:
            sub = sub / part

        submod_path: Optional[Path] = None
        for candidate in (sub.with_suffix(".py"), sub / "__init__.py"):
            if candidate.exists():
                submod_path = candidate.resolve()
                break

        if not submod_path:
            continue

        for alias in node.names:
            raw_name = alias.name
            exported = alias.asname or raw_name

            if raw_name == "*":
                # `from .submodule import *` — pull all names from submodule __all__
                sub_idx = extract_symbols(submod_path)
                all_sym = next(
                    (s for s in sub_idx.symbols if s.kind == "export" and s.name == "__all__"),
                    None,
                )
                if all_sym:
                    for m in re.finditer(r"['\"]([\w]+)['\"]" , all_sym.signature):
                        en = m.group(1)
                        if all_names is None or en in all_names:
                            reexports[en] = submod_path
            else:
                if all_names is None or exported in all_names:
                    reexports[exported] = submod_path

    return reexports


# ── Public: find_definition ───────────────────────────────────────────────────

def find_definition(
    name: str,
    workspace: Path,
    hint_paths: Optional[list[Path]] = None,
    max_results: int = 20,
) -> list[DefinitionSite]:
    """
    Find all definition sites of *name* across the workspace.

    Search order:
      1. hint_paths (active files — fast path, checked first)
      2. Full workspace file scan
      3. __init__.py re-export chains (names that are re-exported from sub-modules)

    Returns DefinitionSite list, deduplicated by (file, line).
    """
    results: list[DefinitionSite] = []
    seen: set[tuple[str, int]] = set()

    # Use the workspace index cache (mtime-aware) to avoid re-parsing files that
    # have already been parsed this session.  Lazy import avoids the circular
    # dependency: index.py → symbols.py → index.py.
    try:
        from nvagent.core.index import get_workspace_index as _get_idx
        _sym_fetcher = _get_idx(workspace).symbols_for
    except Exception:
        _sym_fetcher = extract_symbols  # fallback: parse directly

    def _scan_file(fpath: Path, via_reexport: bool = False, _rpath: Path | None = None) -> None:
        if len(results) >= max_results:
            return
        rpath = _rpath if _rpath is not None else fpath.resolve()
        try:
            idx = _sym_fetcher(fpath)
        except Exception:
            idx = extract_symbols(fpath)
        for sym in idx.symbols:
            if sym.name != name:
                continue
            key = (str(rpath), sym.line)
            if key in seen:
                continue
            seen.add(key)
            results.append(DefinitionSite(
                file=rpath,
                line=sym.line,
                signature=sym.signature.strip(),
                kind=sym.kind,
                via_reexport=via_reexport,
            ))

    # Pass 1: hint paths (active files)
    hint_set: set[str] = set()
    if hint_paths:
        for p in hint_paths:
            rp = p.resolve()
            _scan_file(p, _rpath=rp)
            hint_set.add(str(rp))

    # Pass 2: full workspace
    for fpath in _iter_workspace_files(workspace):
        if len(results) >= max_results:
            break
        rp = fpath.resolve()
        if str(rp) not in hint_set:
            _scan_file(fpath, _rpath=rp)

    # Pass 3: __all__ re-export chaining
    if len(results) < max_results:
        for init_py in workspace.rglob("__init__.py"):
            if str(init_py.parent).startswith(str(workspace / ".venv")):
                continue
            reexports = _resolve_reexports(init_py, workspace)
            if name in reexports:
                target = reexports[name]
                if str(target) not in {str(r.file) for r in results}:
                    _scan_file(target, via_reexport=True)

    return results


# ── Public: find_references ───────────────────────────────────────────────────

def find_references(
    name: str,
    workspace: Path,
    hint_paths: Optional[list[Path]] = None,
    include_definitions: bool = False,
    max_results: int = 100,
) -> list[ReferenceSite]:
    """
    Find all reference sites (usages) of *name* across the workspace.

    Uses word-boundary regex — avoids substring false positives.
    Skips definition lines by default (pass include_definitions=True to include them).
    hint_paths are searched first.

    Speed: reads raw bytes first and skips files that can't contain *name* at
    all before doing the more expensive decode + regex line scan.
    """
    results: list[ReferenceSite] = []
    pattern = re.compile(r'\b' + re.escape(name) + r'\b')
    name_bytes = name.encode()

    hint_set: set[str] = set()
    files_to_search: list[Path] = []

    if hint_paths:
        for p in hint_paths:
            files_to_search.append(p)
            hint_set.add(str(p.resolve()))

    for fp in _iter_workspace_files(workspace):
        rp = str(fp.resolve())
        if rp not in hint_set:
            files_to_search.append(fp)

    for fpath in files_to_search:
        if len(results) >= max_results:
            break
        try:
            raw = fpath.read_bytes()
        except Exception:
            continue

        # Byte-level pre-filter: skip files that definitely don't contain the
        # name. This is a fast memmem search on raw bytes — much cheaper than
        # decode + splitlines + regex when the name is absent.
        if name_bytes not in raw:
            continue

        rpath = fpath.resolve()
        source = raw.decode(encoding="utf-8", errors="replace")
        for lineno, line_text in enumerate(source.splitlines(), start=1):
            if len(results) >= max_results:
                break
            m = pattern.search(line_text)
            if not m:
                continue
            col = m.start()
            ref_kind = _classify_reference(line_text, name, col)
            if not include_definitions and ref_kind == "definition":
                continue
            results.append(ReferenceSite(
                file=rpath,
                line=lineno,
                col=col,
                context=line_text,
                ref_kind=ref_kind,
            ))

    return results
