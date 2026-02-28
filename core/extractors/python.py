"""Python code symbol extractor."""

from __future__ import annotations

import ast
from pathlib import Path

from nvagent.core.extractors.models import Symbol, SymbolIndex


def py_ann(node: ast.expr | None) -> str:
    """Unparse a type annotation node to string."""
    if node is None:
        return ""
    try:
        return ast.unparse(node)
    except Exception:
        return "?"


def py_arg_str(arg: ast.arg) -> str:
    """Format a function argument with type annotation."""
    ann = f": {py_ann(arg.annotation)}" if arg.annotation else ""
    return f"{arg.arg}{ann}"


def py_func_sig(node: ast.FunctionDef | ast.AsyncFunctionDef, prefix: str = "") -> str:
    """Build a one-line function signature."""
    args = node.args
    parts: list[str] = []

    # positional-only args (before /)
    for i, a in enumerate(args.posonlyargs):
        parts.append(py_arg_str(a))
    if args.posonlyargs:
        parts.append("/")

    # regular args
    num_defaults = len(args.defaults)
    num_args = len(args.args)
    for i, a in enumerate(args.args):
        default_idx = i - (num_args - num_defaults)
        arg_s = py_arg_str(a)
        if default_idx >= 0:
            try:
                default_val = ast.unparse(args.defaults[default_idx])
                arg_s += f"={default_val}"
            except Exception:
                arg_s += "=..."
        parts.append(arg_s)

    if args.vararg:
        parts.append(f"*{py_arg_str(args.vararg)}")
    elif args.kwonlyargs:
        parts.append("*")

    for i, a in enumerate(args.kwonlyargs):
        kw_s = py_arg_str(a)
        if i < len(args.kw_defaults) and args.kw_defaults[i] is not None:
            try:
                kw_s += f"={ast.unparse(args.kw_defaults[i])}"
            except Exception:
                kw_s += "=..."
        parts.append(kw_s)

    if args.kwarg:
        parts.append(f"**{py_arg_str(args.kwarg)}")

    ret = f" -> {py_ann(node.returns)}" if node.returns else ""
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


def py_get_docstring(node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) -> str:
    """Extract the first line of a docstring."""
    try:
        doc = ast.get_docstring(node, clean=True)
        if doc:
            return doc.splitlines()[0][:120]
    except Exception:
        pass
    return ""


def py_is_dataclass(node: ast.ClassDef) -> bool:
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


def py_base_names(node: ast.ClassDef) -> set[str]:
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


def py_class_decorators(node: ast.ClassDef) -> str:
    """Build a decorator prefix string for a class (e.g. '@dataclass\\n')."""
    lines: list[str] = []
    for dec in node.decorator_list:
        try:
            lines.append(f"@{ast.unparse(dec)}")
        except Exception:
            pass
    return ("\n".join(lines) + "\n") if lines else ""


def extract_python(path: Path, source: str) -> SymbolIndex:
    """Extract symbols from Python source code using ast module."""
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
            dots = "." * (node.level or 0)  # preserve relative import dots
            mod = node.module or ""
            names = ", ".join((a.asname if a.asname else a.name) for a in node.names)
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
                sig = py_func_sig(node, prefix)
                if class_name:
                    kind = "protocol_method" if is_protocol_ctx else "method"
                else:
                    kind = "function"
                doc = py_get_docstring(node)
                # Skip __init__ bodies for dataclasses — fields are captured as annassign
                if is_dataclass_ctx and node.name == "__init__":
                    continue
                idx.symbols.append(
                    Symbol(
                        kind=kind,
                        name=node.name,
                        signature=sig,
                        line=node.lineno,
                        docstring=doc,
                    )
                )

            elif isinstance(node, ast.ClassDef):
                bases = ", ".join(py_ann(b) for b in node.bases) if node.bases else ""
                base_str = f"({bases})" if bases else ""
                doc = py_get_docstring(node)
                is_dc = py_is_dataclass(node)
                bnames = py_base_names(node)
                is_td = "TypedDict" in bnames
                is_pt = "Protocol" in bnames
                # Build class signature: include decorator(s) and tag special kinds
                dec_str = py_class_decorators(node)
                tag = ""
                if is_td:
                    tag = "  # TypedDict"
                elif is_pt:
                    tag = "  # Protocol"
                sig = f"{dec_str}class {node.name}{base_str}:{tag}"
                idx.symbols.append(
                    Symbol(
                        kind="class",
                        name=node.name,
                        signature=sig,
                        line=node.lineno,
                        docstring=doc,
                    )
                )
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
                                idx.symbols.append(
                                    Symbol(
                                        kind="export",
                                        name="__all__",
                                        signature=f"__all__ = [{preview}]",
                                        line=node.lineno,
                                    )
                                )
                        except Exception:
                            pass
                    # UPPER_CASE top-level constants
                    elif target.id.isupper():
                        try:
                            val = ast.unparse(node.value)
                            if len(val) > 60:
                                val = val[:60] + "..."
                            idx.symbols.append(
                                Symbol(
                                    kind="const",
                                    name=target.id,
                                    signature=f"{target.id} = {val}",
                                    line=node.lineno,
                                )
                            )
                        except Exception:
                            pass

            elif isinstance(node, ast.AnnAssign):
                if not isinstance(node.target, ast.Name):
                    continue
                try:
                    ann = py_ann(node.annotation)
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
                        idx.symbols.append(
                            Symbol(
                                kind=kind,
                                name=name,
                                signature=f"    {name}: {ann}{default_part}",
                                line=node.lineno,
                            )
                        )
                    else:
                        # Top-level annotated name (module-level type alias / var)
                        idx.symbols.append(
                            Symbol(
                                kind="type",
                                name=name,
                                signature=f"{name}: {ann}",
                                line=node.lineno,
                            )
                        )
                except Exception:
                    pass

    _walk_body(tree.body)
    return idx
