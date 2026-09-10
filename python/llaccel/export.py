"""Graph importer: torch.export -> llaccel dialect (docs/DIALECT.md section 1).

    uv run python -m llaccel.export checkpoints/tiny.pt -o build/export/

Pipeline
--------
1. `torch.export.export(model, (tokens,), dynamic_shapes=...)` with batch 1 and a
   dynamic token dimension. The pre-dispatch ATen graph is used as is (no
   `run_decompositions`) so `aten.linear`, `aten.scaled_dot_product_attention`
   and `aten.repeat_interleave` survive; the RMSNorm and RoPE bodies are
   already decomposed into their elementwise forms by autograd tracing.
2. Pattern matching (`Op`/`Cap`/`Lit` trees, see `PATTERNS`) finds the roots of
   composite ops: RMSNorm (`pow/mean/add/rsqrt/mul/mul`, or `aten.rms_norm`),
   RoPE (`rotate_half`: `slice/neg/cat` mixed with `cos/sin` buffers),
   linear (`aten.linear` / `addmm` / `mm` with a transposed weight),
   SDPA (+ optional GQA head replication), `silu`, `mul`, `add`, `embedding`.
   Every node must be either the root of a match, an interior node of exactly
   one match, or a *plumbing* op (view/transpose/reshape/expand/...).
3. Plumbing ops never become llaccel ops. Instead each FX value is tracked as a
   `View` = (llaccel 2-D tensor, `Layout`). A `Layout` describes how the FX
   tensor's dims map onto the rows (`T`) and columns of the 2-D tensor:
   every dim is `B` (size 1), `T` (tokens) or `C(size, stride, div)` meaning
   "index i contributes (i // div) * stride to the column". This is enough to
   prove that `view(1,T,H,D).transpose(1,2)` is the standard head split that
   NUMERICS.md assumes (`x[m][h*D + i]`), and that `repeat_interleave` /
   `expand+reshape` is exactly GQA replication of KV heads (`div = H/Hkv`).
4. Roles (`l{i}.q`, `l{i}.kr`, ...) are assigned *structurally*: attention ops
   number the layers; producers/consumers of each attention decide which linear
   is q/k/v/o, which RMSNorm is attn/ffn/final, etc. Parameter FQNs are only
   used for the dumped weight data, never for naming.
5. Emission: `model.mlir` (strict DIALECT.md syntax), `weights.bin` + `weights.json`
   (raw f32, nn.Linear `[N][K]` convention), `tokenizer.json`, and (via
   `llaccel.calibrate`) `calib.json`.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.fx as fx

from .model import ModelConfig, TinyLlama, load_checkpoint

INT64_MAX = 9223372036854775807


# --------------------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------------------
class ImportError_(RuntimeError):
    """Raised with the offending FX node(s) printed."""


def _fmt_node(n: fx.Node) -> str:
    v = n.meta.get("val")
    shape = list(getattr(v, "shape", [])) if v is not None else "?"
    return f"%{n.name} = {n.op}[target={_opname(n)}](args={n.args}, kwargs={n.kwargs}) shape={shape}"


def fail(msg: str, *nodes: fx.Node) -> None:
    lines = [msg] + ["  offending node: " + _fmt_node(n) for n in nodes]
    raise ImportError_("\n".join(lines))


# --------------------------------------------------------------------------------------
# Pattern-matching mini framework over FX nodes
# --------------------------------------------------------------------------------------
def _opname(n: fx.Node) -> str:
    if n.op != "call_function":
        return n.op
    t = n.target
    return str(t) if hasattr(t, "overloadpacket") or "aten" in str(t) else getattr(t, "__name__", str(t))


class Pat:
    def match(self, x: Any, env: dict, interior: list) -> bool:  # pragma: no cover - abstract
        raise NotImplementedError


class Cap(Pat):
    """Capture a node (or literal) under `name`; the same name must bind the same object."""

    def __init__(self, name: str, pred=None) -> None:
        self.name, self.pred = name, pred

    def match(self, x, env, interior) -> bool:
        if self.pred is not None and not self.pred(x):
            return False
        if self.name in env:
            return env[self.name] is x if isinstance(x, fx.Node) else env[self.name] == x
        env[self.name] = x
        return True


class Lit(Pat):
    def __init__(self, value) -> None:
        self.value = value

    def match(self, x, env, interior) -> bool:
        if isinstance(self.value, (list, tuple)):
            if not isinstance(x, (list, tuple)) or len(x) != len(self.value):
                return False
            return all(_m(p, xi, env, interior) for p, xi in zip(self.value, x))
        return type(x) is type(self.value) and x == self.value if isinstance(self.value, bool) else x == self.value


class Any_(Pat):
    def match(self, x, env, interior) -> bool:
        return True


class Op(Pat):
    """call_function node with target name `name` (overload optional) and positional arg patterns.

    `commutative=True` also tries the two first args swapped. Trailing args of the node that
    are not mentioned by the pattern must equal `defaults` (or are rejected)."""

    def __init__(self, name: str | tuple, *args: Pat, commutative: bool = False, defaults: tuple = ()) -> None:
        self.names = (name,) if isinstance(name, str) else tuple(name)
        self.args, self.commutative, self.defaults = args, commutative, defaults

    def _name_ok(self, n: fx.Node) -> bool:
        on = _opname(n)
        return any(on == nm or on.startswith(nm + ".") for nm in self.names)

    def match(self, x, env, interior) -> bool:
        if not isinstance(x, fx.Node) or x.op != "call_function" or not self._name_ok(x):
            return False
        if x.kwargs:
            return False
        nargs = list(x.args)
        if len(nargs) < len(self.args) or len(nargs) > len(self.args) + len(self.defaults):
            return False
        extra = nargs[len(self.args):]
        if tuple(extra) != tuple(self.defaults[: len(extra)]):
            return False
        orders = [self.args]
        if self.commutative and len(self.args) >= 2:
            orders.append((self.args[1], self.args[0]) + tuple(self.args[2:]))
        for order in orders:
            env2, int2 = dict(env), list(interior)
            if all(_m(p, a, env2, int2) for p, a in zip(order, nargs)):
                env.clear(); env.update(env2)
                interior[:] = int2
                interior.append(x)
                return True
        return False


def _m(p, x, env, interior) -> bool:
    if not isinstance(p, Pat):  # raw literal inside a Lit list
        return Lit(p).match(x, env, interior)
    return p.match(x, env, interior)


def match(p: Pat, node: fx.Node) -> tuple[dict, list] | None:
    env: dict = {}
    interior: list = []
    if p.match(node, env, interior):
        return env, interior
    return None


def is_node(x) -> bool:
    return isinstance(x, fx.Node)


def is_float(x) -> bool:
    return isinstance(x, float)


def is_int(x) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


# ---- the patterns ----------------------------------------------------------------------
X = Cap("x", is_node)
_RMS_CORE = Op("aten.rsqrt", Op("aten.add", Op("aten.mean.dim", Op("aten.pow.Tensor_Scalar", X, Lit(2)), Lit([-1]), Lit(True)),
                                Cap("eps", is_float)))
PAT_RMSNORM = [
    Op("aten.mul", Op("aten.mul", X, _RMS_CORE, commutative=True), Cap("gamma", is_node), commutative=True),
    Op("aten.rms_norm", X, Any_(), Cap("gamma", is_node), Cap("eps", is_float)),
]
_ROT = Op("aten.cat", Lit([Op("aten.neg", Op("aten.slice.Tensor", X, Cap("dim", is_int), Cap("half", is_int), Cap("end", is_int))),
                           Op("aten.slice.Tensor", X, Cap("dim", is_int), Lit(0), Cap("half", is_int))]), Cap("catdim", is_int))
PAT_ROPE = [
    Op("aten.add", Op("aten.mul", X, Cap("cos", is_node)), Op("aten.mul", _ROT, Cap("sin", is_node)), commutative=True),
]
_WT = (Op("aten.t", Cap("w", is_node)), Op("aten.transpose.int", Cap("w", is_node), Lit(0), Lit(1)),
       Op("aten.permute", Cap("w", is_node), Lit([1, 0])))
PAT_LINEAR = [Op("aten.linear", X, Cap("w", is_node), Cap("b", is_node)), Op("aten.linear", X, Cap("w", is_node))]
for _wt in _WT:
    PAT_LINEAR += [Op("aten.addmm", Cap("b", is_node), X, _wt), Op("aten.mm", X, _wt), Op("aten.matmul", X, _wt)]
PAT_SDPA = [Op("aten.scaled_dot_product_attention", Cap("q", is_node), Cap("k", is_node), Cap("v", is_node),
               Lit(None), Lit(0.0), Lit(True), defaults=(None, False))]
PAT_SILU = [Op("aten.silu", X)]
PAT_MUL = [Op("aten.mul", Cap("a", is_node), Cap("b", is_node))]
PAT_ADD = [Op("aten.add", Cap("a", is_node), Cap("b", is_node))]
PAT_EMBED = [Op("aten.embedding", Cap("w", is_node), Cap("tokens", is_node))]

# composite patterns first (their interiors would otherwise match the generic ones)
PATTERNS: list[tuple[str, list[Pat]]] = [
    ("rmsnorm", PAT_RMSNORM), ("rope", PAT_ROPE), ("linear", PAT_LINEAR), ("sdpa", PAT_SDPA),
    ("embedding", PAT_EMBED), ("silu", PAT_SILU), ("mul", PAT_MUL), ("add", PAT_ADD),
]

PLUMBING = ("aten.view", "aten.reshape", "aten._unsafe_view", "aten.transpose.int", "aten.permute", "aten.contiguous",
            "aten.clone", "aten.detach", "aten.alias", "aten._to_copy", "aten.unsqueeze", "aten.squeeze", "aten.expand",
            "aten.repeat_interleave.self_int", "aten.sym_size.int")


# --------------------------------------------------------------------------------------
# Layout tracking
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class C:
    """A column-indexing dim: index i (0 <= i < size) contributes (i // div) * stride to the column."""
    size: int
    stride: int
    div: int = 1


B = "B"  # size-1 dim
T = "T"  # token dim
Dim = Any  # B | T | C


@dataclass(frozen=True)
class Layout:
    dims: tuple

    @staticmethod
    def rows_cols(cols: int, batch_dims: int = 1) -> "Layout":
        return Layout((B,) * batch_dims + (T, C(cols, 1, 1)))

    def is_canonical(self, cols: int) -> bool:
        """[B..., T, C(cols,1,1)] — the 2-D tensor itself (modulo size-1 dims)."""
        core = [d for d in self.dims if d != B]
        return core == [T, C(cols, 1, 1)]

    def heads(self) -> tuple[int, int, int] | None:
        """[B..., C(H, D, div), T, C(D, 1, 1)] -> (H, D, div) — the attention/rope head layout.

        Two degenerate forms are accepted: a single head (`view(1, T, 1, D)` leaves no column dim
        for H, so the layout is [B..., T, C(D)]) and heads replicated from a single KV head
        (`expand`/`repeat_interleave` of a size-1 dim gives C(H, stride=0), i.e. every query head
        reads KV head 0, which is div = H)."""
        core = [d for d in self.dims if d != B]
        if len(core) == 2 and core[0] == T and isinstance(core[1], C):
            dd = core[1]
            if dd.stride == 1 and dd.div == 1:
                return 1, dd.size, 1
        if len(core) == 3 and core[1] == T and isinstance(core[0], C) and isinstance(core[2], C):
            hd, dd = core[0], core[2]
            if dd.stride == 1 and dd.div == 1:
                if hd.stride == dd.size:
                    return hd.size, dd.size, hd.div
                if hd.stride == 0:
                    return hd.size, dd.size, hd.size
        return None


@dataclass
class Tensor2D:
    """A logical llaccel activation `[?][cols]`."""
    cols: int
    name: str | None = None  # llaccel.name, assigned by role
    producer: "IROp | None" = None
    consumers: list = field(default_factory=list)
    ssa: str | None = None

    def __repr__(self) -> str:
        return f"Tensor2D({self.name}, cols={self.cols})"


@dataclass
class View:
    t: Tensor2D
    layout: Layout


@dataclass
class IROp:
    kind: str  # rmsnorm | linear | rope | attention | silu | mul | add
    node: fx.Node
    inputs: list[Tensor2D]
    out: Tensor2D
    params: dict = field(default_factory=dict)  # role -> (fqn, tensor)
    attrs: dict = field(default_factory=dict)
    role: str | None = None
    layer: int | None = None


def _sizes(n: fx.Node) -> list:
    v = n.meta.get("val")
    if v is None or not hasattr(v, "shape"):
        fail("node has no shape metadata", n)
    out = []
    for s in v.shape:
        out.append(int(s) if isinstance(s, int) else T)
    return out


def _reshape_layout(lay: Layout, out_sizes: list, n: fx.Node) -> Layout:
    """Row-major reshape of `lay` (dim sizes implied) to `out_sizes` using merge/split rules."""
    in_dims = list(lay.dims)

    def size_of(d):
        return 1 if d == B else (T if d == T else d.size)

    in_sizes = [size_of(d) for d in in_dims]
    # group dims on both sides so that the products agree (T is symbolic: a group holding T is T + ones)
    out_dims: list = []
    i = j = 0
    while i < len(in_dims) or j < len(out_sizes):
        gi, gj = [], []
        pi = pj = 1
        ti = tj = False

        def push_i():
            nonlocal i, pi, ti
            d = in_dims[i]; i += 1
            gi.append(d)
            if size_of(d) == T: ti = True
            else: pi *= size_of(d)

        def push_j():
            nonlocal j, pj, tj
            s = out_sizes[j]; j += 1
            gj.append(s)
            if s == T: tj = True
            else: pj *= s

        if i < len(in_dims): push_i()
        if j < len(out_sizes): push_j()
        while not (pi == pj and ti == tj):
            if (pi < pj or (tj and not ti)) and i < len(in_dims): push_i()
            elif j < len(out_sizes): push_j()
            else: fail("reshape: cannot align dims", n)
        # absorb trailing size-1 dims into this group
        while i < len(in_dims) and size_of(in_dims[i]) == 1: gi.append(in_dims[i]); i += 1
        while j < len(out_sizes) and out_sizes[j] == 1: gj.append(1); j += 1
        out_dims += _regroup([d for d in gi if size_of(d) != 1], [s for s in gj if s != 1], n, len(gj))
    return Layout(tuple(out_dims))


def _regroup(gi: list, gj: list, n: fx.Node, n_out: int) -> list:
    """Map a group of input dims (product == product of gj) to output sizes gj."""
    if not gi:
        return [B] * n_out
    if gi == [T] and gj == [T]:
        return [T] + [B] * (n_out - 1)
    if T in gi or T in gj:
        fail("reshape mixing the token dim with column dims is not supported", n)
    # merge everything into one C, then split row-major
    merged = gi[0]
    for d in gi[1:]:
        merged = _merge(merged, d, n)
    res = []
    rem = merged
    for s in gj[:-1]:
        head, rem = _split(rem, s, n)
        res.append(head)
    res.append(rem)
    return res + [B] * (n_out - len(gj))


def _merge(a: C, b: C, n: fx.Node) -> C:
    # j = ia * b.size + ib
    if b.stride == 0:  # expanded (broadcast) trailing dim: contribution depends on ia only
        return C(a.size * b.size, a.stride, a.div * b.size)
    if a.div == 1 and b.div == 1 and a.stride == b.size * b.stride:
        return C(a.size * b.size, b.stride, 1)
    if a.stride == 0:
        return C(a.size * b.size, b.stride, b.div) if a.size == 1 else fail("unsupported merge of broadcast dim", n)
    fail(f"reshape merges non-adjacent column ranges {a} x {b}", n)


def _split(d: C, s: int, n: fx.Node) -> tuple[C, C]:
    if d.size % s != 0:
        fail(f"reshape split {d} by {s}", n)
    rest = d.size // s
    if d.div != 1 and d.div % rest != 0:
        fail(f"reshape split of a replicated dim {d} by {s}", n)
    if d.div == 1:
        return C(s, rest * d.stride, 1), C(rest, d.stride, 1)
    # replicated dim split: (i_a * rest + i_b) // div == i_a * rest/div + ... only clean when rest == div
    if rest == d.div:
        return C(s, d.stride, 1), C(rest, 0, 1)
    fail(f"unsupported split of replicated dim {d}", n)


def plumb(kind: str, n: fx.Node, v: View) -> View:
    lay = v.layout
    dims = list(lay.dims)
    if kind in ("aten.view", "aten.reshape", "aten._unsafe_view"):
        return View(v.t, _reshape_layout(lay, _sizes(n), n))
    if kind == "aten.transpose.int":
        d0, d1 = n.args[1], n.args[2]
        dims[d0], dims[d1] = dims[d1], dims[d0]
        return View(v.t, Layout(tuple(dims)))
    if kind == "aten.permute":
        return View(v.t, Layout(tuple(dims[p] for p in n.args[1])))
    if kind in ("aten.contiguous", "aten.clone", "aten.detach", "aten.alias"):
        return v
    if kind == "aten._to_copy":
        val = n.meta.get("val")
        if val is not None and val.dtype != torch.float32:
            fail("only f32 tensors are supported (dtype cast found)", n)
        return v
    if kind == "aten.unsqueeze":
        d = n.args[1]
        d = d if d >= 0 else len(dims) + 1 + d
        dims.insert(d, B)
        return View(v.t, Layout(tuple(dims)))
    if kind == "aten.squeeze":
        out = _sizes(n)
        return View(v.t, _reshape_layout(lay, out, n))
    if kind == "aten.expand":
        sizes = _sizes(n)
        if len(sizes) != len(dims):
            fail("expand changes rank", n)
        new = []
        for d, s in zip(dims, sizes):
            if d == B and s != 1:
                if s == T:
                    fail("expand along the token dim", n)
                new.append(C(s, 0, 1))
            else:
                new.append(d)
        return View(v.t, Layout(tuple(new)))
    if kind == "aten.repeat_interleave.self_int":
        rep, dim = n.args[1], n.args[2]
        d = dims[dim]
        if d == B:  # replicating a size-1 dim: every index maps to the same columns
            dims[dim] = C(rep, 0, 1)
        elif isinstance(d, C):
            dims[dim] = C(d.size * rep, d.stride, d.div * rep)
        else:
            fail("repeat_interleave along the token dim", n)
        return View(v.t, Layout(tuple(dims)))
    fail("unhandled plumbing op", n)


# --------------------------------------------------------------------------------------
# Importer
# --------------------------------------------------------------------------------------
class Importer:
    def __init__(self, ep: torch.export.ExportedProgram) -> None:
        self.ep = ep
        self.gm = ep.graph_module
        self.nodes = list(self.gm.graph.nodes)
        sig = ep.graph_signature
        self.param_of = dict(sig.inputs_to_parameters)
        self.buffer_of = dict(sig.inputs_to_buffers)
        self.const_of = dict(getattr(sig, "inputs_to_lifted_tensor_constants", {}))
        self.user_inputs = list(sig.user_inputs)
        self.views: dict[fx.Node, View] = {}
        self.ops: list[IROp] = []
        self.input_tensor: Tensor2D | None = None
        self.embed: tuple[str, torch.Tensor] | None = None
        self.output: Tensor2D | None = None
        self.rope_tables: dict[fx.Node, tuple[str, torch.Tensor]] = {}
        self.node_of_tensor: dict[int, fx.Node] = {}

    # ---- parameter helpers ---------------------------------------------------------------
    def tensor_of(self, n: fx.Node) -> tuple[str, torch.Tensor] | None:
        if n.op != "placeholder":
            return None
        if n.name in self.param_of:
            fqn = self.param_of[n.name]
            return fqn, self.ep.state_dict[fqn].detach().float().cpu()
        if n.name in self.buffer_of:
            fqn = self.buffer_of[n.name]
            t = self.ep.state_dict.get(fqn, self.ep.constants.get(fqn))
            return fqn, t.detach().float().cpu()
        if n.name in self.const_of:
            fqn = self.const_of[n.name]
            return fqn, self.ep.constants[fqn].detach().float().cpu()
        return None

    def param(self, n, what: str, root: fx.Node) -> tuple[str, torch.Tensor]:
        if not is_node(n):
            fail(f"{what}: expected a tensor node", root)
        r = self.tensor_of(n)
        if r is None:
            fail(f"{what}: expected a parameter/buffer placeholder, got a computed value", root, n)
        return r

    def view_of(self, n: fx.Node, root: fx.Node) -> View:
        if n not in self.views:
            fail("operand is not a tracked activation (unmatched producer?)", root, n)
        return self.views[n]

    def canonical(self, n: fx.Node, root: fx.Node) -> Tensor2D:
        v = self.view_of(n, root)
        if not v.layout.is_canonical(v.t.cols):
            fail(f"operand must be the plain [tokens][{v.t.cols}] tensor but has layout {v.layout}", root, n)
        return v.t

    # ---- pass 1: matching ------------------------------------------------------------------
    def run(self) -> "Importer":
        matched: dict[fx.Node, tuple[str, dict, list]] = {}
        interior_of: dict[fx.Node, fx.Node] = {}
        for n in self.nodes:
            if n.op != "call_function":
                continue
            for kind, pats in PATTERNS:
                if kind in ("silu", "mul", "add"):
                    continue  # generic ones in the second sweep
                hit = None
                for p in pats:
                    hit = match(p, n)
                    if hit:
                        break
                if hit:
                    env, interior = hit
                    matched[n] = (kind, env, [m for m in interior if m is not n])
                    break
        for root, (kind, env, interior) in matched.items():
            for m in interior:
                if m in matched:
                    fail(f"node is both interior of a {kind} match and the root of a {matched[m][0]} match", m, root)
                if m in interior_of:
                    fail("node is interior of two matches", m, root, interior_of[m])
                interior_of[m] = root
                bad = [u for u in m.users if u is not root and u not in interior and u not in interior_of]
                # a user outside the match is OK only if it is also interior of this same match
                bad = [u for u in bad if interior_of.get(u) is not root]
                if bad:
                    fail(f"interior node of a {kind} pattern is used outside the pattern", m, *bad)
        for n in self.nodes:
            if n.op != "call_function" or n in matched or n in interior_of:
                continue
            for kind, pats in PATTERNS:
                if kind not in ("silu", "mul", "add"):
                    continue
                for p in pats:
                    hit = match(p, n)
                    if hit:
                        matched[n] = (kind, hit[0], [])
                        break
                if n in matched:
                    break

        # ---- pass 2: build values in topological order ----------------------------------
        for n in self.nodes:
            if n.op == "placeholder":
                continue
            if n.op == "output":
                outs = n.args[0]
                if not isinstance(outs, (tuple, list)) or len(outs) != 1:
                    fail("expected exactly one output", n)
                self.output = self.canonical(outs[0], n)
                continue
            if n.op != "call_function":
                fail("unsupported node kind", n)
            if n in interior_of:
                continue
            if n in matched:
                kind, env, _ = matched[n]
                getattr(self, "build_" + kind)(n, env)
                continue
            on = _opname(n)
            if on == "aten.sym_size.int":
                continue
            if any(on == p or on.startswith(p + ".") for p in PLUMBING):
                src = n.args[0]
                if is_node(src) and src in self.views:
                    self.views[n] = plumb(next(p for p in PLUMBING if on == p or on.startswith(p + ".")), n, self.views[src])
                    continue
                if is_node(src) and src.op == "placeholder":
                    # plumbing on a parameter/buffer (e.g. slicing the RoPE tables) is resolved at use
                    continue
                fail("plumbing op on an untracked value", n)
            if on == "aten.slice.Tensor" and is_node(n.args[0]) and n.args[0] not in self.views:
                continue  # slices of buffers (RoPE cos/sin) are resolved when the RoPE pattern is built
            fail("unrecognized op (no pattern matched)", n)
        if self.input_tensor is None:
            fail("no aten.embedding found: the model input must be an embedding lookup", self.nodes[-1])
        if self.output is None:
            fail("no output", self.nodes[-1])
        return self

    # ---- builders --------------------------------------------------------------------------
    def new_tensor(self, n: fx.Node, cols: int, layout: Layout | None = None) -> Tensor2D:
        t = Tensor2D(cols)
        self.views[n] = View(t, layout or Layout.rows_cols(cols, batch_dims=len(_sizes(n)) - 2))
        self.node_of_tensor[id(t)] = n
        return t

    def add_op(self, kind: str, n: fx.Node, inputs: list[Tensor2D], out: Tensor2D, **kw) -> IROp:
        op = IROp(kind, n, inputs, out, **kw)
        out.producer = op
        for i in inputs:
            i.consumers.append(op)
        self.ops.append(op)
        return op

    def build_embedding(self, n: fx.Node, env: dict) -> None:
        if self.input_tensor is not None:
            fail("second embedding lookup", n)
        fqn, w = self.param(env["w"], "embedding weight", n)
        if env["tokens"].name not in self.user_inputs:
            fail("embedding must be indexed by the function input tokens", n)
        sizes = _sizes(n)
        if sizes[-1] != w.shape[1] or T not in sizes:
            fail("embedding output shape", n)
        self.embed = (fqn, w)
        self.input_tensor = self.new_tensor(n, w.shape[1])
        self.input_tensor.name = "input"

    def build_rmsnorm(self, n: fx.Node, env: dict) -> None:
        x = self.canonical(env["x"], n)
        fqn, g = self.param(env["gamma"], "rmsnorm gamma", n)
        if tuple(g.shape) != (x.cols,):
            fail(f"rmsnorm gamma shape {tuple(g.shape)} != ({x.cols},)", n)
        out = self.new_tensor(n, x.cols)
        self.add_op("rmsnorm", n, [x], out, params={"gamma": (fqn, g)}, attrs={"eps": float(env["eps"])})

    def build_linear(self, n: fx.Node, env: dict) -> None:
        x = self.canonical(env["x"], n)
        fqn, w = self.param(env["w"], "linear weight", n)
        if w.dim() != 2 or w.shape[1] != x.cols:
            fail(f"linear weight shape {tuple(w.shape)} incompatible with input cols {x.cols} (expected [N][{x.cols}])", n)
        params = {"w": (fqn, w)}
        if "b" in env:
            bf, b = self.param(env["b"], "linear bias", n)
            if tuple(b.shape) != (w.shape[0],):
                fail("linear bias shape", n)
            params["b"] = (bf, b)
        out = self.new_tensor(n, int(w.shape[0]))
        self.add_op("linear", n, [x], out, params=params)

    def resolve_rope_table(self, node: fx.Node, root: fx.Node) -> torch.Tensor:
        """cos/sin operand: a buffer, possibly sliced along dim 0 to T rows and/or unsqueezed."""
        cur = node
        while True:
            on = _opname(cur)
            if cur.op == "placeholder":
                r = self.tensor_of(cur)
                if r is None:
                    fail("RoPE cos/sin must come from a buffer (precomputed table)", root, node)
                return r[1]
            if on.startswith("aten.slice.Tensor") and cur.args[1] == 0 and cur.args[2] in (None, 0):
                cur = cur.args[0]; continue
            if on.startswith("aten.unsqueeze") or on.startswith("aten._to_copy") or on.startswith("aten.alias"):
                cur = cur.args[0]; continue
            fail("RoPE cos/sin operand is not a (sliced) buffer", root, cur)

    def build_rope(self, n: fx.Node, env: dict) -> None:
        v = self.view_of(env["x"], n)
        hd = v.layout.heads()
        if hd is None:
            fail(f"RoPE input must be laid out as [B, H, T, D] (standard head split); got {v.layout}", n, env["x"])
        H, D, div = hd
        if div != 1:
            fail("RoPE applied to replicated KV heads", n)
        if H * D != v.t.cols:
            fail("RoPE head layout does not cover the tensor", n)
        rank = len(v.layout.dims)
        dim = env["dim"] if env["dim"] >= 0 else env["dim"] + rank
        if dim != rank - 1 or env["catdim"] not in (-1, rank - 1):
            fail("rotate_half must slice/cat along the last (head_dim) axis", n)
        if env["half"] != D // 2 or env["end"] < D:
            fail(f"rotate_half halves must be {D//2} wide", n)
        cos = self.resolve_rope_table(env["cos"], n)
        sin = self.resolve_rope_table(env["sin"], n)
        if cos.shape[-1] != D or sin.shape[-1] != D or cos.dim() != 2 or sin.dim() != 2:
            fail(f"RoPE tables must be [max_seq][{D}] (cos, sin duplicated halves)", n)
        out = self.new_tensor(n, v.t.cols, v.layout)
        self.add_op("rope", n, [v.t], out, attrs={"heads": H, "head_dim": D, "cos": cos, "sin": sin})

    def build_sdpa(self, n: fx.Node, env: dict) -> None:
        q, k, v = (self.view_of(env[c], n) for c in "qkv")
        hq, hk, hv = q.layout.heads(), k.layout.heads(), v.layout.heads()
        if hq is None or hk is None or hv is None:
            fail(f"SDPA operands must be [B, H, T, D]; got q={q.layout} k={k.layout} v={v.layout}", n)
        H, D, dq = hq
        if dq != 1 or H * D != q.t.cols:
            fail("query heads must not be replicated", n)
        if hk != hv:
            fail(f"K and V head layouts differ: {hk} vs {hv}", n)
        Hk, Dk, rep = hk
        if Dk != D or Hk != H:
            fail(f"K/V must have {H} heads of {D} after GQA replication", n)
        if H % rep != 0 or k.t.cols != (H // rep) * D or v.t.cols != (H // rep) * D:
            fail("GQA replication factor inconsistent with KV tensor width", n)
        out = self.new_tensor(n, H * D, q.layout)
        self.add_op("attention", n, [q.t, k.t, v.t], out, attrs={"heads": H, "kv_heads": H // rep, "head_dim": D})

    def build_silu(self, n: fx.Node, env: dict) -> None:
        x = self.canonical(env["x"], n)
        self.add_op("silu", n, [x], self.new_tensor(n, x.cols))

    def _binary(self, kind: str, n: fx.Node, env: dict) -> None:
        a, b = env["a"], env["b"]
        if not (is_node(a) and is_node(b)):
            fail(f"{kind} with a scalar operand is not supported", n)
        ta, tb = self.canonical(a, n), self.canonical(b, n)
        if ta.cols != tb.cols:
            fail(f"{kind} operands have different widths {ta.cols} vs {tb.cols}", n)
        self.add_op(kind, n, [ta, tb], self.new_tensor(n, ta.cols))

    def build_mul(self, n, env): self._binary("mul", n, env)
    def build_add(self, n, env): self._binary("add", n, env)


# --------------------------------------------------------------------------------------
# Role assignment (structural) + model attribute derivation
# --------------------------------------------------------------------------------------
@dataclass
class Model:
    cfg: dict
    weights: list[tuple[str, torch.Tensor]]  # (llaccel weight name, tensor) in declaration order
    ops: list[IROp]  # in emission order
    input: Tensor2D
    output: Tensor2D
    node_names: dict[fx.Node, str]  # FX node -> llaccel.name (for calibration)
    ep: torch.export.ExportedProgram


def _only(lst, what: str, op: IROp):
    if len(lst) != 1:
        fail(f"expected exactly one {what}, found {len(lst)}", op.node)
    return lst[0]


def assign_roles(imp: Importer, cfg_hint: ModelConfig | None = None) -> Model:
    attns = [o for o in imp.ops if o.kind == "attention"]
    if not attns:
        fail("no attention op found", imp.nodes[-1])
    used: set[int] = set()
    ordered: list[IROp] = []
    weights: list[tuple[str, torch.Tensor]] = []
    names: dict[fx.Node, str] = {}

    def take(op: IROp, role: str, layer: int | None, wnames: dict[str, str]) -> None:
        if id(op) in used:
            fail(f"op used in two roles ({op.role} and {role})", op.node)
        used.add(id(op))
        op.role, op.layer = role, layer
        op.out.name = role if layer is None else f"l{layer}.{role}"
        names[op.node] = op.out.name
        for prole, wname in wnames.items():
            if prole in op.params:
                fqn, t = op.params[prole]
                op.attrs["w_" + prole] = wname
                weights.append((wname, t))
        ordered.append(op)

    def producer(t: Tensor2D, kind: str, what: str, ctx: IROp) -> IROp:
        p = t.producer
        if p is None or p.kind != kind:
            fail(f"{what}: expected to be produced by {kind}, got {p.kind if p else 'function input'}", ctx.node)
        return p

    def consumer(t: Tensor2D, kind: str, what: str, ctx: IROp, pred=lambda o: True) -> IROp:
        return _only([o for o in t.consumers if o.kind == kind and pred(o)], f"{kind} consuming {what}", ctx)

    residual = imp.input_tensor
    weights.append(("embed", imp.embed[1]))
    H = Hkv = D = ffn = None
    eps_set: set[float] = set()
    rope_tabs = None
    for i, attn in enumerate(attns):
        L = f"l{i}"
        qr_t, kr_t, v_t = attn.inputs
        rq, rk = producer(qr_t, "rope", "attention query", attn), producer(kr_t, "rope", "attention key", attn)
        lq, lk, lv = (producer(t, "linear", w, attn) for t, w in ((rq.inputs[0], "q"), (rk.inputs[0], "k"), (v_t, "v")))
        h_t = lq.inputs[0]
        if lk.inputs[0] is not h_t or lv.inputs[0] is not h_t:
            fail("q/k/v projections must share the same normalized input", attn.node)
        hn = producer(h_t, "rmsnorm", "qkv input", attn)
        if hn.inputs[0] is not residual:
            fail(f"layer {i}: attention norm input is not the residual stream", hn.node)
        lo = consumer(attn.out, "linear", "attention output", attn)
        x1 = consumer(lo.out, "add", "o_proj output", attn)
        if residual not in x1.inputs:
            fail(f"layer {i}: residual add does not add the residual stream", x1.node)
        h2 = consumer(x1.out, "rmsnorm", "x1", attn)
        gate = consumer(h2.out, "linear", "h2", attn, lambda o: any(c.kind == "silu" for c in o.out.consumers))
        sg = consumer(gate.out, "silu", "gate", attn)
        f = consumer(sg.out, "mul", "silu(gate)", attn)
        up = _only([o for o in h2.out.consumers if o.kind == "linear" and o is not gate and f in o.out.consumers],
                   "up projection", attn)
        down = consumer(f.out, "linear", "silu*up", attn)
        x2 = consumer(down.out, "add", "down output", attn)
        if x1.out not in x2.inputs:
            fail(f"layer {i}: second residual add does not add x1", x2.node)
        # dims
        H_, Hkv_, D_ = attn.attrs["heads"], attn.attrs["kv_heads"], attn.attrs["head_dim"]
        if rq.attrs["heads"] != H_ or rk.attrs["heads"] != Hkv_ or rq.attrs["head_dim"] != D_:
            fail("RoPE head counts disagree with attention", attn.node)
        if H is None:
            H, Hkv, D, ffn = H_, Hkv_, D_, gate.out.cols
            rope_tabs = (rq.attrs["cos"], rq.attrs["sin"])
        elif (H, Hkv, D, ffn) != (H_, Hkv_, D_, gate.out.cols):
            fail("layers have different shapes", attn.node)
        if up.out.cols != ffn:
            fail("gate/up widths differ", up.node)
        for r in (rq, rk):
            if not (torch.equal(r.attrs["cos"], rope_tabs[0]) and torch.equal(r.attrs["sin"], rope_tabs[1])):
                fail("all RoPE ops must use the same tables", r.node)
        eps_set.update((hn.attrs["eps"], h2.attrs["eps"]))
        take(hn, "h", i, {"gamma": f"{L}_attn_norm"})
        take(lq, "q", i, {"w": f"{L}_wq", "b": f"{L}_bq"})
        take(lk, "k", i, {"w": f"{L}_wk", "b": f"{L}_bk"})
        take(lv, "v", i, {"w": f"{L}_wv", "b": f"{L}_bv"})
        take(rq, "qr", i, {})
        take(rk, "kr", i, {})
        take(attn, "a", i, {})
        attn.attrs["layer"] = i
        take(lo, "o", i, {"w": f"{L}_wo", "b": f"{L}_bo"})
        take(x1, "x1", i, {})
        x1.inputs = [residual, lo.out]  # canonical operand order: residual first
        take(h2, "h2", i, {"gamma": f"{L}_ffn_norm"})
        take(gate, "g", i, {"w": f"{L}_wg", "b": f"{L}_bg"})
        take(up, "u", i, {"w": f"{L}_wu", "b": f"{L}_bu"})
        take(sg, "sg", i, {})
        take(f, "f", i, {})
        f.inputs = [sg.out, up.out]
        take(down, "d", i, {"w": f"{L}_wd", "b": f"{L}_bd"})
        take(x2, "x2", i, {})
        x2.inputs = [x1.out, down.out]
        residual = x2.out
    fn = consumer(residual, "rmsnorm", "last residual", attns[-1])
    lm = consumer(fn.out, "linear", "final norm", attns[-1])
    if lm.out is not imp.output:
        fail("function output must be the lm_head linear", lm.node)
    eps_set.add(fn.attrs["eps"])
    take(fn, "hn", None, {"gamma": "norm"})
    take(lm, "logits", None, {"w": "lm_head", "b": "lm_head_bias"})
    names[imp.node_of_tensor[id(imp.input_tensor)]] = "input"
    unassigned = [o for o in imp.ops if id(o) not in used]
    if unassigned:
        fail(f"{len(unassigned)} matched op(s) do not belong to a Llama block structure", *[o.node for o in unassigned])
    if len(eps_set) != 1:
        fail(f"all RMSNorms must share one eps, found {sorted(eps_set)}", fn.node)

    dim = imp.input_tensor.cols
    vocab, embed_dim = imp.embed[1].shape
    if embed_dim != dim or lm.out.cols != vocab:
        fail("embedding / lm_head vocab mismatch", lm.node)
    cos, sin = rope_tabs
    max_seq = int(cos.shape[0])
    rope_base = derive_rope_base(cos, sin, D, cfg_hint.rope_base if cfg_hint else None)
    cfg = dict(dim=dim, n_layers=len(attns), n_heads=H, n_kv_heads=Hkv, head_dim=D, ffn=ffn, vocab=int(vocab),
               max_seq=max_seq, rope_base=rope_base, rms_eps=next(iter(eps_set)))
    if cfg_hint is not None:
        for k, v in cfg.items():
            hv = getattr(cfg_hint, k)
            if (abs(v - hv) > 1e-9 if isinstance(v, float) else v != hv):
                fail(f"derived model attribute {k}={v} disagrees with checkpoint config {hv}", lm.node)
    return Model(cfg, weights, ordered, imp.input_tensor, imp.output, names, imp.ep)


def derive_rope_base(cos: torch.Tensor, sin: torch.Tensor, D: int, hint: float | None) -> float:
    """theta_1 = base^(-2/D) is read off the tables at position 1; the whole table is then verified."""
    half = D // 2
    if half < 2:
        base = hint if hint is not None else 10000.0
    else:
        theta1 = math.atan2(float(sin[1, 1]), float(cos[1, 1]))
        base = theta1 ** (-D / 2)
        if hint is not None and abs(base - hint) / hint < 1e-3:
            base = hint
    inv_freq = base ** (-torch.arange(0, D, 2, dtype=torch.float32) / D)
    freqs = torch.outer(torch.arange(cos.shape[0], dtype=torch.float32), inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    if not (torch.allclose(emb.cos(), cos, atol=1e-4) and torch.allclose(emb.sin(), sin, atol=1e-4)):
        raise ImportError_(f"RoPE tables are not rotate_half tables for base={base} (D={D})")
    return float(base)


# --------------------------------------------------------------------------------------
# Emission
# --------------------------------------------------------------------------------------
def mlir_float(x: float) -> str:
    s = repr(float(x))
    if "e" in s or "E" in s:
        mant, exp = s.lower().split("e")
        if "." not in mant:
            mant += ".0"
        return f"{mant}e{exp}"
    if "." not in s:
        s += ".0"
    return s


def _ssa(op: IROp) -> str:
    return f"%{op.out.name.replace('.', '_')}"


def _ref(t: Tensor2D, input_ssa: str = "%x") -> str:
    return input_ssa if t.name == "input" else f"%{t.name.replace('.', '_')}"


def _ty(t: Tensor2D) -> str:
    return f"tensor<?x{t.cols}xf32>"


def emit_mlir(m: Model) -> str:
    c = m.cfg
    out = []
    out.append(f"module attributes {{llaccel.model = {{dim = {c['dim']} : i64, n_layers = {c['n_layers']} : i64, "
               f"n_heads = {c['n_heads']} : i64,")
    out.append(f"    n_kv_heads = {c['n_kv_heads']} : i64, head_dim = {c['head_dim']} : i64, ffn = {c['ffn']} : i64, "
               f"vocab = {c['vocab']} : i64,")
    out.append(f"    max_seq = {c['max_seq']} : i64, rope_base = {mlir_float(c['rope_base'])} : f64, "
               f"rms_eps = {mlir_float(c['rms_eps'])} : f64}}}} {{")
    for name, t in m.weights:
        shape = "x".join(str(int(s)) for s in t.shape)
        out.append(f"  llaccel.weight @{name} : tensor<{shape}xf32>")
    out.append(f"  func.func @forward(%x: {_ty(m.input)} {{llaccel.name = \"input\"}}) -> {_ty(m.output)} {{")
    eps = mlir_float(c["rms_eps"])
    for op in m.ops:
        r, nm = _ssa(op), op.out.name
        ins = op.inputs
        if op.kind == "rmsnorm":
            out.append(f"    {r} = llaccel.rmsnorm {_ref(ins[0])}, @{op.attrs['w_gamma']} {{eps = {eps} : f64, "
                       f"llaccel.name = \"{nm}\"}} : {_ty(op.out)}")
        elif op.kind == "linear":
            ws = f"@{op.attrs['w_w']}" + (f", @{op.attrs['w_b']}" if "w_b" in op.attrs else "")
            out.append(f"    {r} = llaccel.linear {_ref(ins[0])}, {ws} {{llaccel.name = \"{nm}\"}} : "
                       f"{_ty(ins[0])} -> {_ty(op.out)}")
        elif op.kind == "rope":
            out.append(f"    {r} = llaccel.rope {_ref(ins[0])} {{heads = {op.attrs['heads']} : i64, "
                       f"llaccel.name = \"{nm}\"}} : {_ty(op.out)}")
        elif op.kind == "attention":
            a = op.attrs
            out.append(f"    {r} = llaccel.attention {_ref(ins[0])}, {_ref(ins[1])}, {_ref(ins[2])} "
                       f"{{layer = {a['layer']} : i64, heads = {a['heads']} : i64, kv_heads = {a['kv_heads']} : i64, "
                       f"head_dim = {a['head_dim']} : i64, llaccel.name = \"{nm}\"}} : "
                       f"({_ty(ins[0])}, {_ty(ins[1])}, {_ty(ins[2])}) -> {_ty(op.out)}")
        elif op.kind in ("add", "mul"):
            out.append(f"    {r} = llaccel.{op.kind} {_ref(ins[0])}, {_ref(ins[1])} {{llaccel.name = \"{nm}\"}} : {_ty(op.out)}")
        elif op.kind == "silu":
            out.append(f"    {r} = llaccel.silu {_ref(ins[0])} {{llaccel.name = \"{nm}\"}} : {_ty(op.out)}")
        else:  # pragma: no cover
            raise AssertionError(op.kind)
    out.append(f"    return {_ref(m.output)} : {_ty(m.output)}")
    out.append("  }")
    out.append("}")
    return "\n".join(out) + "\n"


def write_weights(m: Model, out_dir: Path) -> list[dict]:
    index = []
    off = 0
    with open(out_dir / "weights.bin", "wb") as f:
        for name, t in m.weights:
            arr = t.detach().cpu().contiguous().float().numpy()
            index.append({"name": name, "shape": [int(s) for s in arr.shape], "offset": off})
            f.write(arr.astype("<f4").tobytes())
            off += arr.nbytes
    with open(out_dir / "weights.json", "w") as f:
        json.dump(index, f, indent=1)
    return index


# --------------------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------------------
def export_program(model: TinyLlama | torch.nn.Module, max_seq: int) -> torch.export.ExportedProgram:
    model = model.eval().cpu()
    tokens = torch.zeros(1, 8, dtype=torch.long)
    Tdim = torch.export.Dim("T", min=2, max=max_seq)
    with torch.no_grad():
        return torch.export.export(model, (tokens,), dynamic_shapes=({1: Tdim},))


def import_model(model: torch.nn.Module, cfg_hint: ModelConfig | None = None, max_seq: int | None = None) -> Model:
    max_seq = max_seq or (cfg_hint.max_seq if cfg_hint else 256)
    ep = export_program(model, max_seq)
    return assign_roles(Importer(ep).run(), cfg_hint)


def export_dir(model: torch.nn.Module, out_dir: Path, cfg_hint: ModelConfig | None, tokenizer: dict | None,
               calib_seqs: int = 64, calib_len: int | None = None, calib_data: Path | None = None) -> Model:
    from .calibrate import calibrate
    out_dir.mkdir(parents=True, exist_ok=True)
    m = import_model(model, cfg_hint)
    (out_dir / "model.mlir").write_text(emit_mlir(m))
    write_weights(m, out_dir)
    if tokenizer is not None:
        with open(out_dir / "tokenizer.json", "w") as f:
            json.dump({"itos": tokenizer["itos"]}, f)
    calib = calibrate(m, n_seqs=calib_seqs, seq_len=calib_len or min(m.cfg["max_seq"], 256), data_dir=calib_data)
    with open(out_dir / "calib.json", "w") as f:
        json.dump(calib, f, indent=1)
    return m


def summarize(m: Model) -> str:
    counts: dict[str, int] = {}
    for op in m.ops:
        counts[op.kind] = counts.get(op.kind, 0) + 1
    n_w = sum(t.numel() for _, t in m.weights)
    return f"ops={counts} weights={len(m.weights)} tensors ({n_w:,} f32) cfg={m.cfg}"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="torch.export -> llaccel dialect")
    ap.add_argument("ckpt")
    ap.add_argument("-o", "--out", default="build/export/")
    ap.add_argument("--calib-seqs", type=int, default=64)
    ap.add_argument("--no-calib", action="store_true")
    args = ap.parse_args(argv)
    model, ckpt = load_checkpoint(args.ckpt)
    cfg = ModelConfig.from_dict(ckpt["config"])
    out = Path(args.out)
    if args.no_calib:
        out.mkdir(parents=True, exist_ok=True)
        m = import_model(model, cfg)
        (out / "model.mlir").write_text(emit_mlir(m))
        write_weights(m, out)
        with open(out / "tokenizer.json", "w") as f:
            json.dump({"itos": ckpt["tokenizer"]["itos"]}, f)
    else:
        m = export_dir(model, out, cfg, ckpt["tokenizer"], calib_seqs=args.calib_seqs)
    print(summarize(m))
    print("wrote", ", ".join(sorted(p.name for p in out.iterdir())))


if __name__ == "__main__":
    main()
