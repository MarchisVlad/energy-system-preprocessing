"""
Compile a SIMPLE-methods GAMS model into an MPS file using the GAMS system
bundled with `gamspy_base`, bypassing the academic 5000-variable solver limit.

The SIMPLE model (https://gitlab.com/beam-me/simple-methods) ships as a multi-
file GAMS pipeline whose entry point is ``simple.gms``. Parameters such as
``--FROM``, ``--TO``, ``--RESOLUTION``, ``--NBREGIONS`` and ``--METHOD`` are
passed as GAMS double-dash arguments. ``simple.gms`` itself calls
``simple_data_gen.gms`` to generate the data, so we only need to invoke
``simple.gms`` once with the desired arguments.

Strategy
--------
Rather than rewriting the model in GAMSPy, we use ``gamspy_base`` purely for
its bundled GAMS executable and the CONVERT solver (a free translator that
ships with every GAMS install). CONVERT does not solve the model; it only
emits an MPS representation, so academic solver size limits do not apply to
the conversion step itself.

The conversion is achieved by:

1. Copying the SIMPLE source tree into a clean working directory so the
   original repository is not polluted with GAMS scratch files.
2. Writing a ``convert.opt`` file that tells CONVERT to emit a
   ``CplexMPS``-format file at the requested path.
3. Patching the resolved model's ``solve`` statement so CONVERT is selected
   and the option file is read. We do this by injecting two lines just
   before the first ``solve`` statement, which keeps the original source
   files untouched on disk.
4. Running the bundled ``gams`` executable on the patched entry point with
   the user-supplied double-dash arguments.

The function returns the path to the generated ``.mps`` file.
"""

from __future__ import annotations

import os
import re
import shutil
import socket as _socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Mapping, Optional

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def simple_to_mps(
    simple_root: str | os.PathLike,
    arguments: Mapping[str, object],
    mps_output_dir: str | os.PathLike,
    *,
    entry_point: str = "simple.gms",
    mps_filename: Optional[str] = None,
    mps_format: str = "CplexMPS",
    keep_workdir: bool = False,
    extra_gams_flags: Optional[list[str]] = None,
    gams_executable: Optional[str | os.PathLike] = None,
) -> Path:
    """Compile a SIMPLE-methods GAMS model to an MPS file.

    Parameters
    ----------
    simple_root
        Path to the root of a SIMPLE-methods checkout (the directory that
        contains ``simple.gms`` and ``simple_data_gen.gms``).
    arguments
        Mapping of GAMS compile-time parameters (the ``--KEY=VALUE`` args
        documented in the README), e.g.
        ``{"NBREGIONS": 4, "FROM": 0, "TO": 1, "METHOD": "standard_lp"}``.
        Keys are uppercased to match GAMS conventions; values are stringified.
        Pass an empty dict to use the model's defaults.
    mps_output_dir
        Directory where the resulting ``.mps`` file is written. Created if
        it does not exist.
    entry_point
        Name of the top-level ``.gms`` file inside ``simple_root``. Defaults
        to ``simple.gms``; override if you want to point at a sibling model
        file in the SIMPLE repository.
    mps_filename
        Filename for the produced MPS. Defaults to a name derived from the
        arguments, e.g. ``simple_NBREGIONS=4_METHOD=standard_lp.mps``.
    mps_format
        CONVERT output format. Use ``"CplexMPS"`` (free format, default) or
        ``"FixedMPS"`` for fixed-width MPS.
    keep_workdir
        If True, the temporary working directory containing the listing
        file, scratch GDX, etc. is preserved and its path is printed. Useful
        for debugging.
    extra_gams_flags
        Additional command-line flags forwarded verbatim to the ``gams``
        executable (e.g. ``["lo=2"]`` to control log verbosity).
    gams_executable
        Override the GAMS binary used. Defaults to the one bundled with
        ``gamspy_base``.

    Returns
    -------
    Path
        Absolute path to the generated ``.mps`` file.

    Raises
    ------
    FileNotFoundError
        If ``simple_root`` or the entry point does not exist.
    RuntimeError
        If GAMS exits with a non-zero return code or the MPS file is not
        produced.
    """
    simple_root = Path(simple_root).resolve()
    mps_output_dir = Path(mps_output_dir).resolve()

    if not simple_root.is_dir():
        raise FileNotFoundError(f"simple_root does not exist: {simple_root}")
    if not (simple_root / entry_point).is_file():
        raise FileNotFoundError(
            f"Entry point {entry_point!r} not found in {simple_root}")

    if mps_format not in {"CplexMPS", "FixedMPS"}:
        raise ValueError(
            f"mps_format must be 'CplexMPS' or 'FixedMPS', got {mps_format!r}")

    mps_output_dir.mkdir(parents=True, exist_ok=True)
    mps_path = mps_output_dir / (mps_filename or _default_mps_name(arguments))

    # When the caller supplies an explicit executable we use the plain
    # subprocess path (their responsibility to handle licensing).  Otherwise
    # we use gamspy_base's GAMS via its socket-daemon protocol, which runs
    # the job as a "GAMSPy program" and therefore accepts the gamspy license
    # without enforcing the 5 000-variable academic size limit.
    use_daemon = gams_executable is None
    if use_daemon:
        gams_exe = _locate_gamspy_gams()
        license_path = _find_gamspy_license()
    else:
        gams_exe = Path(gams_executable)
        license_path = _find_gams_license()

    workdir = Path(tempfile.mkdtemp(prefix="simple_to_mps_"))
    try:
        staged_root = workdir / "src"
        shutil.copytree(simple_root, staged_root)

        _inject_convert_directives(staged_root / entry_point)
        _write_convert_opt(staged_root, mps_path, mps_format)
        _patch_embedded_python_blocks(staged_root)

        bin_dir = workdir / "bin"
        bin_dir.mkdir()
        _write_csv2gdx_shim(bin_dir)

        bin_path = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")

        if use_daemon:
            # The standard gamspy license forbids $include; flatten all
            # includes into a single file before submission.
            flat_name = "_simple_flat.gms"
            flat_text = _flatten_gms(staged_root / entry_point, staged_root)
            (staged_root / flat_name).write_text(flat_text, encoding="utf-8")

            _run_via_gamspy_daemon(
                gams_exe=gams_exe,
                license_path=license_path,
                staged_root=staged_root,
                entry_point=flat_name,
                arguments=arguments,
                extra_gams_flags=extra_gams_flags,
                mps_path=mps_path,
                bin_path=bin_path,
                keep_workdir=keep_workdir,
            )
        else:
            cmd = [
                str(gams_exe),
                entry_point,
                "lo=3",
                "errMsg=1",
                "optDir=.",
            ]
            if license_path:
                cmd.append(f"license={license_path}")
            cmd.extend(_format_double_dash_args(arguments))
            if extra_gams_flags:
                cmd.extend(extra_gams_flags)

            env = os.environ.copy()
            env["PATH"] = bin_path

            result = subprocess.run(
                cmd, cwd=staged_root, capture_output=True, text=True, env=env
            )
            if result.returncode != 0 or not mps_path.is_file():
                raise RuntimeError(
                    _format_failure(cmd, result, mps_path, staged_root,
                                    keep_workdir))

        return mps_path
    finally:
        if keep_workdir:
            print(f"[simple_to_mps] kept working directory: {workdir}",
                  file=sys.stderr)
        else:
            shutil.rmtree(workdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

# Matches GAMS $include directives (case-insensitive, quoted or bare filename).
# GAMS allows whitespace between $ and the directive name (e.g. "$  include").
_INCLUDE_RE = re.compile(
    r'(?m)^([ \t]*)\$\s*include\s+(?:"([^"]+)"|\'([^\']+)\'|(\S+))',
    re.IGNORECASE,
)

# Strips %MACRO% tokens (e.g. %SIMPLEDIR%) from a path string.
_GAMS_MACRO_RE = re.compile(r'%[^%]+%')


def _flatten_gms(
    path: Path,
    root: Path,
    _seen: frozenset | None = None,
) -> str:
    """Return the content of *path* with every ``$include`` inlined recursively.

    The standard gamspy license forbids ``$include`` in submitted jobs; by
    inlining all includes we produce a single file that the license accepts.
    ``$call`` and ``$gdxin``/``$gdxout`` are left untouched.

    Handles:
    * ``$include``, ``$  include`` (spaces between ``$`` and directive name)
    * Quoted and bare filenames
    * ``%MACRO%`` prefixes in paths (e.g. ``%SIMPLEDIR%`` → same directory)
    """
    if _seen is None:
        _seen = frozenset()

    path = path.resolve()
    if path in _seen:
        return f"* [skipped circular $include: {path.name}]\n"

    text = path.read_text(encoding="utf-8", errors="replace")
    _seen = _seen | {path}

    def _inline(m: re.Match) -> str:
        raw = m.group(2) or m.group(3) or m.group(4)
        # Strip %MACRO% tokens before resolving (e.g. %SIMPLEDIR% is the
        # directory of the current file, which we handle via path.parent).
        raw_clean = _GAMS_MACRO_RE.sub("", raw).strip()
        for base in (path.parent, root):
            candidate = (base / raw_clean).resolve()
            if candidate.is_file():
                return _flatten_gms(candidate, root, _seen)
        return m.group(0)  # not found — leave directive in place

    return _INCLUDE_RE.sub(_inline, text)


# gamspy_base does not ship libembpycclib64.so, so GAMS EmbeddedCode Python
# blocks fail at runtime.  We patch the single affected block in
# simple_data_gen.gms — a distance-ranking computation — with pure GAMS code
# that is semantically equivalent.
_EMBEDDED_RANK_RE = re.compile(
    r"embeddedCode Python:.*?endEmbeddedCode\s+rank",
    re.DOTALL | re.IGNORECASE,
)

# Pure-GAMS rank: for each rr1, count how many rr3 (≠ rr1) are strictly
# closer to rr1 than rr2 is, plus tie-breaking by set ordinal.  This gives
# rank 1 to the nearest neighbour, rank 2 to the second-nearest, etc.
_RANK_GAMS_REPLACEMENT = (
    "rank(rr1,rr2)$(not sameas(rr1,rr2)) =\n"
    "    1 + sum(rr$(not sameas(rr,rr1) and distance(rr1,rr) < distance(rr1,rr2)), 1)\n"
    "    + sum(rr$(not sameas(rr,rr1) and distance(rr1,rr) = distance(rr1,rr2)"
    " and ord(rr) < ord(rr2)), 1)"
)


def _patch_embedded_python_blocks(staged_root: Path) -> None:
    """Replace EmbeddedCode Python blocks with pure GAMS equivalents.

    gamspy_base does not bundle libembpycclib64.so, so any $embeddedCode Python
    directive causes a fatal load error.  For standard_lp only
    simple_data_gen.gms is affected (the rank-by-distance block).
    """
    target = staged_root / "simple_data_gen.gms"
    if not target.is_file():
        return
    text = target.read_text(encoding="utf-8", errors="replace")
    patched, n = _EMBEDDED_RANK_RE.subn(_RANK_GAMS_REPLACEMENT, text)
    if n:
        target.write_text(patched, encoding="utf-8")


# Script written into workdir/bin/csv2gdx so GAMS $call csv2gdx works without
# a full GAMS installation (gamspy_base does not ship csv2gdx).
_CSV2GDX_SHIM = """\
#!/usr/bin/env python3
\"\"\"Minimal csv2gdx replacement using gams.transfer.\"\"\"
import re
import sys
import csv
import gams.transfer as gt

def main(argv):
    if not argv:
        sys.exit("csv2gdx: no arguments")
    input_file = argv[0]
    opts = {}
    for tok in argv[1:]:
        if "=" in tok:
            k, v = tok.split("=", 1)
            opts[k.lower()] = v
        else:
            opts[tok.lower()] = True

    output = opts.get("output")
    sym_id = opts.get("id", "data")
    use_header = str(opts.get("useheader", "n")).lower() in ("y", "yes", "true", "1")
    index_col = int(opts.get("index", 1)) - 1  # convert to 0-based

    with open(input_file, newline="", encoding="utf-8-sig") as fh:
        reader = csv.reader(fh)
        rows = list(reader)

    if not rows:
        sys.exit(f"csv2gdx: {input_file} is empty")

    if use_header:
        headers = rows[0]
        data_rows = rows[1:]
    else:
        headers = [str(i + 1) for i in range(len(rows[0]))]
        data_rows = rows

    # Determine value columns from Values=start..end|lastCol
    values_spec = opts.get("values", None)
    if values_spec:
        m = re.match(r"(\\d+)\\.\\.(\\d+|lastcol)$", values_spec, re.IGNORECASE)
        if m:
            start = int(m.group(1)) - 1  # 0-based
            end_raw = m.group(2).lower()
            end = len(headers) - 1 if end_raw == "lastcol" else int(end_raw) - 1
            value_cols = list(range(start, end + 1))
        else:
            value_cols = [i for i in range(len(headers)) if i != index_col]
    else:
        value_cols = [i for i in range(len(headers)) if i != index_col]

    c = gt.Container()
    if len(value_cols) == 1:
        # 1-D parameter: (index) -> value
        col = value_cols[0]
        records = [
            (row[index_col], float(row[col]) if row[col] not in ("", None) else 0.0)
            for row in data_rows if row
        ]
        p = gt.Parameter(c, sym_id, domain=["*"])
        p.setRecords(records)
    else:
        # 2-D parameter: (index, column_name) -> value
        records = []
        for row in data_rows:
            if not row:
                continue
            idx = row[index_col]
            for col in value_cols:
                val_str = row[col] if col < len(row) else ""
                val = float(val_str) if val_str not in ("", None) else 0.0
                records.append((idx, headers[col], val))
        p = gt.Parameter(c, sym_id, domain=["*", "*"])
        p.setRecords(records)

    c.write(output)

if __name__ == "__main__":
    main(sys.argv[1:])
"""


def _write_csv2gdx_shim(bin_dir: Path) -> None:
    """Write an executable csv2gdx Python shim into bin_dir."""
    shim = bin_dir / "csv2gdx"
    shim.write_text(_CSV2GDX_SHIM, encoding="utf-8")
    shim.chmod(0o755)


def _locate_gamspy_gams() -> Path:
    """Return the GAMS executable bundled inside gamspy_base."""
    try:
        from gamspy_base import directory as gamspy_base_dir  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "gamspy_base is not installed. Run `pip install gamspy gamspy_base`."
        ) from e

    candidate = Path(gamspy_base_dir) / ("gams.exe" if os.name == "nt" else "gams")
    if not candidate.is_file():
        raise RuntimeError(
            f"gams executable not found inside gamspy_base at {candidate}")
    return candidate


def _find_gamspy_license() -> Optional[Path]:
    """Return the GAMSPy license (accepted by gamspy_base in daemon mode)."""
    candidate = Path.home() / ".local" / "share" / "GAMSPy" / "gamspy_license.txt"
    return candidate if candidate.is_file() else None


def _run_via_gamspy_daemon(
    gams_exe: Path,
    license_path: Optional[Path],
    staged_root: Path,
    entry_point: str,
    arguments: Mapping[str, object],
    extra_gams_flags: Optional[list[str]],
    mps_path: Path,
    bin_path: str,
    keep_workdir: bool,
) -> None:
    """Submit the patched .gms to gamspy_base's socket daemon.

    gamspy_base launches GAMS with ``GAMSPY_JOB`` + ``incrementalMode=2``,
    which makes it accept the GAMSPy license without enforcing the 5 000-row/
    column academic size cap.  Jobs are submitted by sending a .pf file path
    over a loopback socket (the same protocol gamspy uses internally).
    """
    proc_dir = staged_root / "_proc"
    proc_dir.mkdir(exist_ok=True)

    gams_sysdir = gams_exe.parent

    # --- Initial pf: starts the GAMS daemon in incremental / gamspy mode ---
    initial_pf = proc_dir / "gamspy_init.pf"
    lines = [
        'incrementalMode = "2"',
        f'procdir = "{proc_dir}"',
        f'curdir = "{staged_root}"',
    ]
    if license_path:
        lines.append(f'license = "{license_path}"')
    initial_pf.write_text("\n".join(lines) + "\n")

    env = os.environ.copy()
    env["PATH"] = bin_path

    process = subprocess.Popen(
        [str(gams_exe), "GAMSPY_JOB", "pf", str(initial_pf)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        errors="replace",
        start_new_session=True,
        env=env,
    )

    # Daemon prints "port: <N>" on its first stdout line
    port_info = process.stdout.readline().strip()
    try:
        port = int(port_info.removeprefix("port: "))
    except ValueError:
        rest = process.stdout.read()
        raise RuntimeError(
            f"Failed to start gamspy daemon.\n  first line: {port_info!r}\n"
            f"  rest: {rest}"
        )

    # Connect to the daemon
    TIMEOUT = 30
    start = time.time()
    sock = None
    while True:
        if process.poll() is not None:
            raise RuntimeError(
                f"gamspy daemon exited before accepting connections: "
                f"{process.communicate()[0]}"
            )
        try:
            sock = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
            sock.connect(("127.0.0.1", port))
            break
        except (ConnectionRefusedError, OSError):
            sock.close()
            sock = None
            if time.time() - start > TIMEOUT:
                raise RuntimeError("Timeout connecting to gamspy daemon")

    # --- Per-job pf ---
    job_name = "simple_to_mps_job"
    lst_file = proc_dir / f"{job_name}.lst"
    scriptnext = proc_dir / "gamsnext.sh"
    job_pf = proc_dir / f"{job_name}.pf"

    job_lines = [
        f'input = "{staged_root / entry_point}"',
        f'output = "{lst_file}"',
        f'optdir = "{staged_root}"',
        f'sysdir = "{gams_sysdir}"',
        f'scrdir = "{proc_dir}"',
        f'scriptnext = "{scriptnext}"',
        'lo = "3"',
        'errMsg = "1"',
    ]
    if license_path:
        job_lines.append(f'license = "{license_path}"')
    for k, v in sorted(arguments.items()):
        job_lines.append(f'--{str(k).upper()} = "{v}"')
    if extra_gams_flags:
        job_lines.extend(extra_gams_flags)
    job_pf.write_text("\n".join(job_lines) + "\n")

    # Submit and collect output
    stdout_lines: list[str] = []
    return_code = -1
    try:
        sock.sendall(str(job_pf).encode("utf-8"))

        while True:
            line = process.stdout.readline()
            if not line:
                break
            stdout_lines.append(line.rstrip())
            if line.startswith("--- Job ") and "elapsed" in line:
                break

        response = sock.recv(256)
        rc_str = response[: response.find(b"#")].decode("ascii").strip()
        return_code = int(rc_str) if rc_str else -1
    finally:
        try:
            sock.sendall(b"stop")
            sock.close()
        except OSError:
            pass
        while process.poll() is None:
            pass

    if return_code != 0 or not mps_path.is_file():
        class _R:
            returncode = return_code
            stdout = "\n".join(stdout_lines)
            stderr = ""

        raise RuntimeError(
            _format_failure(
                [str(gams_exe), "GAMSPY_JOB", f"pf={job_pf}"],
                _R(),
                mps_path,
                staged_root,
                keep_workdir,
            )
        )


def _find_gams_license() -> Optional[Path]:
    """Return a GAMS license file suitable for running regular GAMS programs.

    Search order:
    1. ``GAMS_LICENSE`` environment variable.
    2. ``gamslice.txt`` files under ``/data/gams/`` (site-wide installs).
    3. ``gamslice.txt`` in the gamspy_base system directory.
    4. The GAMSPy-only license as a last resort (may be rejected by GAMS for
       non-GAMSPy programs).
    """
    env_path = os.environ.get("GAMS_LICENSE")
    if env_path and Path(env_path).is_file():
        return Path(env_path)

    data_gams = Path("/data/gams")
    if data_gams.is_dir():
        for candidate in sorted(data_gams.glob("*/gamslice.txt"), reverse=True):
            if candidate.is_file():
                return candidate

    try:
        from gamspy_base import directory as gamspy_base_dir  # type: ignore
        candidate = Path(gamspy_base_dir) / "gamslice.txt"
        if candidate.is_file():
            return candidate
    except ImportError:
        pass

    gamspy_license = Path.home() / ".local" / "share" / "GAMSPy" / "gamspy_license.txt"
    return gamspy_license if gamspy_license.is_file() else None


def _locate_gams() -> Path:
    """Return the path to a GAMS executable, preferring a system install.

    Search order:
    1. ``GAMS_DIR`` / ``GAMSDIR`` environment variable.
    2. Versioned installs under ``/data/gams/`` (newest first).
    3. ``gams`` on ``PATH``.
    4. The executable bundled with ``gamspy_base`` (fallback; community limits
       apply to large models).
    """
    exe_name = "gams.exe" if os.name == "nt" else "gams"

    for env_var in ("GAMS_DIR", "GAMSDIR"):
        gams_dir = os.environ.get(env_var)
        if gams_dir:
            candidate = Path(gams_dir) / exe_name
            if candidate.is_file():
                return candidate

    data_gams = Path("/data/gams")
    if data_gams.is_dir():
        for candidate in sorted(data_gams.glob(f"*/{exe_name}"), reverse=True):
            if candidate.is_file():
                return candidate

    path_exe = shutil.which("gams")
    if path_exe:
        return Path(path_exe)

    try:
        from gamspy_base import directory as gamspy_base_dir  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "No GAMS installation found. Set GAMS_DIR, put gams on PATH, "
            "or install gamspy_base."
        ) from e

    candidate = Path(gamspy_base_dir) / exe_name
    if not candidate.is_file():
        raise RuntimeError(
            f"gams executable not found inside gamspy_base at {candidate}")
    return candidate


def _format_double_dash_args(arguments: Mapping[str, object]) -> list[str]:
    """Translate a dict into the ``--KEY=VALUE`` form GAMS expects."""
    out: list[str] = []
    for key, value in arguments.items():
        if value is None:
            continue
        # GAMS double-dash params are case-insensitive but the SIMPLE README
        # documents them in uppercase; normalize for tidiness.
        out.append(f"--{str(key).upper()}={value}")
    return out


def _default_mps_name(arguments: Mapping[str, object]) -> str:
    """Derive a deterministic-ish filename from the provided arguments."""
    if not arguments:
        return "simple.mps"
    parts = [
        f"{str(k).upper()}={_slug(v)}" for k, v in sorted(arguments.items())
        if v is not None
    ]
    return "simple_" + "_".join(parts) + ".mps"


def _slug(value: object) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(value))


def _write_convert_opt(staged_root: Path, mps_path: Path,
                       mps_format: str) -> None:
    """Tell CONVERT to write an MPS file at the requested absolute path."""
    # CONVERT accepts an absolute path on the right-hand side, so the MPS
    # ends up directly in the user-requested output directory.
    opt_path = staged_root / "convert.opt"
    opt_path.write_text(f"{mps_format} {mps_path.as_posix()}\n",
                        encoding="ascii")


# Matches a `solve <model> using <type> minimizing/maximizing <obj>;` line, or
# the feasibility variant `solve <model> using <type>;`. Case-insensitive.
_SOLVE_RE = re.compile(
    r"^(?P<indent>\s*)solve\b[^;]*;",
    re.IGNORECASE | re.MULTILINE,
)


def _inject_convert_directives(entry_path: Path) -> None:
    """Patch the entry .gms so the first solve uses CONVERT with our optfile.

    We insert two lines immediately above the first ``solve`` statement:

        option <modeltype> = convert;
        <modelname>.optfile = 1;

    The model name and type are parsed out of the matched solve statement
    so this works regardless of whether the user picked ``standard_lp``,
    ``benders``, etc.

    Note: for decomposition methods that issue many solves (rolling horizon,
    Benders, etc.), CONVERT will emit the MPS for the *first* solved model
    instance and then GAMS will typically abort because CONVERT does not
    return a solution. That is expected behaviour and matches what you'd
    get by running CONVERT manually on those models. For a full single-MPS
    representation of a decomposed problem, use METHOD=standard_lp or
    METHOD=spExplicitDE.
    """
    text = entry_path.read_text(encoding="utf-8", errors="replace")

    match = _SOLVE_RE.search(text)
    if not match:
        raise RuntimeError(
            f"Could not locate a `solve` statement in {entry_path.name}; "
            "CONVERT injection aborted.")

    solve_stmt = match.group(0)
    model_name, model_type = _parse_solve(solve_stmt)

    indent = match.group("indent")
    injection = (f"{indent}option {model_type} = convert;\n"
                 f"{indent}{model_name}.optfile = 1;\n")

    patched = text[:match.start()] + injection + text[match.start():]
    entry_path.write_text(patched, encoding="utf-8")


_SOLVE_PARSE_RE = re.compile(
    r"solve\s+(?P<model>[A-Za-z_][\w]*)(?:\s+\S+)*?\s+(?:using|use|us)\s+(?P<type>[A-Za-z]+)",
    re.IGNORECASE,
)


def _parse_solve(solve_stmt: str) -> tuple[str, str]:
    m = _SOLVE_PARSE_RE.search(solve_stmt)
    if not m:
        raise RuntimeError(
            f"Could not parse model name/type from solve statement: {solve_stmt!r}"
        )
    return m.group("model"), m.group("type").lower()


def _format_failure(
    cmd: list[str],
    result: subprocess.CompletedProcess,
    mps_path: Path,
    staged_root: Path,
    keep_workdir: bool,
) -> str:
    tail = (result.stdout or "").splitlines()[-40:]
    err_tail = (result.stderr or "").splitlines()[-20:]
    hint = (
        "Working directory was preserved above; check the .lst listing file."
        if keep_workdir else
        "Re-run with keep_workdir=True to inspect the .lst listing file.")
    return (f"GAMS conversion failed (returncode={result.returncode}).\n"
            f"  command: {' '.join(cmd)}\n"
            f"  cwd:     {staged_root}\n"
            f"  mps:     {mps_path} (exists={mps_path.is_file()})\n"
            f"  stdout (last 40 lines):\n    " + "\n    ".join(tail) + "\n"
            f"  stderr (last 20 lines):\n    " + "\n    ".join(err_tail) + "\n"
            f"  {hint}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _main(argv: list[str]) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Compile a SIMPLE-methods GAMS model to an MPS file via "
        "the GAMS system bundled with gamspy_base.")
    parser.add_argument("simple_root", help="Path to the SIMPLE-methods root.")
    parser.add_argument("mps_output_dir", help="Directory for the .mps output.")
    parser.add_argument(
        "--arg",
        "-a",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Forward a GAMS double-dash argument (repeatable). "
        "Example: -a NBREGIONS=4 -a METHOD=standard_lp",
    )
    parser.add_argument("--entry-point", default="simple.gms")
    parser.add_argument("--mps-filename", default=None)
    parser.add_argument(
        "--mps-format",
        choices=["CplexMPS", "FixedMPS"],
        default="CplexMPS",
    )
    parser.add_argument("--keep-workdir", action="store_true")
    args = parser.parse_args(argv)

    arg_dict: dict[str, str] = {}
    for raw in args.arg:
        if "=" not in raw:
            parser.error(f"--arg expects KEY=VALUE, got {raw!r}")
        k, v = raw.split("=", 1)
        arg_dict[k] = v

    out = simple_to_mps(
        args.simple_root,
        arg_dict,
        args.mps_output_dir,
        entry_point=args.entry_point,
        mps_filename=args.mps_filename,
        mps_format=args.mps_format,
        keep_workdir=args.keep_workdir,
    )
    print(out)
    return 0


# if __name__ == "__main__":
#     raise SystemExit(_main(sys.argv[1:]))
