#!/usr/bin/env bash
# Source-only vLLM/Ascend cutover plus the prebuilt Mooncake diagnostic wheel.
set -Eeuo pipefail
umask 077

usage() {
    cat <<'HELP'
Usage: bash tools/deploy_pd_metrics_d.sh [--check | --apply | --rollback STATEFILE]
       [--vllm-dir DIR] [--ascend-dir DIR] [--python ABSOLUTE_PYTHON]
       [--wheel ABSOLUTE_WHEEL] [--backup-wheel ABSOLUTE_ORIGINAL_WHEEL]
       [--state-dir DIR]

Default: --check; no Git fetch, checkout, installation, or data-file writes.
--apply requires compatible diagnostic and exact rollback wheels, clean approved
Git histories, matching editable installs, Python 3.12/Linux, and stopped
vLLM/transfer-client workers. Wheels are discovered automatically; optional
--wheel and --backup-wheel select explicit files. An independent mooncake_master
may stay running unchanged. Missing pinned commits are fetched from the forks
below. vLLM/Ascend are never rebuilt/reinstalled. Detached checkouts preserve
branch tips. The rollback record and wheel copies are saved before cutover.
--rollback STATEFILE restores the recorded source branches/commits and previously
installed Mooncake payload. It refuses dirty trees, unexpected HEADs, changed original branch
tips, changed wheel backups, and running workers. Use the recorded interpreter;
--python may explicitly select it. No state file is sourced as shell code.
Wheel CLI files change for the NEXT launch only; no running master is restarted.

Defaults:
  --vllm-dir   /vllm-workspace/vllm
  --ascend-dir /vllm-workspace/vllm-ascend
  --python     /usr/local/python3.12.16/bin/python
  --state-dir  /vllm-workspace/pd-metrics-deploy-state

Default source pins (override with PD_DEPLOY_CORE_PIN / PD_DEPLOY_ASCEND_PIN;
each must contain exactly 40 hexadecimal characters):
  https://github.com/Emtemf/vllm.git
    f8b9de31a9e3a1b8ad76dee78917534121be39d1
  https://github.com/Emtemf/vllm-ascend.git
    afdbf1e3e3057b32cfe22075244e552a25113330
When installing a newer verified release, set PD_DEPLOY_CORE_PIN to that exact
published core commit on both P and D. The default is a committed source baseline,
not a moving branch tip; current HEAD must be its ancestor (newer HEADs are refused).
Download/copy this script outside both source roots before --apply or --rollback.

Example on either P or D (stop workers through your normal supervisor first):
  bash tools/deploy_pd_metrics_d.sh --check
  bash tools/deploy_pd_metrics_d.sh --apply
  bash tools/deploy_pd_metrics_d.sh --rollback /printed/run/state.json

Discovery searches /data/*/Mooncake-wheelbase (including mooncake-wheel/dist),
/root/Mooncake-wheelbase, /Mooncake-wheelbase, source/calling-directory dist,
/root/mooncake-official-backup and /vllm-workspace staging, up to four directory
levels below each root. Distinct valid candidates are an error listing paths;
byte-identical copies are one candidate. Use an optional override to choose.
If the diagnostic wheel is already installed, its exact matching candidate can
serve as rollback without an original baseline. Otherwise an absent original
may be downloaded from the image's Huawei HTTPS origin and must match installed
payload hashes. --check validates that download in memory without writing files.
No installed distribution is reconstructed into a guessed original wheel.

A check needs the pinned commits already present locally; only --apply may fetch
missing Git objects. No arbitrary snapshot import, reset, stash, force, or cleanup of
user changes is performed. Check mode also refuses active workers: this is a
readiness check, not an online diagnostic. Run inside the P or D container.
HELP
}

die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
mode=check
mode_set=0
statefile=
vllm_dir=/vllm-workspace/vllm
ascend_dir=/vllm-workspace/vllm-ascend
python=/usr/local/python3.12.16/bin/python
wheel=
backup_wheel=
state_dir=/vllm-workspace/pd-metrics-deploy-state
while (($#)); do
    case "$1" in
        --help|-h) usage; exit 0 ;;
        --check|--apply|--rollback)
            ((mode_set == 0)) || die 'Choose only one deployment mode.'
            mode_set=1; mode=${1#--}
            if [[ $mode == rollback ]]; then
                (($# >= 2)) || die '--rollback requires a state file.'
                statefile=$2; shift
            fi
            shift ;;
        --vllm-dir|--ascend-dir|--python|--wheel|--backup-wheel|--state-dir)
            (($# >= 2)) || die "$1 requires a value."
            case "$1" in
                --vllm-dir) vllm_dir=$2 ;;
                --ascend-dir) ascend_dir=$2 ;;
                --python) python=$2 ;;
                --wheel) wheel=$2 ;;
                --backup-wheel) backup_wheel=$2 ;;
                --state-dir) state_dir=$2 ;;
            esac
            shift 2 ;;
        *) die "Unknown option: $1 (use --help)." ;;
    esac
done
command -v git >/dev/null || die 'Required command missing: git'
[[ $python == /* && -x $python ]] || die '--python must name an executable absolute path.'
export PYTHONDONTWRITEBYTECODE=1 GIT_OPTIONAL_LOCKS=0

# Stdlib/packaging inspection only: never import vllm, torch, torch_npu, or .so files.
helper() {
    "$python" - "$@" <<'PY'
import base64
import csv
import hashlib
import importlib.metadata as md
import importlib.util
import io
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import sysconfig
import urllib.parse
import urllib.request
import shutil
import zipfile
from email.parser import BytesParser


def require(ok, message):
    if not ok:
        raise SystemExit("ERROR: " + message)


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def root_info(root):
    root = Path(root).resolve(strict=True)
    require(Path(git(root, "rev-parse", "--show-toplevel")).resolve() == root,
            f"Not the exact Git worktree root: {root}")
    require(not git(root, "status", "--porcelain=v1", "--untracked-files=all", "--ignore-submodules=none"),
            f"Dirty worktree (including untracked files/submodules): {root}")
    gitdir = Path(git(root, "rev-parse", "--absolute-git-dir"))
    require(not any((gitdir / marker).exists() for marker in
                    ("MERGE_HEAD", "CHERRY_PICK_HEAD", "REVERT_HEAD", "rebase-merge", "rebase-apply", "BISECT_LOG")),
            f"Git operation in progress: {root}")
    branch = subprocess.run(["git", "-C", str(root), "symbolic-ref", "--quiet", "--short", "HEAD"],
                            capture_output=True, text=True)
    return {"root": str(root), "precommit": git(root, "rev-parse", "HEAD"),
            "branch": branch.stdout.strip() if branch.returncode == 0 else None}


def runtime():
    require(sys.version_info[:2] == (3, 12), "Selected interpreter must be Python 3.12.")
    require(platform.system() == "Linux" and Path("/proc").is_dir(), "Run this utility in the Linux P or D container.")
    libc, version = platform.libc_ver()
    require(libc == "glibc" and version, "A detectable glibc runtime is required.")
    # Ignore this helper, the deployment shell, and its calling shell ancestors.
    ancestors = {os.getpid()}
    pid = os.getpid()
    while pid > 1:
        try:
            text = Path(f"/proc/{pid}/status").read_text()
            pid = int(re.search(r"^PPid:\s+(\d+)", text, re.M)[1])
            ancestors.add(pid)
        except (OSError, TypeError):
            break
    busy = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit() or int(proc.name) in ancestors:
            continue
        try:
            argv = (proc / "cmdline").read_bytes().split(b"\0")
            cmd = b" ".join(argv).lower()
            comm = (proc / "comm").read_bytes().lower()
        except FileNotFoundError:
            continue
        except PermissionError:
            # An incomplete process view cannot establish that workers are stopped.
            raise SystemExit("ERROR: Cannot inspect PID " + proc.name + "; use an account with process visibility.")
        executable = Path(os.fsdecode(argv[0])).name if argv and argv[0] else ""
        python_master = (executable.startswith("python") and len(argv) > 1
                         and Path(os.fsdecode(argv[1])).name == "mooncake_master")
        if executable == "mooncake_master" or comm.strip() == b"mooncake_master" or python_master:
            continue  # Shared master stays on its already-loaded executable.
        if any(token in cmd + comm for token in
               (b"vllm", b"mooncake", b"torchrun", b"ray::", b"raylet", b"multiprocessing.spawn", b"multiprocessing.forkserver")):
            busy.append(proc.name)
    require(not busy, "Stop serving/worker processes first; candidate PIDs: " + ", ".join(busy))
    return {"python": str(Path(sys.executable).resolve()), "architecture": platform.machine(),
            "glibc": version, "python_version": platform.python_version()}


def editable(name, module, root):
    dist = md.distribution(name)
    direct = json.loads(dist.read_text("direct_url.json") or "{}")
    url = urllib.parse.urlparse(direct.get("url", ""))
    require(direct.get("dir_info", {}).get("editable") is True and url.scheme == "file"
            and url.netloc in ("", "localhost")
            and Path(urllib.parse.unquote(url.path)).resolve() == Path(root),
            f"{name} must be an existing editable install from {root} in this interpreter.")
    spec = importlib.util.find_spec(module)  # Top-level only: no heavy parent import.
    require(spec is not None and spec.origin is not None
            and Path(spec.origin).resolve().is_relative_to(Path(root)),
            f"{module} import resolution does not point into {root}.")
    return {"name": dist.metadata["Name"], "version": dist.version, "origin": spec.origin}


def wheel_info(filename, diagnostic, data=None):
    from packaging.tags import sys_tags, parse_tag
    from packaging.utils import canonicalize_name, parse_wheel_filename
    path = Path(filename)
    require(path.is_absolute() and (data is not None or path.is_file()),
            "Wheel must be an existing absolute file: " + filename)
    name, version, _, tags = parse_wheel_filename(path.name)
    require(str(name) == "mooncake-transfer-engine" and str(version) == "0.3.13",
            "Require non-NPU-distribution mooncake-transfer-engine==0.3.13: " + path.name)
    compatible = set(sys_tags())
    require(bool(tags & compatible), "Wheel Python/ABI/architecture/glibc tags do not match this interpreter: " + path.name)
    require(any(t.platform.startswith("manylinux") for t in tags & compatible),
            "Require a compatible manylinux tag to establish the wheel's glibc floor.")
    with zipfile.ZipFile(io.BytesIO(data) if data is not None else path) as z:
        names = z.namelist()
        require(len(names) == len(set(names)) and all(not n.startswith("/") and ".." not in Path(n).parts for n in names),
                "Unsafe or duplicate wheel members.")
        metas = [n for n in names if n.endswith(".dist-info/METADATA")]
        require(len(metas) == 1, "Wheel must contain one distribution METADATA.")
        prefix = metas[0].rsplit("/", 1)[0]
        meta = BytesParser().parsebytes(z.read(metas[0]))
        require(canonicalize_name(meta["Name"]) == "mooncake-transfer-engine" and meta["Version"] == "0.3.13",
                "Wheel METADATA must be non-NPU mooncake-transfer-engine==0.3.13.")
        wt = BytesParser().parsebytes(z.read(prefix + "/WHEEL"))
        declared = set().union(*(parse_tag(t) for t in wt.get_all("Tag", [])))
        require(declared == tags and bool(declared & compatible), "Wheel WHEEL tags disagree with filename/runtime.")
        records = {row[0]: row[1:] for row in csv.reader(io.StringIO(z.read(prefix + "/RECORD").decode()))}
        payload = {}
        timing = False
        features = {"ascend_transport": False, "store": False, "master": False}
        for n in names:
            if n.endswith("/"):
                continue
            record = records.get(n)
            require(record is not None and len(record) == 2, "Wheel member absent/malformed in RECORD: " + n)
            data_hash = hashlib.sha256()
            previous = b""
            with z.open(n) as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    data_hash.update(chunk)
                    if n.startswith("mooncake/") and "store" in Path(n).name and b"BatchPutTiming" in previous + chunk:
                        timing = True
                    previous = chunk[-32:]
            digest = data_hash.hexdigest()
            if record[0]:
                require(record[0].startswith("sha256="), "Require SHA256 wheel RECORD entries: " + n)
                encoded = base64.urlsafe_b64encode(bytes.fromhex(digest)).rstrip(b"=").decode()
                require(record[0] == "sha256=" + encoded, "Wheel RECORD hash mismatch: " + n)
            else:
                require(n in (prefix + "/RECORD", prefix + "/RECORD.jws", prefix + "/RECORD.p7s"),
                        "Wheel contains unhashed payload: " + n)
            if ".dist-info/" not in n:
                payload[n] = digest
            base = Path(n).name
            features["ascend_transport"] |= n.startswith("mooncake/ascend_transport.") and base.endswith(".so")
            features["store"] |= n.startswith("mooncake/store.") and base.endswith(".so")
            features["master"] |= n == "mooncake/mooncake_master"
        if diagnostic:
            require(all(features.values()) and timing,
                    "Diagnostic wheel needs ascend_transport.so, store.so, mooncake_master and BatchPutTiming in store.")
        else:
            require(not timing, "--backup-wheel must be the original non-diagnostic official wheel, not the P build.")
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(data).hexdigest() if data is not None else sha(path),
            "name": meta["Name"], "version": meta["Version"],
            "payload": payload, "diagnostic": diagnostic}


def installed_matches(wheel):
    dist = md.distribution("mooncake-transfer-engine")
    if dist.version != "0.3.13":
        return False
    for name, digest in wheel["payload"].items():
        if ".data/" in name:
            _, relative = name.split(".data/", 1)
            scheme, relative = relative.split("/", 1)
            mappings = {"purelib": "purelib", "platlib": "platlib", "scripts": "scripts", "data": "data", "headers": "include"}
            require(scheme in mappings, "Unsupported wheel installation scheme: " + scheme)
            path = Path(sysconfig.get_path(mappings[scheme])) / relative
        else:
            path = Path(dist.locate_file(name))
        if not path.is_file() or sha(path) != digest:
            return False
    spec = importlib.util.find_spec("mooncake")
    return (spec is not None and spec.origin is not None
            and Path(spec.origin).resolve() == Path(dist.locate_file("mooncake/__init__.py")).resolve())


def wheel_paths(core, ascend):
    # Only known build/backup/staging locations, with a bounded search depth.
    roots = {Path.cwd() / "dist", Path(core) / "dist", Path(ascend) / "dist",
             Path(core).parent / "dist", Path("/root/mooncake-official-backup"),
             Path("/vllm-workspace"), Path("/data/y00624827/Mooncake-wheelbase"),
             Path("/root/Mooncake-wheelbase"), Path("/Mooncake-wheelbase")}
    roots.update(Path("/data").glob("*/Mooncake-wheelbase"))
    roots.update(Path("/vllm-workspace").glob("*/Mooncake-wheelbase"))
    paths = set()
    for root in sorted(roots):
        for depth in range(5):
            for path in root.glob("*/" * depth + "*.whl"):
                if "mooncake" in path.name.lower() and path.is_file():
                    paths.add(path.resolve())
    return sorted(paths)


def unique_wheel(wheels, label):
    # Identical copies in staging and build output are not distinct builds.
    by_hash = {}
    for wheel in wheels:
        by_hash.setdefault(wheel["sha256"], wheel)
    require(len(by_hash) <= 1, "Ambiguous " + label + " wheels; select an explicit override:\n  "
            + "\n  ".join(w["path"] for w in wheels))
    return next(iter(by_hash.values()), None)


def official_download(url):
    # Default SSL context verifies certificates and hostnames. Never disable TLS.
    with urllib.request.urlopen(url, timeout=60) as response:
        require(urllib.parse.urlparse(response.url).scheme == "https", "Official wheel redirected away from HTTPS.")
        return response.read()


def official_wheel():
    arch = platform.machine()
    require(arch in ("aarch64", "x86_64"), "No known official wheel URL for architecture: " + arch)
    name = f"mooncake_transfer_engine-0.3.13-cp312-cp312-manylinux_2_35_{arch}.whl"
    url = ("https://mindie.obs.cn-north-4.myhuaweicloud.com/"
           "buildtools/docker_images_tools/Mooncake/v0.3.14.rc1/" + name)
    wheel = wheel_info("/" + name, False, official_download(url))
    require(installed_matches(wheel),
            "Huawei official wheel does not match the installed payload; obtain the exact original wheel.")
    wheel["url"] = url
    return wheel


def discover_wheels(core, ascend, new, old):
    diagnostic, originals, rejected = [], [], []
    for path in wheel_paths(core, ascend) if not new or not old else ():
        try:
            diagnostic.append(wheel_info(str(path), True))
        except (SystemExit, OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile):
            try:
                wheel = wheel_info(str(path), False)
                if installed_matches(wheel):
                    originals.append(wheel)
            except (SystemExit, OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile) as exc:
                rejected.append(f"{path}: {exc}")
    candidate = wheel_info(new, True) if new else unique_wheel(diagnostic, "diagnostic (--wheel)")
    require(candidate is not None,
            "No compatible diagnostic wheel found. Copy the P-built .whl from "
            "/data/y00624827/Mooncake-wheelbase/mooncake-wheel/dist/ (or its dist/) "
            "to /vllm-workspace/wheels/ on this container, then rerun. "
            "P and D must have compatible Python/architecture/glibc."
            + ("\nRejected wheels:\n  " + "\n  ".join(rejected) if rejected else ""))
    new_match = installed_matches(candidate)
    if old:
        original = wheel_info(old, False)
    elif new_match:
        # Reinstalling this saved candidate exactly restores the previous payload.
        original = candidate
    else:
        original = unique_wheel(originals, "installed original (--backup-wheel)")
        if original is None:
            try:
                original = official_wheel()
            except (SystemExit, OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile) as exc:
                raise SystemExit("ERROR: No exact original rollback wheel found locally, and the verified-TLS "
                                 "Huawei download could not supply a matching wheel: " + str(exc)
                                 + "\nCopy the exact original from P's /root/mooncake-official-backup/ "
                                 "to /root/mooncake-official-backup/ or /vllm-workspace/wheels/ here. "
                                 "A reconstructed installed RECORD is not an original wheel.") from None
    require(installed_matches(original) or new_match,
            "Installed Mooncake payload matches neither selected wheel; obtain its exact original wheel before proceeding.")
    return candidate, original, new_match


def load_state(path):
    state = json.loads(Path(path).read_text())
    require(state.get("schema") == 1, "Unsupported deployment state schema.")
    require(state["runtime"]["python"] == str(Path(sys.executable).resolve()),
            "Use --python " + state["runtime"]["python"] + " for this rollback.")
    return state


try:
    op, *args = sys.argv[1:]
    if op == "inspect":
        core, ascend, new, old, state_dir = args
        rt = runtime()
        roots = [root_info(core), root_info(ascend)]
        require(roots[0]["root"] != roots[1]["root"], "Source roots must differ.")
        require(not any(Path(state_dir).resolve().is_relative_to(Path(r["root"])) for r in roots),
                "--state-dir must be outside both source trees.")
        try:
            md.distribution("mooncake-transfer-engine-npu")
        except md.PackageNotFoundError:
            pass
        else:
            raise SystemExit("ERROR: Remove the conflicting mooncake-transfer-engine-npu distribution through your normal environment procedure first.")
        installs = [editable("vllm", "vllm", roots[0]["root"]),
                    editable("vllm-ascend", "vllm_ascend", roots[1]["root"])]
        candidate, original, new_match = discover_wheels(core, ascend, new, old)
        dist = md.distribution("mooncake-transfer-engine")
        print(json.dumps({"schema": 1, "runtime": rt, "roots": roots, "editable_installs": installs,
                          "candidate": candidate, "original": original,
                          "installed": {"name": dist.metadata["Name"], "version": dist.version,
                                        "location": str(dist.locate_file("")),
                                        "payload_sha256": (candidate if new_match else original)["payload"],
                                        "matches": "candidate" if new_match else "original"}}))
    elif op == "save":
        snapshot = json.loads(os.environ["PD_DEPLOY_SNAPSHOT"])
        path, core_pin, ascend_pin, run_dir = args
        for root, pin in zip(snapshot["roots"], (core_pin, ascend_pin)):
            root["newcommit"] = pin
        for key in ("candidate", "original"):
            wheel = snapshot[key]
            copy = Path(run_dir) / key / Path(wheel["path"]).name
            if "url" in wheel:
                data = official_download(wheel["url"])
                require(hashlib.sha256(data).hexdigest() == wheel["sha256"],
                        "Official wheel changed after preflight; no cutover performed.")
                with copy.open("xb") as f:
                    f.write(data)
            else:
                shutil.copyfile(wheel["path"], copy)
            require(sha(copy) == wheel["sha256"], "Wheel changed while saving deployment state.")
            wheel["backup_path"] = str(copy.resolve())
        with open(path, "x") as f:
            json.dump(snapshot, f, indent=2)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
    elif op == "report":
        snapshot = json.loads(os.environ["PD_DEPLOY_SNAPSHOT"])
        for key in ("candidate", "original"):
            wheel = snapshot[key]
            label = "Diagnostic" if key == "candidate" else "Rollback"
            print(label + " wheel: " + wheel.get("url", wheel["path"]))
        print("Previous installed payload matches: " + snapshot["installed"]["matches"])
    elif op == "field":
        state = load_state(args[0])
        value = state
        for key in args[1].split("."):
            value = value[int(key)] if isinstance(value, list) else value[key]
        print("" if value is None else value)
    elif op == "guard":
        state = load_state(args[0])
        runtime()
        for root in state["roots"]:
            current = root_info(root["root"])
            require(current["precommit"] in (root["precommit"], root["newcommit"]),
                    "Unexpected HEAD; refuse rollback: " + root["root"])
            require(current["branch"] in (None, root["branch"]),
                    "Unexpected checked-out branch; refuse rollback: " + root["root"])
            if root["branch"]:
                require(git(root["root"], "rev-parse", "refs/heads/" + root["branch"]) == root["precommit"],
                        "Original branch tip changed; refuse rollback: " + root["root"])
        editable("vllm", "vllm", state["roots"][0]["root"])
        editable("vllm-ascend", "vllm_ascend", state["roots"][1]["root"])
        for key in ("candidate", "original"):
            require(sha(state[key]["backup_path"]) == state[key]["sha256"], "Saved wheel hash changed: " + key)
    elif op == "verify-wheel":
        state = load_state(args[0])
        require(installed_matches(state[args[1]]), "Installed Mooncake payload does not match " + args[1] + " wheel.")
    else:
        raise SystemExit("ERROR: Unknown internal operation.")
except (OSError, ValueError, KeyError, IndexError, TypeError, zipfile.BadZipFile, md.PackageNotFoundError, ImportError, subprocess.CalledProcessError) as exc:
    raise SystemExit("ERROR: Preflight/rollback inspection failed: " + str(exc)) from None
PY
}

core_pin=${PD_DEPLOY_CORE_PIN-f8b9de31a9e3a1b8ad76dee78917534121be39d1}
ascend_pin=${PD_DEPLOY_ASCEND_PIN-afdbf1e3e3057b32cfe22075244e552a25113330}
[[ $core_pin =~ ^[[:xdigit:]]{40}$ ]] || die 'PD_DEPLOY_CORE_PIN must contain exactly 40 hexadecimal characters.'
[[ $ascend_pin =~ ^[[:xdigit:]]{40}$ ]] || die 'PD_DEPLOY_ASCEND_PIN must contain exactly 40 hexadecimal characters.'
core_pin=${core_pin,,}
ascend_pin=${ascend_pin,,}
core_base=5df63a7203cdb9328d824a27b682e77c7b9928be
ascend_base=fd4be4dbcf751fcff0dd99d18389caf40680e9e7

field() { helper field "$statefile" "$1"; }
install_wheel() {
    "$python" -m pip install --no-index --force-reinstall --no-deps "$1"
}
recovery_notice() {
    printf '\nDeployment stopped. Services were not restarted. Retain the state and wheel copies.\n' >&2
    printf 'Recovery: bash %q --python %q --rollback %q\n' "$0" "$python" "$statefile" >&2
}

if [[ $mode == rollback ]]; then
    [[ -f $statefile ]] || die 'Rollback state file does not exist.'
    # Never read an interpreter path and execute it from an untrusted JSON file.
    # The default or explicit --python must match the interpreter recorded there.
    helper guard "$statefile"
    previous=$(field installed.matches)
    previous_wheel=$(field "$previous.backup_path")
    roots=("$(field roots.0.root)" "$(field roots.1.root)")
    commits=("$(field roots.0.precommit)" "$(field roots.1.precommit)")
    branches=("$(field roots.0.branch)" "$(field roots.1.branch)")
    trap recovery_notice ERR
    for i in 0 1; do
        if [[ -n ${branches[$i]} ]]; then
            git -C "${roots[$i]}" switch --no-overwrite-ignore -- "${branches[$i]}"
        else
            git -C "${roots[$i]}" switch --no-overwrite-ignore --detach "${commits[$i]}"
        fi
    done
    install_wheel "$previous_wheel"
    helper verify-wheel "$statefile" "$previous"
    printf 'Rollback complete. Recorded source branches/commits and previous Mooncake payload restored. Start services manually.\n'
    exit 0
fi

[[ $vllm_dir == /* && $ascend_dir == /* && $state_dir == /* ]] || die 'Source roots and --state-dir must be absolute paths.'
# No checkout, state-directory creation or pip installation precedes the complete
# runtime/editable/wheel/process preflight. Optional HTTPS downloads stay in memory.
export PD_DEPLOY_SNAPSHOT
PD_DEPLOY_SNAPSHOT=$(helper inspect "$vllm_dir" "$ascend_dir" "$wheel" "$backup_wheel" "$state_dir")
helper report

check_history() {
    local root=$1 base=$2 pin=$3 url=$4
    if [[ $(git -C "$root" rev-parse --is-shallow-repository) == true ]]; then
        [[ $mode == apply ]] || die "Shallow history in $root; --apply fetches full pinned history before checking ancestry."
        git -C "$root" -c protocol.file.allow=never fetch --unshallow --no-tags --no-write-fetch-head "$url" "$pin"
    fi
    if ! git -C "$root" cat-file -e "$pin^{commit}" 2>/dev/null; then
        [[ $mode == apply ]] || die "Pinned commit absent in $root; --check never fetches Git objects. --apply fetches $pin from $url after preflight."
        printf 'Fetching reviewed commit %s from %s\n' "$pin" "$url"
        git -C "$root" -c protocol.file.allow=never fetch --no-tags --no-write-fetch-head "$url" "$pin"
    fi
    git -C "$root" cat-file -e "$base^{commit}" 2>/dev/null || die "Missing approved baseline in $root; use the matching existing Git checkout, not a snapshot."
    git -C "$root" merge-base --is-ancestor "$base" HEAD || die "Different/unrelated baseline: $root"
    git -C "$root" merge-base --is-ancestor "$base" "$pin" || die "Reviewed pin is not descended from approved baseline: $root"
    git -C "$root" merge-base --is-ancestor HEAD "$pin" || die "HEAD is not the baseline or a proven ancestor of the reviewed pin: $root"
    # Prove that baseline->HEAD lies on this exact reviewed diagnostic history.
    # Side branches merged into the diagnostic line are not accepted as bases.
    local head cursor found=0
    head=$(git -C "$root" rev-parse HEAD)
    while IFS= read -r cursor; do
        if [[ $cursor == "$head" ]]; then found=1; break; fi
    done < <(git -C "$root" rev-list --first-parent "$pin" "^$base"; printf '%s\n' "$base")
    ((found == 1)) || die "HEAD is not on the approved first-parent baseline/diagnostic history: $root"
}
check_history "$vllm_dir" "$core_base" "$core_pin" https://github.com/Emtemf/vllm.git
check_history "$ascend_dir" "$ascend_base" "$ascend_pin" https://github.com/Emtemf/vllm-ascend.git
if [[ $mode == check ]]; then
    printf 'READY: clean approved histories, matching editable installs, Python/wheels/backup validated, no candidate workers. No deployment changes made.\n'
    exit 0
fi

# Recheck after any fetch; refuse intervening edits/processes before saving state.
latest=$(helper inspect "$vllm_dir" "$ascend_dir" "$wheel" "$backup_wheel" "$state_dir")
[[ $latest == "$PD_DEPLOY_SNAPSHOT" ]] || die 'Environment changed during preflight; rerun from a stable stopped environment.'
mkdir -p -- "$state_dir"
run_dir=$(mktemp -d "$state_dir/run-XXXXXXXX")
mkdir -- "$run_dir/candidate" "$run_dir/original"
statefile=$run_dir/state.json
helper save "$statefile" "$core_pin" "$ascend_pin" "$run_dir"
printf 'Rollback state (saved before cutover): %s\n' "$statefile"
printf 'Rollback command: bash %q --python %q --rollback %q\n' "$0" "$python" "$statefile"
trap recovery_notice ERR
# Inspect the durable record, wheel copies, both roots and worker state once more.
helper guard "$statefile"
git -C "$vllm_dir" switch --no-overwrite-ignore --detach "$core_pin"
git -C "$ascend_dir" switch --no-overwrite-ignore --detach "$ascend_pin"
install_wheel "$(field candidate.backup_path)"
helper verify-wheel "$statefile" candidate
printf 'Applied pinned source cutover and verified installed Mooncake payload. No services restarted.\n'
printf 'State: %s\n' "$statefile"
