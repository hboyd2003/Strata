#!/usr/bin/env python3
"""Benchmark a running Strata server and compare runs: the harness every change in the gfx906 fork was measured with.

    python3 bench/gfx906/ab.py run OUT_DIR [options]       measure the server as it runs now
    python3 bench/gfx906/ab.py compare DIR [DIR ...] [-o FILE]   one table of several runs, deltas against the first

`run` with no pass options is the **standard** suite (results say `"suite": "standard"`), the one to share and compare
(about 3 minutes on the MI50):
  1. decode, x 2 reps, greedy, thinking off: two code prompts (write a class with tests; find and fix the bugs in a
     function), two writing prompts (an essay, an explanation) and three specific-knowledge questions (chemistry,
     history, physics): tok/s overall and per kind, draft acceptance, answer hashes.
  1b. long context (long_ctx): a ~30K-token document (the engine's reference manual, then one of its source files)
     and a question that needs both, 350 tokens out. Rep 1 reads the document, rep 2 reuses it from the conversation
     cache: read tok/s, decode at ~30K, reuse. A per-run nonce keeps a disk session from an earlier run out of it, so
     its answer hash is reported on its own.
  2. prefill: three ~200-token prompts, ~4K and ~16K-token prompts (a nonce in front, so the conversation cache can't
     reuse them; 16 output tokens), then a follow-up turn on the 16K exchange (~1.8K new tokens over ~16K cached).
The long prompts are built from the engine's own docs and source at a fixed commit (CORPUS_REV), so every machine sends
the same tokens; results record the corpus hash.

Options (any of them makes the suite "custom"):
  --reps N            decode reps (default 2; 0 skips decode and long_ctx)   --no-long   skip long_ctx
  --pf-reps N         prefill reps (default 1)          --no-prefill     skip the prefill pass
  --pf-long N[,N..]   more fresh prefill prompts of N characters after pf16k (~3.34 chars/token: 350000 = ~105K)
  --long N            replace the decode prompts with one N-character prompt (decode at a long context)
  --live              the long-context agent shape: a fresh ~80K-token prompt, then follow-up turns of ~1.2K, ~5K and
                      ~7K new tokens that each generate 400 (~2.5 min a rep). --live-reps N, --live-ctx N_CHARS,
                      --live-turns N,N,..; --live-interleave sends a short other-conversation request before each turn
                      (the turn then pays the disk auto-session save/restore)
Server and machine (these don't change the suite):
  --url URL           server (default http://127.0.0.1:8080)      --model NAME   (default strata)
  --api-key KEY       or STRATA_API_KEY in the environment
  --log SRC           the engine log, for the per-window split: a file path, or docker:CONTAINER:PATH
                      (for the gfx906 image: docker:strata:/tmp/strata-<pack>.log). Without it only the API's numbers
                      are recorded. The split needs STRATA_DECODE_TIMING=1 in the server's env; per-GPU-stage times need
                      STRATA_VERIFY_PROFILE=1 (that one slows decode: don't compare its tok/s).
  --card N            the AMD card to sample (/sys/class/drm/cardN); default: the amdgpu card with the most VRAM
  --label TEXT        a name for the run in compare tables (default: OUT_DIR's name)
  --source DIR        build the prompts from DIR's files instead of CORPUS_REV (results say so; not comparable)

Per request it records the API timings, the answer's hash, the engine log lines it caused (parsed: ms/window split,
verify, GPU-reach wait, host per-layer, draft, tokens/window, CPU experts / VRAM hits / PCIe per layer-window, adapt
wait, expert hit %, prompt reused/read, session save/restore/park), the card's sensors every 0.5 s (sclk, mclk, power,
edge/junction/mem temperature, busy %) and the host CPU %. Outputs: OUT_DIR/results.json (system, server, every row
and the per-pass means) and OUT_DIR/summary.md.

Notes for comparing:
  - Restart the server right before a run. The adaptive expert cache carries over what the server served before, and
    that changes both speed and (by rounding) the answers: hashes only match between runs that start from a fresh start.
  - Even then the cache makes later answers drift, so compare the first pass's hashes; on the gfx906 box the third
    prompt's first-pass hash also flips now and then between runs of the same build.
  - Run the same build twice to see the noise (about +-1% decode tok/s here) before reading a small difference.
  - Nothing else should use the server during a run.
"""
import hashlib, json, os, platform, re, subprocess, sys, threading, time, urllib.request
from pathlib import Path
from statistics import mean

TOOL_VERSION = 2   # 2: the eight-prompt decode set with long_ctx (1: three prompts)
CORPUS_REV = "870aedd9e1ebe1c91f94da93b59af17729f6c5c0"   # upstream Strata v0.2.8: the prompts' source text
REPO = Path(__file__).resolve().parents[2]
CODE_FIX = '''def merge_intervals(intervals):
    """Merge overlapping [start, end] intervals."""
    intervals.sort()
    merged = []
    for start, end in intervals:
        if merged and start < merged[-1][1]:
            merged[-1][1] = end
        else:
            merged.append([start, end])
    return merged'''
# The decode prompts: (tag, prompt, max output tokens). Two code, two writing, three knowledge, and LONG_TAG, whose
# prompt is built per run (a ~30K-token document and a question about it).
PROMPTS = [
    ("code_write", "Write a complete Python implementation of a thread-safe LRU cache with TTL expiry, with docstrings and unit tests.", 800),
    ("code_fix", f"Find every bug in this Python function, explain each one, and give a corrected version:\n\n```python\n{CODE_FIX}\n```", 600),
    ("essay", "Write about 500 words on how a lighthouse works.", 600),
    ("explain", "Explain, in about 500 words, how a refrigerator moves heat out of its inside.", 600),
    ("know_chem", "Explain the Haber-Bosch process: the reaction, the catalyst, the usual temperature and pressure and why they are "
                  "chosen, and why the process matters for agriculture. About 250 words.", 350),
    ("know_hist", "What were the main causes, the key events and the outcome of the War of the Spanish Succession (1701-1714)? "
                  "About 250 words.", 350),
    ("know_phys", "What is the Chandrasekhar limit? Give its approximate value, the physics that sets it, who derived it and when, "
                  "and what happens to a white dwarf that exceeds it. About 250 words.", 350),
    ("long_ctx", None, 350),
]
LONG_TAG = "long_ctx"
GROUPS = {"code": ("code_write", "code_fix"), "writing": ("essay", "explain"), "knowledge": ("know_chem", "know_hist", "know_phys")}
# long_ctx: prose then code from CORPUS_REV (~110K characters, ~30K tokens), and a question that needs both
LONG_DOC = [("docs/DETAILS.md", 70000), ("src/core/conversation_file.cpp", 40000)]
LONG_Q = ("Above are an inference engine's reference manual and one of its source files. Using only them, explain in about "
          "250 words how a conversation is kept between requests: the in-RAM conversation cache, the session files on disk, "
          "the settings that control them, and what the source file does when it writes or reads a session file.")
PREFILL_CHARS = [("pf200a", 700), ("pf200b", 700), ("pf200c", 700), ("pf4k", 14000), ("pf16k", 56000)]
TURN_CHARS = 6300                  # the follow-up turn after pf16k: ~1.8K tokens of the text that follows pf16k's
LIVE_CTX = 267000                  # ~80K tokens
LIVE_TURNS = [4000, 16700, 23400]  # ~1.2K, ~5K, ~7K new tokens a turn
LIVE_GEN = 400
OPT = {"url": "http://127.0.0.1:8080", "model": "strata", "api_key": os.environ.get("STRATA_API_KEY")}


# ---------------------------------------------------------------- the prompts' text
class Corpus:
    """The .cpp then .cu files under src/, in path order, from CORPUS_REV (or a directory), concatenated on demand."""

    def __init__(self, source=None):
        self.source = source
        if source:
            root = Path(source)
            files = sorted(root.rglob("*.cpp")) + sorted(root.rglob("*.cu"))
            self._files = [(str(f.relative_to(root)), f) for f in files]
        else:
            names = subprocess.run(["git", "-C", str(REPO), "ls-tree", "-r", "--name-only", CORPUS_REV, "src"],
                                   capture_output=True, text=True)
            if names.returncode:
                sys.exit(f"this checkout lacks {CORPUS_REV[:8]} (upstream v0.2.8), the prompts' text: fetch the fork's "
                         "main with its history, or pass --source DIR (results are then not comparable)")
            ls = names.stdout.split()
            self._files = [(p[4:], p) for p in sorted(x for x in ls if x.endswith(".cpp")) + sorted(x for x in ls if x.endswith(".cu"))]
        self._cache, self._text = {}, ""

    def _read(self, f):
        if self.source:
            return f.read_text(errors="replace")
        return subprocess.run(["git", "-C", str(REPO), "show", f"{CORPUS_REV}:{f}"], capture_output=True).stdout.decode(errors="replace")

    def long_doc(self):
        """LONG_DOC's files from CORPUS_REV, each cut to its length (with --source: the same amount of its text)."""
        if self.source:
            return self.text(sum(n for _, n in LONG_DOC))
        return "".join(f"=== {p} ===\n{self._read(p)[:n]}\n" for p, n in LONG_DOC)

    def text(self, n_chars):
        while len(self._text) < n_chars and len(self._cache) < len(self._files):
            name, f = self._files[len(self._cache)]
            self._cache[name] = t = self._read(f)
            self._text += f"=== {name} ===\n{t}\n"
        return self._text[:n_chars]


CORPUS = None


# ---------------------------------------------------------------- the engine log
class Log:
    """New lines of the engine log since a mark: a file, or docker:CONTAINER:PATH (read with `docker exec`)."""

    def __init__(self, spec):
        self.spec = spec
        self.docker = spec.split(":", 2)[1:] if spec and spec.startswith("docker:") else None

    def _size(self):
        if self.docker:
            r = subprocess.run(["docker", "exec", self.docker[0], "stat", "-c", "%s", self.docker[1]], capture_output=True, text=True)
            return int(r.stdout.strip() or 0)
        return os.path.getsize(self.spec)

    def mark(self):
        return self._size() if self.spec else 0

    def since(self, off):
        if not self.spec:
            return []
        if self.docker:
            r = subprocess.run(["docker", "exec", self.docker[0], "tail", "-c", f"+{off + 1}", self.docker[1]], capture_output=True)
            data = r.stdout
        else:
            with open(self.spec, "rb") as f:
                f.seek(off)
                data = f.read()
        return data.decode(errors="replace").splitlines()


LOG = Log(None)


# ---------------------------------------------------------------- card / host sampling
def find_card(n=None):
    """(device dir, hwmon dir, facts) of an amdgpu card: cardN, or the one with the most VRAM (not an iGPU)."""
    best = None
    for d in sorted(Path("/sys/class/drm").glob("card*/device")):
        if n is not None and d.parent.name != f"card{n}":
            continue
        try:
            if d.joinpath("vendor").read_text().strip() != "0x1002":
                continue
            vram = int(d.joinpath("mem_info_vram_total").read_text())
            hw = next(d.glob("hwmon/hwmon*"))
        except Exception:
            continue
        if best is None or vram > best[2]["vram_bytes"]:
            facts = {"card": d.parent.name, "device_id": d.joinpath("device").read_text().strip(), "vram_bytes": vram}
            for k, f in (("link_speed", "current_link_speed"), ("link_width", "current_link_width"), ("product", "product_name")):
                try:
                    facts[k] = d.joinpath(f).read_text().strip()
                except Exception:
                    pass
            best = (d, hw, facts)
    return best or (None, None, {})


CARD, HWMON, CARD_FACTS = None, None, {}


def _rd(p, scale=1.0):
    try:
        return float(p.read_text()) * scale
    except Exception:
        return None


def _cpu_times():
    f = [float(x) for x in open("/proc/stat").readline().split()[1:]]
    return sum(f), f[3] + f[4]   # total, idle + iowait


class Sampler:
    """Card sensors + host CPU every `dt` s while a request runs."""

    def __init__(self, dt=0.5):
        self.dt, self.rows, self._stop = dt, [], threading.Event()

    def __enter__(self):
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *a):
        self._stop.set()
        self._t.join()

    def _run(self):
        last = _cpu_times()
        while not self._stop.wait(self.dt):
            cur = _cpu_times()
            dt, di = cur[0] - last[0], cur[1] - last[1]
            last = cur
            r = {"cpu_pct": 100.0 * (1 - di / dt) if dt > 0 else None}
            if HWMON:
                r.update(sclk_mhz=_rd(HWMON / "freq1_input", 1e-6), mclk_mhz=_rd(HWMON / "freq2_input", 1e-6),
                         power_w=_rd(HWMON / "power1_input", 1e-6) or _rd(HWMON / "power1_average", 1e-6),
                         t_edge=_rd(HWMON / "temp1_input", 1e-3), t_junction=_rd(HWMON / "temp2_input", 1e-3),
                         t_mem=_rd(HWMON / "temp3_input", 1e-3), busy_pct=_rd(CARD / "gpu_busy_percent"))
            self.rows.append(r)

    def summary(self):
        out = {}
        for k in (self.rows[0].keys() if self.rows else []):
            v = sorted(r[k] for r in self.rows if r.get(k) is not None)
            if v:
                out[f"hw_{k}_mean"] = round(mean(v), 2)
                out[f"hw_{k}_max"] = round(v[-1], 2)
                if k == "sclk_mhz":
                    out["hw_sclk_mhz_p10"] = round(v[len(v) // 10], 1)
        return out


# ---------------------------------------------------------------- log parsing
NUM = r"(-?\d+(?:\.\d+)?)"
DEC_RE = re.compile(   # engine before 0.1.52
    rf"strata decode timing: {NUM} windows, avg T {NUM}, {NUM} tokens/window, {NUM} ms/window = verify {NUM} "
    rf"\(GPU-reach wait {NUM} \+ per-layer host {NUM} \[plan {NUM} actq {NUM} jobs {NUM} CPU {NUM}\] \+ stage {NUM}\) "
    rf"\+ commit/emit {NUM} \+ draft {NUM}; per layer-window: CPU experts {NUM} \({NUM} entries\), VRAM hits {NUM}, "
    rf"PCIe {NUM}; adapt wait {NUM} join {NUM}, {NUM} swaps/window")
DEC_KEYS = ["windows", "avg_T", "tok_per_window", "ms_per_window", "verify_ms", "gpu_reach_wait_ms", "host_layer_ms",
            "host_plan_ms", "host_actq_ms", "host_jobs_ms", "host_cpu_ms", "stage_ms", "commit_emit_ms", "draft_ms",
            "cpu_experts_per_lw", "cpu_entries_per_lw", "vram_hits_per_lw", "pcie_per_lw", "adapt_wait_ms",
            "adapt_join_ms", "swaps_per_window"]
DEC_RE2 = re.compile(  # engine 0.1.52+: the adaptive waits are inside the sum, "+ other" closes it
    rf"strata decode timing: {NUM} windows, avg T {NUM}, {NUM} tokens/window, {NUM} ms/window = adapt wait {NUM} \+ verify {NUM} "
    rf"\(GPU-reach wait {NUM} \+ per-layer host {NUM} \[plan {NUM} actq {NUM} jobs {NUM} CPU {NUM}\] \+ stage {NUM}\) "
    rf"\+ commit/emit {NUM} \+ draft {NUM} \+ adapt join {NUM} \+ other {NUM}; per layer-window: CPU experts {NUM} \({NUM} entries\), "
    rf"VRAM hits {NUM}, PCIe {NUM}; {NUM} swaps/window")
DEC_KEYS2 = ["windows", "avg_T", "tok_per_window", "ms_per_window", "adapt_wait_ms", "verify_ms", "gpu_reach_wait_ms",
             "host_layer_ms", "host_plan_ms", "host_actq_ms", "host_jobs_ms", "host_cpu_ms", "stage_ms", "commit_emit_ms",
             "draft_ms", "adapt_join_ms", "other_ms", "cpu_experts_per_lw", "cpu_entries_per_lw", "vram_hits_per_lw",
             "pcie_per_lw", "swaps_per_window"]
PROMPT_RE = re.compile(rf"strata serve: prompt {NUM} tokens = {NUM} reused \+ {NUM}(?: of \d+)? read in {NUM} ms \({NUM} tok/s\), "
                       rf"{NUM} generated in {NUM} ms \({NUM} tok/s\)(?:, drafts accepted {NUM} of {NUM})?")
SAVE_RE = re.compile(rf"strata serve: session saved {NUM} tokens, .* in {NUM} ms")
RESTORE_RE = re.compile(rf"strata serve: session restored {NUM} tokens, .* in {NUM} ms")
PARK_RE = re.compile(rf"strata serve: conversation cache: parked {NUM} tokens in {NUM} ms")
PFT_RE = re.compile(rf"strata prefill timing: {NUM} tokens, GPU timeline {NUM} ms, wall {NUM} ms, host staging {NUM} ms:(.*)")
PFT_PHASE_RE = re.compile(r"([a-z][a-z0-9 +/&.-]*?) (\d+) \(\d+(?:\.\d+)?%\)")
HIT_RE = re.compile(rf"decode expert cache hit rate: {NUM}%")
PCIE_RE = re.compile(rf"; {NUM} more read by the GPU over PCIe")


def parse_stages(line):
    """`strata decode GPU stages (ms/window): GDN layers: a 0.1 b c 0.2 | QSA layers: ... | total X ms/window over N
    windows` -> {"gpu_gdn_a": 0.1, ..., "gpu_total_ms": X}."""
    out = {}
    for part in line.split("(ms/window):", 1)[1].split("|"):
        part = part.strip()
        if part.startswith("total"):
            m = re.match(rf"total {NUM} ms/window", part)
            if m:
                out["gpu_total_ms"] = float(m.group(1))
            continue
        pre = "gpu_gdn_" if part.startswith("GDN") else "gpu_qsa_" if part.startswith("QSA") else "gpu_"
        name = []
        for tok in part.split(":", 1)[-1].split():
            try:
                v = float(tok)
            except ValueError:
                name.append(tok)
                continue
            out[pre + re.sub(r"[^a-z0-9]+", "_", " ".join(name).lower()).strip("_")] = v
            name = []
    return out


def parse_request_log(lines):
    p = {}
    for l in lines:
        for rx, keys in ((DEC_RE, DEC_KEYS), (DEC_RE2, DEC_KEYS2)):
            m = rx.search(l)
            if m:
                p.update({"dec_" + k: float(v) for k, v in zip(keys, m.groups())})
        m = PROMPT_RE.search(l)
        if m:
            g = m.groups()
            p.update(prompt_tokens=float(g[0]), prompt_reused=float(g[1]), prompt_read=float(g[2]), prompt_ms=float(g[3]),
                     prefill_tps=float(g[4]), gen_ms=float(g[6]), gen_tps_engine=float(g[7]))
            if g[8] is not None:
                p.update(drafts_accepted=float(g[8]), drafts_offered=float(g[9]),
                         accept=float(g[8]) / float(g[9]) if float(g[9]) else None)
        for rx, k in ((SAVE_RE, "save"), (RESTORE_RE, "restore"), (PARK_RE, "park")):
            m = rx.search(l)
            if m:   # several in one request (a switch can save and park): sum the time
                p[f"{k}_tokens"] = float(m.group(1))
                p[f"{k}_ms"] = p.get(f"{k}_ms", 0.0) + float(m.group(2))
        m = PFT_RE.search(l)   # STRATA_PREFILL_TIMING=1: one line per prompt (or chunk): sum the phases
        if m:
            for k, v in (("tokens", m.group(1)), ("gpu_ms", m.group(2)), ("wall_ms", m.group(3)), ("host_staging_ms", m.group(4))):
                p[f"pft_{k}"] = p.get(f"pft_{k}", 0.0) + float(v)
            for name, v in PFT_PHASE_RE.findall(m.group(5)):
                k = "pft_" + re.sub(r"[^a-z0-9]+", "_", name.strip()).strip("_")
                p[k] = p.get(k, 0.0) + float(v)
        m = HIT_RE.search(l)
        if m:
            p["cache_hit_pct"] = float(m.group(1))
            m = PCIE_RE.search(l)
            p["pcie_read_experts"] = float(m.group(1)) if m else 0.0
        if "decode GPU stages (ms/window):" in l:
            p.update(parse_stages(l))
    return p


# ---------------------------------------------------------------- requests
def http(path, body=None, timeout=3600):
    h = {"Content-Type": "application/json"}
    if OPT["api_key"]:
        h["Authorization"] = f"Bearer {OPT['api_key']}"
    req = urllib.request.Request(OPT["url"] + path, json.dumps(body).encode() if body is not None else None, h)
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())


def chat(content, n, history=()):
    body = {"model": OPT["model"], "max_tokens": n, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False},
            "messages": [*history, {"role": "user", "content": content}]}
    off = LOG.mark()
    with Sampler() as s:
        t0 = time.time()
        r = http("/v1/chat/completions", body)
        wall = time.time() - t0
    if LOG.spec:
        time.sleep(0.5)   # the per-request log lines are written right after the reply
    lines = LOG.since(off)
    t = r.get("timings") or {}
    text = r["choices"][0]["message"].get("content") or ""
    row = {"decode_tps": t.get("predicted_per_second"), "decoded": t.get("predicted_n"),
           "api_prompt_n": t.get("prompt_n"), "api_prompt_tps": t.get("prompt_per_second"),
           "api_prompt_ms": t.get("prompt_ms"), "api_cache_n": t.get("cache_n"),
           "api_drafts": t.get("draft_n"), "api_drafts_accepted": t.get("draft_n_accepted"),
           "wall_s": round(wall, 3), "sha": hashlib.sha1(text.encode()).hexdigest()[:12]}
    if row["api_drafts"]:
        row["api_accept"] = round(row["api_drafts_accepted"] / row["api_drafts"], 4)
    row.update(parse_request_log(lines))
    gen_ms = row.get("gen_ms") if isinstance(row.get("gen_ms"), float) else t.get("predicted_ms")
    prompt_ms = row.get("prompt_ms") if isinstance(row.get("prompt_ms"), float) else t.get("prompt_ms")
    if isinstance(gen_ms, (int, float)):
        row["startup_ms"] = round(wall * 1000 - gen_ms, 1)
        if isinstance(prompt_ms, (int, float)):
            row["overhead_ms"] = round(wall * 1000 - prompt_ms - gen_ms, 1)
    row.update(s.summary())
    row["_log"] = [l for l in lines if l.startswith("strata ")][:40]
    row["_text"] = text   # dropped before writing (the follow-up turns need the reply)
    return row


def tok_tag(n_chars, pre):   # ~3.34 chars/token -> "<pre><K tokens>k"
    return f"{pre}{max(1, round(n_chars / 3340))}k"


def run_live(nonce, rep, rows, interleave=False):
    body = f"[live {nonce} {rep}]\n" + CORPUS.text(LIVE_CTX) + "\n\nIn one sentence: what language is the code above?"
    row = chat(body, 16)
    rows.append({"pass": "live", "rep": rep, "prompt": tok_tag(LIVE_CTX, "lv"), **row})
    hist = [{"role": "user", "content": body}, {"role": "assistant", "content": row["_text"]}]
    off = LIVE_CTX
    for i, nc in enumerate(LIVE_TURNS):
        turn = (f"[live turn {nonce} {rep} {i}]\nMore of the same source:\n" + CORPUS.text(off + nc)[off:] +
                "\n\nSummarize, in about 300 words, what this new part does and how it relates to the code before it.")
        off += nc
        if interleave:   # another conversation takes the slot: the long one is saved now and restored for the turn
            x = chat(f"[live other {nonce} {rep} {i}] Reply with one word: ok.", 4)
            rows.append({"pass": "live", "rep": rep, "prompt": "lvx", **x})
        row = chat(turn, LIVE_GEN, hist)
        rows.append({"pass": "live", "rep": rep, "prompt": tok_tag(nc, "lvt"), **row})
        hist += [{"role": "user", "content": turn}, {"role": "assistant", "content": row["_text"]}]


def measure(reps, do_prefill, nonce, pf_reps=1, live_reps=0, interleave=False, do_long=True):
    rows = []
    long_body = None
    for rep in range(reps):
        for tag, p, n in PROMPTS:
            if tag == LONG_TAG:
                if not do_long:
                    continue
                # the run's nonce: rep 0 reads the document even if an earlier run left it in a disk session; later reps
                # reuse it from the conversation cache (so its answer hash can't match across runs)
                long_body = long_body or f"[ab {tag} {nonce}]\n{CORPUS.long_doc()}\n\n{LONG_Q}"
                row = chat(long_body, n)
                rows.append({"pass": "long", "rep": rep, "prompt": tag, **row})
                print(f"  long {tag} rep {rep}: {row.get('api_prompt_n')} tokens read at {row.get('api_prompt_tps')} tok/s, "
                      f"decode {row['decode_tps']} tok/s", flush=True)
                continue
            rows.append({"pass": "decode", "rep": rep, "prompt": tag, **chat(f"[ab {tag}] {p}", n)})
            print(f"  decode {tag} rep {rep}: {rows[-1]['decode_tps']} tok/s", flush=True)
    if do_prefill:
        for pr in range(pf_reps):
            for tag, nc in PREFILL_CHARS:
                body = f"[run {nonce} {tag} {pr}]\n" + CORPUS.text(nc) + "\n\nIn one sentence: what language is the code above?"
                row = chat(body, 16)
                rows.append({"pass": "prefill", "rep": pr, "prompt": tag, **row})
                print(f"  prefill {tag} rep {pr}: {row.get('prefill_tps') or row.get('api_prompt_tps')} tok/s", flush=True)
                if tag == "pf16k":
                    turn = (f"[turn {nonce} {pr}]\nMore of the same source:\n" + CORPUS.text(nc + TURN_CHARS)[nc:] +
                            "\n\nIn one sentence: does this part use the same language?")
                    hist = [{"role": "user", "content": body}, {"role": "assistant", "content": row["_text"]}]
                    rows.append({"pass": "prefill", "rep": pr, "prompt": "pfturn", **chat(turn, 16, hist)})
    for lr in range(live_reps):
        run_live(nonce, lr, rows, interleave)
    for r in rows:
        r.pop("_text", None)
    return rows


# ---------------------------------------------------------------- aggregation and reports
def _num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def agg(rows, engine=None):
    """Mean of every numeric field per pass -> flat dict (the decode first pass is also kept on its own: the adaptive
    cache makes later reps drift)."""
    out = {}
    for ps in ("decode", "long", "prefill", "live"):
        rs = [r for r in rows if r["pass"] == ps]
        if not rs:
            continue
        for k in sorted({k for r in rs for k, v in r.items() if _num(v) and k != "rep"}):
            v = [r[k] for r in rs if _num(r.get(k))]
            out[f"{ps}.{k}"] = round(mean(v), 4)
        if ps == "decode":
            f = [r for r in rs if r["rep"] == 0]
            out["decode.first_tps"] = round(mean(r["decode_tps"] for r in f), 3)
            out["decode.first_shas"] = " ".join(r["sha"] for r in f)
            for g, tags in GROUPS.items():   # decode speed by kind of prompt (code drafts better than prose)
                v = [r["decode_tps"] for r in rs if r["prompt"] in tags and _num(r.get("decode_tps"))]
                if v:
                    out[f"decode.{g}_tps"] = round(mean(v), 2)
        if ps == "long":   # rep 0 reads the document, later reps reuse it
            first, later = [r for r in rs if r["rep"] == 0], [r for r in rs if r["rep"] > 0]
            for k, f, src in (("read_tokens", "api_prompt_n", first), ("read_tps", "api_prompt_tps", first),
                              ("read_ms", "api_prompt_ms", first), ("reused_tokens", "api_cache_n", later),
                              ("reuse_read_tokens", "api_prompt_n", later), ("startup_ms", "startup_ms", later)):
                v = [r[f] for r in src if _num(r.get(f))]
                if v:
                    out[f"long.{k}"] = round(mean(v), 2)
            out["long.first_sha"] = first[0]["sha"] if first else None
        if ps in ("prefill", "live"):
            fields = [("tps", "prefill_tps"), ("tokens", "prompt_tokens"), ("ms", "prompt_ms"), ("read", "prompt_read"),
                      ("reused", "prompt_reused"), ("overhead_ms", "overhead_ms"), ("api_tps", "api_prompt_tps")]
            if ps == "live":
                fields += [("dec_tps", "decode_tps"), ("ms_win", "dec_ms_per_window"), ("accept", "accept"),
                           ("startup_ms", "startup_ms"), ("save_ms", "save_ms"), ("restore_ms", "restore_ms"), ("park_ms", "park_ms")]
            for tag in dict.fromkeys(r["prompt"] for r in rs):   # mean over reps
                for k, f in fields:
                    v = [r[f] for r in rs if r["prompt"] == tag and _num(r.get(f))]
                    if v:
                        out[f"{ps}.{tag}_{k}"] = round(mean(v), 3)
        if ps == "prefill":
            short = [r for r in rs if r["prompt"].startswith("pf200")]
            for k, f in (("ms", "prompt_ms"), ("tps", "prefill_tps"), ("api_tps", "api_prompt_tps")):
                v = [r[f] for r in short if _num(r.get(f))]
                if v:
                    out[f"prefill.pf200_{k}"] = round(mean(v), 2)
        if ps == "live":
            turns = [r for r in rs if r["prompt"].startswith("lvt")]
            for k in ("decode_tps", "accept", "api_accept", "dec_tok_per_window", "dec_ms_per_window", "dec_verify_ms",
                      "dec_gpu_reach_wait_ms", "dec_draft_ms", "dec_adapt_wait_ms", "dec_cpu_experts_per_lw"):
                v = [r[k] for r in turns if _num(r.get(k))]
                if v:
                    out[f"live.turns_{k}"] = round(mean(v), 3)
            if turns:
                out["live.turn_shas"] = " ".join(r["sha"] for r in turns if r["rep"] == 0)
    for k in ("vram_free_mib", "expert_slots", "expert_cache_mib", "arena_mib", "pcie_frac"):
        if engine and k in engine:
            out[f"engine.{k}"] = engine[k]
    return out


SUMMARY_KEYS = [
    ("decode.first_tps", "decode tok/s (1st pass)"), ("decode.decode_tps", "decode tok/s (all)"),
    ("decode.api_accept", "draft accept"), ("decode.dec_tok_per_window", "tokens/window"),
    ("decode.dec_ms_per_window", "ms/window"), ("decode.dec_verify_ms", "verify ms"),
    ("decode.dec_gpu_reach_wait_ms", "GPU-reach wait ms"), ("decode.dec_host_layer_ms", "host per-layer ms"),
    ("decode.dec_host_cpu_ms", "host CPU-experts ms"), ("decode.dec_draft_ms", "draft ms"),
    ("decode.dec_commit_emit_ms", "commit/emit ms"), ("decode.dec_adapt_wait_ms", "adapt wait ms"),
    ("decode.cache_hit_pct", "expert hit %"), ("decode.dec_pcie_per_lw", "PCIe experts/layer-win"),
    ("decode.dec_cpu_experts_per_lw", "CPU experts/layer-win"), ("decode.hw_power_w_mean", "power W"),
    ("decode.hw_sclk_mhz_mean", "sclk MHz"), ("decode.hw_t_junction_max", "junction max C"),
    ("decode.hw_cpu_pct_mean", "host CPU %"),
    ("decode.code_tps", "decode tok/s: code"), ("decode.writing_tps", "decode tok/s: writing"),
    ("decode.knowledge_tps", "decode tok/s: knowledge"),
    ("long.read_tokens", "long: document tokens"), ("long.read_tps", "long: read tok/s"),
    ("long.decode_tps", "long: decode tok/s at ~30K"), ("long.api_accept", "long: draft accept"),
    ("long.dec_ms_per_window", "long: ms/window"), ("long.reused_tokens", "long: tokens reused (rep 2)"),
    ("long.startup_ms", "long: time to first token, reused (ms)"),
    ("prefill.pf200_api_tps", "prefill ~200 tok/s"), ("prefill.pf4k_api_tps", "prefill ~4K tok/s"),
    ("prefill.pf16k_api_tps", "prefill ~16K tok/s"), ("prefill.pfturn_api_tps", "follow-up turn tok/s"),
    ("prefill.pfturn_ms", "follow-up turn read ms"), ("prefill.pfturn_reused", "follow-up turn tokens reused"),
    ("prefill.pfturn_overhead_ms", "follow-up turn overhead ms"), ("prefill.hw_power_w_mean", "prefill power W"),
    ("live.turns_decode_tps", "live: decode tok/s at ~80K"), ("live.turns_api_accept", "live: draft accept"),
    ("live.turns_dec_ms_per_window", "live: ms/window"), ("live.turns_dec_gpu_reach_wait_ms", "live: GPU-reach wait ms"),
    ("live.turns_dec_draft_ms", "live: draft ms"), ("engine.vram_free_mib", "VRAM free MiB"), ("engine.expert_slots", "expert slots"),
]


def table(runs):
    """Markdown tables of the runs' means, with % deltas against the first."""
    labels = [r["label"] for r in runs]
    aggs = [r["agg"] for r in runs]
    ref = aggs[0]
    keys = sorted({k for a in aggs for k in a})

    def cell(a, k):
        v, r = a.get(k), ref.get(k)
        if not _num(v):
            return str(v) if v not in (None, "") else "–"
        d = f" ({(v - r) / r * 100:+.1f}%)" if _num(r) and r and a is not ref else ""
        return f"{v:.2f}{d}" if isinstance(v, float) else f"{v}{d}"

    head = ["| metric | " + " | ".join(labels) + " |", "|---|" + "---|" * len(runs)]
    extra = [(k, k) for k in keys if (k.startswith("prefill.pf") and k.endswith("_api_tps") and not k.startswith("prefill.pf200")
                                      and not any(k == s for s, _ in SUMMARY_KEYS))
             or (k.startswith("live.lv") and k.endswith(("_api_tps", "_dec_tps", "_startup_ms", "_save_ms", "_restore_ms")))]
    gpu = [(k, k.split(".", 1)[1]) for k in keys if ".gpu_" in k]
    out = []
    for title, ks in (("Key metrics", [(k, n) for k, n in SUMMARY_KEYS if k in keys] + extra),
                      ("GPU stages (needs STRATA_VERIFY_PROFILE=1, ms/window)", gpu)):
        if ks:
            out += [f"## {title}", ""] + head + [f"| {n} | " + " | ".join(cell(a, k) for a in aggs) + " |" for k, n in ks] + [""]
    out += ["## Output (decode first pass)", ""]
    if len(runs) == 1:
        return out + [f"Answer hashes: {ref.get('decode.first_shas')}" +
                      (f" (long_ctx: {ref['long.first_sha']}, not comparable across runs)" if ref.get("long.first_sha") else "")]
    for lab, a in zip(labels, aggs):
        same = a.get("decode.first_shas") == ref.get("decode.first_shas")
        out.append(f"- `{lab}`: {'same as' if same else 'DIFFERENT from'} `{labels[0]}` ({a.get('decode.first_shas')})")
    return out


def system_facts():
    facts = {"host": platform.node(), "kernel": platform.release(), "python": platform.python_version()}
    try:
        facts["cpu"] = next(l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name"))
        facts["cpu_threads"] = os.cpu_count()
        facts["ram_kib"] = int(next(l.split()[1] for l in open("/proc/meminfo") if l.startswith("MemTotal")))
    except Exception:
        pass
    facts["gpu"] = CARD_FACTS
    return facts


# ---------------------------------------------------------------- commands
def arg(name, default=None, cast=str):
    return cast(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else default


def cmd_run():
    global CORPUS, LOG, CARD, HWMON, CARD_FACTS, PROMPTS, LIVE_CTX, LIVE_TURNS
    out = Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    OPT.update(url=arg("--url", OPT["url"]).rstrip("/"), model=arg("--model", OPT["model"]), api_key=arg("--api-key", OPT["api_key"]))
    LOG = Log(arg("--log"))
    CARD, HWMON, CARD_FACTS = find_card(arg("--card", None, int))
    CORPUS = Corpus(arg("--source"))
    suite_opts = [a for a in ("--reps", "--pf-reps", "--no-prefill", "--no-long", "--pf-long", "--long", "--live", "--live-reps",
                              "--live-ctx", "--live-turns", "--live-interleave", "--source") if a in sys.argv]
    if "--long" in sys.argv:
        PROMPTS = [("long", CORPUS.text(arg("--long", cast=int)) + "\n\nAbove is source code from an inference engine. "
                    "Summarize, in about 400 words, what the files do.", 400)]
    for n in (arg("--pf-long", "") or "").split(",") if "--pf-long" in sys.argv else []:
        PREFILL_CHARS.append((tok_tag(int(n), "pf"), int(n)))
    LIVE_CTX = arg("--live-ctx", LIVE_CTX, int)
    if "--live-turns" in sys.argv:
        LIVE_TURNS = [int(n) for n in arg("--live-turns").split(",")]
    live_reps = arg("--live-reps", 1 if "--live" in sys.argv else 0, int)

    try:
        server = {"metrics": http("/metrics", timeout=10)}
    except Exception as e:
        sys.exit(f"no server at {OPT['url']}: {e}")
    engine = server["metrics"].get("engine", {}) if isinstance(server["metrics"], dict) else {}
    print(f"server {OPT['url']}, log {LOG.spec or 'none (API numbers only)'}, card {CARD_FACTS.get('card', 'none')}, "
          f"suite {'custom ' + ' '.join(suite_opts) if suite_opts else 'standard'}", flush=True)
    t0 = time.time()
    rows = measure(arg("--reps", 2, int), "--no-prefill" not in sys.argv, int(t0), arg("--pf-reps", 1, int),
                   live_reps, "--live-interleave" in sys.argv, "--no-long" not in sys.argv)
    res = {
        "tool": "bench/gfx906/ab.py", "tool_version": TOOL_VERSION, "label": arg("--label", out.name),
        "suite": "custom" if suite_opts else "standard", "options": sys.argv[3:],
        "corpus": {"rev": None if CORPUS.source else CORPUS_REV, "source": CORPUS.source,
                   "sha1_first_400k": hashlib.sha1(CORPUS.text(400000).encode()).hexdigest()},
        "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t0)), "seconds": round(time.time() - t0, 1),
        "system": system_facts(), "server": {"url": OPT["url"], "engine": engine}, "rows": rows,
    }
    res["agg"] = agg(rows, engine)
    (out / "results.json").write_text(json.dumps(res, indent=1) + "\n")
    lines = [f"# {res['label']}", "", f"{res['suite']} suite, {res['started']}, {res['seconds']} s. "
             f"GPU {CARD_FACTS.get('product') or CARD_FACTS.get('device_id', '?')}, CPU {res['system'].get('cpu', '?')}.", ""]
    (out / "summary.md").write_text("\n".join(lines + table([res])) + "\n")
    a = res["agg"]
    print(f"decode {a.get('decode.first_tps')} tok/s first pass ({a.get('decode.first_shas')}), "
          f"pf16k {a.get('prefill.pf16k_api_tps')} tok/s -> {out}/summary.md", flush=True)


# options that name the run or say where to measure, not what: runs that differ only in these are comparable
NOT_MEASURED = {"--label", "--log", "--url", "--model", "--api-key", "--card"}


def measured_options(opts):
    """A run's options without NOT_MEASURED and their values."""
    out, skip = [], False
    for o in opts:
        if skip:
            skip = False
        elif o in NOT_MEASURED:
            skip = True
        else:
            out.append(o)
    return out


def cmd_compare():
    dirs = [a for a in sys.argv[2:] if not a.startswith("-") and a != arg("-o")]
    runs = [json.loads((Path(d) / "results.json").read_text()) for d in dirs]
    warn = []
    if len({r.get("tool_version") for r in runs}) > 1:
        warn.append("- The runs come from different harness versions (different prompts): compare only the rows they share.")
    if len({json.dumps(r["corpus"], sort_keys=True) for r in runs}) > 1:
        warn.append("- The runs used different prompt text: their prefill numbers aren't comparable.")
    if len({r["suite"] + json.dumps(measured_options(r["options"])) for r in runs}) > 1:
        warn.append("- The runs used different options: " + "; ".join(
            f"`{r['label']}` {r['suite']} {' '.join(measured_options(r['options']))}" for r in runs))
    lines = ["# Comparison", "", f"Reference: `{runs[0]['label']}`. Cells: value (Δ% vs the reference).", ""]
    lines += [f"- `{r['label']}`: {r['started']}, {r['suite']}, GPU {r['system']['gpu'].get('product') or r['system']['gpu'].get('device_id', '?')}, "
              f"CPU {r['system'].get('cpu', '?')}" for r in runs]
    lines += ([""] + warn if warn else []) + [""] + table(runs)
    text = "\n".join(lines) + "\n"
    if arg("-o"):
        Path(arg("-o")).write_text(text)
    print(text)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "run" and len(sys.argv) > 2:
        cmd_run()
    elif cmd == "compare" and len(sys.argv) > 2:
        cmd_compare()
    else:
        sys.exit(__doc__)
