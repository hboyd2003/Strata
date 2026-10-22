#!/usr/bin/env python3
"""Batching test: the same prompts decoded alone (GEN, the usual path with drafts) and together in the batch
windows (BGEN), greedy.  Every slot's tokens must equal its solo tokens; prints the aggregate decode rate.

  python3 tools/batch_test.py --exe engine/strata --config strata-<model>.json --batch 4 --n 4 \
      --extra "--layer-split 12,24,36 --trim-stage-weights --pcie-frac 0 --adapt-every 1000000"

With --batch-mtp in --extra, --acceptance also compares the drafts themselves: what share of the proposals a
slot was offered it took, against the same decision made alone (see report_acceptance).
"""
import argparse, json, os, re, subprocess, sys, tempfile, threading, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import strata_tokenizer as ST  # noqa: E402

QUESTIONS = [
    "Explique en detail le fonctionnement d'un B-tree.",
    "Write a Python function that merges overlapping intervals, then explain it.",
    "Compare TCP and QUIC: handshake, congestion control, multiplexing.",
    "Raconte l'histoire du calcul de pi, d'Archimede a nos jours.",
    "What are the main causes of the French Revolution?",
    "Explain how a transformer's attention works, step by step.",
    "Donne une recette de tarte aux pommes, etape par etape.",
    "Describe the life cycle of a star like the Sun.",
]


# --- MTP acceptance, read out of the engine's own log lines (all three formats quoted in report_acceptance) ---

# generate.cpp:10283, once per drain of the slots: "strata batch: MTP proposals accepted 30 of 42 (71.4%)"
BATCH_ACCEPT_RE = re.compile(r"strata batch: MTP proposals accepted (\d+) of (\d+) \(([\d.]+)%\)")
# generate.cpp:13354, once per GEN-style request with STRATA_SPEC_DEPTH=1 (counted at :12894):
# "strata serve: spec depth: accepted/tested 3/5 2/5 1/5 0/5 0/5" - one accepted/tested pair per draft depth,
# depth 0 = the window's FIRST proposal (the same decision a batch slot makes), and the first window of a
# request is not counted (it has no draft yet, :12893).
SPEC_DEPTH_RE = re.compile(r"strata serve: spec depth: accepted/tested((?: \d+/\d+)+)")
# The engine says so when it leaves --batch-mtp off (generate.cpp:4543 and :10296), which is why the tally is empty.
MTP_OFF_RE = re.compile(r"--batch-mtp is off|run no slot|drafts do not run there")


def parse_batch_acceptance(text):
    """Sum the batch tally lines: (accepted, offered, lines).  A slot-window verifies its own token plus one MTP
    proposal and keeps the proposal only when the target picked it (generate.cpp:10154), so `offered` is the number
    of slot-windows and the emitted tokens are offered + accepted.  The engine sums every live slot into one
    counter per drain (bt_accepted, generate.cpp:10168), so the batch side has no per-slot split to read.
    A run whose slots drain and refill logs one line per drain, hence the sum."""
    acc = off = n = 0
    for m in BATCH_ACCEPT_RE.finditer(text):
        acc += int(m.group(1))
        off += int(m.group(2))
        n += 1
    return acc, off, n


def parse_spec_depth(text):
    """The solo windows' drafts by depth: one list of (accepted, tested) per line, in request order."""
    out = []
    for m in SPEC_DEPTH_RE.finditer(text):
        out.append([tuple(int(x) for x in p.split("/")) for p in m.group(1).split()])
    return out


def parse_done(line):
    """`DONE <generated> <prompt> <prompt ms> <decode ms> <finish> <drafts accepted> <drafts offered> ...`
    (generate.cpp:13171): the solo request's own tally, every depth together.  None if the line is short."""
    p = line.split()
    if len(p) < 8:
        return None
    try:
        return int(p[6]), int(p[7])
    except ValueError:
        return None


def pct(a, t):
    return 100.0 * a / t if t else 0.0


def report_acceptance(log, n_solo, solo_done, bt_tokens, tol, extra):
    """Print MTP acceptance, batched against solo, and return whether it is within tolerance.

    A batch slot verifies ONE proposal per window (generate.cpp:10154, :10168); a solo window verifies --spec - 1
    of them.  So the two sides' raw rates are not the same measure - the solo window's FIRST proposal is: same
    drafter, same single decision, and that is the depth-0 pair of the spec-depth ladder (generate.cpp:12894).
    The ladder's deeper pairs and the DONE tally are printed for context, not compared.

    The solo lines are the first n_solo 'spec depth' lines: the GEN requests run first, and each BGEN admission
    reads its prompt and decodes its first token alone, which logs one more line per slot before BADM.

    Fails when the difference is larger than --acceptance-tol percentage points AND larger than the band these
    counts could produce by chance (2 sigma of the pooled two-proportion difference, which at a ~69% rate is
    +/-15 pp on 90-vs-60 windows, +/-11 pp on 150-vs-110 and +/-7.5 pp on 300-vs-300).  A short run cannot prove
    a tight tolerance: the printed band says what the run can say, and --max-new is how you sharpen it.
    """
    b_acc, b_off, b_lines = parse_batch_acceptance(log)
    depths = parse_spec_depth(log)
    solo = depths[:n_solo]
    print(f"acceptance: {len(depths)} 'spec depth' line(s) (the first {n_solo} are the solo GEN requests, the rest "
          f"belong to the BGEN prompt-read legs), {b_lines} batch tally line(s)", flush=True)
    if n_solo and len(solo) < n_solo:
        print(f"  note: {len(solo)} of {n_solo} solo requests logged a spec-depth line "
              f"(is STRATA_SPEC_DEPTH reaching the engine?)", flush=True)
    d0 = (sum(p[0][0] for p in solo), sum(p[0][1] for p in solo)) if solo else (0, 0)
    if solo:
        by_depth = {d: (sum(p[d][0] for p in solo), sum(p[d][1] for p in solo))
                    for d in range(1, min(len(p) for p in solo))}
        ladder = "  ".join(f"depth {d}: {a}/{t}" for d, (a, t) in sorted(by_depth.items()) if t)
        print(f"  solo windows: first proposal {d0[0]} of {d0[1]} = {pct(*d0):.1f}%"
              + (f"   ({ladder})" if ladder else ""), flush=True)
        if len(solo) == n_solo:
            print("  solo per prompt (first proposal): " + " ".join(f"{p[0][0]}/{p[0][1]}" for p in solo), flush=True)
    tallied = [x for x in solo_done if x]
    if tallied:
        s_a, s_o = sum(x[0] for x in tallied), sum(x[1] for x in tallied)
        print(f"  solo all drafts (DONE line): {s_a} of {s_o} = {pct(s_a, s_o):.1f}% - context only: a solo window "
              f"offers --spec - 1 proposals and the deeper ones are rejected more often", flush=True)
    if b_lines == 0:
        for line in log.splitlines():
            if MTP_OFF_RE.search(line):
                print(f"  engine: {line.strip()}", flush=True)
        if "--batch-mtp" in extra or os.environ.get("STRATA_BATCH_MTP", "") not in ("", "0"):
            print("  batch: --batch-mtp was asked for and no tally line was logged: FAIL", flush=True)
            return False
        print("  batch: no tally line - plain --batch verifies no proposals at all (a lane of a plain --batch "
              "loses MTP, BATCHING.md), so there is nothing to compare; put --batch-mtp in --extra", flush=True)
        return True
    print(f"  batch slots: {b_acc} of {b_off} proposals accepted = {pct(b_acc, b_off):.1f}% "
          f"(aggregate over slots - the engine sums them, generate.cpp:10168)", flush=True)
    if bt_tokens:
        want = b_off + b_acc
        print(f"  cross-check: {bt_tokens} BT tokens vs {b_off} windows + {b_acc} accepted = {want}"
              + ("" if bt_tokens == want else f"  (off by {bt_tokens - want}: a slot stopped inside a window)"),
              flush=True)
    if not n_solo:
        print("  no solo side to compare against (--skip-solo): the batch rate above stands alone", flush=True)
        return True
    if d0[1] == 0 or b_off == 0:
        print(f"  cannot compare: solo windows with a draft {d0[1]}, batch slot-windows {b_off} "
              f"(is --mtp and --spec T>=2 in the config args?)", flush=True)
        return False
    diff = pct(b_acc, b_off) - pct(*d0)
    p = (b_acc + d0[0]) / (b_off + d0[1])
    band = 200.0 * (p * (1.0 - p) * (1.0 / b_off + 1.0 / d0[1])) ** 0.5
    ok = abs(diff) <= max(tol, band)
    print(f"  first proposal accepted: solo {pct(*d0):.1f}% vs batch {pct(b_acc, b_off):.1f}% = {diff:+.1f} pp "
          f"(tolerance +/-{tol:.1f} pp, counting-noise band +/-{band:.1f} pp): "
          f"{'ACCEPTANCE OK' if ok else 'ACCEPTANCE OFF'}", flush=True)
    return ok


def parse_check():
    """--parse-check: the readers and the comparison on a handcrafted log, no engine and no GPU.
    Every fake line below is copied in shape from what the engine prints (file:line in the regex comments);
    the numbers are consistent the way the engine's are: sum_d spec_depth_ok[d] == DONE accepted,
    and BT tokens == batch offered + accepted.  Run this to prove the parsing without a card."""
    # the engine's stderr, as batch_test.py's log file holds it
    log = """strata generate: --batch-mtp: 4 slot drafters on CUDA1 (884 MiB of private K/V each)
strata serve: prompt 314 tokens = 313 reused + 1 read in 41 ms (24.4 tok/s), 64 generated in 1180 ms (54.2 tok/s), drafts accepted 41 of 96, 1 checkpoints
strata serve: spec depth: accepted/tested 22/31 12/31 7/31 0/0 0/0
strata serve: prompt 96 tokens = 95 reused + 1 read in 12 ms (83.3 tok/s), 64 generated in 1201 ms (53.3 tok/s), drafts accepted 37 of 93, 1 checkpoints
strata serve: spec depth: accepted/tested 20/30 11/30 6/30 0/0 0/0
strata serve: prompt 314 tokens = 0 reused + 314 read in 2210 ms (142.1 tok/s), 1 generated in 19 ms (52.6 tok/s), drafts accepted 0 of 3, 0 checkpoints
strata serve: spec depth: accepted/tested 0/0 0/0 0/0 0/0 0/0
strata batch: 44 windows, avg 6.93 rows, 128.40 ms/window = run 118.20 (CUDA0 GPU-reach wait 37.10 + CPU experts 30.20) + commit 6.10 + emit 2.40 + adapt wait 0.00
strata batch: MTP proposals accepted 43 of 62 (69.4%)
strata batch: MTP proposals accepted 22 of 31 (71.0%)
"""
    # the engine's stdout, which the harness reads as it goes (a DONE line per solo request)
    done = ["DONE 64 314 41.2 1180.3 length 41 96 313 1200 1500 0 0 0.0 1 0",
            "DONE 64 96 12.4 1201.0 length 37 93 95 900 1100 0 0 0.0 1 0"]
    checks = []

    def check(name, got, want):
        checks.append((name, got == want, got, want))

    check("batch tally summed over the two drains", parse_batch_acceptance(log), (43 + 22, 62 + 31, 2))
    check("spec depth ladders", parse_spec_depth(log),
          [[(22, 31), (12, 31), (7, 31), (0, 0), (0, 0)], [(20, 30), (11, 30), (6, 30), (0, 0), (0, 0)],
           [(0, 0), (0, 0), (0, 0), (0, 0), (0, 0)]])
    check("DONE tally", [parse_done(x) for x in done], [(41, 96), (37, 93)])
    check("DONE too short reads as nothing", parse_done("DONE 1 314 2210.0 19.0 length"), None)
    check("MTP-off warning matches", bool(MTP_OFF_RE.search(
        "strata generate: WARNING: --batch-mtp is off: it needs --mtp")), True)
    check("pipelined-groups warning matches", bool(MTP_OFF_RE.search(
        "--batch-mtp's per-slot MTP drafts do not run there (their drafters still hold VRAM")), True)

    # the solo side is the first n_solo ladders: the third line (all 0/0) is a BGEN prompt-read leg and must not be in it
    print("--- report on the fake log: 2 solo requests, batch 65 of 93, 158 BT tokens (= 93 offered + 65 kept) ---")
    ok = report_acceptance(log, 2, [parse_done(x) for x in done], 93 + 65, 5.0, "--batch-mtp")
    check("solo 68.9% vs batch 69.9% passes", ok, True)
    print("--- same log with the slots taking far fewer proposals (12 of 62 + 22 of 31 = 36.6%) ---")
    bad = report_acceptance(log.replace("accepted 43 of 62", "accepted 12 of 62"), 2,
                            [parse_done(x) for x in done], 93 + 34, 5.0, "--batch-mtp")
    check("solo 68.9% vs batch 36.6% fails", bad, False)
    print("--- 158 BT tokens against a tally that does not add up (offered + kept != tokens) ---")
    report_acceptance(log, 2, [parse_done(x) for x in done], 150, 5.0, "--batch-mtp")
    print("--- --skip-solo: the batch rate stands alone ---")
    check("skip-solo is not a failure", report_acceptance(log, 0, [], 93 + 65, 5.0, "--batch-mtp"), True)
    print("--- no tally line: asked for and not asked for ---")
    check("tally asked for but absent fails", report_acceptance("nothing here\n", 2, [parse_done(done[0])],
                                                                0, 5.0, "--batch-mtp"), False)
    check("tally absent and not asked for is a note", report_acceptance(
        "strata generate: WARNING: --batch-mtp is off: it needs --mtp\n", 2, [parse_done(done[0])], 0, 5.0, ""), True)
    failed = [c for c in checks if not c[1]]
    print()
    for name, passed, got, want in checks:
        print(f"  {'ok  ' if passed else 'FAIL'} {name}: {got}" + ("" if passed else f" != {want}"))
    print(f"parse-check: {len(checks)} checks, {len(failed)} failed")
    return 1 if failed else 0


def tokenizer(path):
    t = Path(path)
    vocab = json.loads((t / "vocab.json").read_text(encoding="utf-8"))
    tokens = [None] * len(vocab)
    for s, i in vocab.items():
        tokens[i] = s
    merges = (t / "merges.txt").read_text(encoding="utf-8").split("\n")
    types = json.loads((t / "token_type.json").read_text())
    return ST.Tokenizer(tokens, merges, types)


class Engine:
    def __init__(self, exe, cfg, batch, extra_env, extra_args):
        args = list(cfg["args"])
        if len(cfg.get("gpu") or []) > 1:
            args += ["--layer-split", str(cfg.get("layer_split") or "auto")]
        if batch:
            args += ["--batch", str(batch)]
        args += extra_args
        env = dict(os.environ, **extra_env)
        env["LD_LIBRARY_PATH"] = ":".join(cfg.get("lib_dirs", []) + [env.get("LD_LIBRARY_PATH", "")])
        if os.name == "nt":
            env["PATH"] = os.pathsep.join(cfg.get("lib_dirs", []) + [env.get("PATH", "")])
        self.log_path = os.environ.get("BATCH_TEST_LOG") or os.path.join(tempfile.gettempdir(), "batch_test_engine.log")
        self.log = open(self.log_path, "w")
        self.p = subprocess.Popen([exe, "--serve", *args], cwd=cfg.get("cwd"), stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, stderr=self.log, text=True, bufsize=1, env=env)
        for line in self.p.stdout:
            if line.startswith("READY"):
                return
        raise SystemExit(f"the engine ended before READY - see {self.log_path}")

    def send(self, line):
        self.p.stdin.write(line + "\n")
        self.p.stdin.flush()

    def lines(self):
        for line in self.p.stdout:
            yield line.rstrip("\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--n", type=int, default=4, help="prompts (<= --batch)")
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--skip-solo", action="store_true")
    ap.add_argument("--keys", default="", help='sampling keys for every request, e.g. "temperature=0.7 top_k=20"')
    ap.add_argument("--extra", default="", help='more engine arguments in one string, e.g. "--adapt-every 1000000"')
    ap.add_argument("--dump", default="", help="write every slot's tokens (solo and batch) as JSON: an A/B of two engine builds")
    ap.add_argument("--long-tokens", type=int, default=0,
                    help="the LAST prompt is read from about this many tokens of filler text first: its chunks run "
                         "beside the earlier slots' windows (the prompt path lends experts then)")
    ap.add_argument("--mt-min", default="1", help="STRATA_IQ_MT_MIN for the engine (1: exact; empty: the default)")
    ap.add_argument("--acceptance", action="store_true",
                    help="with --batch-mtp in --extra: also compare MTP acceptance (proposals accepted of offered) "
                         "batched vs solo, not just tokens; sets STRATA_SPEC_DEPTH=1 for the engine")
    ap.add_argument("--acceptance-tol", type=float, default=5.0,
                    help="percentage points the batched first-proposal rate may sit from the solo rate "
                         "(the gate is this or the counting-noise band, whichever is wider)")
    if "--parse-check" in sys.argv:   # no engine, no GPU: prove the readers on a handcrafted log
        return parse_check()
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    tok = tokenizer(cfg["tokenizer"])
    prompts = []
    for q in QUESTIONS[: a.n]:
        text = f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        if a.long_tokens and q is QUESTIONS[a.n - 1]:
            para = ("The committee reviewed item %d of the long agenda, noting the budget, the schedule and the open "
                    "risks, and asked the staff to report back next quarter. ")
            per = max(1, len(tok.encode(para % 100)))
            filler = "".join(para % i for i in range(a.long_tokens // per + 1))
            text = f"<|im_start|>user\n{filler}\n\n{q}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        prompts.append(tok.encode(text, parse_special=True))
    env = {"STRATA_DECODE_TIMING": "1", **({"STRATA_VERIFY_PROFILE": "1"} if os.environ.get("PROF") else {})}
    if a.acceptance:
        env["STRATA_SPEC_DEPTH"] = "1"   # the solo side of the comparison: generate.cpp:13354
    if a.mt_min:
        env["STRATA_IQ_MT_MIN"] = a.mt_min
    eng = Engine(a.exe, cfg, a.batch, env, a.extra.split())
    out = eng.lines()

    solo = []
    solo_done = []   # with --acceptance: each solo request's own DONE tally (generate.cpp:13171)
    if not a.skip_solo:
        for i, ids in enumerate(prompts):
            eng.send(" ".join(x for x in ("GEN", str(a.max_new), a.keys, ",".join(map(str, ids))) if x))
            got, t0 = [], time.time()
            for line in out:
                if line.startswith("T "):
                    got.append(int(line.split()[1]))
                elif line.startswith("DONE") or line.startswith("ERR"):
                    if a.acceptance:
                        solo_done.append(parse_done(line))
                    print(f"solo {i}: {len(got)} tokens in {time.time() - t0:.1f}s  {line[:60]}", flush=True)
                    break
            solo.append(got)

    # batch: admit every prompt, then the slots decode together
    got = {i: [] for i in range(len(prompts))}
    done = {}
    bt_tokens = 0   # tokens that came from a 'BT ' line, that is from a batch window (the acceptance cross-check)
    t_admit = time.time()
    first_bt = None
    for i, ids in enumerate(prompts):
        eng.send(" ".join(x for x in ("BGEN", str(i), str(a.max_new), a.keys, ",".join(map(str, ids))) if x))
        for line in out:
            if line.startswith("T "):
                got[i].append(int(line.split()[1]))
            elif line.startswith("BT "):
                _, s, y = line.split()
                got[int(s)].append(int(y))
                bt_tokens += 1
                first_bt = first_bt or time.time()
            elif line.startswith("BDONE "):
                done[int(line.split()[1])] = line
            elif line.startswith("ERR"):
                print("batch:", line)
                return 1
            elif line.startswith("BADM "):
                if line.split()[2] == "0":
                    done[i] = line
                break
    t_admitted = time.time()
    for line in out:
        if line.startswith("BT "):
            _, s, y = line.split()
            got[int(s)].append(int(y))
            bt_tokens += 1
            first_bt = first_bt or time.time()
        elif line.startswith("BDONE "):
            done[int(line.split()[1])] = line
            if len(done) == len(prompts):
                break
        elif line.startswith("ERR"):
            print("batch:", line)
            return 1
    t_end = time.time()
    total = sum(len(v) for v in got.values())
    print(f"batch: {len(prompts)} slots, {total} tokens; admissions {t_admitted - t_admit:.1f}s; "
          f"then {t_end - t_admitted:.1f}s -> aggregate {total / max(t_end - t_admit, 1e-9):.1f} tok/s overall, "
          f"{sum(len(v) for v in got.values()) / max(t_end - (first_bt or t_admit), 1e-9):.1f} tok/s from the first batch token",
          flush=True)
    ok = True
    for i in range(len(prompts)):
        b = got[i]
        if solo:
            same = b == solo[i]
            ok &= same
            first_diff = next((k for k in range(min(len(b), len(solo[i]))) if b[k] != solo[i][k]), None)
            print(f"slot {i}: {len(b)} tokens, solo {len(solo[i])}: {'IDENTICAL' if same else f'DIFFERS at {first_diff}'}")
        print("   ", ascii(tok.decode(b)[:160]))
    if a.dump:
        Path(a.dump).write_text(json.dumps({"solo": solo, "batch": [got[i] for i in range(len(prompts))]}), encoding="utf-8")
    eng.send("QUIT")
    eng.p.wait(timeout=180)   # the next run needs the GPUs back
    if a.acceptance:
        # the batch tally line is logged when the slots drain (generate.cpp:10244), the engine's stderr is the log file
        with open(eng.log_path, "r", encoding="utf-8", errors="replace") as f:
            log = f.read()
        if not report_acceptance(log, len(prompts) if solo else 0, solo_done, bt_tokens,
                                 a.acceptance_tol, a.extra):
            ok = False
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
