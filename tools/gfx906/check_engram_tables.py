#!/usr/bin/env python3
"""check_engram_tables.py - pre-boot integrity gate for the DS4.1 engram tables (gfx906 fork, P4.0 #18).

The byte-vs-source verifier (tools/v41/verify_engram.py) CANNOT run here: the FP8 source shards were
deleted after extraction and the tables are the sole copy.  This gate mirrors what the engine itself
requires at boot - Ds41Engram::open() (tools/v41/engine/ds41_engram.cpp:186-235):

  * the first --engram dir containing a non-empty model.safetensors.index.json is the one the engine
    uses (both dirs on this box carry the complete 4-entry weight_map, so dir order cannot strand a
    layer - this script re-checks that claim);
  * for every deepseek41.engram.layer_ids entry, layers.<L>.engram.embed.{weight,scale} is in the
    index, its file exists in one of the given dirs, and st_tensor() reads it:
        weight: dtype F8_E4M3, bytes == rows * dim
        scale : dtype F8_E8M0, bytes == rows * dim / 32
    with rows from deepseek41.engram.rows and dim from deepseek41.config's engram_head_dim;

  plus two checks the engine does NOT do, each of which turns a would-be mid-generation I/O failure
  into a pre-boot refusal:  the file on disk is at least 8 + header + the last tensor offset
  (truncation), and -s random 64 KiB reads inside each tensor range read cleanly (EIO/extent rot).

Reads headers + samples only - never the whole ~190 GiB.  Exit 0 = every check passed.

usage: check_engram_tables.py V41.gguf ENGRAM_DIR [ENGRAM_DIR...] [-s SAMPLES_PER_TENSOR]
"""
import json
import os
import random
import re
import struct
import sys

GG_TYPE, GG_SIZE = b"GGUF", 0
T_UINT8, T_INT8, T_UINT16, T_INT16, T_UINT32, T_INT32, T_FLOAT32, T_BOOL = 0, 1, 2, 3, 4, 5, 6, 7
T_STRING, T_ARRAY, T_UINT64, T_INT64, T_FLOAT64 = 8, 9, 10, 11, 12
SCALAR = {T_UINT8: "<B", T_INT8: "<b", T_UINT16: "<H", T_INT16: "<h", T_UINT32: "<I", T_INT32: "<i",
          T_FLOAT32: "<f", T_UINT64: "<Q", T_INT64: "<q", T_FLOAT64: "<d"}
SCALAR_SIZE = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


def gguf_metadata(path):
    """Parse GGUF v2+ KV metadata into {key: (type, value)}; arrays come back as lists, strings as str."""
    with open(path, "rb") as f:
        if f.read(4) != GG_TYPE:
            raise SystemExit(f"{path}: not a GGUF file")
        ver = struct.unpack("<I", f.read(4))[0]
        if ver < 2:
            raise SystemExit(f"{path}: GGUF v{ver} not supported")
        f.read(8)                                    # n_tensors (the tensor-info records follow the KV)
        n_kv = struct.unpack("<Q", f.read(8))[0]

        def rstr():
            n = struct.unpack("<Q", f.read(8))[0]
            return f.read(n).decode("utf-8", "replace")

        def rval(t):
            if t == T_STRING:
                return rstr()
            if t == T_BOOL:
                return f.read(1)[0] != 0
            if t == T_ARRAY:
                et = struct.unpack("<I", f.read(4))[0]
                n = struct.unpack("<Q", f.read(8))[0]
                if et in SCALAR:
                    fmt = "<" + str(n) + SCALAR[et][1]
                    return list(struct.unpack(fmt, f.read(n * SCALAR_SIZE[et]))) if n else []
                return [rval(et) for _ in range(n)]
            return struct.unpack(SCALAR[t], f.read(struct.calcsize(SCALAR[t])))[0]

        meta = {}
        for _ in range(n_kv):
            k, t = rstr(), struct.unpack("<I", f.read(4))[0]
            meta[k] = (t, rval(t))
    return meta


def st_header(path):
    """(header_len, header_dict) of a safetensors file."""
    with open(path, "rb") as f:
        hl = struct.unpack("<Q", f.read(8))[0]
        if hl >= 1 << 30:
            raise SystemExit(f"{path}: safetensors header length {hl} implausible")
        head = json.loads(f.read(hl).decode("utf-8"))
    return hl, head


def main(argv):
    args = list(argv[1:])
    samples = 4
    if "-s" in args:
        i = args.index("-s")
        samples = int(args[i + 1])
        del args[i:i + 2]
    if len(args) < 2:
        print(__doc__.strip().splitlines()[-1])
        return 2
    gguf, dirs = args[0], args[1:]

    # ---- what the engine expects (ds41_geometry.hpp:258-271): dim from the config JSON, rows/layer_ids arrays
    m = gguf_metadata(gguf)
    cfg_s = m.get("deepseek41.config", (None, None))[1]
    if not cfg_s:
        print("FAIL: no deepseek41.config in the GGUF - geometry would not load")
        return 1
    # the engine reads dim with detail::json_num - a loose search of the config STRING (ds41_geometry.hpp:262),
    # which is how it finds engram_head_dim nested under text_config; mirror that, not a strict top-level dict
    mm = re.search(r'"engram_head_dim"\s*:\s*(\d+)', cfg_s)
    if not mm:
        print("FAIL: engram_head_dim not found in deepseek41.config (engine's geometry load would fail too)")
        return 1
    dim = int(mm.group(1))
    layers = m.get("deepseek41.engram.layer_ids", (None, None))[1]
    rows = m.get("deepseek41.engram.rows", (None, None))[1]
    enc = m.get("deepseek41.engram.encoding", (None, None))[1]
    if not layers or not rows or len(layers) != len(rows):
        print("FAIL: deepseek41.engram.layer_ids/rows missing or ragged")
        return 1
    if dim % 32:
        print(f"FAIL: engram_head_dim {dim} not a multiple of 32 (the scale is one E8M0 byte per 32)")
        return 1
    print(f"GGUF {os.path.basename(gguf)}: layers {layers}, rows {rows}, head_dim {dim}, "
          f"encoding {enc or '(absent - not checked by the engine either)'}")
    if isinstance(enc, str) and "row" in enc:
        want = dim + dim // 32
        got = int(enc.rsplit("row", 1)[1])
        if got != want:
            print(f"FAIL: encoding says row{got}, head_dim {dim} implies row{want}")
            return 1

    # ---- the index the engine would actually use: first dir with a non-empty one (ds41_engram.cpp:200-205)
    index, index_dir = None, None
    for d in dirs:
        p = os.path.join(d, "model.safetensors.index.json")
        if os.path.isfile(p) and os.path.getsize(p) > 0:
            index, index_dir = json.load(open(p)), d
            break
    if index is None:
        print("FAIL: no non-empty model.safetensors.index.json in any given dir (engine: 'no model.safetensors.index.json')")
        return 1
    wm = index["weight_map"]
    print(f"index: {index_dir}/model.safetensors.index.json ({len(wm)} entries)")

    rng = random.Random(1234)                                            # deterministic sampling for gate repeatability
    fails = 0
    for lid, lrows in zip(layers, rows):
        for kind, want_dtype, want_bytes in (("weight", "F8_E4M3", lrows * dim),
                                             ("scale", "F8_E8M0", lrows * dim // 32)):
            name = f"layers.{lid}.engram.embed.{kind}"
            tag = f"L{lid}.{kind}"
            fn = wm.get(name)
            if not fn:
                print(f"FAIL {tag}: not in the index (engine: '{name} not in the index')")
                fails += 1
                continue
            path = next((os.path.join(d, fn) for d in dirs if os.path.isfile(os.path.join(d, fn))), None)
            if path is None:
                print(f"FAIL {tag}: {fn} not found in the given dirs")
                fails += 1
                continue
            hl, head = st_header(path)
            ent = head.get(name)
            if ent is None:
                print(f"FAIL {tag}: not a tensor in {fn}")
                fails += 1
                continue
            off0, off1 = ent["data_offsets"]
            bytes_, dt = off1 - off0, ent["dtype"]
            if dt != want_dtype or bytes_ != want_bytes:
                print(f"FAIL {tag}: dtype {dt} / {bytes_} bytes - engine wants {want_dtype} / {want_bytes} "
                      f"(rows {lrows} x dim {dim}{'' if kind == 'weight' else ' / 32'})")
                fails += 1
                continue
            last = max(v["data_offsets"][1] for k, v in head.items() if k != "__metadata__")
            size = os.path.getsize(path)
            need = 8 + hl + last
            if size < need:
                print(f"FAIL {tag}: {path} is {size} bytes, tensor data ends at {need} (TRUNCATED)")
                fails += 1
                continue
            bad = 0
            for _ in range(samples):                                     # readability of random spans (EIO probe)
                with open(path, "rb") as f:
                    f.seek(off0 + rng.randrange(0, max(1, bytes_ - 65536)) + 8 + hl)
                    got = f.read(65536)
                if len(got) != 65536:
                    bad += 1
            if bad:
                print(f"FAIL {tag}: {bad}/{samples} random 64 KiB reads came up short in {fn}")
                fails += 1
                continue
            shape = ent.get("shape") or []
            rows_note = f", shape rows {shape[0]}" if shape and shape[0] != lrows else ""
            print(f"OK   {tag}: {dt} {bytes_:,} B ({bytes_ / 2**30:.1f} GiB) == rows x dim; "
                  f"{os.path.basename(path)} {size:,} B >= {need:,} B; {samples}/{samples} reads{rows_note}")
    if fails:
        print(f"engram gate: {fails} FAIL - the engine would refuse boot the same way")
        return 1
    print(f"engram gate: PASS ({len(layers)} tables x 2 tensors; engine expectations mirrored + truncation + reads)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
