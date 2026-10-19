#!/usr/bin/env python3
"""verify_engram.py SRC_FP8_DIR DST_ENGRAM_DIR...   - byte-exhaustive check of extract_engram.py output.

For every tensor in each DST_DIR/engram-*.safetensors, finds the same tensor in the FP8 source via the
source index and compares EVERY byte in 64 MiB chunks. extract_engram.py already size-checks all tensors
and row-checks 2000 random rows; this closes the remaining gap so the FP8 shards can be deleted safely.
Exit 0 = every byte of every engram tensor matches. ~2x table bytes of sequential reads.
"""
import json, os, struct, sys

def st_header(path):
    with open(path, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        h = json.loads(f.read(n))
    return h, 8 + n

def main():
    src, *dsts = sys.argv[1:]
    if not dsts:
        print(__doc__); sys.exit(2)
    sindex = json.load(open(os.path.join(src, 'model.safetensors.index.json')))['weight_map']
    BUF = 64 << 20
    total = done = 0
    for d in dsts:
        for fn in sorted(os.listdir(d)):
            if not fn.endswith('.safetensors'):
                continue
            out = os.path.join(d, fn)
            h, dsrc = st_header(out)
            for nm, t in h.items():
                sfile = os.path.join(src, sindex[nm])
                sh, sds = st_header(sfile)
                a, b = sindex[nm], None
                sa, sb = sh[nm]['data_offsets']
                nbytes = sb - sa
                assert t['data_offsets'][1] - t['data_offsets'][0] == nbytes, f'size mismatch {nm}'
                da = dsrc + t['data_offsets'][0]
                with open(sfile, 'rb', buffering=0) as fs, open(out, 'rb', buffering=0) as fd:
                    fs.seek(sds + sa); fd.seek(da); left = nbytes
                    while left:
                        n = min(BUF, left)
                        x, y = fs.read(n), fd.read(n)
                        if x != y:
                            sys.exit(f'MISMATCH {nm} at offset {nbytes-left}')
                        left -= n; done += n
                        if done % (16 << 30) < BUF:
                            print(f'  {done/1e9:7.1f} GB compared', flush=True)
                total += nbytes
    print(f'OK: all bytes of all engram tensors match ({total/1e9:.1f} GB)')

if __name__ == '__main__':
    main()
