import collections, pathlib, sys
sys.path.insert(0, "/home/zhuojun/ninfer")
from tools.artifact.reader import Artifact

HIDDEN = 5120; ATTN_Q = 24 * 256; ATTN_K = 4 * 256
GDN_K = 16 * 128; GDN_V = 48 * 128; INTER = 17408

def classify(name, shape):
    n, k = (shape[0], shape[1]) if len(shape) >= 2 else (0, 0)
    if name.startswith("mtp/"):
        return "rep"
    if name.endswith("token_embedding") or name.endswith("output_head"):
        return "rep"
    if n == 2 * ATTN_Q + 2 * ATTN_K and k == HIDDEN: return "half"
    if n == 2 * GDN_K + 2 * GDN_V and k == HIDDEN: return "half"
    if n == 4 and k == 2 * GDN_K + GDN_V: return "half"
    if n == HIDDEN and k == ATTN_Q and name.endswith("attention/output"): return "half"
    if n == HIDDEN and k == GDN_V and name.endswith("gdn/output"): return "half"
    if n == 2 * INTER and k == HIDDEN: return "half"
    if n == HIDDEN and k == INTER: return "half"
    return "rep"

p = pathlib.Path("/home/zhuojun/models/qwen3_8_27b_nvfp4.ninfer")
with Artifact(p) as a:
    d = a.directory
    by_id = a.by_id
    names_by_obj = collections.defaultdict(set)
    for name, b in d.bindings.items():
        if name.split("/")[0] not in ("text", "mtp", "proposal"):
            continue
        oids = [b["object"]] if "object" in b else [pt["object"] for pt in b.get("parts", [])]
        for oid in oids:
            names_by_obj[oid].add(name)

    stats = collections.Counter()
    comp = collections.Counter()
    for oid, names in names_by_obj.items():
        o = by_id.get(oid)
        if o is None or not hasattr(o, "shape"):
            continue
        primary = next((n for n in sorted(names) if n.startswith("text/")), sorted(names)[0])
        kind = classify(primary, list(o.shape))
        top = primary.split("/")[0]
        bucket = "text_layers" if primary.startswith("text/layers") else primary
        comp[(bucket, kind)] += o.bytes
        stats[("all", kind)] += o.bytes
        if top in ("proposal", "mtp"):
            stats[(top, kind)] += o.bytes

    def mib(x): return x / 1048576
    def per_shard(exclude_proposal):
        tot = 0
        for oid, names in names_by_obj.items():
            o = by_id.get(oid)
            if o is None or not hasattr(o, "shape"): continue
            primary = next((n for n in sorted(names) if n.startswith("text/")), sorted(names)[0])
            if exclude_proposal and primary.startswith("proposal/"): continue
            b = o.bytes
            tot += b if classify(primary, list(o.shape)) == "rep" else b // 2
        return tot
    print("per-shard weights without proposal: %.1f MiB (%.2f GiB)" % (mib(per_shard(True)), per_shard(True)/2**30))
    print("per-shard weights with proposal:    %.1f MiB (%.2f GiB)" % (mib(per_shard(False)), per_shard(False)/2**30))
    print("--- component/kind (MiB)")
    for (name, kind), v in sorted(comp.items()):
        print("  %-28s %-5s %10.1f" % (name, kind, mib(v)))
    print("--- groups (MiB)")
    for (grp, kind), v in sorted(stats.items()):
        print("  %-10s %-5s %10.1f" % (grp, kind, mib(v)))
