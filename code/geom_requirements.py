"""geom_requirements.py — C-spike FROZEN geometry-side assertion module.

Preregistered metric implementation for the C-spike (see
_paper_artifacts_rc-fdedd0/Cspike_预注册判据_20260706.md §2.1). Frozen by sha256 in
that document's Appendix A BEFORE the smoke run; changes afterwards only via the
prereg's bug-exception clause with a written trail.

Protocol (adopted from CADTestBench, MIT, github.com/dimitrismallis/CADTestBench):
each intent tag is a requirement group evaluated by executable assertions (check())
against the LIVE B-rep produced by running the generated program — never against
the program text. The geometry core is lifted from the audited holdout_validate.py
(paper Table 6), with the GEOMERR slice bug fixed (line[8:], not line[5:]).

Normative assertions (prereg §2.1 — exactly these decide tag membership):
  <CIRCLE>     any edge with geomType()=="CIRCLE"
  <NGON>       any face containing >=5 LINE edges
  <THIN>       executed-bbox aspect ratio z/max(x,y) < 0.3
  <TALL>       same ratio > 2.0
  <MULTI_PART> len(solids())>1        [SENSITIVITY ONLY — excluded from the primary
                                       4-tag metric per prereg §2.1]

Execution failure (subprocess error or 10s timeout) defines "not executable" for the
primary execution-inclusive score; this is THE executability definition of the spike
(not the oracle `valid` bit, which is carried through for the 2x2 decomposition only).

Usage:
  .venv/bin/python geom_requirements.py score --inp spike_results/adherence_samples.jsonl \
      --out spike_results/geom_scores.jsonl [--workers 8]
  .venv/bin/python geom_requirements.py summary --scores spike_results/geom_scores.jsonl \
      --out spike_results/evaluation_summary.json
"""
import argparse, collections, json, subprocess, sys, textwrap, time
from multiprocessing import Pool
from pathlib import Path

TAGS_ALL = ["<CIRCLE>", "<NGON>", "<THIN>", "<TALL>", "<MULTI_PART>"]
TAGS_PRIMARY = ["<CIRCLE>", "<NGON>", "<THIN>", "<TALL>"]  # 4-tag primary (prereg §2.1)
TIMEOUT_S = 10

REQUIREMENT_DESC = {
    "<CIRCLE>": "solid has at least one circular edge (geomType CIRCLE)",
    "<NGON>": "solid has a face bounded by >=5 straight (LINE) edges",
    "<THIN>": "executed bounding-box aspect z/max(x,y) < 0.3",
    "<TALL>": "executed bounding-box aspect z/max(x,y) > 2.0",
    "<MULTI_PART>": "execution yields more than one disjoint solid",
}

# Geometry core: executes the program in an isolated interpreter and inspects the
# resulting B-rep. Identical assertions to holdout_validate.py (audited Table 6 path).
_GEOM_SRC = '''
import cadquery as cq, sys
try:
{body}
    r = result
    f=[]
    if any(e.geomType()=="CIRCLE" for e in r.edges().vals()): f.append("<CIRCLE>")
    for face in r.faces().vals():
        if sum(1 for e in face.Edges() if e.geomType()=="LINE")>=5: f.append("<NGON>"); break
    if len(r.solids().vals())>1: f.append("<MULTI_PART>")
    bb=r.val().BoundingBox(); foot=max(bb.xlen,bb.ylen); d=bb.zlen
    if foot>1e-9:
        ratio=d/foot
        if ratio<0.3: f.append("<THIN>")
        elif ratio>2.0: f.append("<TALL>")
    print("GEOM:"+",".join(f))
except Exception as e:
    print("GEOMERR:"+type(e).__name__)
'''


def geom_features(code, timeout=TIMEOUT_S):
    """Return (set_of_tags | None, status). None => not executable (spike definition)."""
    if not code or not code.strip():
        return None, "empty_program"
    src = _GEOM_SRC.format(body=textwrap.indent(code, "    "))
    try:
        p = subprocess.run([sys.executable, "-c", src],
                           capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None, "timeout"
    out = (p.stdout or "").strip().splitlines()
    line = next((l for l in out if l.startswith("GEOM:") or l.startswith("GEOMERR:")), "")
    if line.startswith("GEOM:"):
        return set(t for t in line[5:].split(",") if t), "ok"
    if line.startswith("GEOMERR:"):
        return None, line[8:] or "err"
    return None, "no_output"


def check(condition, pass_msg, fail_msg):
    """CADTestBench-style single-assertion record."""
    return {"passed": bool(condition), "message": pass_msg if condition else fail_msg}


def check_requirements(program):
    """Run the requirement groups on one program. Returns the per-sample record."""
    tags, status = geom_features(program)
    executable = tags is not None
    checks = {}
    for tag in TAGS_ALL:
        if not executable:
            checks[tag] = {"passed": False,
                           "message": f"not executable ({status}); requirement untestable, scored fail"}
        else:
            checks[tag] = check(tag in tags,
                                f"requirement met: {REQUIREMENT_DESC[tag]}",
                                f"requirement NOT met: {REQUIREMENT_DESC[tag]}")
    return {
        "executable": executable,
        "exec_status": status,
        "geom_produced": sorted(tags) if executable else [],
        "checks": checks,
    }


def _score_row(row):
    rec = check_requirements(row.get("program") or "")
    out = {k: row.get(k) for k in ("condition", "seed", "idx", "intended", "produced",
                                   "valid", "header_source", "header_injected",
                                   "prefix_fraction", "prefix_features")}
    out.update(rec)
    return out


def cmd_score(inp, out, workers):
    rows = [json.loads(l) for l in Path(inp).read_text().splitlines() if l.strip()]
    missing = sum(1 for r in rows if "program" not in r)
    if missing:
        raise SystemExit(f"FATAL: {missing}/{len(rows)} rows lack the 'program' field — "
                         "raw-program logging patch not active for these rows.")
    t0 = time.time()
    with Pool(workers) as pool:
        scored = pool.map(_score_row, rows, chunksize=8)
    with open(out, "w") as f:
        for s in scored:
            f.write(json.dumps(s) + "\n")
    n_exec = sum(1 for s in scored if s["executable"])
    print(f"[geom] scored {len(scored)} rows in {time.time()-t0:.0f}s; "
          f"executable {n_exec}/{len(scored)}")


def cmd_summary(scores, out):
    rows = [json.loads(l) for l in Path(scores).read_text().splitlines() if l.strip()]
    agg = collections.defaultdict(lambda: {"n": 0, "executable": 0,
                                           "status_counts": collections.Counter()})
    for r in rows:
        a = agg[(r["condition"], r["seed"])]
        a["n"] += 1
        a["executable"] += int(r["executable"])
        a["status_counts"][r["exec_status"]] += 1
    summary = {f"{c}|seed{s}": {"n": a["n"], "executable": a["executable"],
                                "executability_rate": a["executable"] / max(a["n"], 1),
                                "status_counts": dict(a["status_counts"])}
               for (c, s), a in sorted(agg.items())}
    Path(out).write_text(json.dumps(summary, indent=2))
    print(f"[geom] summary for {len(summary)} cells -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s1 = sub.add_parser("score"); s1.add_argument("--inp", required=True)
    s1.add_argument("--out", required=True); s1.add_argument("--workers", type=int, default=8)
    s2 = sub.add_parser("summary"); s2.add_argument("--scores", required=True)
    s2.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "score":
        cmd_score(a.inp, a.out, a.workers)
    else:
        cmd_summary(a.scores, a.out)
