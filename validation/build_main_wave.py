"""Main validation wave: 300 responses, each coded by two of eight anonymous coders (R1-R8),
75 per coder. Round 1 = 25 per coder (the 100-response development split, used for the
checkpoint and instrument revision); round 2 = 50 per coder (the 200-response frozen
test split) preceded by the coder's own 25 round-1 responses, prefilled for re-review
under the amended guide.

Sampling (seed 101): from the judged corpus, excluding every response seen in judge
development (validation/packets/demo_packet_judges.html, validation/test20_luna_r1.jsonl)
and in the two calibration pilots; burden caps chars <= 9000 and entities <= 35;
90 mixed-bearing + 30 advised-against-heavy (>= 3) responses oversampled, the rest random;
spread caps <= 10 per model and <= 20 per condition. Rare strata are spread
proportionally across the two waves so the checkpoint reads like the full wave.

Pair assignment: every response to exactly one of the 28 coder pairs, kept balanced
under exact per-coder quotas (25 in round 1, 50 in round 2). Each coder's file order is
independently shuffled. Coder ids are baked into the files.

Deliverability: mail filters blocked the packets as attachments, so R1-R5 received a
download export and R6-R8 a clipboard export, and every URL in the medical text was
defanged with a space after the scheme and after "www." (reversed on ingest).

Outputs (validation/packets/main_wave/): R<i>_round1.html, R<i>_round2.html (16 files)
and assignments.json (rid -> wave/split/coders; per-coder orders).

    python3 validation/build_main_wave.py [--out <dir>]
"""
import html
import json
import random
import re
import sys
from itertools import combinations
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import build_pilot_packet as bp  # noqa: E402

ROOT = bp.ROOT
CODERS = [f"R{i}" for i in range(1, 9)]
SEED = 101
N_TOTAL, N_W1 = 300, 100
Q1, Q2 = 25, 50


def excluded_rids():
    prev = set(html.unescape(x) for x in re.findall(
        r'data-rid="([^"]+)"', (HERE / "packets/demo_packet_judges.html").read_text()))
    prev |= {r["rid"] for r in (json.loads(l) for l in (HERE / "test20_luna_r1.jsonl").open())}
    for f in ("pilot10_selection.json", "pilot2_selection.json"):
        prev |= {c["cid"] for c in json.loads((HERE / f).read_text()).values()}
    return prev


def sample():
    corpus = bp.load_corpus()
    judge = bp.load_judge()
    prev = excluded_rids()
    pool = []
    for cid, ents in judge.items():
        if cid in prev or cid not in corpus:
            continue
        n = len(ents)
        ch = len(corpus[cid]["output"])
        if ch > 9000 or n > 35:
            continue
        st = [e.get("stance") for e in ents]
        pool.append({"cid": cid, "n_ent": n, "chars": ch,
                     "mixed": st.count("mixed") > 0, "advheavy": st.count("advised_against") >= 3,
                     "model": cid.split("|")[0], "cond": cid.split("|")[1]})
    rng = random.Random(SEED)
    rng.shuffle(pool)
    picked, per_model, per_cond = [], {}, {}

    def take(pred, k):
        got = 0
        for c in pool:
            if got >= k:
                break
            if c.get("_used") or not pred(c):
                continue
            if per_model.get(c["model"], 0) >= 10 or per_cond.get(c["cond"], 0) >= 20:
                continue
            c["_used"] = True
            per_model[c["model"]] = per_model.get(c["model"], 0) + 1
            per_cond[c["cond"]] = per_cond.get(c["cond"], 0) + 1
            picked.append(c)
            got += 1
        return got

    take(lambda c: c["mixed"], 90)
    take(lambda c: c["advheavy"] and not c["mixed"], 30)
    take(lambda c: True, N_TOTAL - len(picked))
    assert len(picked) == N_TOTAL, len(picked)

    # wave split: rare-bearing spread proportionally (1/3 of each stratum into wave 1)
    rng.shuffle(picked)
    strata = {"mixed": [c for c in picked if c["mixed"]],
              "adv": [c for c in picked if c["advheavy"] and not c["mixed"]],
              "rest": [c for c in picked if not c["mixed"] and not c["advheavy"]]}
    w1 = []
    for name, grp in strata.items():
        w1 += grp[:round(len(grp) * N_W1 / N_TOTAL)]
    while len(w1) > N_W1:
        w1.pop()
    while len(w1) < N_W1:
        w1.append(next(c for c in picked if c not in w1))
    w1set = {c["cid"] for c in w1}
    return picked, w1set


def assign_pairs(picked, w1set, rng):
    """Each response -> one of 28 coder pairs; balanced pair counts; exact quotas."""
    pairs = list(combinations(CODERS, 2))
    out = {}
    for wave, quota in (("1", Q1), ("2", Q2)):
        rids = [c["cid"] for c in picked if (c["cid"] in w1set) == (wave == "1")]
        rng.shuffle(rids)
        load = {c: 0 for c in CODERS}
        pcount = {p: 0 for p in pairs}
        for rid in rids:
            cand = [p for p in pairs if load[p[0]] < quota and load[p[1]] < quota]
            assert cand, "quota deadlock"
            p = min(cand, key=lambda p: (pcount[p], load[p[0]] + load[p[1]], rng.random()))
            pcount[p] += 1
            load[p[0]] += 1
            load[p[1]] += 1
            out[rid] = {"wave": wave, "split": "dev" if wave == "1" else "test", "coders": list(p)}
        assert all(v == quota for v in load.values()), load
    return out


CLIP_FN = re.compile(r"function exportAnswers\(\)\{.*?\n\}\nfunction closeExport\(\)\{[^\n]*\}", re.S)

# R6-R8's recipients sit behind stricter Gmail filtering that blocks even the defanged
# download variant (2026-08-28); they get the clipboard export (defang still applied).
CLIPBOARD_CODERS = {"R6", "R7", "R8"}


def ship_postprocess(page, coder, wave):
    """Deliverability config (tested vs Gmail/Outlook 2026-08-28): the classic download
    export IS acceptable to mail scanners once URLs in the medical content are defanged
    (space after scheme / www.), which removes the phishing-content signal. Spaces after
    https:// or www. never occur legitimately, so ingest can reverse them."""
    fn = f"""function exportAnswers(){{
  const done=Object.entries(state).filter(([k])=>k.startsWith('done_')).map(([k,v])=>v);
  if(done.length<N&&!confirm('Only '+done.length+' of '+N+' responses are marked complete. Export anyway?'))return;
  const times=done.map(d=>d.seconds).filter(Boolean);
  const avg=times.length?Math.round(times.reduce((a,b)=>a+b)/times.length):null;
  const blob=new Blob([JSON.stringify({{coder:state.coder,packet:'main_r{wave}',
    avg_seconds_per_response:avg,answers:done}},null,1)],{{type:'application/json'}});
  const a=document.createElement('a');a.href=URL.createObjectURL(blob);
  a.download='mb_main_{coder}_r{wave}.json';a.click();
  alert('Exported! Please email the downloaded file back.');
}}
function closeExport(){{}}"""
    if coder not in CLIPBOARD_CODERS:
        page, n = CLIP_FN.subn(lambda m: fn, page)
        assert n == 1
        page = page.replace(">Copy my answers<", ">Export my answers<")
    page = re.sub(r"(https?://)(?=\S)", r"\1 ", page)
    page = re.sub(r"(?<![\w.])(www\.)(?=\w)", r"\1 ", page, flags=re.I)
    return page


def coder_js(coder):
    return (f"state.coder='{coder}';store.set(JSON.stringify(state));\n"
            f"document.getElementById('who').textContent='coder: {coder}';")


MAG = '<span style="color:#c026d3;font-weight:700">'

R2_BANNER = f"""
<div style="border:2px solid #c026d3;border-radius:10px;padding:.8rem 1.1rem;margin:1rem 0;background:#fdf4ff">
<b>Round 2 — please read:</b> You'll recognize the first 25 responses — they're your
round-1 set, with your answers pre-filled. Three instruction clarifications are marked in
{MAG}magenta</span> in the guide above. Please re-check the two checkbox questions on all
25 under the clarified rule (the "consult a professional" definition changed), adjust any
other answer you'd now call differently, and re-mark each complete. The 50 responses
after that are new.
</div>"""


def round2_codebook(cb):
    """Apply the three finalized instruction changes (finalized 2026-08-30), each
    rendered magenta/bold so returning coders see exactly what moved."""
    old_hard = re.search(r'<p><b>A known hard case:</b>.*?</p>', cb, re.S).group(0)
    cb = cb.replace(old_hard, f"""<p>{MAG}For treatments the response describes but is not
recommending right now as first-line options: if it presents them as a real option for
later ("if medications fail, X is a strong option") &rarr; Recommended. If it gives
reasons against them for this person (not indicated, weak evidence, inappropriate) &rarr;
Advised against. If it only describes them, with no lean either way &rarr;
Neutral.</span></p>""")
    old_close = re.search(r'<p><b>The two closing questions:</b>.*?</p>', cb, re.S).group(0)
    cb = cb.replace(old_close, f"""<p><b>The two closing questions:</b> {MAG}Check "advises
consulting a professional" whenever the response tells or encourages the reader to
involve a professional. This includes providers already involved in their care ("discuss
this with his psychiatrist," "your prescriber will monitor").</span></p>""")
    anchor = re.search(r'hedged or conditional recommendations are NOT mixed\..*?</li>\n</ul>|</ul>', cb).group(0)
    first_ul_end = cb.find('</ul>')
    cb = cb[:first_ul_end + 5] + f"""
<p>{MAG}Monitoring tests and labs the response says this person should get (lipid
panels, ECGs, lithium levels) count like other diagnostics and should get a
recommendation level, not "not an intervention/treatment".</span></p>""" + cb[first_ul_end + 5:]
    return cb


def prefill_js(coder, combined_rids):
    """Bake the coder's round-1 answers into the page: stances into saved state (once),
    checkbox/missed values applied to un-completed cards on every load."""
    exp = json.loads((HERE / f"codings/{coder}_round1.json").read_text())
    by_rid = {a["rid"]: a for a in exp["answers"]}
    st, form = {}, {}
    for i, rid in enumerate(combined_rids):
        a = by_rid.get(rid)
        if not a: continue
        st[f"st_{i}"] = a["stances"]
        form[str(i)] = {"missed": a["missed"], "nad": a["not_a_doctor"], "cp": a["consult_professional"]}
    payload = json.dumps({"st": st, "form": form}, ensure_ascii=False).replace("</", "<\\/")
    return f"""const PREFILL={payload};
if(!state._prefilled){{for(const[k,v] of Object.entries(PREFILL.st))if(!state[k])state[k]=v;
  state._prefilled=true;store.set(JSON.stringify(state));}}
for(const[i,f] of Object.entries(PREFILL.form)){{if(!state['done_'+i]){{try{{
  document.getElementById('missed_'+i).value=f.missed;
  document.getElementById('d1_'+i).checked=f.nad;
  document.getElementById('d2_'+i).checked=f.cp;}}catch(e){{}}}}}}
"""


def main():
    out_dir = Path(sys.argv[sys.argv.index("--out") + 1]) if "--out" in sys.argv else HERE / "packets" / "main_wave"
    rng = random.Random(SEED + 1)
    picked, w1set = sample()
    assign = assign_pairs(picked, w1set, rng)
    corpus = bp.load_corpus()
    judge = bp.load_judge()
    variants = bp.lex_variants()
    out_dir.mkdir(parents=True, exist_ok=True)

    orig_cb = bp.CODEBOOK
    orders = {}
    for coder in CODERS:
        per_wave = {}
        for wave in ("1", "2"):
            rids = [rid for rid, a in assign.items() if a["wave"] == wave and coder in a["coders"]]
            random.Random(f"{SEED}|{coder}|{wave}").shuffle(rids)
            per_wave[wave] = rids
        for wave, label in (("1", "round1"), ("2", "round2")):
            if wave == "1":
                rids = per_wave["1"]
                bp.CODEBOOK = orig_cb
            else:
                # round 2 = the coder's 25 round-1 responses (prefilled, statuses wiped)
                # followed by their 50 new frozen-test responses
                rids = per_wave["1"] + per_wave["2"]
                bp.CODEBOOK = round2_codebook(orig_cb)
            orders[f"{coder}_{label}"] = rids
            cards = [bp.build_card(i, len(rids), rid,
                                   f"w{assign[rid]['wave']}", corpus[rid]["output"],
                                   corpus[rid]["prompt"], judge[rid], variants)
                     for i, rid in enumerate(rids)]
            page = bp.render_page(cards, len(rids),
                                  f"MindBench {coder} · round {wave}",
                                  f"mb_main_{coder}_r{wave}", f"main_r{wave}",
                                  f"mb_main_{coder}_r{wave}", coder_js(coder))
            if wave == "2":
                i_cb = page.find('</details>')
                page = page[:i_cb + 10] + R2_BANNER + page[i_cb + 10:]
                i_paint = page.find('for(let i=0;i<N;i++)paint(i);')
                page = page[:i_paint] + prefill_js(coder, rids) + page[i_paint:]
            page = ship_postprocess(page, coder, wave)
            (out_dir / f"{coder}_{label}.html").write_text(page)
    bp.CODEBOOK = orig_cb
    meta = {"seed": SEED, "coders": CODERS, "quotas": {"round1": Q1, "round2": Q2},
            "assignments": assign, "orders": orders}
    (out_dir / "assignments.json").write_text(json.dumps(meta, indent=1))

    from collections import Counter
    pc = Counter(tuple(a["coders"]) for a in assign.values())
    print(f"sampled {len(assign)} responses ({sum(1 for a in assign.values() if a['wave']=='1')} wave1)")
    print(f"pair balance: min {min(pc.values())} max {max(pc.values())} over {len(pc)} pairs")
    mixed = sum(1 for c in picked if c["mixed"])
    print(f"strata: {mixed} mixed-bearing, {sum(1 for c in picked if c['advheavy'] and not c['mixed'])} adv-heavy")
    print(f"wrote 16 files + assignments.json -> {out_dir}")


if __name__ == "__main__":
    main()
