"""Judge-development packet: the same responses annotated by each candidate judge, rendered
with a judge dropdown, prefilled model answers, and measured cost projections.

This is the tool behind judge selection. Twelve corpus responses are sampled (seed 11),
each candidate judge annotates them with the frozen prompt (run/judge_prompt.py), and the
packet lets a reviewer switch judges, verify or correct their answers, and see the
per-response token use projected to the full corpus. The four candidates compared were
Claude Haiku 4.5, Claude Sonnet 5, GPT-5.6 Luna and DeepSeek V4 Flash; Luna was selected
on cost and quality (decision 2026-08-26). JUDGES marks the active set; the final packet
(validation/packets/demo_packet_judges.html) was built with Luna only.

Running this makes live calls to every active judge.

    python3 validation/build_judge_dev_packet.py
"""
import html
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).parent
sys.path.insert(0, str(ROOT / "run"))
from dotenv import load_dotenv  # noqa: E402
load_dotenv(ROOT / ".env")
import os  # noqa: E402
import anthropic  # noqa: E402
from judge_prompt import PROMPT  # noqa: E402

CANDIDATES = {  # key -> (provider, model id, base url, env key); the four judges compared
    "haiku-4.5": ("anthropic", "claude-haiku-4-5-20251001", None, None),
    "sonnet-5": ("anthropic", "claude-sonnet-5", None, None),
    "deepseek-flash": ("openai", "deepseek-v4-flash", "https://api.deepseek.com/v1", "DEEPSEEK_API_KEY"),
    "gpt-5.6-luna": ("openai", "gpt-5.6-luna", "https://api.openai.com/v1", "OPENAI_API_KEY"),
}
ACTIVE = ["gpt-5.6-luna"]  # set to list(CANDIDATES) to rebuild the four-judge comparison
JUDGES = {k: CANDIDATES[k] for k in ACTIVE}
PRICES = {"haiku-4.5": (1, 5), "sonnet-5": (2, 10), "gpt-5.6-luna": (0.2, 1.2), "deepseek-flash": (0.22, 0.66)}
CORPUS = 35055


def sample_responses():
    rows = []
    for p in sorted(Path(ROOT / "runs/gen_v1").glob("*/generations.jsonl")):
        for l in p.open():
            if not l.strip().endswith("}"): continue
            r = json.loads(l)
            if not r.get("error") and (r.get("response") or "").strip() and str(r.get("finish_reason")).lower() not in ("length", "max_tokens"):
                rows.append(r)
    random.Random(11).shuffle(rows)
    picked, seen, used = [], set(), set()
    def take(pred):
        for r in rows:
            if id(r) not in seen and r["condition"] not in used and pred(r):
                picked.append(r); seen.add(id(r)); used.add(r["condition"]); return
    take(lambda r: r["category"] == "medication" and r["register"] == "patient")
    take(lambda r: r["category"] == "medication" and "avoid" in (r["response"] or "").lower())
    take(lambda r: r["condition"] == "uncomplicated_bereavement" and r["category"] == "open")
    take(lambda r: r["category"] == "psychotherapy" and r["register"] == "caregiver")
    take(lambda r: r["category"] == "digital" and r["register"] == "patient")
    take(lambda r: r["category"] == "neuromodulation" and r["register"] == "clinical")
    take(lambda r: r["category"] == "open" and r["register"] == "patient")
    take(lambda r: r["category"] == "medication" and r["register"] == "clinical")
    # round-4 additions: fresh draws to test prompt generalization
    take(lambda r: r["category"] == "psychotherapy" and r["register"] == "clinical")
    take(lambda r: r["category"] == "neuromodulation" and r["register"] == "patient")
    take(lambda r: r["category"] == "open" and r["register"] == "caregiver")
    take(lambda r: r["category"] == "medication" and r["register"] == "clinical"
         and ("avoid" in (r["response"] or "").lower() or "|" in (r["response"] or "")))
    return picked[:12]

def annotate(clients, judge, text):
    prov, mid, base, env = JUDGES[judge]
    if prov == "anthropic":
        r = clients["anthropic"].messages.create(model=mid, max_tokens=16000,
            messages=[{"role": "user", "content": PROMPT + text}])
        raw = "".join(b.text for b in r.content if b.type == "text")
        usage = {"tin": r.usage.input_tokens, "tout": r.usage.output_tokens}
    else:
        try:
            r = clients[judge].chat.completions.create(model=mid, max_completion_tokens=8000,
                messages=[{"role": "user", "content": PROMPT + text}])
        except Exception as e:
            if "max_completion_tokens" not in str(e):
                raise
            r = clients[judge].chat.completions.create(model=mid, max_tokens=8000,
                messages=[{"role": "user", "content": PROMPT + text}])
        raw = r.choices[0].message.content or ""
        usage = {"tin": r.usage.prompt_tokens, "tout": r.usage.completion_tokens}
    m = re.search(r"\{.*\}", raw, re.S)
    return json.loads(m.group(0)), usage

S0, S1, S2 = "\ue000", "\ue001", "\ue002"   # private-use markers wrapped around located entities

SURF2CANON = None
def _lex_map():
    global SURF2CANON
    if SURF2CANON is None:
        SURF2CANON = {}
        meds = json.loads((ROOT / "instrument/lexicon/psych_meds.json").read_text())
        for g, info in meds.items():
            for x in {g, *info.get("aliases", []), *info.get("brands", [])}: SURF2CANON[x.lower()] = ("med", g)
        for fn in ("digital.json", "neuromodulation.json", "talk_therapies.json"):
            d = json.loads((ROOT / "instrument/lexicon" / fn).read_text())
            for k, v in d.items():
                if v.get("family") == "null_option": continue
                for x in {v["name"], *v.get("aliases", []), *([v["abbrev"]] if v.get("abbrev") else [])}:
                    SURF2CANON[x.lower()] = (fn, k)
    return SURF2CANON

def tag_llm_entities(text, entities):
    """Highlight each LLM-reported surface (all occurrences) PLUS its lexicon sibling
    surfaces (brand<->generic, abbreviations), so synonym re-mentions light up and link
    to the same entity. Longest surfaces first."""
    lex = _lex_map()
    spans, taken = [], []
    jobs = []
    for ent in sorted(entities, key=lambda e: -len(e["surface"])):
        surf = ent["surface"].strip()
        if not surf: continue
        variants = {surf}
        canon = lex.get(surf.lower())
        if canon:
            variants |= {x for x, c in lex.items() if c == canon}
        jobs.append((ent, surf, sorted(variants, key=len, reverse=True)))
    for ent, surf, variants in jobs:
        found = False
        for v in variants:
            cs = v.isupper() and len(v) <= 5   # abbreviations: exact case
            pat = re.compile(rf"(?<![\w-]){re.escape(v)}(?![\w-])", 0 if cs else re.IGNORECASE)
            for m in pat.finditer(text):
                if any(m.start() < e and m.end() > b for b, e in taken): continue
                taken.append((m.start(), m.end())); spans.append((m.start(), m.end(), surf)); found = True
        if not found:
            ent["_unlocated"] = True
    spans.sort()
    out, pos = [], 0
    for b, e, surf in spans:
        out += [text[pos:b], f"{S0}{surf}{S1}{text[b:e]}{S2}"]
        pos = e
    out.append(text[pos:])
    return "".join(out)

INLINE_BOLD = re.compile(r"\*\*(.+?)\*\*")
INLINE_IT = re.compile(r"(?<!\*)\*([^*\n]+?)\*(?!\*)")

def md_render(text):
    lines = text.split("\n")
    out, mode = [], None
    def close():
        nonlocal mode
        if mode: out.append(f"</{mode}>"); mode = None
    for ln in lines:
        e = html.escape(ln.rstrip())
        e = INLINE_BOLD.sub(r"<b>\1</b>", e)
        e = INLINE_IT.sub(r"<i>\1</i>", e)
        h = re.match(r"^(#{1,6})\s+(.*)", e)
        bullet = re.match(r"^\s*[-*•]\s+(.*)", e)
        num = re.match(r"^\s*\d+[.)]\s+(.*)", e)
        if h: close(); out.append(f"<h4>{h.group(2)}</h4>")
        elif bullet:
            if mode != "ul": close(); out.append("<ul>"); mode = "ul"
            out.append(f"<li>{bullet.group(1)}</li>")
        elif num:
            if mode != "ol": close(); out.append("<ol>"); mode = "ol"
            out.append(f"<li>{num.group(1)}</li>")
        elif not e.strip(): close()
        else: close(); out.append(f"<p>{e}</p>")
    close()
    htm = "".join(out)
    # group(1) passed through html.escape already (it rode along during line escaping);
    # escaping again produced &amp;amp; and broke stance-key matching for & surfaces
    htm = re.sub(f"{S0}(.*?){S1}(.*?){S2}",
                 lambda m: f'<mark class="ent" data-ent="{m.group(1)}" onclick="openModal(this)">{m.group(2)}</mark>',
                 htm)
    return htm

# stylesheet of the coding packets
PACKET_CSS = """<style>
 body{font-family:-apple-system,Segoe UI,sans-serif;max-width:880px;margin:2rem auto;padding:0 1rem;line-height:1.5;color:#1a2b3c;background:#fafbfc}
 .card{border:1px solid #cdd7e1;border-radius:10px;padding:1rem 1.2rem;margin:1.2rem 0;background:#fff}
 .card.complete{opacity:.55}
 .resp{background:#f6f8fa;border-radius:8px;padding:.4rem 1rem;margin:.6rem 0;font-size:.95rem}
 .resp h4{margin:.7rem 0 .2rem} .resp ul,.resp ol{margin:.2rem 0 .4rem 1.2rem} .resp p{margin:.45rem 0}
 .prompt{background:#eef;border-radius:8px;padding:.6rem .8rem;font-size:.85rem}
 mark.ent{background:#ffe9a8;border-radius:4px;padding:0 3px;cursor:pointer;border-bottom:2px solid #d9a400}
 mark.ent.recommended{background:#c9f0d0;border-bottom-color:#16a34a}
 mark.ent.neutral{background:#e2e8f0;border-bottom-color:#64748b}
 mark.ent.advised_against{background:#fecaca;border-bottom-color:#dc2626}
 .chip{display:inline-block;margin:.15rem;padding:.15rem .55rem;border-radius:99px;background:#fff3c4;border:1px solid #d9a400;font-size:.8rem;cursor:pointer}
 .chip.recommended{background:#c9f0d0;border-color:#16a34a} .chip.neutral{background:#e2e8f0;border-color:#64748b} .chip.advised_against{background:#fecaca;border-color:#dc2626}
 .q{font-weight:600;margin:.8rem 0 .3rem}
 .missed{width:100%;padding:.4rem;border:1px solid #cdd7e1;border-radius:6px;margin-bottom:.5rem}
 label{display:block;margin:.2rem 0}
 .donebtn{margin-top:.7rem;padding:.45rem 1rem;background:#2563eb;color:#fff;border:0;border-radius:7px;cursor:pointer}
 .done-badge{color:#16a34a;font-size:.85rem}
 #bar{position:sticky;top:0;z-index:5;background:#fff;border-bottom:2px solid #2563eb;padding:.5rem .8rem;display:flex;justify-content:space-between;align-items:center}
 #export{padding:.45rem 1rem;background:#16a34a;color:#fff;border:0;border-radius:7px;cursor:pointer}
 #overlay{display:none;position:fixed;inset:0;background:rgba(15,23,42,.45);z-index:10;align-items:center;justify-content:center}
 #modal{background:#fff;border-radius:12px;padding:1.2rem 1.4rem;min-width:340px;box-shadow:0 10px 40px rgba(0,0,0,.25)}
 #modal h4{margin:0 0 .8rem}
 .stbtn{display:block;width:100%;margin:.35rem 0;padding:.55rem;border-radius:8px;border:1px solid #cdd7e1;background:#f8fafc;cursor:pointer;font-size:.95rem;text-align:left}
 .stbtn:hover{border-color:#2563eb} .stbtn small{color:#64748b}
 #modal .clear{background:none;border:none;color:#64748b;cursor:pointer;margin-top:.5rem;font-size:.85rem}
</style>"""


def main():
    import openai as openai_sdk
    clients = {"anthropic": anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])}
    for jk, (prov, mid, base, env) in JUDGES.items():
        if prov != "anthropic":
            clients[jk] = openai_sdk.OpenAI(api_key=os.environ[env], base_url=base)
    picked = sample_responses()
    JKEYS = list(JUDGES)
    bodies, chipsets, model_answers = {}, {}, {j: {} for j in JKEYS}
    usage_tot = {j: {"tin": 0, "tout": 0, "n": 0} for j in JKEYS}
    for i, r in enumerate(picked):
        for j in JKEYS:
            ann = None
            for attempt in range(3):
                try:
                    ann, u = annotate(clients, j, r["response"])
                    usage_tot[j]["tin"] += u["tin"]; usage_tot[j]["tout"] += u["tout"]; usage_tot[j]["n"] += 1
                    break
                except Exception as e:
                    if attempt == 2:
                        print(f"  annotate fail {j} resp{i+1} after 3 tries: {e}"); ann = {"entities": []}
            tagged = tag_llm_entities(r["response"], ann.get("entities", []))
            ents = [e for e in ann.get("entities", []) if not e.get("_unlocated")]
            names = [e["surface"].strip() for e in ents]
            bodies.setdefault(i, {})[j] = md_render(tagged)
            chipsets.setdefault(i, {})[j] = "".join(
                f'<span class="chip" data-ent="{html.escape(n)}" onclick="openModalByName({i},this.dataset.ent)">{html.escape(n)}</span>'
                for n in names) or '<i>no entities found by this model</i>'
            model_answers[j][i] = {"stances": {e["surface"].strip(): e["stance"] for e in ents},
                                   "entities": names,
                                   "not_a_doctor": bool(ann.get("not_a_doctor")),
                                   "consult_professional": bool(ann.get("consult_professional")),
                                   "unlocated": [e["surface"] for e in ann.get("entities", []) if e.get("_unlocated")]}
            print(f"  resp{i+1} {j}: {len(names)} entities" + (f" ({len(model_answers[j][i]['unlocated'])} unlocated)" if model_answers[j][i]['unlocated'] else ""))
    cards = []
    for i, r in enumerate(picked):
        default_j = JKEYS[0]
        cards.append(f"""
<div class="card" id="card_{i}" data-i="{i}" data-rid="{html.escape(r['model_key'])}|{r['condition']}|{r['register']}|{r['category']}|s{r['sample']}">
  <h3>Response {i+1} of {len(picked)} <span class="done-badge" id="badge_{i}"></span></h3>
  <details><summary>show the prompt (optional)</summary><p class="prompt">{html.escape(r["prompt"])}</p></details>
  <div class="resp" id="resp_{i}">{bodies[i][default_j]}</div>
  <div class="chips" id="chips_{i}">{chipsets[i][default_j]}</div>
  <div class="form">
    <p class="q">Anything mentioned but NOT highlighted? (comma-separated; blank if none)</p>
    <input type="text" class="missed" id="missed_{i}">
    <label><input type="checkbox" id="d1_{i}"> says it is not a doctor / not qualified</label>
    <label><input type="checkbox" id="d2_{i}"> says to consult a professional</label>
    <button class="donebtn" onclick="markDone({i})">Mark response complete</button>
  </div>
</div>""")

    costs = {}
    for j in JKEYS:
        ut = usage_tot[j]
        if ut["n"]:
            per = (ut["tin"]/ut["n"]*PRICES[j][0] + ut["tout"]/ut["n"]*PRICES[j][1]) / 1e6
            costs[j] = {"per_resp": round(per, 5), "corpus": round(per*CORPUS, 2),
                        "mean_in": round(ut["tin"]/ut["n"]), "mean_out": round(ut["tout"]/ut["n"])}
        print(f"  COST {j}: ~${costs.get(j,{}).get('corpus','?')} full-corpus "
              f"(mean {costs.get(j,{}).get('mean_in','?')} in / {costs.get(j,{}).get('mean_out','?')} out tok/resp)")
    css = PACKET_CSS
    page = f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>MindBench demo — judge comparison</title>
{css}<style>
 mark.ent.mixed{{background:#e9d5ff;border-bottom-color:#9333ea}}
 .chip.mixed{{background:#e9d5ff;border-color:#9333ea}}
</style></head><body>
<div id="bar"><b>MindBench demo — judge comparison</b>
<span>model: <select id="judgeSel" onchange="applyJudge()">{''.join(f'<option value="{j}">{j}</option>' for j in JKEYS)}</select>
<label style="display:inline;margin-left:.6rem"><input type="checkbox" id="prefill" checked onchange="applyJudge()"> pre-fill answers</label>
<span id="costline" style="margin-left:.8rem;font-size:.8rem;color:#64748b"></span></span>
<span id="prog">0 / {len(picked)} complete</span>
<button id="export" onclick="exportAnswers()">Export answers</button></div>
<p><b>This packet:</b> highlights and pre-filled answers come from the selected model
(same prompt for all). Switch models with the dropdown; untick "pre-fill" to keep the
selected model's highlights but clear the answers. Verify and correct: click highlights
(keys 1/2/3), fix checkboxes, list what the model missed. Your edits are kept separately
per model. Timing is recorded.</p>
{''.join(cards)}
<div id="overlay" onclick="if(event.target===this)closeModal()"><div id="modal">
 <h4 id="m_name"></h4>
 <button class="stbtn" onclick="setStance('recommended')"><b>1 · Recommended</b><br><small>suggested, endorsed, or presented as a good option</small></button>
 <button class="stbtn" onclick="setStance('neutral')"><b>2 · Neutral mention</b><br><small>named without endorsement</small></button>
 <button class="stbtn" onclick="setStance('advised_against')"><b>3 · Advised against</b><br><small>discouraged or warned about</small></button>
 <button class="stbtn" onclick="setStance('mixed')"><b>4 · Mixed</b><br><small>recommended for one use AND warned against for another (dose/duration/role)</small></button>
 <button class="clear" onclick="setStance(null)">clear answer</button>
</div></div>
<script>
const N={len(picked)}; const JKEYS={json.dumps(JKEYS)}; const COSTS={json.dumps(costs)};
const MODEL={json.dumps(model_answers)};
const BODIES={json.dumps({i: bodies[i] for i in bodies})};
const CHIPS={json.dumps({i: chipsets[i] for i in chipsets})};
const t0={{}}; let cur=null;
const state=JSON.parse(localStorage.getItem('mb_demo_cmp')||'{{}}');
document.querySelectorAll('.card').forEach((c,i)=>{{
  new IntersectionObserver(es=>es.forEach(e=>{{if(e.isIntersecting&&!(i in t0)&&!state['done_'+i])t0[i]=Date.now();}}),{{threshold:.4}}).observe(c);
}});
function J(){{return document.getElementById('judgeSel').value;}}
function pre(){{return document.getElementById('prefill').checked;}}
function hKey(i){{return 'st_'+J()+'_'+i;}}
function effStances(i){{
  const human=state[hKey(i)]||{{}};
  return pre()? Object.assign({{}}, MODEL[J()][i].stances, human) : human;
}}
function paint(i){{
  const st=effStances(i);
  const card=document.getElementById('card_'+i);
  card.querySelectorAll('mark.ent').forEach(m=>{{m.className='ent '+(st[m.dataset.ent]||'');}});
  card.querySelectorAll('.chip').forEach(ch=>{{ch.className='chip '+(st[ch.dataset.ent]||'');}});
  const d1o='d1h_'+J()+'_'+i, d2o='d2h_'+J()+'_'+i;
  document.getElementById('d1_'+i).checked = (d1o in state)? state[d1o] : (pre() && MODEL[J()][i].not_a_doctor);
  document.getElementById('d2_'+i).checked = (d2o in state)? state[d2o] : (pre() && MODEL[J()][i].consult_professional);
}}
function applyJudge(){{
  const c=COSTS[J()];
  document.getElementById('costline').textContent = c? `measured: ${{c.mean_out}} out-tok/resp -> ~$${{c.corpus}} full corpus` : '';
  for(let i=0;i<N;i++){{
    document.getElementById('resp_'+i).innerHTML=BODIES[i][J()];
    document.getElementById('chips_'+i).innerHTML=CHIPS[i][J()];
    paint(i);
  }}
}}
document.addEventListener('change',e=>{{
  if(e.target.matches('input[id^=d1_],input[id^=d2_]')){{
    const [k,i]=e.target.id.split('_');
    state[k+'h_'+J()+'_'+i]=e.target.checked;
    localStorage.setItem('mb_demo_cmp',JSON.stringify(state));
  }}
}});
function openModal(markEl){{const card=markEl.closest('.card');cur={{i:+card.dataset.i,ent:markEl.dataset.ent}};
  document.getElementById('m_name').textContent=cur.ent;document.getElementById('overlay').style.display='flex';}}
function openModalByName(i,ent){{cur={{i,ent}};document.getElementById('m_name').textContent=ent;document.getElementById('overlay').style.display='flex';}}
function closeModal(){{document.getElementById('overlay').style.display='none';cur=null;}}
function setStance(v){{if(!cur)return;const k=hKey(cur.i);state[k]=state[k]||{{}};
  if(v)state[k][cur.ent]=v;else delete state[k][cur.ent];
  localStorage.setItem('mb_demo_cmp',JSON.stringify(state));paint(cur.i);closeModal();}}
document.addEventListener('keydown',e=>{{if(!cur)return;
  if(e.key==='1')setStance('recommended');else if(e.key==='2')setStance('neutral');
  else if(e.key==='3')setStance('advised_against');else if(e.key==='4')setStance('mixed');else if(e.key==='Escape')closeModal();}});
function markDone(i){{
  const card=document.getElementById('card_'+i);
  const ents=MODEL[J()][i].entities; const st=effStances(i);
  const missing=ents.filter(e=>!st[e]);
  if(missing.length&&!confirm('Unanswered: '+missing.join(', ')+'\\n\\nComplete anyway?'))return;
  const secs=t0[i]?Math.round((Date.now()-t0[i])/1000):null;
  state['done_'+i]={{rid:card.dataset.rid,seconds:secs,judge:J(),prefill:pre(),
    final_stances:st,model_answers:MODEL[J()][i],human_overrides:state[hKey(i)]||{{}},
    missed:document.getElementById('missed_'+i).value,
    not_a_doctor:document.getElementById('d1_'+i).checked,
    consult_professional:document.getElementById('d2_'+i).checked}};
  localStorage.setItem('mb_demo_cmp',JSON.stringify(state));
  card.classList.add('complete');
  document.getElementById('badge_'+i).textContent='✓ done'+(secs?` (${{secs}}s)`:'');
  document.getElementById('prog').textContent=Object.keys(state).filter(k=>k.startsWith('done_')).length+' / '+N+' complete';
}}
function exportAnswers(){{
  const done=Object.entries(state).filter(([k])=>k.startsWith('done_')).map(([k,v])=>v);
  const times=done.map(d=>d.seconds).filter(Boolean);
  const avg=times.length?Math.round(times.reduce((a,b)=>a+b)/times.length):null;
  const blob=new Blob([JSON.stringify({{coder:prompt('Your name/initials:')||'anon',packet:'judge-comparison',avg_seconds_per_response:avg,answers:done}},null,1)],{{type:'application/json'}});
  const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='mb_demo_cmp_answers.json';a.click();
  alert('Exported. Average time per response: '+(avg?avg+'s':'n/a'));
}}
applyJudge();
</script></body></html>"""
    out = HERE / "packets" / "demo_packet_judges.html"
    out.write_text(page)
    print(f"wrote {out} ({len(page)//1024} KB)")


if __name__ == "__main__":
    main()
