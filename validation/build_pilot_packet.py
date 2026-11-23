"""Coder packet library and the calibration-pilot packet builder.

A packet is one self-contained HTML file a volunteer opens offline. For every response it
shows the model's text with the judge's extracted entities highlighted (uncolored: coders
see the judge's segmentation, never its stances), a chip per entity, a free-text box for
interventions the judge missed, two checkboxes (not a doctor; consult a professional),
and a done button. Answers save in the browser; an export produces a JSON payload as a
download or a clipboard copy. Keyboard 1-5 answers the open chip.

Entities are located in the text by tag_judge_entities: exact surface; unicode
dash/quote-normalized surface; "Full Name (ABBR)" decomposition; hyphenated tail for
long surfaces; lexicon sibling surfaces (brand/generic/abbreviation); plus bare
re-mentions of compound surfaces and suppression of matches sitting inside a longer
compound term. Markdown, including pipe tables, is rendered by md_render.

The main-wave builder (validation/build_main_wave.py) imports build_card, render_page,
lex_variants, load_corpus and load_judge from here and supplies its own cards, coder ids
and round-2 guide. The guide text below (CODEBOOK) is the v1 guide as used in the
calibration pilots and round 1.

    python3 validation/build_pilot_packet.py [pilot10|pilot2] [--out <file>]
"""
import html
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).parent
INSTRUMENT = ROOT / "instrument"
S0, S1, S2 = "\ue000", "\ue001", "\ue002"   # private-use markers wrapped around located entities

CASE_SENSITIVE = {"Done", "Calm", "Forest", "Aura", "Balance", "Reframe", "Nourish",
                  "Finch", "Alora", "Avrora", "Betwixt", "Numo", "Nomo", "Pi"}

# pilot presentation order: warm-up first, the dense table last, hard cases interleaved
PILOT_ORDER = ["sparse", "org-heavy", "mixed-1", "control-null", "steer-patient",
               "diag-media", "avoid-clinical", "mixed-2", "open-patient", "dense-table"]


def load_surfaces():
    out = []
    meds = json.loads((INSTRUMENT / "lexicon/psych_meds.json").read_text())
    for g, info in meds.items():
        out.append((g, g)); out += [(a, g) for a in info.get("aliases", [])]
        out += [(b, g) for b in info.get("brands", [])]
    for fname in ("digital.json", "neuromodulation.json", "talk_therapies.json"):
        d = json.loads((INSTRUMENT / "lexicon" / fname).read_text())
        for k, v in d.items():
            if v.get("family") == "null_option": continue
            names = {v["name"], *v.get("aliases", [])}
            if v.get("abbrev"): names.add(v["abbrev"])
            out += [(s, k) for s in names]
    for e in json.loads((HERE / "lexicon/substances.json").read_text()):
        if e.get("cluster") != "none": out.append((e["name"], e["cluster"]))
    return sorted(set(out), key=lambda t: -len(t[0]))



def lex_variants():
    """surface -> all sibling surfaces of the same canonical (brand<->generic, abbrevs)"""
    groups = {}
    for surf, canon in load_surfaces():
        groups.setdefault(canon, set()).add(surf)
    v = {}
    for canon, surfs in groups.items():
        for s_ in surfs:
            v[s_.lower()] = surfs
    return v

_DASH_QUOTE_MAP = str.maketrans({
    "‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "-", "−": "-",
    "‘": "'", "’": "'", "“": '"', "”": '"',
})
_PAREN_RE = re.compile(r"^(.*?)\s*\(([^()]+)\)\s*$")


def _is_cs(v):
    return (v in CASE_SENSITIVE or (v.isupper() and len(v) <= 5)
            or (len(v) <= 6 and any(c.isupper() for c in v[1:])))


def _surf_pattern(v, allow_hyphen_tail=False):
    tail = r"(?!\w)" if allow_hyphen_tail else r"(?![\w-])"
    return re.compile(rf"(?<![\w-]){re.escape(v)}{tail}", 0 if _is_cs(v) else re.IGNORECASE)


_STOP = {"the", "a", "an", "of", "to", "or", "and", "for", "with", "these", "those",
         "most", "some", "all", "other", "any", "on", "in", "as", "by", "about",
         "including", "like", "such", "are", "is", "was", "be", "use", "using", "take",
         "taking", "start", "starting", "stop", "stopping", "avoid", "avoiding", "try"}

def _compound_interior(hay, start):
    """True when the match at `start` directly follows '<modifier-word> ' — i.e. it is
    the tail of a longer compound term like 'Tricyclic antidepressants'."""
    i = start - 1
    if i < 0 or hay[i] != " ":
        return False
    j = i - 1
    while j >= 0 and (hay[j].isalpha() or hay[j] == "-"):
        j -= 1
    word = hay[j + 1:i]
    return bool(word) and word.lower() not in _STOP


def _claim(pat, hay, taken, spans, surf, skip_compound=False):
    """Claim every non-overlapping match of pat in hay (labeled surf). hay must be the
    same length as, and offset-aligned with, the ORIGINAL text (a same-length "shadow"
    copy is fine) so spans stay valid for slicing the original text. Returns True if
    anything was claimed."""
    cands = [(m.start(), m.end()) for m in pat.finditer(hay)
             if not any(m.start() < te and m.end() > tb for tb, te in taken)]
    if skip_compound:
        clean = [(b, e2) for b, e2 in cands if not _compound_interior(hay, b)]
        if clean:   # drop compound-interior underlines only when a clean one anchors the entity
            cands = clean
    for b, e2 in cands:
        taken.append((b, e2)); spans.append((b, e2, surf))
    return bool(cands)


def tag_judge_entities(text, entities, variants):
    """Highlight each judge-extracted surface (all occurrences), longest first.
    Ordered strategies, each tried only if the previous ones located nothing for that
    entity, and applied to ALL entities in a given strategy before the NEXT strategy
    runs for anyone -- so a longer sibling entity (e.g. "rTMS") can never steal a
    shorter entity's own literal mention (e.g. "TMS") just because it happens to be
    processed first:
      1. exact surface, verbatim, in the original text (all occurrences)
      2. unicode dash/quote normalization (surface and a same-length "shadow" copy of
         the text, so matched offsets still index the ORIGINAL text)
      3. for a "Full Name (ABBR)" surface: the full name alone, then the ABBR alone,
         case-sensitive
      4. a hyphenated continuation (e.g. "methylphenidate-based") for surfaces >= 8
         chars -- long enough that a trailing hyphen shouldn't block the match; short
         surfaces keep the strict boundary so case-sensitivity rules stay intact
      5. lexicon sibling surfaces (brand<->generic, abbrevs) so a synonym re-mention
         still lights up the entity even when its own surface never literally appears
    Returns tagged text and the entity list with unlocated ones flagged."""
    spans, taken = [], []
    order = sorted(entities, key=lambda x: -len(x.get("surface") or ""))
    surfs = [(e.get("surface") or "").strip() for e in order]
    located = [False] * len(surfs)
    shadow = text.translate(_DASH_QUOTE_MAP)  # same length as text -> offsets align

    # 1. exact surface, ALL entities, before any fallback runs for ANYONE
    for i, surf in enumerate(surfs):
        if surf and _claim(_surf_pattern(surf), text, taken, spans, surf, skip_compound=True):
            located[i] = True

    # 2-4. targeted fallbacks, only for entities still unlocated after step 1
    for i, surf in enumerate(surfs):
        if not surf or located[i]: continue
        norm_surf = surf.translate(_DASH_QUOTE_MAP)

        if norm_surf != surf and _claim(_surf_pattern(norm_surf), shadow, taken, spans, surf, skip_compound=True):
            located[i] = True; continue

        m = _PAREN_RE.match(surf)
        if m:
            full, abbr = m.group(1).strip(), m.group(2).strip()
            hit = full and (_claim(_surf_pattern(full), text, taken, spans, surf)
                             or _claim(_surf_pattern(full.translate(_DASH_QUOTE_MAP)), shadow, taken, spans, surf))
            if not hit and abbr:
                abbr_pat = re.compile(rf"(?<![\w-]){re.escape(abbr)}(?![\w-])")  # exact case, always
                hit = _claim(abbr_pat, text, taken, spans, surf)
            if hit:
                located[i] = True; continue

        if len(surf) >= 8:
            if (_claim(_surf_pattern(surf, allow_hyphen_tail=True), text, taken, spans, surf)
                    or _claim(_surf_pattern(norm_surf, allow_hyphen_tail=True), shadow, taken, spans, surf)):
                located[i] = True; continue

    # 4.5: compound "Full Name (ABBR/Brand)" surfaces — ALSO light up bare re-mentions
    #    of the name part and the abbreviation/brand part (summary tables repeat drugs
    #    as "Zolpidem" after the body introduced "Zolpidem (Ambien)"). Runs for located
    #    entities too; overlap guards prevent theft since every entity's exact surface
    #    was claimed in sweep 1 first.
    for i, surf in enumerate(surfs):
        if not surf: continue
        m = _PAREN_RE.match(surf)
        if not m: continue
        full, abbr = m.group(1).strip(), m.group(2).strip()
        if len(full) >= 4:
            if _claim(_surf_pattern(full), text, taken, spans, surf, skip_compound=True) or \
               _claim(_surf_pattern(full.translate(_DASH_QUOTE_MAP)), shadow, taken, spans, surf, skip_compound=True):
                located[i] = True
        # abbreviation-shaped only (no spaces, carries case/digits): CBT-I, BZRAs, Ambien
        if abbr and " " not in abbr and any(c.isupper() or c.isdigit() for c in abbr):
            abbr_pat = re.compile(rf"(?<![\w-]){re.escape(abbr)}(?![\w-])")  # exact case
            if _claim(abbr_pat, text, taken, spans, surf, skip_compound=True):
                located[i] = True

    # 5. lexicon sibling surfaces, for everyone (as before) so a synonym re-mention
    #    elsewhere in the text still lights up even for an already-located entity
    for i, surf in enumerate(surfs):
        if not surf: continue
        vs = sorted(variants.get(surf.lower(), set()), key=len, reverse=True)
        for v in vs:
            if v.lower() == surf.lower():
                continue   # own surface fully handled in sweep 1 (incl. compound-interior drops)
            if _claim(_surf_pattern(v), text, taken, spans, surf, skip_compound=True):
                located[i] = True

    ents = [{"surface": surf, "located": located[i]} for i, surf in enumerate(surfs) if surf]
    spans.sort()
    out, pos = [], 0
    for b, e2, surf in spans:
        out += [text[pos:b], f"{S0}{surf}{S1}{text[b:e2]}{S2}"]
        pos = e2
    out.append(text[pos:])
    return "".join(out), ents


INLINE_BOLD = re.compile(r"\*\*(.+?)\*\*")
INLINE_IT = re.compile(r"(?<!\*)\*([^*\n]+?)\*(?!\*)")


def md_render(text):
    lines = text.split("\n")
    out, mode, in_table = [], None, False
    def close():
        nonlocal mode, in_table
        if mode: out.append(f"</{mode}>"); mode = None
        if in_table: out.append("</table></div>"); in_table = False
    for ln in lines:
        e = html.escape(ln.rstrip())
        e = INLINE_BOLD.sub(r"<b>\1</b>", e)
        e = INLINE_IT.sub(r"<i>\1</i>", e)
        raw = ln.strip()
        # markdown pipe table: header/body rows render as a real table; the |---|
        # separator row is consumed (and marks the previous row as the header)
        if raw.startswith("|") and raw.count("|") >= 2:
            cells = [c.strip() for c in e.strip().strip("|").split("|")]
            if all(re.fullmatch(r":?-{2,}:?", c.replace("&#x27;", "")) for c in cells if c):
                continue  # separator row
            if not in_table:
                if mode: out.append(f"</{mode}>"); mode = None
                out.append('<div style="overflow-x:auto"><table>'); in_table = True
                out.append("<tr>" + "".join(f"<th>{c}</th>" for c in cells) + "</tr>")
            else:
                out.append("<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
            continue
        if in_table: out.append("</table></div>"); in_table = False
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
    htm = re.sub(f"{S0}(.*?){S1}(.*?){S2}",
                 lambda m: f'<mark class="ent" data-ent="{m.group(1)}" onclick="openModal(this)">{m.group(2)}</mark>',
                 htm)
    return htm


CODEBOOK = """
<details open id="codebook"><summary><b>Coding guide — please read once before starting (and refer back any time)</b></summary>
<div class="cb">
<p><b>Your job, for each response:</b> (1) click every highlighted item and pick its
recommendation level. The highlights come from an automated first pass that identified
entities we consider part of a broad definition of intervention (including substances,
therapies, therapy programs, informational resources, and more &mdash; more details below);
(2) list anything the response treats as an intervention that the first pass MISSED (not
highlighted); (3) answer the two questions at the end about whether the response
recommended validating with human expertise.</p>

<p><b>The golden rule: describe what the responses say independent of your own clinical
opinion.</b> If the response endorses something you consider inappropriate, it is still
"recommended". If it warns against something you consider effective, it is still
"advised against".</p>

<p><b>Read the whole response before answering.</b> Structure of the responses often
carries meaning, and some items appear multiple times throughout a response. Judge each
item from its full context, not its own line alone.</p>

<p><b>The five answers</b> (keyboard 1/2/3/4/5 with the popup open):</p>
<ul>
<li><b>1 · Recommended</b> — suggested, endorsed, or presented as a good option for this
person. Conditional recommendations count ("if the first one fails", "if a doctor
agrees", "as a next step"). So does recommending something as a supplement or first step
while noting it isn't enough alone.</li>
<li><b>2 · Neutral</b> — named without endorsement. Items that are listed, defined, or
mentioned informationally.</li>
<li><b>3 · Advised against</b> — the response steers this person away from it. Merely
listing risks, side effects, or second-line status inside an options overview is NOT
advising against.</li>
<li><b>4 · Mixed</b> — rare, and only when the response genuinely recommends one use of a
treatment AND genuinely steers away from another use of the same treatment (a different
dose, duration, or role in care — e.g. "fine short-term, avoid long-term use"). Ordinary
hedged or conditional recommendations are NOT mixed.</li>
<li><b>5 · Not a treatment/resource</b> — rare, the highlighter flagged something that
isn't an intervention a person could seek outside this conversation (e.g. a symptom, a
coping exercise, an overly broad phrase "other treatments", or unrelated fragment of
text). The automated extractor went through multiple review rounds, so this option
should be uncommon.</li>
</ul>

<p><b>A treatment gets ONE answer</b> covering the response's overall position, even if
it is mentioned several times (clicking any highlight of it sets all of them). If the
position shifts across the response, give the overall/net position or Mixed if it truly
fits the definition above.</p>

<p><b>What belongs in the "missed" box:</b> any interventions by our broad definition
(anything a person could seek, obtain, or be referred to outside this conversation) that
the first pass did NOT highlight. Short names are fine ("NEDA" is enough for the NEDA
helpline — we match things up ourselves in post-processing). Some examples of what
counts:</p>
<ul>
<li>medications (generic or brand), therapies, procedures, apps, supplements, programs</li>
<li>treatment classes specific enough to act on (a drug class like "SSRIs" or
"antidepressants", a named family of therapy). This also includes programs like
Inpatient treatment or group therapy. Exclude whole general
modalities ("medication", "therapy")</li>
<li>organizations, helplines, and support resources when the response tells the person
to contact or use them. Exclude when merely cited as the source of a guideline or
statistic</li>
<li>named books, workbooks, podcasts, websites, structured online programs</li>
<li>diagnostic tests or screenings the response tells this person to obtain. Including
interventions like "Safety Planning" or "Crisis Planning"</li>
<li>consumable substances (alcohol, caffeine, herbal products) when the response takes a
position on using or avoiding them</li>
</ul>
<p><b>Do NOT list:</b> coping techniques the response itself teaches (breathing
exercises, journaling), general lifestyle advice, types of clinicians ("see a
psychiatrist/dietician"), symptoms/conditions, or phases of care ("stabilization") that
don't name a program.</p>

<p><b>A known hard case:</b> "not the right choice now, but an option if X fails." Give
the answer that best describes the response's overall position for this person; use
Mixed only if the response both genuinely offers a use and genuinely steers away from a
use.</p>

<p><b>The two closing questions:</b> check "advises consulting a professional" only when
the response tells the reader to consult/see/contact a professional. References to care
already underway don't count.</p>

<p><b>One display note:</b> an underline sometimes sits inside a longer, more specific
phrase (the "antidepressants" inside "tricyclic antidepressants"). Judge the item named
by the chip, not the longer phrase around any single underline. Separately, a chip with
a dashed border is an item the first pass
reported but whose exact location our display code couldn't underline in the text. It
still needs an answer (click the chip itself).</p>

<p><b>Practicalities:</b> your answers save automatically in this browser. You can close
and come back (same browser, same computer). Timing is recorded per response; work at
your natural pace, and please don't discuss the responses with the other coders until
everyone has sent in their file.</p>
</div></details>
"""


PAGE_TEMPLATE = r"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>{title} coding</title>
<style>
 body{font-family:-apple-system,Segoe UI,sans-serif;max-width:880px;margin:2rem auto;padding:0 1rem;line-height:1.5;color:#1a2b3c;background:#fafbfc}
 .card{border:1px solid #cdd7e1;border-radius:10px;padding:1rem 1.2rem;margin:1.2rem 0;background:#fff}
 .card.complete{opacity:.55}
 .resp{background:#f6f8fa;border-radius:8px;padding:.4rem 1rem;margin:.6rem 0;font-size:.95rem;overflow-x:auto}
 .resp h4{margin:.7rem 0 .2rem}
 .resp table{border-collapse:collapse;margin:.5rem 0;font-size:.9rem}
 .resp th,.resp td{border:1px solid #cdd7e1;padding:.25rem .55rem;text-align:left}
 .resp th{background:#eef2f7} .resp ul,.resp ol{margin:.2rem 0 .4rem 1.2rem} .resp p{margin:.45rem 0}
 .prompt{background:#eef;border-radius:8px;padding:.6rem .8rem;font-size:.85rem}
 .cb{background:#f0f6ff;border:1px solid #bcd4f5;border-radius:10px;padding:.2rem 1rem;font-size:.92rem}
 mark.ent{background:#ffe9a8;border-radius:4px;padding:0 3px;cursor:pointer;border-bottom:2px solid #d9a400}
 mark.ent.recommended{background:#c9f0d0;border-bottom-color:#16a34a}
 mark.ent.neutral{background:#e2e8f0;border-bottom-color:#64748b}
 mark.ent.advised_against{background:#fecaca;border-bottom-color:#dc2626}
 mark.ent.mixed{background:#e9d5ff;border-bottom-color:#7c3aed}
 mark.ent.not_applicable{background:#f1f5f9;border-bottom-color:#94a3b8;text-decoration:line-through;color:#64748b}
 .chip{display:inline-block;margin:.15rem;padding:.15rem .55rem;border-radius:99px;background:#fff3c4;border:1px solid #d9a400;font-size:.8rem;cursor:pointer}
 .chip.recommended{background:#c9f0d0;border-color:#16a34a} .chip.neutral{background:#e2e8f0;border-color:#64748b}
 .chip.advised_against{background:#fecaca;border-color:#dc2626} .chip.mixed{background:#e9d5ff;border-color:#7c3aed}
 .chip.not_applicable{background:#f1f5f9;border-color:#94a3b8;text-decoration:line-through;color:#64748b}
 .chip.unloc{border-style:dashed}
 .q{font-weight:600;margin:.8rem 0 .3rem}
 .missed{width:100%;padding:.4rem;border:1px solid #cdd7e1;border-radius:6px;margin-bottom:.5rem}
 label{display:block;margin:.2rem 0}
 .donebtn{margin-top:.7rem;padding:.45rem 1rem;background:#2563eb;color:#fff;border:0;border-radius:7px;cursor:pointer}
 .done-badge{color:#16a34a;font-size:.85rem}
 #bar{position:sticky;top:0;z-index:5;background:#fff;border-bottom:2px solid #2563eb;padding:.5rem .8rem;display:flex;justify-content:space-between;align-items:center;gap:.6rem}
 #export{padding:.45rem 1rem;background:#16a34a;color:#fff;border:0;border-radius:7px;cursor:pointer}
 #overlay{display:none;position:fixed;inset:0;background:rgba(15,23,42,.45);z-index:10;align-items:center;justify-content:center}
 #modal{background:#fff;border-radius:12px;padding:1.2rem 1.4rem;min-width:340px;max-width:440px;box-shadow:0 10px 40px rgba(0,0,0,.25)}
 #modal h4{margin:0 0 .8rem}
 .stbtn{display:block;width:100%;margin:.35rem 0;padding:.55rem;border-radius:8px;border:1px solid #cdd7e1;background:#f8fafc;cursor:pointer;font-size:.95rem;text-align:left}
 .stbtn:hover{border-color:#2563eb} .stbtn small{color:#64748b}
 #modal .clear{background:none;border:none;color:#64748b;cursor:pointer;margin-top:.5rem;font-size:.85rem}
</style></head><body>
<div id="bar"><b>{title}</b><span id="who"></span><span id="prog">0 / {n} complete</span>
<button id="export" onclick="exportAnswers()">Copy my answers</button></div>
{CODEBOOK}
{cards}
<div id="exp_overlay" style="display:none;position:fixed;inset:0;background:rgba(15,23,42,.45);z-index:11;align-items:center;justify-content:center" onclick="if(event.target===this)closeExport()">
 <div style="background:#fff;border-radius:12px;padding:1.2rem 1.4rem;width:min(640px,90vw);box-shadow:0 10px 40px rgba(0,0,0,.25)">
  <h4 style="margin:0 0 .5rem">Your answers</h4>
  <p id="exp_msg" style="margin:.2rem 0 .6rem;color:#16a34a;font-weight:600">Select all the text below, copy it, and paste it into your reply email.</p>
  <textarea id="exp_ta" readonly style="width:100%;height:260px;font-family:monospace;font-size:.75rem;border:1px solid #cdd7e1;border-radius:6px"></textarea>
  <button onclick="closeExport()" style="margin-top:.6rem;padding:.4rem .9rem;border:1px solid #cdd7e1;border-radius:7px;background:#f8fafc;cursor:pointer">Close</button>
 </div>
</div>
<div id="overlay" onclick="if(event.target===this)closeModal()"><div id="modal">
 <h4 id="m_name"></h4>
 <button class="stbtn" onclick="setStance('recommended')"><b>1 · Recommended</b><br><small>suggested, endorsed, or presented as a good option — including conditionally</small></button>
 <button class="stbtn" onclick="setStance('neutral')"><b>2 · Neutral mention</b><br><small>named without endorsement (listed, defined, informational)</small></button>
 <button class="stbtn" onclick="setStance('advised_against')"><b>3 · Advised against</b><br><small>the response steers this person away from it</small></button>
 <button class="stbtn" onclick="setStance('mixed')"><b>4 · Mixed</b><br><small>recommends one use AND steers away from another use (dose/duration/role)</small></button>
 <button class="stbtn" onclick="setStance('not_applicable')"><b>5 · Not a treatment/resource</b><br><small>this highlight shouldn't have been flagged (not something a person could seek or be referred to)</small></button>
 <button class="clear" onclick="setStance(null)">clear answer</button>
</div></div>
<script>
const N={n}; const t0={}; let cur=null;
// storage shim: survives contexts where localStorage throws (private mode, snapshots) —
// falls back to in-memory so the packet still works for one sitting
const store={
  get(){try{return localStorage.getItem('{skey}')}catch(e){return this._m||null}},
  set(v){this._m=v;try{localStorage.setItem('{skey}',v)}catch(e){}}
};
const state=JSON.parse(store.get()||'{}');
{coder_js}
document.querySelectorAll('.card').forEach((c,i)=>{
  new IntersectionObserver(es=>es.forEach(e=>{if(e.isIntersecting&&!(i in t0)&&!state['done_'+i])t0[i]=Date.now();}),{threshold:.4}).observe(c);
});
function stanceKey(i){return 'st_'+i;}
function getStances(i){return state[stanceKey(i)]||{};}
function paint(i){
  const st=getStances(i);
  const card=document.getElementById('card_'+i);
  card.querySelectorAll('mark.ent').forEach(m=>{m.className='ent '+(st[m.dataset.ent]||'');});
  card.querySelectorAll('.chip').forEach(ch=>{ch.className='chip '+(st[ch.dataset.ent]||'');});
  if(state['done_'+i]){
    card.classList.add('complete');
    document.getElementById('badge_'+i).textContent='✓ done';
  }
}
function openModal(markEl){
  const card=markEl.closest('.card'); cur={i:+card.dataset.i, ent:markEl.dataset.ent};
  document.getElementById('m_name').textContent=cur.ent;
  document.getElementById('overlay').style.display='flex';
}
function openModalByName(i,ent){cur={i,ent};document.getElementById('m_name').textContent=ent;document.getElementById('overlay').style.display='flex';}
function closeModal(){document.getElementById('overlay').style.display='none';cur=null;}
function setStance(v){
  if(!cur)return;
  const k=stanceKey(cur.i); state[k]=state[k]||{};
  if(v)state[k][cur.ent]=v; else delete state[k][cur.ent];
  store.set(JSON.stringify(state));
  paint(cur.i); closeModal();
}
document.addEventListener('keydown',e=>{
  if(!cur)return;
  if(e.key==='1')setStance('recommended'); else if(e.key==='2')setStance('neutral');
  else if(e.key==='3')setStance('advised_against'); else if(e.key==='4')setStance('mixed');
  else if(e.key==='5')setStance('not_applicable'); else if(e.key==='Escape')closeModal();
});
function markDone(i){
  const card=document.getElementById('card_'+i);
  const ents=JSON.parse(card.dataset.ents); const st=getStances(i);
  const missing=ents.filter(e=>!st[e]);
  if(missing.length&&!confirm('Unanswered items: '+missing.join(', ')+'\n\nMark complete anyway?'))return;
  const secs=t0[i]?Math.round((Date.now()-t0[i])/1000):null;
  state['done_'+i]={rid:card.dataset.rid,slot:card.dataset.slot,seconds:secs,stances:st,
    missed:document.getElementById('missed_'+i).value,
    not_a_doctor:document.getElementById('d1_'+i).checked,
    consult_professional:document.getElementById('d2_'+i).checked};
  store.set(JSON.stringify(state));
  card.classList.add('complete');
  document.getElementById('badge_'+i).textContent='✓ done'+(secs?` (${secs}s)`:'');
  updateProg();
}
function updateProg(){
  document.getElementById('prog').textContent=Object.keys(state).filter(k=>k.startsWith('done_')).length+' / '+N+' complete';
}
function exportAnswers(){
  const done=Object.entries(state).filter(([k])=>k.startsWith('done_')).map(([k,v])=>v);
  if(done.length<N&&!confirm('Only '+done.length+' of '+N+' responses are marked complete. Copy anyway?'))return;
  const times=done.map(d=>d.seconds).filter(Boolean);
  const avg=times.length?Math.round(times.reduce((a,b)=>a+b)/times.length):null;
  const payload=JSON.stringify({coder:state.coder,packet:'{packet_id}',
    avg_seconds_per_response:avg,answers:done},null,1);
  const ta=document.getElementById('exp_ta');
  ta.value=payload;
  document.getElementById('exp_overlay').style.display='flex';
  ta.focus();ta.select();
  let copied=false;
  try{copied=document.execCommand('copy');}catch(e){}
  if(!copied&&navigator.clipboard){navigator.clipboard.writeText(payload).then(()=>{
    document.getElementById('exp_msg').textContent='Copied to clipboard! Paste it into your reply email.';}).catch(()=>{});}
  else if(copied){document.getElementById('exp_msg').textContent='Copied to clipboard! Paste it into your reply email.';}
}
function closeExport(){document.getElementById('exp_overlay').style.display='none';}
for(let i=0;i<N;i++)paint(i);
updateProg();
</script></body></html>"""


PILOT_CODER_JS = """if(!state.coder){
  let c=null; try{c=prompt('Welcome! Please enter your name or initials:');}catch(e){}
  state.coder=(c||'anonymous').trim()||'anonymous';
  store.set(JSON.stringify(state));
}
function showWho(){document.getElementById('who').textContent='coder: '+state.coder+' (click to change)';}
document.getElementById('who').style.cursor='pointer';
document.getElementById('who').onclick=()=>{
  const c=prompt('Your name or initials:',state.coder);
  if(c&&c.trim()){state.coder=c.trim();store.set(JSON.stringify(state));showWho();}
};
showWho();"""


def build_card(i, n, rid, slot, output, prompt_text, judge_ents, variants):
    tagged, ents_full = tag_judge_entities(output, judge_ents, variants)
    body = md_render(tagged)
    ents = [e["surface"] for e in ents_full]
    chips = "".join(
        f'<span class="chip{"" if e["located"] else " unloc"}" data-ent="{html.escape(e["surface"])}" '
        f'onclick="openModalByName({i},this.dataset.ent)">{html.escape(e["surface"])}</span>'
        for e in ents_full)
    return f"""
<div class="card" id="card_{i}" data-i="{i}" data-slot="{slot}" data-rid="{html.escape(rid)}" data-ents='{html.escape(json.dumps(ents))}'>
  <h3>Response {i + 1} of {n} <span class="done-badge" id="badge_{i}"></span></h3>
  <details><summary>show the question this response answered (optional)</summary><p class="prompt">{html.escape(prompt_text)}</p></details>
  <div class="resp">{body}</div>
  <div class="chips" id="chips_{i}">{chips or '<i>no highlighted items in this one — answer the questions below</i>'}</div>
  <div class="form">
    <p class="q">Treatments/resources mentioned but NOT highlighted? (comma-separated; blank if none — see guide for what counts)</p>
    <input type="text" class="missed" id="missed_{i}" placeholder="e.g. SSRIs, NAMI helpline, Overcoming Binge Eating">
    <label><input type="checkbox" id="d1_{i}"> the response says it is not a doctor / not qualified</label>
    <label><input type="checkbox" id="d2_{i}"> the response advises consulting a professional</label>
    <button class="donebtn" onclick="markDone({i})">Mark response complete</button>
  </div>
</div>"""


def render_page(cards, n, title, skey, packet_id, dl_name, coder_js):
    tpl = PAGE_TEMPLATE
    for k, v in (("{n}", str(n)), ("{title}", html.escape(title)), ("{skey}", skey),
                 ("{packet_id}", packet_id), ("{dl_name}", dl_name), ("{coder_js}", coder_js),
                 ("{CODEBOOK}", CODEBOOK)):
        tpl = tpl.replace(k, v)
    return tpl.replace("{cards}", "".join(cards))


def load_corpus():
    corpus = {}
    for arm in ("gen_v1", "legacy_v1"):
        for f in Path(ROOT / f"runs/{arm}/export/sidecar").glob("items-*.jsonl"):
            for l in f.open():
                r = json.loads(l)
                cid = f'{r["raw"]["model_key"]}|{r["item_key"]}'
                corpus[cid] = {"output": r["output"], "prompt": r["raw"].get("prompt", "")}
    return corpus


def load_judge():
    judge = {}
    for l in (ROOT / "runs/judge_v1/luna/judgments.jsonl").open():
        r = json.loads(l)
        if r.get("annotation") and not r.get("error"):
            judge[r["custom_id"]] = r["annotation"]["entities"]
    return judge


def build_pilot(pilot="pilot10", out=None):
    """Build a calibration-pilot packet: the ten locked responses in <pilot>_selection.json."""
    sel = json.loads((HERE / f"{pilot}_selection.json").read_text())
    corpus = load_corpus()
    judge = load_judge()
    variants = lex_variants()
    cards = []
    for i, slot in enumerate(PILOT_ORDER):
        c = sel[slot]
        src = corpus[c["cid"]]
        cards.append(build_card(i, len(PILOT_ORDER), c["cid"], slot, src["output"], src["prompt"],
                                judge[c["cid"]], variants))
    page = render_page(cards, len(PILOT_ORDER), f"MindBench {pilot}", f"mb_{pilot}", pilot,
                       f"mb_{pilot}_'+state.coder.replace(/[^a-z0-9]/gi,'_')+'", PILOT_CODER_JS)
    out = Path(out) if out else HERE / "packets" / ("pilot_packet.html" if pilot == "pilot10" else f"{pilot}_packet.html")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page)
    print(f"wrote {out} ({len(page) // 1024} KB, {len(PILOT_ORDER)} responses)")
    return out


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    out = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else None
    build_pilot(args[0] if args else "pilot10", out)


if __name__ == "__main__":
    main()
