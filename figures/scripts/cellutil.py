"""Shared labels and loaders for the per-cell figures (data/cells.json)."""

import json
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
OUT = os.path.join(ROOT, "figures", "out")
CELLS_PATH = os.path.join(ROOT, "data", "cells.json")

LISTS = ["pharmacological", "psychotherapy", "neuromodulation", "digital"]
LIST_LABEL = {
    "pharmacological": "Pharmacological",
    "psychotherapy": "Psychotherapy",
    "neuromodulation": "Procedures",
    "digital": "Digital",
}

CONTROLS = ["situational_stress_subclinical", "uncomplicated_bereavement"]

# The 41 generation models, grouped by developer and in release order within a developer
# (oldest first); the roster records no release dates, so version number is the proxy.
# Same order as disclosures_by_model.py.
MODEL_ORDER = ["claude-sonnet-4.5", "claude-haiku-4.5", "claude-opus-4.5", "claude-opus-4.6",
               "claude-sonnet-4.6", "claude-opus-4.8", "claude-sonnet-5", "claude-opus-5",
               "claude-fable-5",
               "gpt-3.5-turbo", "gpt-4.1", "gpt-5-2025-snapshot", "gpt-5-mini-2025-snapshot",
               "gpt-5.5", "gpt-5.6-luna", "gpt-5.6-terra", "gpt-5.6-sol",
               "gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-3.5-flash",
               "gemini-3.5-flash-lite", "gemini-3.6-flash", "gemini-3.7-flash",
               "grok-4.3", "grok-4.5", "grok-4.6",
               "mistral-large-3", "ministral-3-8b", "ministral-3-14b", "mistral-small-4",
               "mistral-medium-3.5",
               "qwen3-8b", "qwen-flash", "qwen-3.5-plus", "qwen-3.7-plus", "qwen-3.8-max",
               "deepseek-v4-flash", "deepseek-v4-pro",
               "kimi-k2.6", "kimi-k3"]

COND_DISPLAY = {
    "adhd_adult": "ADHD (adult)",
    "alcohol_use_disorder": "Alcohol use disorder",
    "anorexia_nervosa": "Anorexia nervosa",
    "autism_spectrum_disorder": "Autism spectrum",
    "binge_eating_disorder": "Binge-eating disorder",
    "bipolar_1_maintenance": "Bipolar I (maint.)",
    "borderline_personality_disorder": "Borderline PD",
    "bulimia_nervosa": "Bulimia nervosa",
    "chronic_insomnia": "Chronic insomnia",
    "generalized_anxiety_disorder": "GAD",
    "major_depressive_disorder": "MDD",
    "narcissistic_personality_disorder": "Narcissistic PD",
    "ocd": "OCD",
    "panic_disorder": "Panic disorder",
    "ptsd": "PTSD",
    "schizophrenia_first_episode": "Schizophrenia (1st ep.)",
    "situational_stress_subclinical": "Subclinical stress",
    "social_anxiety_disorder": "Social anxiety",
    "uncomplicated_bereavement": "Bereavement",
}

# short display names for options; anything not listed is used as-is and truncated
OPT_SHORT = {
    "eye movement desensitization and reprocessing": "EMDR",
    "repetitive transcranial magnetic stimulation": "rTMS",
    "interpersonal and social rhythm therapy": "IPSRT",
    "enhanced cognitive behavioral therapy": "CBT-E",
    "transference-focused psychotherapy": "TFP",
    "acceptance and commitment therapy": "ACT",
    "exposure and response prevention": "ERP",
    "cognitive behavioral therapy": "CBT",
    "cognitive processing therapy": "CPT",
    "dialectical behavior therapy": "DBT",
    "mentalization-based therapy": "MBT",
    "interpersonal therapy": "IPT",
    "cbt for insomnia": "CBT-I",
    "electroconvulsive therapy": "ECT",
    "vagus nerve stimulation": "VNS",
    "deep brain stimulation": "DBS",
    "family-based treatment": "FBT",
    "amphetamine-dextroamphetamine": "amphetamine-dex.",
    "prolonged exposure therapy": "prolonged exposure",
    "complicated grief therapy": "complicated grief",
    "motivational interviewing": "motivational int.",
    "supportive psychotherapy": "supportive therapy",
    "behavioral activation": "behav. activation",
    "family-focused therapy": "family-focused",
    "anterior capsulotomy": "capsulotomy",
    "anterior cingulotomy": "cingulotomy",
    "adhd coaching": "ADHD coaching",
    "nocd": "NOCD",
    "ptsd coach": "PTSD coach",
    "cbt-i coach": "CBT-i coach",
    "dbt coach": "DBT coach",
    "i am sober": "I am sober",
    "deep transcranial magnetic stimulation": "deep TMS",
    "transcranial direct current stimulation": "tDCS",
    "transcranial alternating current stimulation": "tACS",
    "transcranial random noise stimulation": "tRNS",
    "intermittent theta-burst stimulation": "theta-burst (iTBS)",
    "accelerated theta-burst stimulation": "accel. theta-burst",
    "transcutaneous auricular vagus nerve stimulation": "taVNS",
    "trigeminal nerve stimulation": "trigeminal nerve stim.",
    "cranial electrotherapy stimulation": "cranial electro. stim.",
    "ketamine / esketamine (clinic-administered)": "ketamine (clinic)",
    "psilocybin-assisted therapy": "psilocybin-assisted",
    "mdma-assisted therapy": "MDMA-assisted",
    "stellate ganglion block": "stellate ganglion blk.",
    "metrazol convulsive therapy": "metrazol convulsive",
    "bright light therapy": "bright light",
    "magnetic seizure therapy": "magnetic seizure ther.",
    "responsive neurostimulation": "responsive neurostim.",
    "epidural cortical stimulation": "epidural cortical stim.",
    "focused ultrasound capsulotomy": "focused US capsulotomy",
    "mindfulness-based cognitive therapy": "MBCT",
    "mindfulness-based stress reduction": "MBSR",
    "applied behavior analysis": "ABA",
    "serdexmethylphenidate-dexmethylphenidate": "serdex-dexmethylph.",
    "endoscopic thoracic sympathectomy": "endoscopic sympath.",
    "community reinforcement approach": "community reinforce.",
    "solution-focused brief therapy": "solution-focused",
    "good psychiatric management": "good psych. mgmt.",
    "relapse prevention therapy": "relapse prevention",
    "imagery rehearsal therapy": "imagery rehearsal",
    "hyperbaric oxygen therapy": "hyperbaric oxygen",
    "accelerated tms": "accelerated TMS",
    "trauma-focused cbt": "trauma-focused CBT",
    "act coach": "ACT Coach",
    "ptsd family coach": "PTSD family coach",
}


def short(name, maxlen=24):
    s = OPT_SHORT.get(name, name)
    if len(s) > maxlen:
        s = s[: maxlen - 1].rstrip() + "…"
    return s


def load_cells(path=CELLS_PATH):
    with open(path) as fh:
        return json.load(fh)


def ordered_conditions():
    """17 clinical presentations A-Z by display name, then the two controls."""
    clinical = sorted((c for c in COND_DISPLAY if c not in CONTROLS),
                      key=lambda c: COND_DISPLAY[c].lower())
    return clinical, list(CONTROLS)


def sorted_options(cell):
    """(name, option) pairs of one cell, highest named (recommended) rate first."""
    return sorted(cell.get("options", {}).items(),
                  key=lambda kv: (-kv[1]["named_rate"], kv[0]))


def fmt_pct(x):
    p = 100 * x
    if p >= 0.95:
        return "%.0f%%" % p
    if p > 0:
        return "<1%"
    return "0%"
