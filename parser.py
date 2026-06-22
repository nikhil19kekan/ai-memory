"""
Deterministic grammar parser using spaCy.
Extracts structured atoms from English sentences.
LLM is NOT called here — unknown patterns are flagged, not guessed.
"""

import warnings
warnings.filterwarnings("ignore", message=".*dtype.*align.*")  # lemminflect NumPy compat noise
import spacy
import lemminflect  # registers token._.lemma(); also used directly as lemminflect.getLemma()
from spacy.language import Language
from datetime import datetime
from lookups import (
    VERB_EDGE_MAP, STATE_ADJECTIVES,
    QUANTITY_NER_LABELS, IMPLIED_OBJECTS, POSSESSIVE_PREDICATES,
)
from learned_rules import get_rules as _get_learned_rules

# suppress "entity_ruler has no patterns yet" warning that fires before
# any person names have been registered via register_person()
warnings.filterwarnings("ignore", message=".*W036.*")

nlp = spacy.load("en_core_web_lg")

# ── EntityRuler ──────────────────────────────────────────────────
# Placed before ner so known names always get correct PERSON label.
# graph_manager calls register_person() as new people are learned.
_ruler = nlp.add_pipe("entity_ruler", before="ner", config={"overwrite_ents": True})


def register_person(name: str):
    """Register a name as PERSON so NER is consistent across all future sentences."""
    _ruler.add_patterns([{"label": "PERSON", "pattern": name}])
    _ruler.add_patterns([{"label": "PERSON", "pattern": name.lower()}])


# ── post-parse fixer component ───────────────────────────────────
# Runs last (after NER so doc.ents is fully populated).
# Corrects spaCy mis-parse patterns caused by unusual proper nouns.

VERB_PENN_TAGS = {"VBZ", "VB", "VBP", "VBD", "VBG", "VBN"}
NOUN_PENN_TAGS = {"NN", "NNS"}


@Language.component("kgraph_post_parse_fixer")
def kgraph_post_parse_fixer(doc):
    root = next((t for t in doc if t.dep_ == "ROOT"), None)
    if root is None:
        return doc

    # Pattern A: ROOT is NOUN + has a VERB compound/amod child in our verb table
    # Example: "Nikhil dislikes coffee"  → coffee=ROOT, dislikes=compound(VERB)
    # Example: "Nikhil hated coffee last year" → coffee=ROOT, hated=amod(VBN)
    # Fix: swap so verb=ROOT, noun=dobj
    if root.tag_ in NOUN_PENN_TAGS:
        verb_compound = next(
            (c for c in root.children
             if c.dep_ in ("compound", "amod")
             and c.tag_ in VERB_PENN_TAGS
             and VERB_EDGE_MAP.get(c._.lemma())),
            None,
        )
        if verb_compound is not None:
            # Snapshot children and person indices BEFORE any mutations
            # (spaCy's children generator can be invalidated by head changes).
            old_root_children = list(root.children)
            person_indices = {
                t.i for ent in doc.ents if ent.label_ == "PERSON"
                for t in ent
            }

            root.dep_ = "dobj"
            root.head = verb_compound
            verb_compound.dep_ = "ROOT"
            verb_compound.head = verb_compound

            # Any PERSON amod child of the old root is actually the subject,
            # not a modifier. Re-tag it as nsubj of the new ROOT so that
            # get_noun_phrase(object) won't include it (e.g. "Nikhil hated coffee").
            for child in old_root_children:
                is_person_ner = child.i in person_indices
                is_likely_subject = (
                    child.dep_ == "amod"
                    and child.tag_ not in VERB_PENN_TAGS
                    and child.i < verb_compound.i
                    and len(child.text) > 1
                    and child.text[0].isupper()
                )
                if is_person_ner or is_likely_subject:
                    child.dep_ = "nsubj"
                    child.head = verb_compound
            return doc

        # Pattern C: ROOT is a mis-tagged verb (NNS instead of VBZ)
        # Example: "Nikhil lives in Mumbai" → lives=ROOT(NNS), lemma="life"
        # Fix: force verb POS so the action-verb path resolves correctly
        verb_lemmas = lemminflect.getLemma(root.text, "VERB")
        if verb_lemmas and VERB_EDGE_MAP.get(verb_lemmas[0]):
            root.tag_ = "VBZ"
            root.pos_ = "VERB"
            return doc

    # Pattern B: ROOT is past participle (VBN) + auxpass "be" child + is a state adjective
    # Example: "Nikhil is stressed" → stressed=ROOT(VBN), is=auxpass
    # Fix: swap so is=ROOT, stressed=acomp
    if root.tag_ == "VBN" and root.text.lower() in STATE_ADJECTIVES:
        be_aux = next(
            (c for c in root.children
             if c.dep_ == "auxpass" and c._.lemma() == "be"),
            None,
        )
        if be_aux is not None:
            root.dep_ = "acomp"
            root.head = be_aux
            be_aux.dep_ = "ROOT"
            be_aux.head = be_aux

    return doc


nlp.add_pipe("kgraph_post_parse_fixer", last=True)   # must run after NER so doc.ents is populated


# ── helper functions ─────────────────────────────────────────────

def get_entity_type(phrase, token, doc):
    """
    Determine entity type deterministically.
    Priority: spaCy NER > PROPN heuristic (unknown names) > CONCEPT fallback.
    NER is checked before PROPN so that Google→ORG and Japan→PLACE are
    not incorrectly overridden to PERSON by the PROPN tag.
    No world-knowledge lookups — entity types are derived from grammar/NER only.
    """
    for ent in doc.ents:
        if ent.start <= token.i < ent.end:
            if ent.label_ == "PERSON":       return "PERSON"
            if ent.label_ in ("GPE", "LOC"): return "PLACE"
            if ent.label_ == "ORG":          return "ORG"

    # PROPN fallback: unrecognized proper nouns (e.g. rare personal names)
    if token.pos_ == "PROPN":
        return "PERSON"

    return "CONCEPT"


def get_noun_phrase(token, doc):
    """
    Reconstruct full noun phrase: 'Indian food', 'software engineer'.
    Skips VERB tokens — verbs are never legitimate compound modifiers here.
    """
    modifiers = [
        c.text for c in token.children
        if c.dep_ in ("amod", "compound", "nmod")
        and c.i < token.i
        and c.pos_ != "VERB"
    ]
    return " ".join(modifiers + [token.text])


def get_attr_phrase(token):
    """
    Build the full text of an attr/acomp token for attribute storage.
    Includes all modifiers (age, nationality, etc.) from the subtree but excludes:
    - determiners ("a", "the")
    - conjunctions ("and"), cc deps, punctuation
    - other conjunct / appositive subtrees (those become separate attribute atoms)
    - relative clauses ("who loves indian food" — those become verb_relation atoms)
    """
    exclude = set()
    for child in token.children:
        if child.dep_ in ("conj", "appos", "cc", "punct", "relcl"):
            for t in child.subtree:
                exclude.add(t.i)

    parts = [
        t.text for t in sorted(token.subtree, key=lambda x: x.i)
        if t.i not in exclude and t.dep_ != "det"
    ]
    return " ".join(parts).strip()


def _collect_conjuncts(token):
    """Return token plus all conj/appos descendants (any depth).
    Handles both coordination ("A and B") and apposition ("A, B") patterns.
    """
    result = [token]
    queue  = list(token.children)
    while queue:
        child = queue.pop(0)
        if child.dep_ in ("conj", "appos"):
            result.append(child)
            queue.extend(child.children)
    return result


def _subtree_text(token):
    return " ".join(t.text for t in sorted(token.subtree, key=lambda t: t.i))


def is_quantity_token(token, doc):
    """True if token is part of a CARDINAL/QUANTITY/TIME NER entity."""
    return any(
        ent.start <= token.i < ent.end and ent.label_ in QUANTITY_NER_LABELS
        for ent in doc.ents
    )


def _is_date_time_token(token, doc):
    """True if token is part of a DATE or TIME NER entity."""
    return any(
        ent.start <= token.i < ent.end and ent.label_ in ("DATE", "TIME")
        for ent in doc.ents
    )


def _is_named_entity(token, doc):
    """True if token is a GPE/LOC/ORG/PERSON entity — a real object, not an adverb."""
    return any(
        ent.start <= token.i < ent.end and ent.label_ in ("GPE", "LOC", "ORG", "PERSON")
        for ent in doc.ents
    )


def get_qualifier(verb_token, doc):
    """
    Extract adverb qualifier: 'sometimes', 'a lot', 'half of the times'.
    Uses NER labels for quantity detection — no hardcoded word lists.
    DATE/TIME tokens are skipped: they belong to the edge date, not the qualifier.
    Named-entity npadvmod tokens (places, people) are skipped: they are objects.
    """
    parts = []
    for child in verb_token.children:
        if _is_date_time_token(child, doc):
            continue  # "yesterday", "last year" → edge date, not qualifier
        if child.dep_ == "npadvmod" and _is_named_entity(child, doc):
            continue  # "visits Delhi" → Delhi is object, not qualifier
        if child.dep_ in ("advmod", "npadvmod"):
            parts.append(_subtree_text(child))
        elif is_quantity_token(child, doc):
            parts.append(_subtree_text(child))
    return " ".join(parts)


def get_tense(verb_token):
    morph = verb_token.morph.to_dict()
    return morph.get("Tense", "Pres")


def get_modal(doc):
    for token in doc:
        if token.pos_ == "AUX" and token.lemma_ in (
            "could", "might", "should", "would", "can", "may"
        ):
            return token.lemma_
    return None


def get_date_from_doc(doc):
    for ent in doc.ents:
        if ent.label_ == "DATE":
            return ent.text
    return datetime.now().strftime("%Y-%m-%d")


def get_negation(verb_token):
    return any(c.dep_ == "neg" for c in verb_token.children)


def get_conjunct_objects(obj_token, doc):
    """Collect this object token and all its conj descendants (any depth).
    Handles chained conjuncts: "biryani, dosa and samosa" where samosa
    is conj of dosa (not biryani) — BFS collects all levels.
    """
    result = [obj_token]
    queue  = list(obj_token.children)
    while queue:
        child = queue.pop(0)
        if child.dep_ == "conj":
            result.append(child)
            queue.extend(child.children)
    return result


# ── main parse function ──────────────────────────────────────────

def parse(text, date_override=None):
    """
    Parse a sentence into a list of structured atoms.

    Atom types:
      verb_relation   — Nikhil likes Indian food
      attribute       — Nikhil is a software engineer
      state           — Nikhil is stressed  (temporal)
      modal_category  — biryani could be Indian food
      possessive      — Nikhil's mother lives in Delhi
      unknown_verb    — verb not in lookup table → goes to LLM
    """
    doc   = nlp(text)
    atoms = []
    date  = date_override or get_date_from_doc(doc)
    modal = get_modal(doc)

    root = next((t for t in doc if t.dep_ == "ROOT"), None)
    if root is None:
        return atoms

    # find subject (nsubj or nsubjpass); fall back to NER PERSON before root
    subj_token = next(
        (t for t in doc if t.dep_ in ("nsubj", "nsubjpass") and t.head == root), None
    )
    if subj_token is None:
        person_ents = {t.i for ent in doc.ents if ent.label_ == "PERSON" for t in ent}
        subj_token = next(
            (t for t in doc if t.i < root.i and t.i in person_ents), None
        )
    if subj_token is None:
        subj_token = next(
            (t for t in doc
             if t.i < root.i
             and t.pos_ in ("PROPN", "NOUN")
             and t.dep_ not in ("compound", "amod")),
            None,
        )
    if subj_token is None:
        return atoms

    subject_phrase = get_noun_phrase(subj_token, doc)
    subj_type      = get_entity_type(subject_phrase, subj_token, doc)

    # ── POSSESSIVE SUBJECT DETECTION ─────────────────────────────
    # "Nikhil's mother lives in Delhi" → poss_owner=Nikhil, subj_token=mother
    # Emits a possessive atom and overrides subject_phrase to the placeholder.
    poss_owner = next(
        (c for c in subj_token.children if c.dep_ == "poss"), None
    )
    if poss_owner and subj_token.lemma_.lower() in POSSESSIVE_PREDICATES:
        relation     = POSSESSIVE_PREDICATES[subj_token.lemma_.lower()]
        owner_phrase = get_noun_phrase(poss_owner, doc)
        owner_type   = get_entity_type(owner_phrase, poss_owner, doc)

        # Check for named appositive: "Nikhil's mother Sunita lives in Delhi"
        appos_child = next(
            (c for c in subj_token.children if c.dep_ == "appos"), None
        )
        if appos_child:
            entity_name  = appos_child.text.lower()
            entity_type  = get_entity_type(appos_child.text, appos_child, doc)
            entity_named = True
        else:
            entity_name  = f"{poss_owner.text.lower()}:{subj_token.lemma_.lower()}"
            entity_type  = "PERSON"   # relatives/relations are PERSON by default
            entity_named = False

        atoms.append({
            "type":         "possessive",
            "owner":        owner_phrase,
            "owner_type":   owner_type,
            "relation":     relation,
            "entity":       entity_name,
            "entity_type":  entity_type,
            "entity_named": entity_named,
            "date":         date,
        })

        # Override subject for all subsequent atoms in this sentence
        subject_phrase = entity_name
        subj_type      = entity_type

    negated    = get_negation(root)
    has_be_aux = any(
        t._.lemma() == "be" and t.dep_ in ("aux", "auxpass") and t.head == root
        for t in doc
    )

    # ── LINKING VERB / STATE (is/are/was) ────────────────────────
    if root._.lemma() == "be" or has_be_aux:
        primary_attrs = [
            t for t in doc
            if t.dep_ in ("attr", "acomp") and t.head == root
        ]
        all_attr_tokens = []
        for primary in primary_attrs:
            all_attr_tokens.extend(_collect_conjuncts(primary))

        for token in all_attr_tokens:
            obj_phrase = get_attr_phrase(token)
            obj_type   = get_entity_type(obj_phrase, token, doc)

            if modal:
                atoms.append({
                    "type":         "modal_category",
                    "subject":      subject_phrase,
                    "subject_type": subj_type,
                    "object":       obj_phrase,
                    "object_type":  obj_type,
                    "modal":        modal,
                    "date":         date,
                })
            elif token.text.lower() in STATE_ADJECTIVES:
                atoms.append({
                    "type":         "state",
                    "subject":      subject_phrase,
                    "subject_type": subj_type,
                    "state":        token.text.lower(),
                    "date":         date,
                })
            else:
                atoms.append({
                    "type":         "attribute",
                    "subject":      subject_phrase,
                    "subject_type": subj_type,
                    "object":       obj_phrase,
                    "object_type":  obj_type,
                    "date":         date,
                })

            # ── relative clauses on this attr token ──────────────────
            # "male who loves indian food" → nikhil LOVES indian food
            for rc in token.children:
                if rc.dep_ != "relcl":
                    continue
                rc_edge = VERB_EDGE_MAP.get(rc._.lemma()) or _get_learned_rules().get_verb_predicate(rc._.lemma())
                rc_neg  = get_negation(rc)
                if rc_neg and rc_edge in ("LIKES", "LOVES"):
                    rc_edge = "DISLIKES"
                elif rc_neg and rc_edge == "HATES":
                    rc_edge = "LIKES"
                if rc_edge is None:
                    atoms.append({
                        "type":    "unknown_verb",
                        "subject": subject_phrase,
                        "verb":    rc._.lemma(),
                        "text":    text,
                        "date":    date,
                    })
                    continue
                for obj_tok in rc.children:
                    if obj_tok.dep_ in ("dobj", "pobj") and not is_quantity_token(obj_tok, doc):
                        for conj_obj in get_conjunct_objects(obj_tok, doc):
                            op = get_noun_phrase(conj_obj, doc)
                            ot = get_entity_type(op, conj_obj, doc)
                            atoms.append({
                                "type":         "verb_relation",
                                "subject":      subject_phrase,
                                "subject_type": subj_type,
                                "edge_type":    rc_edge,
                                "object":       op,
                                "object_type":  ot,
                                "qualifier":    get_qualifier(rc, doc),
                                "tense":        get_tense(rc),
                                "negated":      rc_neg,
                                "date":         date,
                            })

        _scan_learned_prep_patterns(doc, root, atoms, subject_phrase, subj_type, date)
        return atoms

    # ── ACTION / STATIVE VERB ────────────────────────────────────
    edge_type = VERB_EDGE_MAP.get(root._.lemma()) or _get_learned_rules().get_verb_predicate(root._.lemma())
    qualifier = get_qualifier(root, doc)
    tense     = get_tense(root)

    if negated and edge_type in ("LIKES", "LOVES"):
        edge_type = "DISLIKES"
    elif negated and edge_type == "HATES":
        edge_type = "LIKES"

    if edge_type is None:
        atoms.append({
            "type":    "unknown_verb",
            "subject": subject_phrase,
            "verb":    root._.lemma(),
            "text":    text,
            "date":    date,
        })
        return atoms

    # collect objects; skip CARDINAL/QUANTITY tokens (they are frequency qualifiers)
    # Also collect npadvmod when the token is a named entity (e.g. "visits Delhi" →
    # Delhi is dep_=npadvmod but is a real object, not an adverbial modifier).
    obj_tokens = []
    for token in doc:
        is_direct_obj = token.dep_ in ("dobj", "pobj") and (
            token.head == root or token.head.head == root
        )
        is_place_advmod = (
            token.dep_ == "npadvmod"
            and token.head == root
            and _is_named_entity(token, doc)
        )
        if (is_direct_obj or is_place_advmod) and not is_quantity_token(token, doc):
            obj_tokens.extend(get_conjunct_objects(token, doc))

    # fallback: if parser made the object the ROOT (e.g. "dislikes coffee" before fixer),
    # use the original ROOT noun as the object
    if not obj_tokens:
        orig_root = next((t for t in doc if t.dep_ == "ROOT"), None)
        if (orig_root and orig_root != root
                and orig_root.pos_ in ("NOUN", "PROPN")
                and orig_root.i > root.i):
            obj_tokens.append(orig_root)

    # deduplicate while preserving order
    seen, unique_objs = set(), []
    for t in obj_tokens:
        if t.i not in seen:
            seen.add(t.i)
            unique_objs.append(t)

    # if still no object and verb implies one (COOKS → food), use implied object
    if not unique_objs and edge_type in IMPLIED_OBJECTS:
        atoms.append({
            "type":         "verb_relation",
            "subject":      subject_phrase,
            "subject_type": subj_type,
            "edge_type":    edge_type,
            "object":       IMPLIED_OBJECTS[edge_type],
            "object_type":  "CONCEPT",
            "qualifier":    qualifier,
            "tense":        tense,
            "negated":      negated,
            "date":         date,
        })
        return atoms

    for obj_token in unique_objs:
        obj_phrase = get_noun_phrase(obj_token, doc)
        obj_type   = get_entity_type(obj_phrase, obj_token, doc)
        atoms.append({
            "type":         "verb_relation",
            "subject":      subject_phrase,
            "subject_type": subj_type,
            "edge_type":    edge_type,
            "object":       obj_phrase,
            "object_type":  obj_type,
            "qualifier":    qualifier,
            "tense":        tense,
            "negated":      negated,
            "date":         date,
        })

    _scan_learned_prep_patterns(doc, root, atoms, subject_phrase, subj_type, date)
    return atoms


def _scan_learned_prep_patterns(doc, root, atoms, subject, subj_type, date):
    """
    Check for prepositional patterns from learned rules.
    Handles structures like "into running" → INTERESTED_IN.
    Only fires for preps that are direct children or conjuncts of root
    and have a matching learned pattern rule.
    """
    learned = _get_learned_rules()
    consumed = {a.get("object", "").lower() for a in atoms}

    for token in doc:
        if token.pos_ != "ADP":
            continue
        pred = learned.get_prep_predicate(token.lemma_)
        if not pred:
            continue
        # only handle preps structurally related to root
        related = (
            token.head == root
            or (token.dep_ == "conj" and token.head == root)
            or (token.dep_ == "prep" and token.head == root)
        )
        if not related:
            continue
        obj_tok = next(
            (c for c in token.children if c.dep_ in ("pcomp", "pobj")), None
        )
        if not obj_tok:
            continue
        obj_phrase = get_noun_phrase(obj_tok, doc)
        if obj_phrase.lower() in consumed:
            continue
        obj_type = get_entity_type(obj_phrase, obj_tok, doc)
        atoms.append({
            "type":         "verb_relation",
            "subject":      subject,
            "subject_type": subj_type,
            "edge_type":    pred,
            "object":       obj_phrase,
            "object_type":  obj_type,
            "qualifier":    "",
            "tense":        "Pres",
            "negated":      False,
            "date":         date,
        })
