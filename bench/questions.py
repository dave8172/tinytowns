"""The next-symbol task, worded for decision models (Jev in step 2, Laya in step 3).

Our models read '.' + letters and give 28 probabilities. A decision model reads text and answers
typed questions, so the same task is asked in words: one shared description (the "state"), then
one Choice question per position, each offering the same 28 options. Both models get exactly
this wording, so any difference between them is the model, not the prompt.

'paris' becomes 6 questions, one per prediction our models make:
    p0  written so far: (nothing)  -> real answer 'p'
    p1  written so far: "p"        -> 'a'
    ...
    p5  written so far: "paris"    -> 'end'
"""
ALPHABET = ". abcdefghijklmnopqrstuvwxyz"  # same order as model.py: index 0 = end, 1 = space

# Option names, in ALPHABET order. "end" and "space" are words because a bare "." or " " is
# easy to misread; their descriptions say exactly what they mean.
OPTIONS = ["end", "space"] + list("abcdefghijklmnopqrstuvwxyz")
CRITERIA = {"end": "the name is complete: nothing more comes after what is written",
            "space": "a space between two words of the name"}
CRITERIA.update({ch: None for ch in OPTIONS[2:]})  # a letter needs no description

# Everything the model knows about the task. The same facts our models learned from the data:
# real place names, lowercase a-z and spaces only, at most 20 symbols.
STATE = {
    "task": "A real place name is being written one symbol at a time. Guess what comes next.",
    "names": "Cities, towns and villages anywhere in the world with at least 1,000 people "
             "(the GeoNames list). Written in lowercase, letters a-z and spaces only, "
             "accents removed, at most 20 symbols long.",
}


def questions(name):
    """One question per prediction: ({id: {instructions, criteria}}, {id: real answer})."""
    qs, answers = {}, {}
    for t in range(len(name) + 1):  # each letter, then "end of name"
        so_far = name[:t]
        written = f'Written so far: "{so_far}".' if so_far else "Nothing is written yet."
        qs[f"p{t}"] = {"instructions": f"{written} Which symbol comes next?",
                       "criteria": CRITERIA}
        answers[f"p{t}"] = OPTIONS[ALPHABET.index(name[t])] if t < len(name) else "end"
    return qs, answers
