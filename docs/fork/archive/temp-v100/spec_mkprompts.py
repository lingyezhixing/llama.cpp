import io, os

BASE = r"<TEMP>\v100"

# repetitive doc pool (same as t20_mkprompt) - easy to continue / copy for ngram tests
POOL = [
    "The history of computing spans many centuries and many cultures.",
    "Early humans used tally sticks, knotted cords, and counting boards to keep records.",
    "The abacus was one of the first tools designed to help with arithmetic.",
    "In the seventeenth century, mechanical calculators began to appear in Europe.",
    "Blaise Pascal built a gear-driven adding machine for his father's tax work.",
    "Gottfried Leibniz extended the idea with a device that could multiply and divide.",
    "Charles Babbage designed the Difference Engine and later the Analytical Engine.",
    "Ada Lovelace wrote notes describing how the Analytical Engine could compute sequences.",
    "Herman Hollerith used punched cards to tabulate the American census of 1890.",
    "Punched card systems spread into business and government around the world.",
    "Alan Turing formalized computation with an abstract machine and proved deep limits.",
    "Claude Shannon showed that electrical circuits could implement Boolean logic.",
    "The Second World War accelerated work on electronic codebreaking machines.",
    "Colossus and ENIAC demonstrated the speed of vacuum tube electronics.",
    "The stored program concept allowed instructions and data to share memory.",
    "Early commercial computers such as UNIVAC found use in administration.",
    "The invention of the transistor replaced fragile vacuum tubes.",
    "Integrated circuits put many transistors on a single piece of silicon.",
    "Moore's law described the steady shrinking of transistors for decades.",
    "Mainframes served large organizations with batch processing.",
    "Time sharing let many users interact with a single machine.",
    "The microprocessor put a complete processor on one chip.",
    "Personal computers brought computing into homes and small offices.",
    "Networking grew from local links into a global internet.",
    "The web made information accessible through browsers and hyperlinks.",
    "Mobile devices put powerful computers into pockets and hands.",
    "Graphics processors enabled simulation, games, and machine learning.",
    "Cloud computing shifted workloads into large data centers.",
    "Open source software let communities collaborate on shared code.",
    "Each generation of hardware opened new kinds of software.",
    "Programming languages evolved from machine code to high level abstractions.",
    "Operating systems manage memory, files, processes, and devices.",
    "Databases organize vast collections of structured information.",
    "Compilers translate human readable code into machine instructions.",
    "Cryptography protects communication and commerce in digital networks.",
    "Computer graphics render images from geometric and physical models.",
    "Artificial intelligence seeks to automate perception and reasoning.",
    "Machine learning finds patterns in large collections of data.",
    "The story of computing continues to unfold with every new discovery.",
]

def make_doc(target_chars, path):
    parts, n, i = [], 0, 0
    while n < target_chars:
        if i % 20 == 0:
            chunk = "\n\n## Section %d\n\n" % (i // 20)
            parts.append(chunk)
            n += len(chunk)
        s = POOL[i % len(POOL)] + " "
        parts.append(s)
        n += len(s)
        i += 1
    text = "".join(parts)
    io.open(path, "w", encoding="ascii", newline="\n").write(text)
    print("%-28s chars=%-8d approx_tokens=%-7d" % (os.path.basename(path), len(text), len(text) // 4))

CODE = '''"""small utilities for record processing."""

import os
import json


class Record:
    def __init__(self, name, value):
        self.name = name
        self.value = value

    def as_dict(self):
        return {"name": self.name, "value": self.value}


def load_records(path):
    """Load records from a jsonl file."""
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(Record(**json.loads(line)))
    return out


def save_records(path, records):
    """Save records to a jsonl file."""
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec.as_dict()) + "\\n")


def summarize(records):
    """Return a summary dict for a list of records."""
    total = 0
    for rec in records:
        total += rec.value
    return {"count": len(records), "total": total}

'''

PROSE = '''The harbor town woke slowly. Fishermen checked their nets while the sky turned from grey to pale gold, and gulls wheeled over the water in loose, noisy circles. A baker rolled up the metal shutter of her shop and set out trays of bread that steamed in the cold air. Down by the pier, an old man repaired a wooden crate with short, careful strokes of a hammer, pausing now and then to watch a ferry slide past the breakwater. Children on their way to school stopped to count the boats, arguing about which one was the fastest. By mid morning the market square was full of voices, crates of oranges, and the smell of coffee from a small cart near the fountain. A musician tuned a battered guitar under the arcade, waiting for the crowd to thicken before he began to play. Nobody was in a hurry. The town had learned long ago that the day would take care of itself, and that the tide would come back whether or not anyone worried about it.'''

def make_text(text, path):
    io.open(path, "w", encoding="ascii", newline="\n").write(text)
    print("%-28s chars=%-8d approx_tokens=%-7d" % (os.path.basename(path), len(text), len(text) // 4))

# ratio: chars per token, calibrate after first tokenize probe
ratio = float(os.environ.get("RATIO", "4.2"))
make_text(CODE,  os.path.join(BASE, "spec_code_0.txt"))
make_text(PROSE, os.path.join(BASE, "spec_prose_0.txt"))
make_doc(int(600    * ratio), os.path.join(BASE, "spec_doc_0.txt"))
make_doc(int(32768  * ratio), os.path.join(BASE, "spec_doc_32k.txt"))
make_doc(int(131072 * ratio), os.path.join(BASE, "spec_doc_128k.txt"))
