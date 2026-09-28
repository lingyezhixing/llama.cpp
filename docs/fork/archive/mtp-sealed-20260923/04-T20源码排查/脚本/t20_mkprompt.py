import io

sentences = [
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

def make(target_chars, path):
    parts = []
    n = 0
    i = 0
    while n < target_chars:
        if i % 20 == 0:
            chunk = "\n\n## Section %d\n\n" % (i // 20)
            parts.append(chunk)
            n += len(chunk)
        s = sentences[i % len(sentences)] + " "
        parts.append(s)
        n += len(s)
        i += 1
    parts.append("\n\n# Question\n\nWrite a long, detailed essay about the history of computing. Be verbose.\n")
    text = "".join(parts)
    io.open(path, "w", encoding="ascii", newline="\n").write(text)
    print(path, "chars:", len(text), "approx tokens:", len(text)//4)

make(180000,  r"<TEMP>\v100\t20_prompt32k.txt")
make(720000,  r"<TEMP>\v100\t20_prompt128k.txt")
