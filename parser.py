import nltk
nltk.download('punkt_tab')
import sys

TERMINALS = """
Adj -> "country" | "dreadful" | "enigmatical" | "little" | "moist" | "red"
Adv -> "down" | "here" | "never"
Conj -> "and" | "until"
Det -> "a" | "an" | "his" | "my" | "the"
N -> "armchair" | "companion" | "day" | "door" | "hand" | "he" | "himself"
N -> "holmes" | "home" | "i" | "mess" | "paint" | "palm" | "pipe" | "she"
N -> "smile" | "thursday" | "walk" | "we" | "word"
P -> "at" | "before" | "in" | "of" | "on" | "to"
V -> "arrived" | "came" | "chuckled" | "had" | "lit" | "said" | "sat" | "eat"
V -> "smiled" | "tell" | "were"
"""

NONTERMINALS = """
S -> N V | NP VP | S Conj S | S Conj VP | VP NP | VP
NP -> N | Det N | Adj NP | NP PP | Det AP | NP Adv
VP -> V | V NP | VP PP | VP Adv | Adv VP
AP -> Adj | Adj AP
PP -> P NP
"""

grammar = nltk.CFG.fromstring(NONTERMINALS + TERMINALS)
parser = nltk.ChartParser(grammar)


def main():

    # If filename specified, read sentence from file
    if len(sys.argv) == 2:
        with open(sys.argv[1]) as f:
            s = f.read()

    # Otherwise, get sentence as input
    else:
        s = input("Sentence: ")

    # Convert input into list of words
    s = preprocess(s)

    # Attempt to parse sentence
    try:
        trees = list(parser.parse(s))
    except ValueError as e:
        print(e)
        return
    if not trees:
        print("Could not parse sentence.")
        return

    # Print each tree with noun phrase chunks
    for tree in trees:
        tree.pretty_print()

        print("Chunks")
        for np in np_chunk(tree):
            print(" ".join(np.flatten()))


def preprocess(sentence):
    """
    Convert `sentence` to a list of its words.
    Pre-process sentence by converting all characters to lowercase
    and removing any word that does not contain at least one alphabetic
    character.
    """
    # Tokenize sentence into words
    words = nltk.word_tokenize(sentence)
    # Convert to lowercase and filter out non-alphabetic words
    words = [w.lower() for w in words if w.isalpha()]
    return words


def np_chunk(tree):
    """
    Return a list of all noun phrase chunks in the sentence tree.
    A noun phrase chunk is defined as any subtree of the sentence
    whose label is "NP" that does not itself contain any other
    noun phrases as subtrees.
    """
    chunks = []
    # Iterate through all subtrees labeled "NP"
    for subtree in tree.subtrees(lambda t: t.label() == "NP"):
        # Check if this NP contains any other NP as a subtree (excluding itself)
        if not any(child.label() == "NP" for child in subtree.subtrees(lambda t: t != subtree and t.label() == "NP")):
            chunks.append(subtree)
    return chunks
    


if __name__ == "__main__":
    main()
