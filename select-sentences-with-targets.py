# Selects all sentences containing at least one target word, and outputs them to a file.
# These are the potential positive instances for target predictors.


import lib.utility as u
import config as c


def select_sentences(f, words):
    """
    From file f, we read in sentences and create and return two sets:
    s_yes - sentences containing a target word
    s_no - sentences not containing a target word
    """
    s_yes = set()
    s_no = set()
    contains_target = False
    
    with open(f) as file:
        for line in file:
            # skip sentences less than 2 words
            if len(line.split()) < 2:
                continue
            
            contains_target = False
            for w in words:
                if w in line:
                    contains_target = True
                    break
            
            if contains_target:
                s_yes.add(line.strip())
            else:
                s_no.add(line.strip())
    
    return s_yes, s_no


if __name__ == "__main__":
    # Load target words into memory for ease
    words = u.get_target_words()
    
    pos_yes_sentences, pos_no_sentences = select_sentences("../data/imdb-all-sentences-pos.txt", words)
    neg_yes_sentences, neg_no_sentences = select_sentences("../data/imdb-all-sentences-neg.txt", words)
    #print len(pos_yes_sentences), len(pos_no_sentences), len(neg_yes_sentences), len(neg_no_sentences)
    
    # Write sentences containing a target word to a file
    with open(c.output_dir + "sentences-with.txt", "w") as file:
        for s in pos_yes_sentences.union(neg_yes_sentences):
            file.write(s + "\n")
    
    # Write sentences not containing a target word to a file
    with open(c.output_dir + "sentences-without.txt", "w") as file:
        for s in pos_no_sentences.union(neg_no_sentences):
            file.write(s + "\n")
    