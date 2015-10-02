# Select target words using Log Likelihood Ratio (LLR) to measure pos/neg association.

# To compute LLR:
# P(t = +1 | w) = total positives containing w / total containing w
# P(t = -1 | w) = total negatives containing w / total containing w
# then take logarithm of ratio - if it's highly positive it's more associated with positive class (t = +1)...


import math
import random
import lib.utility as u
import config as c


def count_occurrences(f, words):
    """
    From f, takes a count of occurrences for each word in words
    and returns dictionary
    """
    counts = {}
    with open(f) as file:
        for line in file:
            for w in words:
                if w in line:
                    try:
                        counts[w] += 1
                    except:
                        counts[w] = 1
    return counts


if __name__ == "__main__":
    words = u.get_lines(c.output_dir + "frequent-words.txt")
    
    # Get count in positive sentences, and negative sentences to use to compute LLR
    pos_counts = count_occurrences("../data/imdb-all-sentences-pos.txt", words)
    neg_counts = count_occurrences("../data/imdb-all-sentences-neg.txt", words)
    
    # Compute log likelihood ratio for each word and store in dictionary
    llrs = {}
    for w in words:
        # Get each count from dicts
        pos = pos_counts[w]
        neg = neg_counts[w]
        # Compute each conditional prob, then take log as score
        # P(t = +1 | w)
        p1 = pos / float(pos + neg)
        # P(t = -1 | w)
        p2 = neg / float(pos + neg)
        llrs[w] = math.log(p1 / p2)
    
    # remove NUM feat from calculations
    del llrs["NUM"]
    
    # Write top pos/neg target words to file
    # Taking the N/2 highest positive and N/2 highest negative words from LLR score
    N = c.num_targets
    with open(c.output_dir + "target-words2.txt", "w") as file:
        # Get highest negative words and write to file
        count = 0
        for x in sorted(llrs, key=llrs.get):
            if count == N/2.0:
                break
            #print x, llrs[x]
            if x.isalpha():
                file.write(x + " " + str(llrs[x]) + "\n")
                count += 1
        
        # Get highest positive words and write to file
        count = 0
        for x in sorted(llrs, key=llrs.get, reverse=True):
            if count == N/2.0:
                break
            #print x, llrs[x]
            if x.isalpha():
                file.write(x + " " + str(llrs[x]) + "\n")
                count += 1
    