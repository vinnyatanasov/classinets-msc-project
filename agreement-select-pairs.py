# Take in pairs of targets and their cosine scores and select the highest N pairs
# to then compute agreement in normal way


import lib.utility as u
import config as c


if __name__ == "__main__":
    # Read in all pairs with cosine scores
    pairs = u.get_lines(c.output_dir + "edges-cosine.txt")
    
    # Iterate each pair and add to dictionary, where key is tuple and
    # value is score - then we can sort by value
    pairs_dict = {}
    for p in pairs:
        p = p.split()
        t1 = p[0]
        t2 = p[1]
        score = float(p[2])
        
        # check reverse isn't in dict already before inserting
        if (t2, t1) not in pairs_dict:
            pairs_dict[(t1, t2)] = score
    
    #total = len(pairs_dict)
    #print "total number of pairs =", total
    
    # Set N (number of pairs to select)
    N = 5000
    print "chosen number of pairs =", N
    
    # Iterate through sorted pairs, and take top N
    i=0
    top_pairs = []
    for key in sorted(pairs_dict, key=pairs_dict.get, reverse=True):
        if i == N:
            break
        #print key, pairs_dict[key]
        top_pairs.append((key[0], key[1], pairs_dict[key]))
        i+=1
    
    # For interest, count how many unique terms are in the top_pairs
    terms = set()
    for x in top_pairs:
        terms.add(x[0])
        terms.add(x[1])
    print len(terms), "from 700 have at least one connection..."
    
    # Print new pairs to file
    #with open(c.output_dir + "target-word-high-pairs.txt", "w") as file:
    #    for x in top_pairs:
    #        file.write(x[0] + " " + x[1] + " " + str(x[2]) + "\n")
    