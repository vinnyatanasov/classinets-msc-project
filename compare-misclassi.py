# Script to compare lists of misclassified sentences to see which ones were originally
# wrong then corrected (and vise-versa).


if __name__ == "__main__":
    # read in baseline's misclassified indices from file
    baseline_mis = set()
    with open("output-700/mis-baseline.txt") as file:
        for line in file:
            baseline_mis.add(int(line.strip()))
    
    # method to compare against
    method = "wordnet"
    
    # read in compare method's misclassified indices from file
    method_mis = set()
    with open("output-700/mis-" + method + ".txt") as file:
        for line in file:
            method_mis.add(int(line.strip()))
    
    #print len(baseline_mis)
    
    # do set operations on two sets
    # those made right (originally misclassified and then correctly classified) given
    # by set difference
    corrected = baseline_mis.difference(method_mis)
    
    # those made wrong (opposite)
    mistaken = method_mis.difference(baseline_mis)
    
    print "Corrected:"
    print corrected
    
    print "Mistakes:"
    print mistaken
    
    
    # Read in sentences, and save the corrected and mistakes to arrays
    # corrected/mistakes are 0 indexed, so when read into array we can use
    # indexes directly to access sentences
    sents = []
    with open("output-700/expanded-data/test-all-neighb.txt") as file:
        for line in file:
            sents.append(line.strip())
    
    # Corrected sentences
    corrected_sents = []
    for i in corrected:
        corrected_sents.append(sents[i])
    
    # Mistake sentences
    mistaken_sents = []
    for i in mistaken:
        mistaken_sents.append(sents[i])
    
    
    #print len(corrected_sents), len(mistaken_sents)
    
    
    # Print corrected and mistakes to files
    #with open("corrected-sents-all-neighbs.txt", "w") as file:
    #    for x in corrected_sents:
    #        file.write(x + "\n")
    
    #with open("mistaken-sents-all-neighbs.txt", "w") as file:
    #    for x in mistaken_sents:
    #        file.write(x + "\n")
    
    
    
    print "##"
    print "Method:", method
    print "Corrected instances:", len(corrected)
    print "Mistaken instances:", len(mistaken)
    print "Net gain:", len(baseline_mis)-len(method_mis)
    print "##"
    