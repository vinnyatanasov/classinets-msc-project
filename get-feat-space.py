# Select feature space using features in sentences selected in train sets for target predictors.


import lib.utility as u
import config as c
import os


if __name__ == "__main__":
    # Store target words in array
    targets = u.get_target_words()
    
    # Set up feats dict - to count number of occurrences of each
    f_dict = {}
    
    # For each target word go and get their data and iterate through features in each sentence and add log in f_dict
    for word in targets:
        path = c.train_data_dir + word + ".txt"
        with open(path) as file:
            for line in file:
                for word in line.strip().split()[1:]:
                    if word.isalpha():
                        try:
                            f_dict[word] += 1
                        except:
                            f_dict[word] = 1
    
    # remove unwanted features
    #del f_dict["<NUM>"]
    
    # Get top occurring N features
    feats = []
    N = c.feat_space_size
    i = 0
    for word in sorted(f_dict, key=f_dict.get, reverse=True):
        if i == N:
            #print f_dict[word]
            break
        feats.append(word)
        i += 1
    print "There are", len(feats), "features..."
    
    # Print feat space to file
    #with open(c.output_dir + "feat-space.txt", "w") as file:
    #    for feat in feats:
    #        file.write(feat + " ")
    