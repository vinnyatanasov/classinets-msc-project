# ClassiNet basic expansion method using predictions and a threshold.


import lib.utility as u
import lib.expand as ex
import config as c
import numpy


if __name__ == "__main__":
    # Method variables
    predict_threshold = 0.5
    input_file = "../data/tree-test/test-full.txt"
    output_file = "test-expanded-1.txt"
    
    # Get feature space and target words
    features_index = u.read_features_file(c.output_dir + "feat-space.txt")
    target_words = u.get_target_words()
    
    # Get sentences/vectors of data to expand
    sentences, data = ex.get_expansion_data(input_file, features_index)
    
    # Get matrix of weight vectors
    print "generating weight matrix..."
    W, b_arr = u.get_weight_matrix(target_words)
    
    # Iterate over instances and expand them
    print "expanding feature vectors..."
    i = 0
    for vect in data:
        # Get prediction values for each feature using weight matrix and bias terms
        values = ex.predict_feats(W, b_arr, vect)
        
        # Dict to hold expanded features
        expanded_feats = {}
        
        # Iterate values and save those above threshold into expanded_feats dict
        for j in xrange(len(values)):
            this_target_word = target_words[j]
            value = values[j]
            index = features_index[this_target_word]
            # only continue if the feature is not already present in vect
            if vect[index] == 0:
                # if predicted value exceeds threshold, add it
                if value >= predict_threshold:
                    #print key, value
                    expanded_feats[this_target_word] = float("{0:.4f}".format(value))
        
        # Add all new feats to vector
        for key, value in expanded_feats.iteritems():
            sentences[i].append(key + ":" + str(value))
        
        #print sentences[i]
        #print i
        i += 1
    
    # Write new, expanded data to file
    #with open(c.expanded_data_dir + output_file, "w") as file:
    #    for s in sentences:
    #        for i in s:
    #            file.write(i + " ")
    #        file.write("\n")
    