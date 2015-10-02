# ClassiNet N-path expansion method.


import lib.utility as u
import lib.expand as ex
import config as c
import numpy
import networkx as nx


if __name__ == "__main__":
    # Method variables
    # n (path length) should be either 1 or 2
    # note: tried 3, but didn't give good results, so now code is limited to 1 or 2
    n = 2
    input_file = "../data/tree-test/test-full.txt"
    output_file = "test-expanded-3.txt"
    
    # Generate graph from edges file
    G = ex.read_graph(c.output_dir + "edges-directed-6.txt")
    
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
        #if i == 3:
        #    break
        
        # Store array of current feats
        feats = [f.split(":")[0] for f in sentences[i][1:]]
        
        # Get prediction values for each feature using weight matrix and bias terms
        values = ex.predict_feats(W, b_arr, vect)
        
        # create dict to hold expanded features
        predicted_feats = {}
        path_feats = {}
        threshold = 0.9
        min_value = 0.5
        
        # go through values and save those above threshold into expanded_feats dict
        for j in xrange(len(values)):
            this_target_word = target_words[j]
            value = values[j]
            # only continue if not in feats already
            if this_target_word not in feats:
                # if predicted value exceeds threshold, we continue with it
                if value >= threshold:
                    # add to expanded feats
                    predicted_feats[this_target_word] = float("{0:.4f}".format(value))
                    
                    # first-level path feats
                    for edge in nx.edges_iter(G, [this_target_word]):
                        nodea = edge[0]
                        nodeb = edge[1]
                        edge_weight = G[nodea][nodeb]["weight"]
                        feat_val = value * edge_weight
                        print "*", nodea, nodeb, edge_weight
                        
                        # only continue if value is reasonable
                        if feat_val >= min_value:
                            if nodeb not in feats and nodeb not in predicted_feats:
                                path_feats[nodeb] = float("{0:.4f}".format(feat_val))
                            
                            # include second-level path feats if set
                            if n == 2:
                                for edge2 in nx.edges_iter(G, [nodeb]):
                                    nodea2 = edge2[0]
                                    nodeb2 = edge2[1]
                                    edge_weight2 = G[nodea2][nodeb2]["weight"]
                                    feat_val2 = feat_val * edge_weight2
                                    print "**", nodeb2, feat_val2
                                    
                                    if nodeb2 not in feats and nodeb2 not in predicted_feats:
                                        if nodeb2 not in path_feats:
                                            path_feats[nodeb2] = float("{0:.4f}".format(feat_val2))
        
        # Add predicted feats
        for key, value in predicted_feats.iteritems():
            if key not in feats:
                sentences[i].append(key + ":" + str(value))
        
        # Add path feats with weighted edges
        for key, value in path_feats.iteritems():
            if key not in feats:
                sentences[i].append(key + ":" + str(value))
        
        print i
        i += 1
    
    # Write new, expanded data to file
    #with open(c.expanded_data_dir + output_file, "w") as file:
    #    for s in sentences:
    #        for i in s:
    #            file.write(i + " ")
    #        file.write("\n")
    