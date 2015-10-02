# ClassiNet mutual neighbour method.

# Note: file can be used for just predicted or both (predicted+original).
# When used for both, we need to change feature values.


import lib.utility as u
import lib.expand as ex
import config as c
import numpy
import networkx as nx


if __name__ == "__main__":
    # Method variables
    with_original_feats = False
    min_sup = 2
    w1 = 1.05 # neighb feat weight
    w2 = 0.9 # predicted feat weight
    
    input_file = "../data/tree-test/test-full.txt"
    output_file = "test-mutual-pred.txt"
    
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
        if i == 3:
            break
        
        # Store array of current feats
        feats = [f.split(":")[0] for f in sentences[i][1:]]
        
        # Get prediction values for each feature using weight matrix and bias terms
        values = ex.predict_feats(W, b_arr, vect)
        
        # Dict for neighb feats
        neighb_feats = {}
        
        # First get all predicted feats
        predicted_feats = []
        predict_threshold = 0.85
        
        for j in xrange(len(values)):
            word = target_words[j]
            value = values[j]
            # only continue if not in feats already
            if word not in feats:
                if value >= predict_threshold:
                    predicted_feats.append(word)
                    
                    # get neighbours and put into dict, logging count
                    for edge in nx.edges_iter(G, [word]):
                        nodeb = edge[1]
                        try:
                            neighb_feats[nodeb] += 1
                        except:
                            neighb_feats[nodeb] = 1
        
        #
        # Original feats addition
        #
        # We add to neighbour feats dict by also logging neighbours of original feats
        if with_original_feats:
            print "yes"
            # traverse graph to get list of words in instance that have nodes in graph
            original_feats = []
            for feat in feats:
                for node in G.nodes_iter():
                    if node == feat:
                        original_feats.append(feat)
             
            # now get neighbouring nodes in graph from starting points given by instance feats and predicted feats
            for feat in original_feats:
                for edge in nx.edges_iter(G, [feat]):
                    nodeb = edge[1]
                    try:
                        neighb_feats[nodeb] += 1
                    except:
                        neighb_feats[nodeb] = 1
        
        
        # Here we try and give more weight to those features occurring in both
        # predicted and mut. neighbours
        new_feats = {}
        
        # first we add in all neighb feats that aren't in predicted feats
        for key, value in neighb_feats.iteritems():
            if value < min_sup:
                continue
            if key not in predicted_feats:
                new_feats[key] = w1 * 0.5
        
        # Get all predicted feats, and  if they're in neighb feats too they get more weight
        for feat in predicted_feats:
            feat_val = 0.5
            
            # if it's also in neighb feats with good support then it gets weighted higher
            if feat in neighb_feats:
                if neighb_feats[feat] >= min_sup:
                    feat_val = 0.8
            
            new_feats[feat] = w2 * feat_val
        
        # Add all new feats to sentence
        for key, value in new_feats.iteritems():
            if key not in feats:
                sentences[i].append(key + ":" + str(value))
        
        print sentences[i]
        print i
        print ""
        i += 1
    
    # Write new, expanded data to file
    #with open(c.expanded_data_dir + output_file, "w") as file:
    #    for s in sentences:
    #        for i in s:
    #            file.write(i + " ")
    #        file.write("\n")
    