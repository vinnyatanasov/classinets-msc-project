# ClassiNet all neighbour expansion method.


import lib.utility as u
import lib.expand as ex
import config as c
import numpy
import networkx as nx


if __name__ == "__main__":
    # Method variables
    w1 = 1.0 # weight for neighbour feats
    w2 = 1.0 # weight for predicted feats
    predict_threshold = 0.9 # prediction threshold
    # various feature values
    val_both = 0.95
    val_first = 0.6
    val_second = 0.3
    val_pred = 0.45
    
    input_file = "../data/tree-test/test-full.txt"
    output_file = "test-expanded-z1.txt"
    
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
    
    
    print "expanding feature vectors..."
    i = 0
    for vect in data:
        #if i == 1:
        #    break
        
        # Store array of current feats
        feats = [f.split(":")[0] for f in sentences[i][1:]]
        
        # Get prediction values for each feature using weight matrix and bias terms
        values = ex.predict_feats(W, b_arr, vect)
        
        # Part 1:
        # Get all neighbour feats of instance feats
        start_feats = set()
        expanded_feats_1 = {} # first-level neighbs
        expanded_feats_2 = {} # second-level neighbs
        
        for feat in feats:
            for node in G.nodes_iter():
                # if feat is found in the graph, then we get its neighbours and then break to
                # next feat
                if node == feat:
                    start_feats.add(feat)
                    
                    # first level neighbours
                    for edge in nx.edges_iter(G, [node]):
                        nodeb = edge[1]
                        
                        if nodeb not in start_feats:
                            try:
                                expanded_feats_1[nodeb] += 1
                            except:
                                expanded_feats_1[nodeb] = 1
                        
                        # second level neighbours
                        for edge2 in nx.edges_iter(G, [nodeb]):
                            nodeb2 = edge2[1]
                            
                            if nodeb2 not in start_feats:
                                try:
                                    expanded_feats_2[nodeb2] += 1
                                except:
                                    expanded_feats_2[nodeb2] = 1
                    break
        
        # Dict to hold all new feats
        new_feats = {}
        
        # First add those in both first-level and second-level with more weight
        in_both = set(expanded_feats_1).intersection(set(expanded_feats_2))
        for word in in_both:
            new_feats[word] = w1 * val_both
        
        # We add all from first and second levels, with different values - first weighted more
        for key, value in expanded_feats_1.iteritems():
            if key not in feats:
                if key not in in_both:
                    new_feats[key] = w1 * val_first
        for key, value in expanded_feats_2.iteritems():
            if key not in feats:
                if key not in in_both:
                    if key not in expanded_feats_1:
                        new_feats[key] = w1 * val_second
        
        
        # Part 2:
        # Get predicted feats, and add them to new_feats if not there already
        for j in xrange(len(values)):
            word = target_words[j]
            value = values[j]
            if value >= predict_threshold:
                if word not in feats:
                    if word not in new_feats:
                        new_feats[word] = w2 * val_pred
        
        # add all new feats to sentence
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
    