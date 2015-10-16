# ClassiNet all neighbour expansion method. Here we tweak the method slightly for the paper.


import lib.utility as u
import lib.expand as ex
import config as c
import numpy
import networkx as nx


if __name__ == "__main__":
    # Method variables
    #predict_threshold = 0.9 # prediction threshold
    
    input_file = "test-full.txt"
    output_file = "test-expanded-all-neighb.txt"
    
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
        #if i == 5:
        #    break
        
        # Store array of current feats
        feats = [f.split(":")[0] for f in sentences[i][1:]]
        
        # Get prediction values for each feature using weight matrix and bias terms
        values = ex.predict_feats(W, b_arr, vect)
        
        # Part 1:
        # Get all neighbour feats of instance feats
        start_feats = set()
        expanded_feats = {} # first-level neighbs
        
        for feat in feats:
            for node in G.nodes_iter():
                # if feat is found in the graph, then we get its neighbours and then break to
                # next feat
                if node == feat:
                    start_feats.add(feat)
                    
                    # first level neighbours
                    for edge in nx.edges_iter(G, [node]):
                        nodea = edge[0]
                        nodeb = edge[1]
                        ew = G[nodea][nodeb]["weight"]
                        
                        if nodeb not in start_feats:
                            expanded_feats[nodeb] = ew
                    
                    # go to next feature
                    break
        
        # Dict to hold all new feats
        new_feats = {}
        
        # We add all neighbs
        for key, value in expanded_feats.iteritems():
            if key not in feats:
                sentences[i].append(key + ":" + str(value))
            else:
                print key
        
        print sentences[i]
        print i
        print ""
        i += 1
    
    # Write new, expanded data to file
    with open(c.expanded_data_dir + output_file, "w") as file:
        for s in sentences:
            for i in s:
                file.write(i + " ")
            file.write("\n")
    