# ClassiNet mutual neighbour method. Here we tweak the method slightly for the paper.

# Note: file can be used for just predicted or both (predicted+original).
# When used for both, we need to change feature values.


import lib.utility as u
import lib.expand as ex
import config as c
import numpy
import networkx as nx


if __name__ == "__main__":
    # Method variables
    min_sup = 2
    
    input_file = "test-full.txt"
    output_file = "test-expand-mutual-neighb-two2.txt"
    
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
        #if i == 5:
        #    break
        
        # Store array of current feats
        feats = [f.split(":")[0] for f in sentences[i][1:]]
        
        # Get prediction values for each feature using weight matrix and bias terms
        values = ex.predict_feats(W, b_arr, vect)
        
        # Get predicted feats to get neighbours of too
        predicted_feats = []
        for j in xrange(len(values)):
            word = target_words[j]
            value = values[j]
            # only continue if not in feats already
            if word not in feats:
                if value >= 0.9:
                    predicted_feats.append(word)
        
        # Get all neighbour feats of instance feats
        start_feats = set()
        expanded_feats = {} # first-level neighbs
        # to add only mutual, we record an array of all nodes each is a neighbour of, then at the end
        # we can check it's neighb array and see if it's more than 1
        counts = {}
        
        # instance feats
        for feat in feats:
            for node in G.nodes_iter():
                # if feat is found in the graph, then we get its neighbours and then break to next feat
                if node == feat:
                    start_feats.add(feat)
                    
                    # first level neighbours
                    for edge in nx.edges_iter(G, [node]):
                        nodeb = edge[1]
                        ew = G[node][nodeb]["weight"]
                        
                        if nodeb not in start_feats:
                            # add weight to dict
                            expanded_feats[nodeb] = ew
                            
                            # and also log node in set of neighbs
                            try:
                                n = counts[nodeb]
                                n.add(node)
                                counts[nodeb] = n
                            except:
                                counts[nodeb] = set([node])
                        
                        # second level neighbours
                        for edge2 in nx.edges_iter(G, [nodeb]):
                            nodeb2 = edge2[1]
                            ew2 = G[nodeb][nodeb2]["weight"]
                            
                            if nodeb2 not in start_feats:
                                # add weight to dict
                                expanded_feats[nodeb2] = ew2 * ew
                                
                                # and also log node in set of neighbs
                                try:
                                    n = counts[nodeb2]
                                    n.add(node)
                                    counts[nodeb2] = n
                                except:
                                    counts[nodeb2] = set([node])
                    
                    # go to next feature
                    break
        
        # We now add the expanded feats
        for key, value in expanded_feats.iteritems():
            if key not in feats:
                count = len(counts[key])
                if count >= min_sup:
                    #print count
                    sentences[i].append(key + ":" + str(value))
        
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
    