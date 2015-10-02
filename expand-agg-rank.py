# ClassiNet aggregated rank feature expansion method.


import lib.utility as u
import lib.expand as ex
import config as c
import numpy
import networkx as nx


if __name__ == "__main__":
    # Method variables
    # Weight for prediction rank
    w1 = 1.0
    # Weight for neighbour rank
    w2 = 1.5
    
    input_file = "../data/tree-test/test-full.txt"
    output_file = "test-1.txt"
    
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
        
        # Dict to hold expanded features
        predicted_feats = {}
        threshold = 0.85
        
        # Iterate values and save those above threshold into predicted_feats array
        for j in xrange(len(values)):
            this_target_word = target_words[j]
            value = values[j]
            # only continue if not in feats already
            if this_target_word not in feats:
                if value >= threshold:
                    predicted_feats[this_target_word] = value
        
        # Compile prediction rank list, by adding predicted feats in descending order of prediction value
        ranked_by_pred = []
        for key in sorted(predicted_feats, key=predicted_feats.get, reverse=True):
            ranked_by_pred.append(key)
        
        # Get instance feats that are in ClassiNet
        instance_feats = []
        for feat in feats:
            for node in G.nodes_iter():
                if node == feat:
                    instance_feats.append(feat)
        
        # Get neighbouring nodes in graph from starting points given by instance feats and predicted feats
        neighb_feats = {}
        for feat in instance_feats:
            for edge in nx.edges_iter(G, [feat]):
                nodea = edge[0]
                nodeb = edge[1]
                #edge_weight = G[nodea][nodeb]
                #print nodea, nodeb
                if nodeb not in feats:
                    try:
                        neighb_feats[nodeb] += 1
                    except:
                        neighb_feats[nodeb] = 1
        for feat in predicted_feats:
            for edge in nx.edges_iter(G, [feat]):
                nodea = edge[0]
                nodeb = edge[1]
                #edge_weight = G[nodea][nodeb]
                #print nodea, nodeb
                if nodeb not in feats:
                    try:
                        neighb_feats[nodeb] += 1
                    except:
                        neighb_feats[nodeb] = 1
        
        # Compile neighbour rank list, by adding neighbour features in descending order of frequency
        ranked_by_neighb = []
        for key in sorted(neighb_feats, key=neighb_feats.get, reverse=True):
            ranked_by_neighb.append(key)
        
        total_pred = len(ranked_by_pred)
        total_neighb = len(ranked_by_neighb)
        
        # Iterate through union of ranked_by_pred and ranked_by_neighb and aggregate ranks
        exp_candidates = {}
        for word in set(ranked_by_pred).union(ranked_by_neighb):
            # if a feature is not in either list, it gets a 0 for that rank
            # get score in prediction list
            try:
                pred_r = 1 - (ranked_by_pred.index(word) / float(total_pred))
            except:
                pred_r = 0
            # get score in neighbour list
            try:
                neighb_r = 1 - (ranked_by_neighb.index(word) / float(total_neighb))
            except:
                neighb_r = 0
            
            # Score is weighted sum of ranks
            score = (w1 * pred_r) + (w2 * neighb_r)
            exp_candidates[word] = score
        
        
        # Add the top scoring features
        feat_val = 0.85
        for key in sorted(exp_candidates, key=exp_candidates.get, reverse=True):
            if exp_candidates[key] < 1.0 or feat_val < 0.2:
                break
            print key, exp_candidates[key]
            sentences[i].append(key + ":" + str(feat_val))
            feat_val -= 0.05
        
        # try different weighting method for feat values
        #k = 1
        #feat_val = 0.85
        #for key in sorted(exp_candidates, key=exp_candidates.get, reverse=True):
        #    if exp_candidates[key] < 1.0 or k > 10:
        #        break
        #    #print key, exp_candidates[key]
        #    val = (1 / float(k)) - 0.05
        #    #print val
        #    sentences[i].append(key + ":" + str(val))
        #    k += 1
        #    #feat_val -= 0.05
        
        print sentences[i]
        print i
        i += 1
    
    # Write new, expanded data to file
    #with open(c.expanded_data_dir + output_file, "w") as file:
    #    for s in sentences:
    #        for i in s:
    #            file.write(i + " ")
    #        file.write("\n")
    