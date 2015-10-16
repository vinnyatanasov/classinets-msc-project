# ClassiNet local path expansion method using shortest paths.


import lib.utility as u
import lib.expand as ex
import config as c
import numpy
import networkx as nx
import math


def get_path_feats(path):
    """
    Returns dict of features and their weight from the given path.
    """
    #print path
    feats = {}
    max = len(path)-1
    for i in xrange(max):
        a = path[i]
        b = path[i+1]
        weight = G[a][b]["weight"]
        feats[b] = weight
    
    return feats


def get_path_weight(path):
    """
    Computes path weight of given path (array of vertices) by taking sum of logs of each
    edge weight probability. Returns exponentiated value (back as a probability)
    """
    total_weight = 0
    max = len(path)-1
    for i in xrange(max):
        a = path[i]
        b = path[i+1]
        # get weight and take log so no super tiny numbers
        weight = G[a][b]["weight"]
        #print a, b, weight
        weight = math.log(weight)
        # negate them so we get positive values
        total_weight += weight
    
    return numpy.exp(total_weight)


if __name__ == "__main__":
    # Method variables
    n = 1
    input_file = "test-full.txt"
    output_file = "test-expand-short-path1.txt"
    
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
    print "##### BEGIN #####"
    i = 0
    for vect in data:
        #if i == 4:
        #    break
        
        # Store array of instance feats
        feats = [f.split(":")[0] for f in sentences[i][1:]]
        
        # Get those instance feats in ClassiNet
        inst_feats = set()
        for feat in feats:
            for node in G.nodes_iter():
                if node == feat:
                    inst_feats.add(feat)
                    break
        
        print "Instance feats:", inst_feats
        
        # Get prediction values for each feature using weight matrix and bias terms
        values = ex.predict_feats(W, b_arr, vect)
        
        # Create dict to hold expanded features
        pred_feats = {}
        threshold = 0.9
        
        # Get predicted feats
        for j in xrange(len(values)):
            this_target_word = target_words[j]
            value = values[j]
            # only continue if not in feats already
            if this_target_word not in feats:
                # if predicted value exceeds threshold, we continue with it
                if value >= threshold:
                    # add to expanded feats
                    pred_feats[this_target_word] = float("{0:.4f}".format(value))
        
        # Filter out any predicted feats that aren't in graph (because we're using approximation, remember)
        not_in = set()
        for feat in pred_feats:
            found = False
            for node in G.nodes_iter():
                if node == feat:
                    found = True
                    break
            if found == False:
                not_in.add(feat)
        
        if len(not_in) > 0:
            for feat in not_in:
                #print "deleting", feat
                del pred_feats[feat]
        
        # Get path features from shortest paths
        path_feats = {}
        for feat in pred_feats:
            #print feat
            for feat2 in inst_feats:
                # get shortest path from feat to feat2
                if nx.has_path(G, feat, feat2):
                    path = nx.shortest_path(G, feat, feat2, weight="weight")
                    
                    if len(path) <= 4:
                        #print feat, feat2
                        #print path
                        #path_feats[feat2] = get_path_weight(path)
                        this_path_feats = get_path_feats(path)
                        
                        # add all from this_path_feats to path_feats
                        for key, value in this_path_feats.iteritems():
                            path_feats[key] = value
                    else:
                        continue
        
        print "Path feats:", path_feats
        
        # Add predicted feats
        for key, value in pred_feats.iteritems():
            if key not in feats:
                sentences[i].append(key + ":" + str(value))
        
        # Add path feats
        for key, value in path_feats.iteritems():
            if key not in feats and key not in pred_feats:
                sentences[i].append(key + ":" + str(value))
        
        #print sentences[i]
        print i
        i += 1
    
    # Write new, expanded data to file
    #with open(c.expanded_data_dir + output_file, "w") as file:
    #    for s in sentences:
    #        for i in s:
    #            file.write(i + " ")
    #        file.write("\n")
    