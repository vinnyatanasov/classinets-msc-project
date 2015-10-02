# ClassiNet bayesian expansion method.

# For each candidate feat c, we need to compute P(c|x), by bayes rule, is given as P(x|c)P(c)
# So, we need to compute P(x1|c),...,P(xn|c) where x1,...,xn are the instance feats
# P(xi|c) is given by the (directed) edge weight between c and xi (assuming xi is in ClassiNet)
# then we multiply by P(c) - which is likelihood of candidate feat (optional)
#
# Note:
# For computational reasons, we don't consider all features as candidates. Instead, only those
# neighbours of neighbours of original feats, or highly predicted feats


import lib.utility as u
import lib.expand as ex
import config as c
import numpy
import math
import networkx as nx


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
        #print a, b
        # get weight and take log so no super tiny numbers
        weight = G[a][b]["weight"]
        weight = math.log(weight)
        # negate them so we get positive values
        total_weight += weight
    
    return numpy.exp(total_weight)


if __name__ == "__main__":
    # Method variables
    input_file = "../data/tree-test/test-full.txt"
    output_file = "expanded-bayesian1.txt"
    
    # Generate graph from edges file
    G = ex.read_graph(c.output_dir + "edges-directed-6-m.txt")
    
    # get array of nodes
    nodes = [x for x in G.nodes_iter()]
    
    # Get feature space and target words
    features_index = u.read_features_file(c.output_dir + "feat-space.txt")
    target_words = u.get_target_words()
    
    # Get sentences/vectors of data to expand
    sentences, data = ex.get_expansion_data(input_file, features_index)
    
    # Get matrix of weight vectors
    print "generating weight matrix..."
    W, b_arr = u.get_weight_matrix(target_words)
    
    # get likelihoods from file and store in dict
    #likelihoods = {}
    #with open("_700/target-likelihoods.txt") as file:
    #    for line in file:
    #        line = line.strip().split()
    #        likelihoods[line[0]] = float(line[1])
    
    # Iterate through sentences
    i = 0
    for sent in sentences:
        #if i == 3:
        #   break
        
        # Store array of current feats
        feats = [f.split(":")[0] for f in sentences[i][1:]]
        
        # Find instance features that are in ClassiNet
        feats_in_graph = set()
        for feat in feats:
            if feat in nodes:
                feats_in_graph.add(feat)
        
        # Get neighbouring nodes as candidates
        candidates = set()
        for feat in feats_in_graph:
            # get each neighbour
            for neighb in nx.all_neighbors(G, feat):
                candidates.add(neighb)
                # include neighbours of neighbours as candidates
                for neighb2 in nx.all_neighbors(G, neighb):
                    candidates.add(neighb2)
                    # neighbours of neighbours of neighbours?!
                    #for neighb3 in nx.all_neighbors(G, neighb2):
                    #    candidates.add(neighb3)
        
        # Get prediction values for each target word
        values = ex.predict_feats(W, b_arr, data[i])
        
        # Take highest predicted features and store in set
        predicted = set()
        for x in xrange(len(values)):
            this_target_word = target_words[x]
            value = values[x]
            if value >= 0.9:
                if this_target_word in nodes:
                    predicted.add(this_target_word)
        
        # Add the predicted features to set of candidates
        candidates = candidates.union(predicted)
        print len(candidates), "expansion candidates"
        
        # Iterate ClassiNet, and compute candidate score for each node
        scores = {}
        for node in candidates:
            # Skip node if it's already in graph
            if node in feats_in_graph:
                continue
            
            # We compute P(xi|c) for each feature i in feature vect x (those in ClassiNet, at least)
            # and store in cond_probs dict
            cond_probs = {}
            for feat in feats_in_graph:
                # P(x|c) is given by shortest path in classinet between c and x
                # First, check path exists
                if nx.has_path(G, node, feat):
                    p = nx.shortest_path(G, node, feat, weight="weight")
                    # if it exists, we need to get the weight - so we can iterate
                    # through the list returned by the shortest_path function to get each weight
                    # along the way and multiply them
                    
                    # only consider 'small-ish' paths
                    # [one, two, three, four, five]
                    # len3=2 hops, len4=3 hops, len5=4hops
                    max_len = 3
                    if len(p) <= max_len:
                        weight = get_path_weight(p)
                    else:
                        # if there's no path of length max_len
                        # then we can give it a fixed value which is an approximation
                        weight = 0.45**(max_len-1)
                        #weight = 0
                    
                    cond_probs[feat] = weight
                else:
                    #print "no path between", node, "+", feat
                    continue
            
            # Now we compute P(x|c) by adding the logarithms of each p(xi|c)
            score = 0
            for value in cond_probs.itervalues():
                if value != 0:
                    score += math.log(value)
            
            #print score, float(numpy.exp(score))
            
            # It may be that the score is 0, meaning none of the features are within N jumps
            # in that case, it's a bad candidate, so we can ignore this candidate
            if score != 0:
                # Multiply by likelihood of c (P(c)) to get final candidate score
                #likelihood = likelihoods[node]
                #score += math.log(likelihood)
                
                # exponentiate to get back to probability value
                scores[node] = float(numpy.exp(score))
        
        # Add the top k candidates
        k = 0
        f_val = 0.9
        for key in sorted(scores, key=scores.get, reverse=True):
            if k == 15:
                break
            
            if key not in feats:
                # use weighted sigmoid for feat value
                #f_val = 1.0/(1.0 + math.exp(-3 * scores[key]))
                
                #print key, scores[key], f_val
                sent.append(key + ":" + str(f_val))
                
                f_val -= 0.05
                k += 1
        
        print sent
        print i
        #print ""
        i += 1
    
    print ""
    
    # Write new, expanded data to file
    #with open(c.expanded_data_dir + output_file, "w") as file:
    #    for s in sentences:
    #        for i in s:
    #            file.write(i + " ")
    #        file.write("\n")
    