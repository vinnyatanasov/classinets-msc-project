# Methods commonly used in feature expansion scripts.


import lib.utility as u
import numpy
import networkx as nx


def read_graph(edge_file):
    """
    Reads edge_file and creates graph G using networkx.
    """
    G = nx.DiGraph()
    
    with open(edge_file) as file:
        for line in file:
            l = line.strip().split()
            # get each node and weight from line
            node1 = l[0]
            node2 = l[1]
            w = float(l[2])
            G.add_edge(node1, node2, weight=w)
    
    return G


def get_expansion_data(data_file, feat_index):
    """
    Reads sentences from data_file and stores in sents as arrays of words.
    Creates binary feature vectors from sents and stores in vects.
    """
    # arrays for sentences and vectors
    sents = []
    vects = []
    
    print "generating feature vectors..."
    
    # get sentences
    with open(data_file) as file:
        for line in file:
            sents.append(line.strip().split())
    
    # convert sentences to binary feature vects
    feat_size = len(feat_index)
    for line in sents:
        # create binary feature vectors and add to array
        vect = numpy.zeros(feat_size)
        for i in line[1:]:
            i = i.split(":")
            word = i[0]
            value = i[1]
            try:
                vect[feat_index[word]] = float(value)
            except:
                pass
        vects.append(vect)
    
    return sents, vects


def predict_feats(weight_matrix, bias_array, instance_vect):
    """
    Compute and return predicted values for each target word.
    We do matrix multiplication with weight_matrix with instance_vect then add
    on bias terms from bias_array and put through sigmoid function.
    """
    # Matrix multiplication between weights and instance vect gives column vector of all results
    values = numpy.dot(weight_matrix, instance_vect.T)
    # + bias to each value
    values = [values[x] + bias_array[x] for x in xrange(len(values))]
    # Finally put each value through sigmoid function
    values = [u.sigmoid(x) for x in values]
    
    return values


