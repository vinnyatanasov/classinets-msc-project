# SCL implementation for feature expansion using target word weight vectors learnt for ClassiNet.


import lib.utility as u
import numpy


def get_sentence_data(data_file):
    """
    Returns binary feature vectors (d) and corresponding labels (l)
    made from sentences in data_file.
    """
    # Read sentences from file
    sents = []
    with open(data_file) as file:
        for line in file:
            # add them as arrays to make expansion easier
            sents.append(line.strip().split())
    
    # Get binary feature vects (d) and labels (l) from sents
    d = []
    l = []
    for line in sents:
        vect = numpy.zeros(feature_size)
        for i in line[1:]:
            i = i.split(":")
            word = i[0]
            value = i[1]
            #print word, value
            try:
                vect[features_index[word]] = float(value)
            except:
                pass
        l.append(line[0])
        d.append(vect)
    
    return d, l


if __name__ == "__main__":
    # Get feature vectors
    print "generating feature vectors..."
    features_index = u.read_features_file("output-700/feat-space.txt")
    feature_size = len(features_index)
    target_words = u.get_target_words()
    
    # Now let's do SCL
    print "getting targets..."
    targets = u.get_target_words()
    
    # Load target word weight vectors as rows into matrix W
    print "creating W..."
    W, _ = u.get_weight_matrix(targets)
    
    # Perform SVD on W
    print "performing SVD(W)..."
    WT = W.T
    # SVD returns U, S, V.T
    U, S, VT = numpy.linalg.svd(WT, full_matrices=False)
    
    #print "W"
    #print W.shape
    #print "WT"
    #print WT.shape
    #print ""
    
    #print U
    #print "U"
    #print U.shape
    #print ""
    #print "UT"
    #print U.T.shape
    #print ""
    
    #print D
    #print "S"
    #print S.shape
    #print ""
    
    #print V
    #print "VT"
    #print VT.shape
    #print ""
    
    # Projection (theta) needs to contain the top columns of U (or rows of U.T) as rows
    # then if we transpose it, we get it back to columns (so it'll be n x h)
    # which means when we dot with x (which is 1 x n), we get a 1 x h matrix (array) of augmented feats
    h = 25
    #theta2 = U.T[:h]
    #theta = U[:h]
    theta = U.T[:h].T
    
    print "Theta"
    print "h:", h
    print theta.shape
    print ""
    
    
    # Get the sentences to expand with SCL
    # Test sentences
    #test_data, test_labels = get_sentence_data("tree-test/test-full.txt")
    # Train sentences
    train_data, train_labels = get_sentence_data("train-6000.txt")
    
    # Compute extra features using theta
    # Test data
    #print "computing new feats (test) using projection..."
    #extra_feats_test = []
    #for vect in test_data:
    #    new_feats = numpy.dot(vect, theta)
    #    extra_feats_test.append(new_feats)
    
    # Write new feats for test data to file (in same order as data)
    #with open("scl-proj-feats-testa.txt", "w") as file:
    #    for x in extra_feats_test:
    #        for i in x:
    #            file.write(str(i) + " ")
    #        file.write("\n")
    #print "done test expansion"
    
    # Train data
    print "computing new feats (train) using projection..."
    extra_feats_train = []
    for vect in train_data:
        new_feats = numpy.dot(vect, theta)
        extra_feats_train.append(new_feats)
    
    # Write new feats for train data to file (in same order as data)
    with open("scl-proj-feats-train6.txt", "w") as file:
        for x in extra_feats_train:
            for i in x:
                file.write(str(i) + " ")
            file.write("\n")
    print "done train expansion"
    