# Compute cosine similarity between all pairs of target words to then sample
# the most related pairs to compute full agreement on.


import lib.utility as u
import config as c
import numpy


if __name__ == "__main__":
    # Read target words into array
    target_words = u.get_target_words()
    
    # Create matrix of weight vectors (weight vectors are rows)
    # Note: weight vectors have been L2 normalised in get_weight_vector method (with True param)
    W = numpy.array([u.get_weight_vector(word, True) for word in target_words])
    print W.shape
    
    # Dot W with itself to get cosine similarities of every pair
    M = numpy.dot(W, W.T)
    print M.shape
    
    # Print matrix of cosine sim edges to file - two ways to do this
    #dimensions = len(target_words)
    # Print only half (as it's symmetric)
    #with open(c.output_dir + "edges-cosine.txt", "w") as file:
    #    # loop over matrix, ignoring diagonals
    #    for i in xrange(dimensions):
    #        for j in xrange(i+1, dimensions):
    #            if i != j:
    #                file.write(target_words[i] + " " + target_words[j] + " " + str(M[i][j]) + "\n")
    #
    # Print all edges (except diagonals)
    #with open(c.output_dir + "edges-cosine.txt", "w") as file:
    #    # loop over matrix, ignoring diagonals
    #    for i in xrange(dimensions):
    #        for j in xrange(dimensions):
    #            if i != j:
    #                file.write(target_words[i] + " " + target_words[j] + " " + str(M[i][j]) + "\n")
    