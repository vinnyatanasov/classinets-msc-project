# Creates k-NN graph and mutual k-NN graph from edge list file.


import config as c


if __name__ == "__main__":
    # k-NN config variables
    directed = True
    k = 5
    
    # Read edges from file into dictionary in format:
    # { word1: {word:2, word2:3, word4:4},
    #   word2: {...}, ... }
    graph_dict = {}
    with open(c.output_dir + "edges-directed.txt") as file:
        for line in file:
            l = line.strip().split()
            # get each node and weight from line
            node1 = l[0]
            node2 = l[1]
            weight = float(l[2])
            
            try:
                edges = graph_dict[node1]
            except:
                edges = {}
            
            edges[node2] = weight
            graph_dict[node1] = edges
            
            # if undirected, then we have to include reverse edge too
            if not directed:
                try:
                    edges = graph_dict[node2]
                except:
                    edges = {}
                
                edges[node1] = weight
                graph_dict[node2] = edges
    
    # Top k edges selection
    # first, we must choose the top k edges for each node (and ignore possible duplicates for now)
    # we only choose edges for k highest weighted words for each vertex
    top_k_edges = []
    
    # loop through dict and select k edges from each vertex
    for w, d in graph_dict.iteritems():
        #print w, "..."
        # w is current word, d is associated dict
        count = 0
        for i in sorted(d, key=d.get, reverse=True):
            # stop at k
            if count == k:
                break
            # add edge to top k array
            top_k_edges.append((w, i, d[i]))
            count += 1
    
    
    # k-NN graph edge selection
    # now, using the top k edges, we can convert to usable k-NN format by simply removing
    # duplicates if we're doing an undirected graph, or doing nothing if directed.
    if directed:
        knn_edges = list(top_k_edges)
    else:
        # make a copy of the top k edges to begin with
        knn_edges = []
        
        # loop through top k edges and look for duplicates - we first check to see we haven't added
        # reverse already, and if not, we add it to knn_edges - that we, we eliminate duplicates
        for edge_a in top_k_edges:
            # get nodes
            i = edge_a[0]
            j = edge_a[1]
            
            # check we don't have reverse already in knn_edges
            found_reverse = False
            for edge_b in knn_edges:
                ii = edge_b[0]
                jj = edge_b[1]
                
                if i == jj and j == ii:
                    found_reverse = True
                    break
            
            if not found_reverse:
                knn_edges.append(edge_a)
    
    # Print k-NN edges
    #for i in knn_edges:
    #    print i
    #print len(knn_edges)
    #print ""
    
    # Mutual k-NN graph edge selection
    # We go through the top k edges, and check if they are in the opposite's selection
    # if so, we keep them, if not, we ignore them
    mutual_knn_edges = []
    
    # we call first node i, and second node j
    for edge_a in top_k_edges:
        # get nodes
        i = edge_a[0]
        j = edge_a[1]
        
        # first check if it's in list already (for directed method)...
        if edge_a in mutual_knn_edges:
            continue
        
        # loop through other edges, to find potential match
        for edge_b in top_k_edges:
            # skip itself...
            if edge_a != edge_b:
                # get new set of nodes
                ii = edge_b[0]
                jj = edge_b[1]
                
                # check if reverse edge is a match
                if i == jj and j == ii:
                    if directed:
                        # if directed version, we just add them both to array
                        mutual_knn_edges.append(edge_a)
                        mutual_knn_edges.append(edge_b)
                    else:
                        # if it's undirected, we first make sure opposite isn't in there already
                        # - because there's no point as it's symmetric - and then add it
                        if edge_b not in mutual_knn_edges:
                            mutual_knn_edges.append(edge_a)
    
    # Print mutual k-NN edges
    #for i in mutual_knn_edges:
    #    print i
    #print len(mutual_knn_edges)
    
    # Save k-NN edges to new file
    #with open(c.output_dir + "edges-directed-5.txt", "w") as file:
    #    for i in knn_edges:
    #        file.write(i[0] + " " + i[1] + " " + str(i[2]) + "\n")
    
    # Save mutual k-NN edges to new file
    #with open(c.output_dir + "edges-directed-5-m.txt", "w") as file:
    #    for i in mutual_knn_edges:
    #        file.write(i[0] + " " + i[1] + " " + str(i[2]) + "\n")
    