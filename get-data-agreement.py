# Get training data (sentences) to learn agreement with.
# We select different datasets for every pair of words to optimise the agreement measure.
# dDtaset for pair x and y: N sents containing x, N sents containing y, and N sents containing neither.


import lib.utility as u
import config as c
import random
import itertools


if __name__ == "__main__":
    # Load target words into memory
    target_words = u.get_target_words()
    
    # Get highly related pairs of words to get agreement data for
    #pairs = u.get_lines(c.output_dir + "target-word-high-pairs.txt")
    
    # Store all sentences in memory
    sentences_with = set(u.get_lines(c.output_dir + "sentences-with.txt"))
    sentences_without = set(u.get_lines(c.output_dir + "sentences-without.txt"))
    
    # Get all sentences used already - using a dict (so we only care about sentences used for specific word)
    used_sentences = {}
    for word in target_words:
        with open(c.train_data_dir + word + ".txt") as file:
            sents = set()
            for line in file:
                sents.add(line[3:].strip())
            used_sentences[word] = sents
    
    # Get sentences with that we haven't used, and sentences without that we haven't used, using set difference
    #sentences_with_available = list(sentences_with.difference(used_sentences))
    sentences_without_available = list(sentences_without.difference(used_sentences))
    
    
    # We now select agreement sentences from the remaining available batch
    # N is number of sentences containing x, y, and neither - 3*N is total number of sentences we will have
    N = c.agreement_data_size/3.0
    
    # Speed up process by using a dict to store all sentences for one word
    # now on every iteration we can check if the word's sentences have been selected already
    # this is also good because we use the same sets of sentences as much as possible
    sent_dict = {}
    
    i = 0
    # Iterate through pairs
    #for pair in pairs:
    for pair in itertools.combinations(target_words, r=2):
        #if i == 10:
        #    break
        
        # Get words from pair (take just first two elements from pair)
        #pair = pair.split()[0:2]
        print pair
        t1 = pair[0]
        t2 = pair[1]
        
        # Get new sentences with that are available to use (using those used already for each word in pair)
        used = used_sentences[t1].union(used_sentences[t2])
        sentences_with_available = list(sentences_with.difference(used))
        
        # Shuffle the available sentences each time
        random.shuffle(sentences_with_available)
        random.shuffle(sentences_without_available)
        
        # Array to hold all selected sentences from this pair
        selected_sentences = []
        
        # Go through both words in pair and select their sentences
        for word in pair:
            # get N sentences containing word
            #print "--containing", word
            
            # Check if we have selected sentences for this word already
            # if we have, we can just get them from the dict and add directly to the selected_sentences array
            if word in sent_dict:
                print "Yes, getting from dict"
                selected_sentences = selected_sentences + sent_dict[word]
            # if we haven't we need to go ahead and get them from the file, and then add to array AND dictionary
            else:
                print "No, selecting sentences..."
                
                # temp array to hold current word sentences
                temp_sents = []
                
                count = 0
                for x in sentences_with_available:
                    if count == N:
                        break
                    if word in x and len(x.split()) >= 4:
                        temp_sents.append(x)
                        count += 1
                print count
                
                # add another loop looking at shorter sentences, only if we didn't get enough longer sentences
                if count < N:
                    print "finding shorter sentences..."
                    for x in sentences_with_available:
                        if count == N:
                            break
                        # get some more sentences, this time lowering the threshold length
                        # and checking we don't have the sentence already
                        #if word in x and len(x.split()) >= 1 and x not in selected_sentences:
                        if word in x and x not in selected_sentences:
                            temp_sents.append(x)
                            count += 1
                    print count
                
                # Add selected sents to array and dict
                selected_sentences = selected_sentences + temp_sents
                sent_dict[word] = temp_sents
            
            print "running total:", len(selected_sentences)
        
        # Finally select N sentences containing neither target word
        print "--containing neither"
        count = 0
        for x in sentences_without_available:
            if count == N:
                break
            if len(x.split()) >= 10:
                selected_sentences.append(x)
                count += 1
        print count
        
        print "running total:", len(selected_sentences)
        
        # Print message alerting if not enough sentences were found for pair
        if len(selected_sentences) < 3*N:
            print "Oops: There were not enough sentences for", t1, t2, "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        print ""
        
        # Save sentences to file with label indicating which words are present:
        # 0 - neither, 1 - t1 present, 2 - t2 present
        with open(c.output_dir + "agreement-data/" + t1 + "-" + t2 + ".txt", "w") as file:
            for x in selected_sentences:
                l = 0
                if t1 in x:
                    l = 1
                elif t2 in x:
                    l = 2
                file.write(str(l) + " " + x + "\n")
        
        i += 1
    
    #print len(sent_dict)
    