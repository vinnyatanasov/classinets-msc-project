# Get training data or each target word to train binary predictor.


import lib.utility as u
import config as c
import random


if __name__ == "__main__":
    # Load the targets into memory
    target_words = u.get_target_words()
    
    # Store all sentences in memory
    sentences_with = u.get_lines(c.output_dir + "sentences-with.txt")
    sentences_without = u.get_lines(c.output_dir + "sentences-without.txt")
    all_sentences = sentences_with + sentences_without
    
    # For each target word, find sample sentences (pos and neg)
    # N is number of pos/neg instances to select
    N = c.train_data_size/2.0
    i = 0
    for word in target_words:
        #if i == 20:
        #    break
        
        # Shuffle data first
        random.shuffle(sentences_with)
        random.shuffle(all_sentences)
        
        # Get positive instances (containing word)
        pos_data = []
        count = 0
        for s in sentences_with:
            # stop when we have enough
            if count == N:
                break
            
            words = s.split()
            # make sure sentence contains target word and meets length threshold
            if (word in words and len(words) >= 8):
                pos_data.append(s)
                count += 1
        
        if len(pos_data) < N:
            N = len(pos_data)
        
        # Get negative instances (not containing word)
        neg_data = []
        count = 0
        for s in all_sentences:
            # stop when we have enough
            if count == N:
                break
            
            words = s.split()
            # make sure the sentence is long enough, hasn't been chosen already, and doesn't contain target word
            if (len(words) >= 10 and word not in words):
                # just to be doubly sure (it can't really be in there, but let's be safe)
                if s not in pos_data:
                    neg_data.append(s)
                    count += 1
        
        print word, len(pos_data), len(neg_data)
        
        i += 1
        
        # Write the train data to file
        #with open(c.output_dir + "target-train-data/" + word + ".txt", "w") as file:
        #    for s in pos_data:
        #        file.write("+1 " + s + "\n")
        #    for s in neg_data:
        #        file.write("-1 " + s + "\n")
    