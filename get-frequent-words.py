# Finds how many sentences each word appears in and saves those over some threshold to file.


import config as c
import nltk


def read_words(f, words):
    """
    Reads lines from f, and logs how many times each word occurs by updating
    entries in words dict.
    """
    with open(f) as file:
        for line in file:
            w = tokenizer.tokenize(line.strip())
            for word in w:
                try:
                    words[word] += 1
                except:
                    words[word] = 1


if __name__ == "__main__":
    # Use regex tokenizer to keep contractions together
    tokenizer = nltk.RegexpTokenizer("[\w']+")
    words = {}
    
    # Read in words from pos/neg data files
    read_words("../data/imdb-all-sentences-pos.txt", words)
    read_words("../data/imdb-all-sentences-neg.txt", words)
    
    # Put the words that occur in at least m sentences into a new dictionary
    m = 1300
    freq_words = {}
    count = 0
    for w in sorted(words, key=words.get, reverse=True):
        # Stop when we get to threshold
        if (words[w] < m):
            break
        #print w, words[w]
        freq_words[w] = words[w]
        count+=1
    
    # Remove stop words from contention
    with open("stopwords.txt") as file:
        for word in file:
            freq_words.pop(word.strip(), None)
    
    # Print out frequent words
    #for i in sorted(freq_words, key=freq_words.get, reverse=True):
    #    print i, freq_words[i]
    
    # Save frequent words to file
    with open(c.output_dir + "frequent-words.txt", "w") as file:
        for w in freq_words:
            if w.isalpha():
                file.write(w + "\n")
    