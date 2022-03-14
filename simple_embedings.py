import re
import numpy as np
import matplotlib.pyplot as plt

class SimpleEmbedings:
    def __init__(self, emb_dim, window=2):
        self.emb_dim = emb_dim
        self.window = window
        
    def tokenize(self, text):
        pattern = re.compile(r'[A-Za-z]{2,}')
        self.tokens = list(set(pattern.findall(text.lower())))
    
    def make_word_id(self):
        self.word_to_id = {}
        self.id_to_word = {}
        for i, token in enumerate(self.tokens):
            self.word_to_id[token] = i
            self.id_to_word[i] = token
        self.vocab_size = len(self.word_to_id)
        assert self.vocab_size == len(self.tokens), "self.vocab_size != len(tokens)"
    
    def generate_training_data(self):
        X, y = [], []
        for i in range(self.vocab_size):
            idx = list(range(max(0, i - self.window), i)) + list(range(i, min(self.vocab_size, i + self.window + 1)))
            for j in idx:
                if j != i:
                    center_word = self.word_to_id[self.tokens[i]]
                    one_hot = [0] * self.vocab_size
                    one_hot[center_word] = 1
                    X.append(one_hot)
                    
                    word_in_window = self.word_to_id[self.tokens[j]]                                      
                    one_hot = [0] * self.vocab_size
                    one_hot[word_in_window] = 1
                    y.append(one_hot)
        self.X = np.array(X)
        self.y = np.array(y)
    
    def train(self, n_iter=50, learning_rate=0.05, info=True):
        self.output = {}
        self.output['weight_1'] = np.random.randn(self.vocab_size, self.emb_dim)
        self.output['weight_2'] = np.random.randn(self.emb_dim, self.vocab_size)
        
        history = []
        for i in range(n_iter):
            self.output["a1"] = self.X @ self.output['weight_1']
            self.output["a2"] = self.output["a1"] @ self.output['weight_2']
            self.output["prob"] = self.softmax(self.output["a2"])
            
            da2 = self.output["prob"] - self.y
            dw2 = self.output["a1"].T @ da2
            da1 = da2 @ self.output['weight_2'].T
            dw1 = self.X.T @ da1
            self.output['weight_1'] -= learning_rate * dw1
            self.output['weight_2'] -= learning_rate * dw2

            cross_entropy_loss = - np.sum(np.log(self.output["prob"]) * self.y)
            history.append(cross_entropy_loss)
        
        if info:
            plt.figure(figsize=(7,2))
            plt.title("Cross entropy loss / Iterations")
            plt.plot(range(len(history)), history)
            plt.show()
        
    def softmax(self, X):
        res = [0] * len(X) 
        for i, x in enumerate(X):
            exp = np.exp(x)
            res[i] = exp / exp.sum()
        return np.array(res)
    
    
    def make_embedings(self, text, info=True):
        self.tokenize(text)
        if info: print(f"Tokenize is done.\nNumber of tokens: {len(self.tokens)}")
        self.make_word_id()
        if info: print(f"Words-id-words is done.")
        self.generate_training_data()
        if info: print(f"Generating of training data with the window size({self.window}) is done.\nLength of train data: {len(self.X)}\nTraining start.")
        self.train(info=info)
        if info: print(f"Training is done.\nYou may get embedings for 'word' try: model.get_embedding('word')\n")
    
    
    def get_embedding(self, word):
        try:
            idx = self.word_to_id[word]
        except KeyError:
            print(f"There is not the word '{word}' in tokens")
        one_hot = [0] * self.vocab_size
        one_hot[idx] = 1
        return one_hot @ self.output['weight_1']
    
    
    def similarity(self, word, n_first=10, outlist=False):
        try:
            idx = self.word_to_id[word]
        except KeyError:
            print(f"There is not the word '{word}' in tokens")
        one_hot = [0] * self.vocab_size
        one_hot[idx] = 1
        a1 = one_hot @ self.output['weight_1']
        a2 = a1 @ self.output['weight_2']
        prob = self.softmax([a2])
        out = []
        for i, id in enumerate(np.argsort(prob[0])[::-1][:n_first]):
            if not outlist: print(f"{i+1}) {self.id_to_word[id]}")
            else: out.append(self.id_to_word[id]) 
        if outlist: return out 