import numpy as np
class word_embedings(object):
    dimension = 0
    def __init__(self,path=None,dimension=300,debug=False):
        if path ==None:
            # path = "data/glove_6B/glove_6B_300d.txt"
            # path = r"C:\Users\pravi\Downloads\InferSent-master\InferSent-master\dataset\GloVe\glove.840B.300d.txt"
            path = r'C:\Users\pravi\Downloads\wiki-news-300d-1M.vec\wiki-news-300d-1M.vec'
        f = open(path,'r',encoding='utf8')
        data = f.readlines()
        self.word2vec = {}
        self.word2vec['end'] = np.random.randn(300)
        self.word2vec['unknown'] = np.random.randn(300)
        if debug == False:

            self.dimension = dimension
            if debug==True:
                for line in data[:2000]:
                    entries = line.split(" ")
                    self.word2vec[entries[0]] = self.convert_To_Vec(entries[1:])
            else:
                for line in data:
                    entries = line.strip(' \n').strip('\n').split(" ")
                    if len(entries) > 300:
                        self.word2vec[entries[0]] = self.convert_To_Vec(entries[1:])
        print("word vectors are loaded successful!!")


    def convert_To_Vec(self,elements):
        vec = np.zeros(self.dimension)
        for i,element in enumerate(elements):
            vec[i] = float(element)
        return vec

# w = word_embedings(debug=False)
#
# print(":)")