import numpy as np
def splitToWords(text):
    if len(text) > 0:
        return text.split()
    else:
        return None


def load_data(path,word2Id = None,Id2Word = None,max_sequence_length = None):
    """

    :param paths: list of paths
    :return: dictionary which contains train data,vocab size, maximum sequence length
    """
    if max_sequence_length == None:
        max_sequence_length = 0
    if word2Id == None:
        word2Id = {}
        word2Id['end'] = 0
        word2Id['unknown'] = 1
    if Id2Word == None:
        Id2Word = {}
        Id2Word[0] = 'end'
        Id2Word[1] = 'unknown'
    result = {}

    f = open(path,'r',encoding='utf8')
    lines = f.readlines()
    for line in lines:
        splts = line.strip('\n').split('\t')
        sent1 = splts[1]
        sent2 = splts[2]
        words = splitToWords(sent1)
        if len(words) > max_sequence_length:
            max_sequence_length = len(words)
        words = splitToWords(sent2)
        if len(words) > max_sequence_length:
            max_sequence_length = len(words)

    classes = 1
    data_count = len(lines)
    data_1 = np.zeros([data_count,max_sequence_length])
    data_2 = np.zeros([data_count, max_sequence_length])

    labels = np.zeros([data_count])
    i = 0
    lengths_1 = np.zeros([data_count])
    lengths_2 = np.zeros([data_count])
    f = open(path,'r',encoding='utf8')
    lines = f.readlines()
    for line in lines[1:]:
        line_splits = line.strip('\n').split('\t')
        sent1 = line_splits[1]
        sent2 = line_splits[2]
        score = float(line_splits[3])
        words_1 = splitToWords(sent1)
        words_2 = splitToWords(sent2)
        update(word2Id,Id2Word,words_1)
        update(word2Id, Id2Word, words_2)

        lengths_1[i] = len(words_1)
        lengths_2[i] = len(words_2)
        data_1[i,:] = text2Ids(words_1,word2Id,max_sequence_length)
        data_2[i,:] = text2Ids(words_2,word2Id,max_sequence_length)
        labels[i] = score
        i += 1
    print("data loaded successful!!!\n")
    result['data_1'] = np.array(data_1)
    result['data_2'] = np.array(data_2)
    result['length1'] = np.array(lengths_1)
    result['length2'] = np.array(lengths_2)
    result['label'] = np.array(labels)
    result['word2Id'] = word2Id
    result['Id2Word'] = Id2Word
    result['max_sequence_length'] = max_sequence_length
    result['vocab_size'] = len(word2Id.keys())-2
    result['total_classes'] = classes
    return result

def load_test_data(path,max_sequence_length,word2Id,Id2Word):
    """

    :param paths: list of paths
    :return: dictionary which contains train data,vocab size, maximum sequence length
    """

    f = open(path,'r',encoding='utf8')
    lines = f.readlines()
    data_length = 0
    for line in lines:
        line_splits = line.strip('\n').split('\t')
        sent1 = line_splits[1]
        sent2 = line_splits[2]
        words1 = splitToWords(sent1)
        words2 = splitToWords(sent2)
        if len(words1) <= max_sequence_length and len(words2) <= max_sequence_length:
            data_length += 1

    classes = 1

    data_1 = np.zeros([data_length,max_sequence_length])
    data_2 = np.zeros([data_length,max_sequence_length])
    lengths_1 = np.zeros([data_length])
    lengths_2 = np.zeros([data_length])
    labels = np.zeros([data_length])
    i = 0
    result = {}
    f = open(path,'r',encoding='utf8')
    lines = f.readlines()
    for line in lines[1:]:
        line_splits = line.strip('\n').split('\t')
        sent1 = line_splits[1]
        sent2 = line_splits[2]
        score = float(line_splits[3])
        words_1 = splitToWords(sent1)
        words_2 = splitToWords(sent2)

        if len(words_1) <= max_sequence_length and len(words_2) <= max_sequence_length:
            lengths_1[i] = len(words_1)
            lengths_2[i] = len(words_2)
            update(word2Id,Id2Word,words_1)
            update(word2Id, Id2Word, words_2)

            data_1[i,:] = text2Ids(words_1,word2Id,max_sequence_length)
            data_2[i,:] = text2Ids(words_2,word2Id,max_sequence_length)
            labels[i] = score
            i += 1
    print("data loaded successful!!!\n")
    result['data_1'] = np.array(data_1)
    result['data_2'] = np.array(data_2)
    result['length1'] = np.array(lengths_1)
    result['length2'] = np.array(lengths_2)
    result['label'] = np.array(labels)
    result['word2Id'] = word2Id
    result['Id2Word'] = Id2Word
    result['vocab_size'] = len(word2Id.keys())-2
    result['total_classes'] = classes
    return result


def load_data_chars(paths=None,Id2Vec=None,score_position=0):
    """
    :param paths: list of paths
    :return: dictionary which contains train data,vocab size, maximum sequence length
    """

    if paths == None:
        ## default we load this paths
        paths = []
        paths.append(r'C:\Users\pravi\PycharmProjects\Sentence_similarity\data\sts\semeval-sts\2012\MSRpar.train.tsv')
        paths.append(r'C:\Users\pravi\PycharmProjects\Sentence_similarity\data\sts\semeval-sts\2012\MSRpar.test.tsv')
        paths.append(r'C:\Users\pravi\PycharmProjects\Sentence_similarity\data\sts\semeval-sts\2012\OnWN.test.tsv')
        paths.append(r'C:\Users\pravi\PycharmProjects\Sentence_similarity\data\sts\semeval-sts\2012\SMTeuroparl.test.tsv')
        paths.append(r'C:\Users\pravi\PycharmProjects\Sentence_similarity\data\sts\semeval-sts\2012\SMTeuroparl.train.tsv')
        paths.append(r'C:\Users\pravi\PycharmProjects\Sentence_similarity\data\sts\semeval-sts\2012\SMTnews.test.tsv')

    max_sequence_length = 0
    max_word_len = 0
    word2Id = {}
    char2Id = {}
    char2Id['unknown'] = 0
    word2Id['end'] = 0
    word2Id['unknown'] = 1
    Id2Word = {}
    Id2Word[0] = 'end'
    Id2Word[1] = 'unknown'
    Id2Char = {}
    Id2Char[0] = 'unknown'
    result = {}
    labels = []
    data_count = 0
    classes = 1
    for path in paths:
        f = open(path,'r',encoding='utf8')
        lines = f.readlines()
        for line in lines[1:]:
            splits = line.strip('\n').split('\t')
            score = float(splits[score_position])
            sentence_1 = splits[1]
            sentence_2 = splits[2]
            words_1 = splitToWords(sentence_1)
            words_2 = splitToWords(sentence_2)
            if len(words_1) > max_sequence_length:
                max_sequence_length = len(words_1)
            if len(words_2) > max_sequence_length:
                max_sequence_length = len(words_2)
            words_ = words_1 + words_2
            for word in words_:
                if len(list(word)) > max_word_len:
                    max_word_len = len(list(word))
        data_count += len(lines)
    data_1 = np.zeros([data_count,max_sequence_length],dtype=int)
    data_2 = np.zeros([data_count, max_sequence_length], dtype=int)
    data_char_1 = np.zeros([data_count,max_sequence_length,max_word_len],dtype=int)
    data_char_2 = np.zeros([data_count, max_sequence_length, max_word_len], dtype=int)
    labels = np.zeros([data_count,classes])
    i = 0
    for path in paths:
        f = open(path,'r',encoding='utf8')
        lines = f.readlines()
        for line in lines[1:]:

            splits = line.strip('\n').split('\t')
            score = float(splits[score_position])
            sentence_1 = splits[1]
            sentence_2 = splits[2]
            words_1 = splitToWords(sentence_1)
            words_2 = splitToWords(sentence_2)
            words_ = words_1 + words_2
            for word in words_:
                update(char2Id,Id2Char,list(word))

            update(word2Id,Id2Word,words_)
            data_1[i,:] = text2Ids(words_1,word2Id,max_sequence_length)
            data_2[i, :] = text2Ids(words_2, word2Id, max_sequence_length)
            for j in range(len(words_1)):
                data_char_1[i,j,:] = text2Ids(words_1[j],char2Id,max_word_len)
            for j in range(len(words_2)):
                data_char_2[i, j, :] = text2Ids(words_2[j], char2Id, max_word_len)
            labels[i,0] = score
            i += 1

    print("data loaded successful!!!\n")
    result['data_1'] = np.array(data_1)
    result['data_2'] = np.array(data_2)
    result['label'] = np.array(labels)
    result['word2Id'] = word2Id
    result['Id2Word'] = Id2Word
    result['char2Id'] = char2Id
    result['Id2Char'] = Id2Char
    result['data_char_1'] = np.array(data_char_1)
    result['data_char_2'] = np.array(data_char_2)
    result['max_word_len'] = max_word_len
    result['max_sequence_length'] = max_sequence_length
    result['vocab_size'] = len(word2Id.keys())-2
    result['total_classes'] = classes
    return result

def load_test_data_chars(paths,char2Id,word2Id,max_sequence_length,max_word_len,score_position=0):
    result = {}
    data_length = 0
    classes = 1
    for path in paths:
        f = open(path,'r',encoding='utf8')
        lines = f.readlines()
        data_length = len(lines)

    data_1 = np.zeros([data_length,max_sequence_length])
    data_2 = np.zeros([data_length,max_sequence_length])
    data_char_1 = np.zeros([data_length,max_sequence_length,max_word_len])
    data_char_2 = np.zeros([data_length,max_sequence_length,max_word_len])
    labels = np.zeros([data_length,classes])
    i = 0
    for path in paths:
        f = open(path,'r',encoding='utf8')
        lines = f.readlines()
        for line in lines[1:]:

            splits = line.strip('\n').split('\t')
            score = float(splits[score_position])
            sentence_1 = splits[1]
            sentence_2 = splits[2]
            words_1 = splitToWords(sentence_1)
            words_2 = splitToWords(sentence_2)

            data_1[i,:] = text2Ids(words_1,word2Id,max_sequence_length)
            data_2[i, :] = text2Ids(words_2, word2Id, max_sequence_length)
            for j in range(len(words_1)):
                data_char_1[i,j,:] = text2Ids(words_1[j],char2Id,max_word_len)
            for j in range(len(words_2)):
                data_char_2[i, j, :] = text2Ids(words_2[j], char2Id, max_word_len)
            labels[i,0] = score
            i += 1

    print("data loaded successful!!!\n")
    result['data_1'] = np.array(data_1)
    result['data_2'] = np.array(data_2)
    result['label'] = np.array(labels)
    result['word2Id'] = word2Id
    result['char2Id'] = char2Id
    result['data_char_1'] = np.array(data_char_1)
    result['data_char_2'] = np.array(data_char_2)

    return result

def update(word2Id,Id2Word,words):
    keys = list(word2Id.keys())
    for word in words:
        if word not in keys:
            word2Id[word] = len(keys)
            Id2Word[len(keys)] = word
            keys.append(word)

def text2Ids(words,word2Id,max_sequence_len):
    a = np.zeros(max_sequence_len,dtype=int)
    keys = word2Id.keys()
    for i,word in enumerate(words):
        if word in keys:
            a[i] = word2Id[word]
        else:
            a[i] = word2Id['unknown']
    return a
