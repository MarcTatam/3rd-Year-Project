from nltk.corpus import stopwords

class Word:
    """Class for representing a word"""
    def __init__(self, word:str):
        """Constructor
        
        Args
        self - instance identifier
        word - word being represented"""
        self.word = word
        self.distribution = [[0]*24]*61

