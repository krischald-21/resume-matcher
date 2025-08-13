# Preprocess the text from resumes
class text_preprocess:
    def __init__(self, text):
        self.text = text

    def to_lowercase(self):
        return self.text.lower()

    def remove_special_characters(self):
        import re
        return re.sub(r'[^a-zA-Z0-9\s]', '', self.text)

    def tokenize(self):
        return self.text.split()
    
    def stopword_removal(self, stopwords):
        tokens = self.tokenize()
        return [word for word in tokens if word not in stopwords]
    
    def preprocess(self, stopwords):
        self.to_lowercase()
        self.remove_special_characters()
        tokens = self.stopword_removal(stopwords)
        return tokens