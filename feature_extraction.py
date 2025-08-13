# Extract features from job descriptions and resumes using TF-IDF
import re
from sklearn.feature_extraction.text import TfidfVectorizer
class feature_extraction:
    def __init__(self, documents):
        self.documents = documents
        self.vectorizer = TfidfVectorizer()

    def fit_transform(self):
        return self.vectorizer.fit_transform(self.documents)

    def get_feature_names(self):
        return self.vectorizer.get_feature_names_out()

    def transform(self, new_documents):
        return self.vectorizer.transform(new_documents)
    
    # Using NER to extract skills from resumes and job descriptions
    def extract_skills(self, nlp_model):
        skills = []
        for doc in self.documents:
            nlp_doc = nlp_model(doc)
            doc_skills = [ent.text for ent in nlp_doc.ents if ent.label_ == 'SKILL']
            skills.append(doc_skills)
        return skills
    
    # Using NER to extract job titles from resumes and job descriptions
    def extract_job_titles(self, nlp_model):
        job_titles = []
        for doc in self.documents:
            nlp_doc = nlp_model(doc)
            doc_job_titles = [ent.text for ent in nlp_doc.ents if ent.label_ == 'JOB_TITLE']
            job_titles.append(doc_job_titles)
        return job_titles
    
    # Using Regex or Date Ranges to extract experience from resumes and job descriptions
    def extract_experience(self):
        experience = []
        for doc in self.documents:
            years = re.findall(r'\b\d{1,2}\s*(?:years?|yrs?)\b', doc, re.IGNORECASE)
            experience.append(years)
        return experience   

    # Using NER to extract education from resumes and job descriptions
    def extract_education(self, nlp_model):
        education = []
        for doc in self.documents:
            nlp_doc = nlp_model(doc)
            doc_education = [ent.text for ent in nlp_doc.ents if ent.label_ == 'EDUCATION']
            education.append(doc_education)
        return education
    
    # Using TF-IDF to extract keywords from resumes and job descriptions
    def extract_keywords(self, n=15):
        tfidf_matrix = self.fit_transform()
        feature_names = self.get_feature_names()
        keywords = []
        for i in range(tfidf_matrix.shape[0]):
            tfidf_vector = tfidf_matrix[i].toarray()[0]
            top_indices = tfidf_vector.argsort()[-n:][::-1]
            top_keywords = [feature_names[j] for j in top_indices]
            keywords.append(top_keywords)
        return keywords
    
    