# Match skills, job titles, experience, and education from resumes and job descriptions
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
class resume_matcher:
    def __init__(self, resumes, job_descriptions, nlp_model):
        self.resumes = resumes
        self.job_descriptions = job_descriptions
        self.nlp_model = nlp_model

    def match_skills(self):
        resume_skills = self.extract_skills(self.resumes)
        job_skills = self.extract_skills(self.job_descriptions)
        return resume_skills, job_skills

    def match_job_titles(self):
        resume_job_titles = self.extract_job_titles(self.resumes)
        job_job_titles = self.extract_job_titles(self.job_descriptions)
        return resume_job_titles, job_job_titles

    def match_experience(self):
        resume_experience = self.extract_experience(self.resumes)
        job_experience = self.extract_experience(self.job_descriptions)
        return resume_experience, job_experience

    def match_education(self):
        resume_education = self.extract_education(self.resumes)
        job_education = self.extract_education(self.job_descriptions)
        return resume_education, job_education
    
    def match_keywords(self, n=15):
        resume_keywords = self.extract_keywords(self.resumes, n)
        job_keywords = self.extract_keywords(self.job_descriptions, n)
        return resume_keywords, job_keywords

    # use cosine similarity to match resumes and job descriptions
    # keyword matching using cosine similarity
    def keyword_matching(self, resume_keywords, job_keywords):
        vectorizer = TfidfVectorizer().fit_transform([resume_keywords, job_keywords])
        cos_sim = cosine_similarity(vectorizer[0:1], vectorizer[1:2])[0][0]
        return cos_sim
    
    # skill matching using cosine similarity
    def skill_matching(self, resume_skills, job_skills):
        vectorizer = TfidfVectorizer().fit_transform([' '.join(resume_skills), ' '.join(job_skills)])
        cos_sim = cosine_similarity(vectorizer[0:1], vectorizer[1:2])[0][0]
        return cos_sim
    
    # job title matching using cosine similarity
    def job_title_matching(self, resume_job_titles, job_job_titles):
        vectorizer = TfidfVectorizer().fit_transform([' '.join(resume_job_titles), ' '.join(job_job_titles)])
        cos_sim = cosine_similarity(vectorizer[0:1], vectorizer[1:2])[0][0]
        return cos_sim
    
    # experience matching using cosine similarity
    def experience_matching(self, resume_experience, job_experience):
        vectorizer = TfidfVectorizer().fit_transform([' '.join(resume_experience), ' '.join(job_experience)])
        cos_sim = cosine_similarity(vectorizer[0:1], vectorizer[1:2])[0][0]
        return cos_sim
    
    # education matching using cosine similarity
    def education_matching(self, resume_education, job_education):
        vectorizer = TfidfVectorizer().fit_transform([' '.join(resume_education), ' '.join(job_education)])
        cos_sim = cosine_similarity(vectorizer[0:1], vectorizer[1:2])[0][0]
        return cos_sim
    
    # overall matching score in the following order:
    # Skills       → 0.40
    # Experience   → 0.25
    # Education    → 0.15
    # Job Titles   → 0.10
    # Keywords     → 0.10
    def overall_matching_score(self, resume, job):
        skills_score = self.skill_matching(resume['skills'], job['skills']) * 0.40
        experience_score = self.experience_matching(resume['experience'], job['experience']) * 0.25
        education_score = self.education_matching(resume['education'], job['education']) * 0.15
        job_title_score = self.job_title_matching(resume['job_titles'], job['job_titles']) * 0.10
        keywords_score = self.keyword_matching(resume['keywords'], job['keywords']) * 0.10
        
        return skills_score + experience_score + education_score + job_title_score + keywords_score
