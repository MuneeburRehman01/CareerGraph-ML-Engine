import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

class CareerGraphEngine:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.role_database = {
            "Software Engineer": "python, java, algorithm, system design, git",
            "Data Scientist": "python, statistics, machine learning, sql, pandas",
            "Cloud Architect": "aws, azure, terraform, networking, security",
            "Product Manager": "strategy, roadmap, user experience, agile, leadership"
        }
        self._fit_roles()

    def _fit_roles(self):
        """Prepare role vectors for similarity calculation."""
        self.roles = list(self.role_database.keys())
        self.role_vectors = self.vectorizer.fit_transform(self.role_database.values())

    def predict_role_match(self, user_skills: str) -> dict:
        """Find the best matching role for a given skill string."""
        user_vector = self.vectorizer.transform([user_skills])
        similarities = cosine_similarity(user_vector, self.role_vectors).flatten()
        best_match_idx = np.argmax(similarities)
        
        return {
            "best_match": self.roles[best_match_idx],
            "confidence": round(similarities[best_match_idx], 2),
            "similarity_scores": dict(zip(self.roles, similarities))
        }

    def identify_skill_gaps(self, user_skills: str, target_role: str) -> list:
        """Identify missing skills for a target role."""
        if target_role not in self.role_database:
            return []
        
        role_skills = set(self.role_database[target_role].split(", "))
        user_skills_set = set(user_skills.lower().replace(",", " ").split())
        return list(role_skills - user_skills_set)

if __name__ == "__main__":
    engine = CareerGraphEngine()
    my_skills = "python, git, sql, pandas"
    
    match = engine.predict_role_match(my_skills)
    print(f"Prediction for '{my_skills}':\n- Role: {match['best_match']} (Confidence: {match['confidence']})")
    
    gaps = engine.identify_skill_gaps(my_skills, "Data Scientist")
    print(f"- Skill Gaps for Data Scientist: {gaps}")