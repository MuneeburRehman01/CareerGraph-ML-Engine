# 📈 CareerGraph-ML-Engine

[![ML-Science](https://img.shields.io/badge/Focus-Machine%20Learning%20Science-red.svg)]()
[![Scikit-learn](https://img.shields.io/badge/Stack-Scikit--learn-orange.svg)]()
[![Pandas](https://img.shields.io/badge/Stack-Pandas-blue.svg)]()

CareerGraph is a graph-based **Machine Learning Engine** designed to analyze career trajectories, predict career paths, and identify critical skill gaps. It uses semantic similarity and graph centrality to model the relationship between professional roles and technical skills.

## 🌟 Core Functionality

- **Skill Gap Discovery:** Compares a user's current skill set against the requirements of their target role.
- **Trajectory Prediction:** Predicts the most likely 'next role' based on historical data of career transitions.
- **Semantic Role Mapping:** Clusters similar job titles using NLP-based embeddings to handle role variations (e.g., 'Data Scientist' vs. 'ML Engineer').
- **Graph-Based Importance:** identifies 'pivot skills' that are essential for multiple high-growth career paths.

## 🛠️ Machine Learning Pipeline

1.  **Data Ingestion:** Loads career histories and job descriptions (standardized to JSON).
2.  **Vectorization:** Encodes skills and roles using TF-IDF or pre-trained sentence transformers.
3.  **Similarity Analysis:** Uses Cosine Similarity to find roles with high skill overlap.
4.  **Clustering:** groups roles based on shared competencies.

## 🚀 Installation & Usage

`ash
# Clone the repository
git clone https://github.com/MuneeburRehman01/CareerGraph-ML-Engine.git

# Install requirements
pip install numpy pandas scikit-learn sentence-transformers

# Train the model with local dataset
python engine.py --train --data ./datasets/professional_profiles.csv

# Run prediction for a profile
python engine.py --predict --role "Junior Developer" --skills "Python, Git, SQL"
`

## 📜 Roadmap

- [ ] Implementation of Graph Neural Networks (GNNs) for more complex path prediction.
- [ ] Real-time job market data integration via APIs (LinkedIn, Indeed).
- [ ] Skill-based salary forecasting using regression models.

---
Developed with 🚀 by [Muneeb ur Rehman](https://www.linkedin.com/in/muneeb-ur-rehman-a13977207/)