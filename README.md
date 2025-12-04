🥊 UFC Style Analyzer: The Geometry of Combat

👉 Click Here to Launch the Live App

(Replace the link above with your actual Streamlit Cloud URL once deployed)

📖 Project Overview

This project is an interactive data science application that applies Unsupervised Machine Learning to mixed martial arts.

By analyzing career statistics of UFC veterans (5+ fights), this tool mathematically identifies "fighting archetypes" (e.g., Strikers, Grapplers, Hybrids) without relying on human opinion. It uses Principal Component Analysis (PCA) to reduce multidimensional performance metrics into a 2D "Style Map" and K-Means Clustering to automatically classify fighters.

🧮 The Math Behind the Map

1. Feature Engineering

We process raw fight data to create a "Fighter Fingerprint" based on 4 key dimensions:

Striking Volume: Significant strikes landed per fight.

Striking Accuracy: Landed / Attempted ratio.

Grappling Threat: Takedowns landed + Submission attempts per fight.

Control Time: Average time spent controlling the opponent (seconds).

2. Dimensionality Reduction (PCA)

The data is standardized (Z-Score Normalization) and fed into a PCA algorithm.

PC1 (The X-Axis): Typically represents the Grappling vs. Striking spectrum.

PC2 (The Y-Axis): Typically represents Aggression/Volume vs. Passivity.

3. Clustering (K-Means)

We apply K-Means clustering ($k=4$) to the PCA output to detect latent groups of fighters who share statistically similar approaches to combat.

✨ Key Features

Interactive 2D Map: Built with Plotly, allowing users to hover over any dot to see raw stats.

Fighter Search: Type any veteran's name (e.g., "Jon Jones") to highlight their exact location in the style space.

The Doppelgänger Engine: Uses Euclidean Distance to find the 5 fighters who are mathematically most similar to your search target.

Robust Data Loading: Automatically pulls data from GitHub mirrors, with a manual drag-and-drop fallback for reliability.

🛠️ How to Run Locally

Clone the repository:

git clone [https://github.com/YOUR_USERNAME/ufc-style-analyzer.git](https://github.com/YOUR_USERNAME/ufc-style-analyzer.git)
cd ufc-style-analyzer


Install requirements:

pip install -r requirements.txt


Run the App:

streamlit run app.py


📂 File Structure

app.py: The main application logic (Data processing, PCA, Visualization).

requirements.txt: List of Python dependencies.

📊 Data Source

Data is sourced from UFC-Fight-Data and scraped via ufcstats.com.
