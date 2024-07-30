from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

app = Flask(__name__)

# Load and preprocess the training dataset
data = pd.read_csv('train.csv')
data['Mileage'] = data['Mileage'].str.replace(' km', '').str.replace(',', '').astype(float)
data['Levy'] = data['Levy'].replace('-', np.nan).astype(float)
data['Levy'] = data['Levy'].fillna(data['Levy'].mean())

# Convert categorical columns to strings
categorical_cols = ['Manufacturer', 'Model', 'Category', 'Leather interior', 'Fuel type', 'Gear box type',
                    'Drive wheels', 'Color']
for col in categorical_cols:
    data[col] = data[col].astype(str)


# Feature Engineering
def combine_features(row):
    return ' '.join([
        str(row['Manufacturer'] or ''),
        str(row['Model'] or ''),
        str(row['Category'] or ''),
        str(row['Leather interior'] or ''),
        str(row['Fuel type'] or ''),
        str(row['Gear box type'] or ''),
        str(row['Drive wheels'] or ''),
        str(row['Color'] or ''),
        str(row['Engine volume'] or ''),
        str(row['Mileage'] or ''),
        str(row['Price'] or '')
    ])


data['combined_features'] = data.apply(combine_features, axis=1)

# Create TF-IDF Matrix and compute Similarity Scores
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(data['combined_features'])
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/recommend_by_features', methods=['POST'])
def recommend_by_features():
    features = request.json

    # Fill missing features with default or mean values
    filled_features = {
        'Manufacturer': features.get('Manufacturer', ''),
        'Model': features.get('Model', ''),
        'Category': features.get('Category', ''),
        'Leather interior': features.get('Leather interior', ''),
        'Fuel type': features.get('Fuel type', ''),
        'Engine volume': features.get('Engine volume', ''),
        'Mileage': features.get('Mileage', np.nan),
        'Gear box type': features.get('Gear box type', ''),
        'Drive wheels': features.get('Drive wheels', ''),
        'Color': features.get('Color', ''),
        'Price': features.get('Price', np.nan)
    }

    features_df = pd.DataFrame([filled_features])
    features_df['Mileage'] = features_df['Mileage'].replace(' km', '').astype(float)
    features_df['Price'] = pd.to_numeric(features_df['Price'], errors='coerce')

    # Ensure Mileage and Price are not NaN
    features_df['Mileage'] = features_df['Mileage'].fillna(data['Mileage'].mean())
    features_df['Price'] = features_df['Price'].fillna(data['Price'].mean())

    features_df['combined_features'] = features_df.apply(combine_features, axis=1)
    test_tfidf_matrix = tfidf_vectorizer.transform(features_df['combined_features'])

    test_cosine_sim = linear_kernel(test_tfidf_matrix, tfidf_matrix)

    # Get recommendations
    sim_scores = list(enumerate(test_cosine_sim[0]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]  # Exclude the first result (itself)
    car_indices = [i[0] for i in sim_scores]

    recommended_cars = data.iloc[car_indices][['ID', 'Manufacturer', 'Model', 'Price', 'Mileage']].to_dict(
        orient='records')
    return jsonify(recommended_cars)


if __name__ == '__main__':
    app.run(debug=True)
