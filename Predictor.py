import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Load and preprocess data as before
data = pd.read_csv('train.csv')
data['Mileage'] = data['Mileage'].str.replace(' km', '').str.replace(',', '').astype(float)
data['Levy'] = data['Levy'].replace('-', np.nan).astype(float)
data['Levy'] = data['Levy'].fillna(data['Levy'].mean())

categorical_cols = ['Manufacturer', 'Model', 'Category', 'Leather interior', 'Fuel type', 'Gear box type',
                    'Drive wheels', 'Color']
for col in categorical_cols:
    data[col] = data[col].astype(str)


def combine_features(row):
    return ' '.join([
        row['Manufacturer'],
        row['Model'],
        row['Category'],
        row['Leather interior'],
        row['Fuel type'],
        row['Gear box type'],
        row['Drive wheels'],
        row['Color'],
        str(row['Engine volume']),
        str(row['Mileage']),
        str(row['Price'])
    ])


data['combined_features'] = data.apply(combine_features, axis=1)
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(data['combined_features'])
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Load and preprocess test data similarly
test_data = pd.read_csv('test.csv')
test_data['Mileage'] = test_data['Mileage'].str.replace(' km', '').str.replace(',', '').astype(float)
test_data['Levy'] = test_data['Levy'].replace('-', np.nan).astype(float)
test_data['Levy'] = test_data['Levy'].fillna(data['Levy'].mean())

for col in categorical_cols:
    test_data[col] = test_data[col].astype(str)

test_data['combined_features'] = test_data.apply(combine_features, axis=1)
test_tfidf_matrix = tfidf_vectorizer.transform(test_data['combined_features'])
test_cosine_sim = linear_kernel(test_tfidf_matrix, tfidf_matrix)


# Recommendation Function based on features
def recommend_car_based_on_features(features_dict):
    # Create a DataFrame with the provided features
    features_df = pd.DataFrame([features_dict])

    # Ensure all required columns are present
    for col in categorical_cols + ['Engine volume', 'Mileage', 'Price', 'Levy']:
        if col not in features_df.columns:
            features_df[col] = np.nan

    # Preprocess the provided features
    features_df['Mileage'] = features_df['Mileage'].astype(float, errors='ignore')
    features_df['Levy'] = features_df['Levy'].astype(float, errors='ignore').fillna(data['Levy'].mean())

    for col in categorical_cols:
        features_df[col] = features_df[col].astype(str)

    # Combine features
    features_df['combined_features'] = features_df.apply(combine_features, axis=1)

    # Convert to TF-IDF matrix
    feature_tfidf_matrix = tfidf_vectorizer.transform(features_df['combined_features'])

    # Compute similarity scores
    feature_cosine_sim = linear_kernel(feature_tfidf_matrix, tfidf_matrix)

    # Get pairwise similarity scores
    sim_scores = list(enumerate(feature_cosine_sim[0]))

    # Sort cars based on similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the indices of the top 10 most similar cars
    sim_scores = sim_scores[:10]
    car_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar cars from the training data
    return data.iloc[car_indices]


# Example usage
example_features = {
    'Manufacturer': 'VOLKSWAGEN',
    'Model': 'Golf',
    'Category': 'Hatchback',
    'Leather interior': 'No',
    'Fuel type': 'Diesel',
    'Engine volume': '2.0 Turbo',
    'Mileage': 50000,
    'Gear box type': 'Manual',
    'Drive wheels': 'Front',
    'Color': 'Grey',
    'Price': 15000
}

recommended_cars = recommend_car_based_on_features(example_features)
print("Recommended cars based on the given features:")
print(recommended_cars[['ID', 'Manufacturer', 'Model', 'Price', 'Mileage']])
