from flask import Flask, request, render_template, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

# Load the trained model
with open('Models/final_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load processed data
df = pd.read_csv('Data/Processed Data/processed_data.csv')
df = df.drop(columns=['interactions'], errors='ignore')  # ← add this line

# Feature columns used during training
feature_cols = ['user_id', 'product_id', 'carts_x', 'views_x', 'avg_seconds', 'avg_price', 
                'total_views', 'total_purchases', 'total_carts', 'products', 'sessions', 
                'purchase_rate_x', 'views_y', 'carts_y', 'purchases', 'users', 
                'purchase_rate_y', 'cat_purchase_rate', 'brand_purchase_rate', 
                'category_encoded', 'brand_encoded']


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/recommend', methods=['POST'])
def recommend():
    user_id = int(request.form.get('user_id'))

    # Filter for this user
    user_data = df[df['user_id'] == user_id].copy()

    if len(user_data) == 0:
        return render_template('index.html', error=f'User {user_id} not found in dataset')

    # Keep only products user has not purchased
    user_data = user_data[user_data['purchased'] == 0]

    if len(user_data) == 0:
        return render_template('index.html', error='User has interacted with all products')

    # Predict purchase probability
    user_data['probability'] = model.predict_proba(user_data[feature_cols])[:, 1]

    # Top 10 recommendations
    recommendations = user_data[['product_id', 'probability']]\
        .sort_values('probability', ascending=False)\
        .head(10)\
        .to_dict(orient='records')

    return render_template('index.html', recommendations=recommendations, user_id=user_id)


if __name__ == '__main__':
    app.run(debug=True)