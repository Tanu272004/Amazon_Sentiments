# filepath: /amazon-sentiments/amazon-sentiments/Sentimental.py
# -------------------------------
# Amazon Review Sentiment Analysis
# -------------------------------

# ✅ Import Libraries
import pandas as pd
import numpy as np
from faker import Faker
import random
from textblob import TextBlob

# -------------------------------
# STEP 1: Generate Synthetic Customers Review Dataset
# -------------------------------

fake = Faker()
Faker.seed(42)
random.seed(42)

# Define product categories for realism
categories = ['Electronics', 'Books', 'Clothing', 'Sports', 'Home']

# Number of reviews
num_reviews = 500

data = []
for i in range(1, num_reviews + 1):
    review_id = i
    customer_id = fake.uuid4()
    product_id = random.randint(1000, 9999)
    category = random.choice(categories)
    
    # Generate random review text (short sentences)
    review_text = fake.sentence(nb_words=random.randint(8, 20))
    
    # Simulate a rating (1-5 stars)
    rating = random.randint(1, 5)
    
    data.append([review_id, customer_id, product_id, category, review_text, rating])

# Create DataFrame
df = pd.DataFrame(data, columns=['review_id', 'customer_id', 'product_id', 'category', 'review_text', 'rating'])

# -------------------------------
# STEP 2: Sentiment Analysis using TextBlob
# -------------------------------

def get_sentiment(text):
    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity  # range: -1 (negative) to +1 (positive)
    if polarity > 0.1:
        return 'Positive'
    elif polarity < -0.1:
        return 'Negative'
    else:
        return 'Neutral'

def get_polarity(text):
    return round(TextBlob(text).sentiment.polarity, 3)

# Apply sentiment analysis
df['sentiment'] = df['review_text'].apply(get_sentiment)
df['polarity'] = df['review_text'].apply(get_polarity)

# -------------------------------
# STEP 3: Save to CSV
# -------------------------------
# Unique Customers
customers_df = df[['customer_id']].drop_duplicates()
customers_df['name'] = [fake.name() for _ in range(len(customers_df))]
customers_df['region'] = [fake.state() for _ in range(len(customers_df))]
customers_df.to_csv('customers.csv', index=False)

# Unique Products
products_df = df[['product_id', 'category']].drop_duplicates()
products_df.to_csv('products.csv', index=False)

# Reviews (already have df)
df.to_csv('reviews.csv', index=False)
