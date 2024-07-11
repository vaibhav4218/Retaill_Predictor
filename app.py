from flask import Flask, request, jsonify, render_template_string
import numpy as np
import pandas as pd
import os
import datetime as dt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

app = Flask(__name__)

class DataLoader:
    def __init__(self, filepath):
        self.filepath = filepath
    
    def load_data(self):
        return pd.read_csv(self.filepath)

class DataCleaner:
    @staticmethod
    def drop_missing_customer_ids(data):
        return data.dropna(subset=['CustomerID'])

class FeatureEngineer:
    @staticmethod
    def add_recency_frequency_monetary(data):
        snapshot_date = max(data['InvoiceDate']) + dt.timedelta(days=1)
        rfm_data = data.groupby('CustomerID').agg({
            'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
            'InvoiceNo': 'count',
            'UnitPrice': lambda x: (x * data.loc[x.index, 'Quantity']).sum()
        })
        rfm_data.rename(columns={
            'InvoiceDate': 'Recency',
            'InvoiceNo': 'Frequency',
            'UnitPrice': 'Monetary'
        }, inplace=True)
        return rfm_data

    @staticmethod
    def add_is_repeat_purchase(data):
        data['IsRepeatPurchase'] = data.duplicated(subset=['CustomerID', 'StockCode'], keep=False).astype(int)
        agg_data = data.groupby(['CustomerID', 'StockCode']).agg(
            TotalQuantity=('Quantity', 'sum'),
            NumTransactions=('InvoiceNo', 'nunique'),
            AvgUnitPrice=('UnitPrice', 'mean'),
            LastPurchase=('InvoiceDate', 'max'),
            IsRepeatPurchase=('IsRepeatPurchase', 'max')
        ).reset_index()
        agg_data['RecencyDays'] = (data['InvoiceDate'].max() - agg_data['LastPurchase']).dt.days
        agg_data.drop(columns=['LastPurchase'], inplace=True)
        return agg_data

    @staticmethod
    def calculate_total_revenue(data):
        data['TotalRevenue'] = data['Quantity'] * data['UnitPrice']
        return data

class ModelTrainer:
    def __init__(self, model, features, target):
        self.model = model
        self.features = features
        self.target = target
    
    def train_model(self, X_train, y_train):
        self.model.fit(X_train, y_train)
    
    def evaluate_model(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        return accuracy, report

class ProductRecommender:
    @staticmethod
    def recommend_top_products(data, top_n=10):
        product_revenue = data.groupby(['StockCode', 'Description'])['TotalRevenue'].sum().reset_index()
        top_products = product_revenue.sort_values(by='TotalRevenue', ascending=False).head(top_n)
        return top_products

# Load data globally for simplicity
data_loader = DataLoader("online_retail.csv")
try:
    data = data_loader.load_data()
    print("Data loaded successfully.")
except FileNotFoundError:
    print("File not found. Please ensure 'online_retail.csv' is in the correct directory.")
    raise

data_cleaner = DataCleaner()
data = data_cleaner.drop_missing_customer_ids(data)
feature_engineer = FeatureEngineer()
data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'])

@app.route('/')
def index():
    return render_template_string('''
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Welcome</title>
        </head>
        <body>
            <h1>Welcome to the Retail Analysis API</h1>
            <button onclick="location.href='/customer_return'">Customer Return</button>
            <button onclick="location.href='/repurchase'">Repurchase</button>
            <button onclick="location.href='/recommend_products'">Recommend Products</button>
        </body>
        </html>
    ''')

@app.route('/customer_return', methods=['GET', 'POST'])
def customer_return():
    if request.method == 'POST':
        try:
            rfm_data = feature_engineer.add_recency_frequency_monetary(data)
            rfm_data['ReturningCustomer'] = np.where(rfm_data['Frequency'] > 1, 1, 0)

            X_returning = rfm_data[['Recency', 'Frequency', 'Monetary']]
            y_returning = rfm_data['ReturningCustomer']
            X_train_ret, X_test_ret, y_train_ret, y_test_ret = train_test_split(X_returning, y_returning, test_size=0.2, random_state=42)

            returning_model_trainer = ModelTrainer(RandomForestClassifier(n_estimators=100, random_state=42), X_returning, y_returning)
            returning_model_trainer.train_model(X_train_ret, y_train_ret)
            accuracy_ret, report_ret = returning_model_trainer.evaluate_model(X_test_ret, y_test_ret)

            return jsonify({
                "accuracy": accuracy_ret,
                "classification_report": report_ret
            })
        except Exception as e:
            return jsonify({"error": str(e)})
    return render_template_string('''
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Customer Return Prediction</title>
        </head>
        <body>
            <h1>Customer Return Prediction</h1>
            <form action="/customer_return" method="post">
                <input type="submit" value="Predict">
            </form>
        </body>
        </html>
    ''')

@app.route('/repurchase', methods=['GET', 'POST'])
def repurchase():
    if request.method == 'POST':
        try:
            agg_data = feature_engineer.add_is_repeat_purchase(data)

            feature_columns_rep = ['TotalQuantity', 'NumTransactions', 'AvgUnitPrice', 'RecencyDays']
            X_rep = agg_data[feature_columns_rep]
            y_rep = agg_data['IsRepeatPurchase']
            X_train_rep, X_test_rep, y_train_rep, y_test_rep = train_test_split(X_rep, y_rep, test_size=0.2, random_state=42)

            repeat_model_trainer = ModelTrainer(RandomForestClassifier(n_estimators=100, random_state=42), X_rep, y_rep)
            repeat_model_trainer.train_model(X_train_rep, y_train_rep)
            accuracy_rep, report_rep = repeat_model_trainer.evaluate_model(X_test_rep, y_test_rep)

            return jsonify({
                "accuracy": accuracy_rep,
                "classification_report": report_rep
            })
        except Exception as e:
            return jsonify({"error": str(e)})
    return render_template_string('''
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Repurchase Prediction</title>
        </head>
        <body>
            <h1>Repurchase Prediction</h1>
            <form action="/repurchase" method="post">
                <input type="submit" value="Predict">
            </form>
        </body>
        </html>
    ''')

@app.route('/recommend_products', methods=['GET', 'POST'])
def recommend_products():
    if request.method == 'POST':
        try:
            data_with_revenue = feature_engineer.calculate_total_revenue(data)
            product_recommender = ProductRecommender()
            top_products = product_recommender.recommend_top_products(data_with_revenue)
            
            return jsonify(top_products.to_dict(orient="records"))
        except Exception as e:
            return jsonify({"error": str(e)})
    return render_template_string('''
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Recommend Products</title>
        </head>
        <body>
            <h1>Recommend Products</h1>
            <form action="/recommend_products" method="post">
                <input type="submit" value="Get Recommendations">
            </form>
        </body>
        </html>
    ''')

if __name__ == "__main__":
    app.run(debug=True)
