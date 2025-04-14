import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re
import joblib
from urllib.parse import urlparse

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


def count_dot(url):
    return url.count(r'.')

def no_of_dir(url):
    return urlparse(url).path.count(r'/')

def no_of_embed(url):
    return urlparse(url).path.count(r'//')

def shortening_service(url):
    match = re.search(r'bit\.ly|goo\.gl|shorte\.st|go2l\.ink|x\.co|ow\.ly|t\.co|tinyurl|tr\.im|is\.gd|cli\.gs|'
                     r'yfrog\.com|migre\.me|ff\.im|tiny\.cc|url4\.eu|twit\.ac|su\.pr|twurl\.nl|snipurl\.com|'
                     r'short\.to|BudURL\.com|ping\.fm|post\.ly|Just\.as|bkite\.com|snipr\.com|fic\.kr|loopt\.us|'
                     r'doiop\.com|short\.ie|kl\.am|wp\.me|rubyurl\.com|om\.ly|to\.ly|bit\.do|t\.co|lnkd\.in|'
                     r'db\.tt|qr\.ae|adf\.ly|goo\.gl|bitly\.com|cur\.lv|tinyurl\.com|ow\.ly|bit\.ly|ity\.im|'
                     r'q\.gs|is\.gd|po\.st|bc\.vc|twitthis\.com|u\.to|j\.mp|buzurl\.com|cutt\.us|u\.bb|yourls\.org|'
                     r'x\.co|prettylinkpro\.com|scrnch\.me|filoops\.info|vzturl\.com|qr\.net|1url\.com|tweez\.me|v\.gd|'
                     r'tr\.im|link\.zip\.net', url)
    return 1 if match else 0

def count_http(url):
    return url.count('http')

def count_per(url):
    return url.count(r'%')

def count_ques(url):
    return url.count(r'?')

def count_hyphen(url):
    return url.count(r'-')

def count_equal(url):
    return url.count(r'=')

def url_length(url):
    return len(str(url))

def hostname_length(url):
    return len(urlparse(url).netloc)

def suspicious_words(url):
    score_map = {
        'error': 30, 'SELECT': 50, 'FROM': 50, 'WHERE': 50, 'javascript': 20,
        'cookie': 25, '--': 30, 'admin': 10, '\'': 30, 'password': 15,
        'UNION': 35, 'script': 25, 'eval': 25, 'document': 20
    }
    matches = re.findall(r'(?i)' + '|'.join(score_map.keys()), url)
    return sum(score_map.get(match.lower(), 0) for match in matches)

def digit_count(url):
    return sum(1 for i in url if i.isnumeric())

def letter_count(url):
    return sum(1 for i in url if i.isalpha())

def count_special_characters(url):
    return len(re.sub(r'[a-zA-Z0-9\s]', '', url))

def number_of_parameters(url):
    params = urlparse(url).query
    return 0 if params == '' else len(params.split('&'))

def is_encoded(url):
    return int('%' in url.lower())

def apply_to_content(content, function):
    if pd.isna(content) or not isinstance(content, str):
        return 0
    return function(content)

def preprocess_data(df, label_encoders=None, fit_encoders=True):
    """Process raw data into features (works for both training and new data)"""
    # Initial processing
    df = df.rename(columns={'lenght': 'content_length'})
    
    # Process content_length
    df['content_length'] = df['content_length'].astype(str)
    df['content_length'] = df['content_length'].str.extract(r'(\d+)')
    df['content_length'] = pd.to_numeric(df['content_length'], errors='coerce').fillna(0)
    
    # URL feature engineering
    url_features = {
        'count_dot_url': count_dot,
        'count_dir_url': no_of_dir,
        'count_embed_url': no_of_embed,
        'short_url': shortening_service,
        'count-http': count_http,
        'count%_url': count_per,
        'count?_url': count_ques,
        'count-_url': count_hyphen,
        'count=_url': count_equal,
        'url_length': url_length,
        'hostname_length': hostname_length,
        'sus_url': suspicious_words,
        'count-digits_url': digit_count,
        'count-letters_url': letter_count,
        'param_count': number_of_parameters,
        'is_encoded_url': is_encoded,
        'special_chars_url': count_special_characters
    }
    
    for feature_name, func in url_features.items():
        df[feature_name] = df['URL'].apply(func)
    
    # Content feature engineering
    content_features = {
        'count_dot_content': count_dot,
        'count%_content': count_per,
        'count-_content': count_hyphen,
        'count=_content': count_equal,
        'sus_content': suspicious_words,
        'count-digits_content': digit_count,
        'count-letters_content': letter_count,
        'special_chars_content': count_special_characters,
        'is_encoded_content': is_encoded
    }
    
    for feature_name, func in content_features.items():
        df[feature_name] = df['content'].apply(lambda x: apply_to_content(x, func))
    
    # Encode categorical features
    categorical_cols = ['Method', 'host', 'Accept']
    if label_encoders is None:
        label_encoders = {}
    
    for col in categorical_cols:
        if col in df.columns:
            if fit_encoders or col not in label_encoders:
                le = LabelEncoder()
                df[col+'_enc'] = le.fit_transform(df[col].astype(str))
                label_encoders[col] = le
            else:
                le = label_encoders[col]
                df[col+'_enc'] = le.transform(df[col].astype(str))
    
    # Final feature selection
    final_features = [
        'count_dot_url', 'count_dir_url', 'count_embed_url', 'short_url',
        'count-http', 'count%_url', 'count?_url', 'count-_url', 'count=_url',
        'url_length', 'hostname_length', 'sus_url', 'count-digits_url',
        'count-letters_url', 'param_count', 'is_encoded_url', 'special_chars_url',
        'content_length', 'count_dot_content', 'count%_content', 'count-_content',
        'count=_content', 'sus_content', 'count-digits_content', 'count-letters_content',
        'special_chars_content', 'is_encoded_content',
        'Method_enc', 'host_enc', 'Accept_enc'
    ]
    
    return df[final_features], label_encoders

def prepare_new_data(url, content=None, method="GET", label_encoders=None):
    """Prepare a single new URL for prediction"""
    data = {
        'URL': [url],
        'content': [content] if content is not None else [None],
        'Method': [method],
        'host': [urlparse(url).netloc],
        'Accept': ['*/*'],
        'content_length': [len(content) if content is not None else 0]
    }
    df = pd.DataFrame(data)
    features, _ = preprocess_data(df, label_encoders, fit_encoders=False)
    return features

def train_models(X_train, y_train, X_test, y_test):
    """Train and evaluate multiple models"""
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Support Vector Machine": SVC(random_state=42, probability=True),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "XGBoost": XGBClassifier(random_state=42, eval_metric='logloss')
    }
    
    results = []
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        results.append({
            'Model': name,
            'Accuracy': accuracy,
            'Precision': report['weighted avg']['precision'],
            'Recall': report['weighted avg']['recall'],
            'F1-Score': report['weighted avg']['f1-score']
        })
        
        print(f"{name} Accuracy: {accuracy:.4f}")
    
    return pd.DataFrame(results), models['Random Forest']

def plot_model_comparison(results_df):
    """Plot model comparison results"""
    plt.figure(figsize=(14, 8))
    sns.barplot(x='Accuracy', y='Model', data=results_df, palette='mako', hue='Model', legend=False)
    plt.title('Web Attack Detection: Model Comparison - Accuracy Scores')
    plt.xlabel('Accuracy Score')
    plt.ylabel('Model')
    plt.xlim(0.75, 1.0)
    plt.tight_layout()
    plt.show()

def plot_feature_importance(model, features):
    """Plot feature importance"""
    importance = pd.DataFrame({
        'Feature': features,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=importance.head(20))
    plt.title('Top 20 Important Features')
    plt.tight_layout()
    plt.show()

def save_model_pipeline(model, scaler, label_encoder, label_encoders, features, filepath):
    """Save all components needed for predictions"""
    pipeline = {
        'model': model,
        'scaler': scaler,
        'label_encoder': label_encoder,
        'label_encoders': label_encoders,
        'features': features
    }
    joblib.dump(pipeline, filepath)
    print(f"Model pipeline saved to {filepath}")

def load_model_pipeline(filepath):
    """Load saved model pipeline"""
    return joblib.load(filepath)

def predict_web_attack(url, content=None, method="GET", pipeline=None):
    """Make prediction for a new URL with robust handling for unseen categories"""
    if pipeline is None:
        pipeline = load_model_pipeline('web_attack_model.pkl')
    
    # Prepare the data with safe encoding
    data = {
        'URL': [url],
        'content': [content] if content is not None else [None],
        'Method': [method],
        'host': [urlparse(url).netloc],
        'Accept': ['*/*'],
        'content_length': [len(content) if content is not None else 0]
    }
    df = pd.DataFrame(data)
    
    # Safe feature engineering (same as training)
    df = safe_feature_engineering(df)
    
    # Safe categorical encoding
    df = safe_encode_categoricals(df, pipeline['label_encoders'])
    
    # Select final features
    features = df[pipeline['features']]
    
    # Scale and predict
    try:
        new_data_scaled = pipeline['scaler'].transform(features)
        proba = pipeline['model'].predict_proba(new_data_scaled)[0]
        prediction = pipeline['model'].predict(new_data_scaled)[0]
        class_name = pipeline['label_encoder'].inverse_transform([prediction])[0]
        
        return {
            'url': url,
            'prediction': class_name,
            'probability_normal': float(proba[0]),
            'probability_anomalous': float(proba[1]),
            'is_anomalous': bool(prediction == 1)
        }
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return {
            'url': url,
            'error': str(e),
            'is_anomalous': True  # Default to anomalous if there's an error
        }

def safe_feature_engineering(df):
    """Apply feature engineering safely"""
    # URL features
    df['count_dot_url'] = df['URL'].apply(count_dot)
    df['count_dir_url'] = df['URL'].apply(no_of_dir)
    df['count_embed_url'] = df['URL'].apply(no_of_embed)
    df['short_url'] = df['URL'].apply(shortening_service)
    df['count-http'] = df['URL'].apply(count_http)
    df['count%_url'] = df['URL'].apply(count_per)
    df['count?_url'] = df['URL'].apply(count_ques)
    df['count-_url'] = df['URL'].apply(count_hyphen)
    df['count=_url'] = df['URL'].apply(count_equal)
    df['url_length'] = df['URL'].apply(url_length)
    df['hostname_length'] = df['URL'].apply(hostname_length)
    df['sus_url'] = df['URL'].apply(suspicious_words)
    df['count-digits_url'] = df['URL'].apply(digit_count)
    df['count-letters_url'] = df['URL'].apply(letter_count)
    df['param_count'] = df['URL'].apply(number_of_parameters)
    df['is_encoded_url'] = df['URL'].apply(is_encoded)
    df['special_chars_url'] = df['URL'].apply(count_special_characters)
    
    # Content features
    df['count_dot_content'] = df['content'].apply(lambda x: apply_to_content(x, count_dot))
    df['count%_content'] = df['content'].apply(lambda x: apply_to_content(x, count_per))
    df['count-_content'] = df['content'].apply(lambda x: apply_to_content(x, count_hyphen))
    df['count=_content'] = df['content'].apply(lambda x: apply_to_content(x, count_equal))
    df['sus_content'] = df['content'].apply(lambda x: apply_to_content(x, suspicious_words))
    df['count-digits_content'] = df['content'].apply(lambda x: apply_to_content(x, digit_count))
    df['count-letters_content'] = df['content'].apply(lambda x: apply_to_content(x, letter_count))
    df['special_chars_content'] = df['content'].apply(lambda x: apply_to_content(x, count_special_characters))
    df['is_encoded_content'] = df['content'].apply(lambda x: apply_to_content(x, is_encoded))
    
    return df

def safe_encode_categoricals(df, label_encoders):
    """Safely encode categorical features with fallback for unseen values"""
    categorical_cols = ['Method', 'host', 'Accept']
    
    for col in categorical_cols:
        if col in df.columns:
            le = label_encoders[col]
            # Handle unseen values by mapping them to a special category
            df[col+'_enc'] = df[col].astype(str).apply(
                lambda x: le.transform([x])[0] if x in le.classes_ else len(le.classes_)
            )
    
    return df