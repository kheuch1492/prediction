import streamlit as st
import joblib
import nltk
import urllib.request
import re
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer

# Charger le modèle et le vectoriseur TF-IDF (ajustez les chemins si nécessaire)
model = joblib.load('C:/Users/sall1/OneDrive/Bureau/Projet NLP/modele_regression_logistique.joblib')
tfidf_vectorizer = joblib.load('C:/Users/sall1/OneDrive/Bureau/Projet NLP/tfidf_vectorizer.joblib') 

st.title('Prédiction de Sentiment')

# Créer une zone de texte pour l'entrée de l'utilisateur
user_input = st.text_area("Entrez le texte à analyser")

# Télécharger les stopwords et le tokenizer si nécessaire
nltk.download('stopwords')
nltk.download('punkt')

# Charger les stop words arabes
url = "https://raw.githubusercontent.com/mohataher/arabic-stop-words/master/list.txt"
arabicStopWords = set(urllib.request.urlopen(url).read().decode('utf-8').splitlines())

# Initialisation du stemmer
stemmer = SnowballStemmer("arabic")

# Dictionnaire pour la conversion de l'arabe au format Buckwalter
buckArab = {"'":"ء", "|":"آ", "?":"أ", "&":"ؤ", "<":"إ", "}":"ئ", "A":"ا", "b":"ب", 
            "p":"ة", "t":"ت", "v":"ث", "g":"ج", "H":"ح", "x":"خ", "d":"د", "*":"ذ",
            "r":"ر", "z":"ز", "s":"س", "$":"ش", "S":"ص", "D":"ض", "T":"ط", "Z":"ظ",
            "E":"ع", "G":"غ", "_":"ـ", "f":"ف", "q":"ق", "k":"ك", "l":"ل", "m":"م",
            "n":"ن", "h":"ه", "w":"و", "Y":"ى", "y":"ي", "F":"ً", "N":"ٌ", "K":"ٍ",
            "~":"ّ", "o":"ْ", "u":"ُ", "a":"َ", "i":"ِ"}

# Fonction de prétraitement du texte
def preprocess_text(text):
    # Convertir le texte en minuscules
    text = text.lower()

    # Remplacer les caractères spéciaux par leur équivalent en arabe
    for k, v in buckArab.items():
        text = text.replace(v, k)

    # Tokenisation
    words = word_tokenize(text)

    # Suppression des stop words et stemming
    words = [stemmer.stem(word) for word in words if word not in arabicStopWords]

    # Recombiner les mots en une seule chaîne de texte
    return ' '.join(words)

if st.button('Prédire'):
    # Prétraiter l'entrée utilisateur
    processed_input = preprocess_text(user_input)

    # Transformer le texte traité en vecteurs TF-IDF
    transformed_input = tfidf_vectorizer.transform([processed_input])

    # Faire la prédiction
    prediction = model.predict(transformed_input)

    # Afficher le résultat
    st.write("Prédiction :", prediction[0])
