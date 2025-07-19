  # Chatbot ML App using Dash

An interactive chatbot web application built using **Python**, **Dash**, and **Scikit-learn**.  
This chatbot uses a **Naive Bayes classifier** trained on a custom Q&A dataset to generate relevant responses based on user input.

## 🚀 Features
- Machine learning-based response generation
- TF-IDF vectorization for text preprocessing
- User-friendly web interface with Dash
- Real-time chatbot interaction
- Custom dataset support

## 🧠 How It Works
1. The dataset (`chatbot_dataset.csv`) contains question-answer pairs.
2. Questions are tokenized using NLTK and vectorized using TF-IDF.
3. A Naive Bayes model is trained to classify the most likely answer.
4. The app takes user input and returns a predicted answer instantly.

## 📁 Project Structure
chatbot-ml-dash/
│
├── chatbot_app.py # Main Dash app
├── chatbot_dataset.csv # Custom Q&A dataset
├── README.md # Project documentation

## 🔧 Requirements

Install dependencies using pip:

pip install pandas nltk scikit-learn dash

Also, make sure to download NLTK tokenizer data:

import nltk
nltk.download('punkt')

▶️ Running the App
Run the following command in your terminal:

python chatbot_app.py

Then open your browser and go to: http://127.0.0.1:8050


📌 Use Case
This app is ideal for:

Learning basic NLP and ML integration

Building FAQ bots for educational or demo purposes

Creating simple ML-powered Dash applications

📜 License
This project is open-source and free to use under the MIT License.
