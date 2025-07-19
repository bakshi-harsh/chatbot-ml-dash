import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import random

# Load the dataset
data = pd.read_csv('chatbot_dataset.csv')

# Preprocess the data
nltk.download('punkt')
data['Question'] = data['Question'].apply(lambda x: ' '.join(nltk.word_tokenize(x.lower())))

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(data['Question'], data['Answer'], test_size=0.2, random_state=42)

# Create a model pipeline
model = make_pipeline(TfidfVectorizer(), MultinomialNB())
model.fit(X_train, y_train)

def get_response(question):
    question = ' '.join(nltk.word_tokenize(question.lower()))
    answer = model.predict([question])[0]
    return answer

# Initialize the Dash app
app = dash.Dash(__name__)

# Define the layout
pp.layout = html.Div(
    style={
        'fontFamily': 'Arial, sans-serif',
        'backgroundColor': '#121212',
        'color': '#FFFFFF',
        'backgroundImage': 'url("/assets/robot-human-hands-interacting.jpg")', 
        'backgroundSize': 'cover',
        'backgroundPosition': 'center',
        'backgroundRepeat': 'no-repeat',
        'minHeight': '100vh',
        'padding': '40px',
        'maxWidth': '800px',
        'margin': 'auto',
        'borderRadius': '12px',
        'boxShadow': '0 0 20px rgba(0, 0, 0, 0.5)'
    },
    children=[
        html.H1("🤖 AI Chatbot", style={
            'textAlign': 'center',
            'color': '#00FFCC',
            'marginBottom': '30px',
            'textShadow': '0 0 10px #00FFCC'
        }),
        
        dcc.Textarea(
            id='user-input',
            value='Type your question here...',
            style={
                'width': '100%',
                'height': 100,
                'padding': '15px',
                'border': '1px solid #333',
                'borderRadius': '8px',
                'fontSize': '16px',
                'resize': 'none',
                'backgroundColor': '#1e1e1e',
                'color': '#FFFFFF',
                'boxShadow': 'inset 0 1px 3px rgba(255,255,255,0.1)'
            }
        ),
        
        html.Button('🧠 Submit', id='submit-button', n_clicks=0, style={
            'marginTop': '20px',
            'padding': '10px 25px',
            'backgroundColor': '#00FFCC',
            'color': '#000',
            'border': 'none',
            'borderRadius': '6px',
            'fontSize': '16px',
            'cursor': 'pointer',
            'boxShadow': '0 2px 5px rgba(0,0,0,0.3)'
        }),
        
        html.Div(id='chatbot-output', style={
            'padding': '20px',
            'marginTop': '30px',
            'backgroundColor': 'rgba(0, 0, 0, 0.6)',
            'borderRadius': '8px',
            'border': '1px solid #444',
            'minHeight': '100px',
            'color': '#FFFFFF',
            'boxShadow': 'inset 0 1px 3px rgba(255,255,255,0.1)'
        })
    ]
)

# Define callback to update chatbot response
@app.callback(
    Output('chatbot-output', 'children'),
    Input('submit-button', 'n_clicks'),
    [dash.dependencies.State('user-input', 'value')]
)
def update_output(n_clicks, user_input):
    if n_clicks > 0:
        response = get_response(user_input)
        return html.Div([
            html.P(f"You: {user_input}", style={'margin': '10px'}),
            html.P(f"Bot: {response}", style={'margin': '10px', 'backgroundColor': '#f0f0f0', 'padding': '10px', 'borderRadius': '5px'})
        ])
    return "Ask me something!"

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
