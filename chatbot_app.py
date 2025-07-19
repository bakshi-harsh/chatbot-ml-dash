import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
import dash
from dash import dcc, html
from dash.dependencies import Input, Output

# Load dataset
data = pd.read_csv('chatbot_dataset.csv')

# Preprocess text
nltk.download('punkt')
data['Question'] = data['Question'].apply(lambda x: ' '.join(nltk.word_tokenize(x.lower())))

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(data['Question'], data['Answer'], test_size=0.2, random_state=42)

# Build model
model = make_pipeline(TfidfVectorizer(), MultinomialNB())
model.fit(X_train, y_train)

# Get bot response
def get_response(question):
    question = ' '.join(nltk.word_tokenize(question.lower()))
    return model.predict([question])[0]

# Initialize Dash app
app = dash.Dash(__name__)

# App layout
app.layout = html.Div(
    style={
        'fontFamily': 'Arial, sans-serif',
        'backgroundColor': '#ffffff',
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
                'border': '1px solid #444',
                'borderRadius': '8px',
                'fontSize': '16px',
                'resize': 'none',
                'backgroundColor': 'rgba(0, 0, 0, 0.6)',
                'color': '#FFFFFF',
                'boxShadow': 'inset 0 1px 3px rgba(255,255,255,0.1)',
                'marginBottom': '20px'
            }
        ),

        html.Button('🧠 Submit', id='submit-button', n_clicks=0, style={
            'marginTop': '10px',
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

# Callback for chatbot interaction
@app.callback(
    Output('chatbot-output', 'children'),
    Input('submit-button', 'n_clicks'),
    [dash.dependencies.State('user-input', 'value')]
)
def update_output(n_clicks, user_input):
    message_style = {
        'margin': '10px',
        'padding': '10px',
        'borderRadius': '5px',
        'backgroundColor': 'rgba(0, 0, 0, 0.6)',
        'color': '#FFFFFF',
        'border': '1px solid #444'
    }

    if n_clicks > 0:
        response = get_response(user_input)
        return html.Div([
            html.P(f"You: {user_input}", style=message_style),
            html.P(f"Bot: {response}", style=message_style)
        ])

    return html.P("Ask me something!", style=message_style)

# Run server
if __name__ == '__main__':
    app.run(debug=True)
