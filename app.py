import os
import re
import nltk
import requests
import datetime
from flask import Flask, render_template, request, redirect, session, url_for, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from collections import Counter
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import pipeline
from dotenv import load_dotenv
import google.generativeai as genai


# ---------- ENV SETUP ----------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")  # ✅ Correct
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
  # ✅ FIXED
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")  # ✅ FIXED
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))  # ✅ FIXED

# ---------- INIT ----------
app = Flask(__name__)
app.secret_key = 'supersecretkey'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///diary.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# ---------- NLTK + EMOTION ----------
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

emotion_classifier = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    top_k=None,
    framework="pt"
)

# ---------- DB MODELS ----------
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)

class JournalEntry(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    journal_text = db.Column(db.Text)
    mood_emoji = db.Column(db.String(10))
    sentiment = db.Column(db.String(20))
    sentiment_score = db.Column(db.Float)
    emotions = db.Column(db.String(100))
    stress_score = db.Column(db.Float)
    timestamp = db.Column(db.String(30))
    ai_suggestions = db.Column(db.Text)

# ---------- HELPERS ----------
def clean_text(text):
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    return text.lower()

def get_emotions(text):
    try:
        results = emotion_classifier(text[:512])
        top_emotions = []
        for r in sorted(results[0], key=lambda x: x['score'], reverse=True):
            if r['score'] > 0.1:
                top_emotions.append(r['label'])
            if len(top_emotions) >= 3:
                break
        return top_emotions or ["neutral"]
    except Exception as e:
        print("Emotion error:", e)
        return ["neutral"]

def determine_sentiment(vader, emotions):
    compound = vader["compound"]
    if any(e in emotions for e in ['sadness', 'anger', 'fear', 'disappointment', 'grief', 'frustration', 'anxiety']):
        return "mixed" if compound > 0.3 else "negative"
    if any(e in emotions for e in ['joy', 'love']):
        return "positive"
    return "positive" if compound >= 0.3 else "negative" if compound <= -0.3 else "neutral"

def estimate_stress(vader_scores, detected_emotions):
    compound_score = vader_scores["compound"]
    neg_score = vader_scores["neg"]
    base_stress = 9.0 if compound_score < -0.7 else 7.5 if compound_score < -0.4 else 5.0 if compound_score < -0.1 else 2.0
    if neg_score > 0.2:
        base_stress += neg_score * 5
    emotion_weights = {
        'anger': 2.5, 'sadness': 2.0, 'fear': 3.0,
        'disappointment': 1.5, 'grief': 4.0,
        'frustration': 2.0, 'anxiety': 2.5
    }
    addition = sum(emotion_weights.get(e, 0.0) for e in detected_emotions)
    return round(min(10.0, base_stress + addition), 2)

def calculate_streak(entries):
    from datetime import datetime, timedelta
    dates = sorted(list(set(datetime.strptime(e.timestamp.split()[0], "%Y-%m-%d") for e in entries)), reverse=True)
    if not dates: return 0
    streak = 1
    for i in range(1, len(dates)):
        if dates[i - 1] - dates[i] == timedelta(days=1):
            streak += 1
        else:
            break
    return 0 if dates[0].date() != datetime.now().date() else streak

def generate_ai_suggestion(emotions, sentiment):
    prompt = f"""
You're an empathetic mental wellness coach.

A user has just written in their diary. Their overall emotional tone is "{sentiment}", and the specific emotions they are experiencing include: {', '.join(emotions)}.

Please provide thoughtful, emotionally intelligent support by offering the following:
1. Two simple and uplifting activities the user can do today to feel better, reduce stress, or regain emotional balance.
2. One short motivational quote that relates to the mood.
3. One gentle reflection question that encourages self-awareness and growth.

Make sure your language is warm, comforting, and friendly—like a wise friend. Avoid clinical tone. Do not mention that this is AI-generated.
"""
    model = genai.GenerativeModel("gemini-2.0-flash")
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print("Gemini error:", e)
        return "Unable to fetch suggestions at the moment. Please try again later."

def get_youtube_videos(emotions, sentiment, max_results=3):
    search_query = f"{' '.join(emotions)} {sentiment} motivation or relaxation"
    url = "https://www.googleapis.com/youtube/v3/search"
    params = {
        "part": "snippet",
        "q": search_query,
        "key": YOUTUBE_API_KEY,  # ✅ FIXED
        "maxResults": max_results,
        "type": "video",
        "safeSearch": "strict"
    }
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        items = response.json().get("items", [])
        return [
            {
                "title": item["snippet"]["title"],
                "video_id": item["id"]["videoId"],
                "thumbnail": item["snippet"]["thumbnails"]["high"]["url"]
            }
            for item in items
        ]
    except Exception as e:
        print("YouTube API Error:", e)
        return []


# ---------- ROUTES ----------
@app.route("/")
def index():
    return redirect(url_for('dashboard') if 'user_id' in session else url_for('login'))

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        if User.query.filter_by(username=username).first():
            flash("Username already exists")
            return redirect("/register")
        hashed = generate_password_hash(password)
        db.session.add(User(username=username, password_hash=hashed))
        db.session.commit()
        flash("Registered! Please log in.")
        return redirect("/login")
    return render_template("register.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password_hash, password):
            session["user_id"] = user.id
            session["username"] = user.username
            return redirect("/dashboard")
        flash("Invalid credentials")
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    flash("Logged out")
    return redirect("/login")

@app.route("/dashboard")
def dashboard():
    if "user_id" not in session:
        return redirect("/login")
    return render_template("mood.html", username=session["username"])

@app.route("/journal", methods=["GET", "POST"])
def journal():
    if "user_id" not in session:
        return redirect("/login")

    result = None
    if request.method == "POST" and "mood_emoji" in request.form and "journal_text" not in request.form:
        session["selected_mood"] = request.form.get("mood_emoji")

    if request.method == "POST" and "journal_text" in request.form:
        text = request.form["journal_text"]
        mood = session.get("selected_mood", "😐")
        vader = sia.polarity_scores(text)
        emotions = get_emotions(clean_text(text))
        sentiment = determine_sentiment(vader, emotions)
        stress = estimate_stress(vader, emotions)
        ai_suggestions = generate_ai_suggestion(emotions, sentiment)
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

        entry = JournalEntry(
            user_id=session["user_id"],
            journal_text=text,
            mood_emoji=mood,
            sentiment=sentiment,
            sentiment_score=vader["compound"],
            emotions=", ".join(emotions),
            stress_score=stress,
            timestamp=timestamp,
            ai_suggestions=ai_suggestions
        )
        db.session.add(entry)
        db.session.commit()

        session["last_entry_id"] = entry.id

        result = {
            "sentiment": sentiment,
            "sentiment_score": vader["compound"],
            "emotions": emotions,
            "stress_score": stress,
            "mood_emoji": mood,
            "timestamp": timestamp
        }

        return redirect("/suggestions")

    entries = JournalEntry.query.filter_by(user_id=session["user_id"]).all()
    streak = calculate_streak(entries)
    return render_template("home.html", result=result, username=session["username"], streak=streak)

@app.route("/vlogs", methods=["GET", "POST"])
def vlogs():
    if "user_id" not in session:
        return redirect("/login")

    result = None
    if request.method == "POST" and "vlog_text" in request.form:
        text = request.form["vlog_text"]
        mood = session.get("selected_mood", "😐")
        vader = sia.polarity_scores(text)
        emotions = get_emotions(clean_text(text))
        sentiment = determine_sentiment(vader, emotions)
        stress = estimate_stress(vader, emotions)
        ai_suggestions = generate_ai_suggestion(emotions, sentiment)
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

        entry = JournalEntry(
            user_id=session["user_id"],
            journal_text=text,
            mood_emoji=mood,
            sentiment=sentiment,
            sentiment_score=vader["compound"],
            emotions=", ".join(emotions),
            stress_score=stress,
            timestamp=timestamp,
            ai_suggestions=ai_suggestions
        )
        db.session.add(entry)
        db.session.commit()
        session["last_entry_id"] = entry.id

        return redirect("/suggestions")

    entries = JournalEntry.query.filter_by(user_id=session["user_id"]).all()
    streak = calculate_streak(entries)
    return render_template("vlogs.html", result=result, username=session["username"], streak=streak)

@app.route("/suggestions")
def suggestions():
    if "user_id" not in session or "last_entry_id" not in session:
        return redirect("/login")

    entry = JournalEntry.query.filter_by(id=session["last_entry_id"], user_id=session["user_id"]).first()
    if not entry:
        flash("Suggestion not found.")
        return redirect("/dashboard")

    videos = get_youtube_videos(entry.emotions.split(","), entry.sentiment)

    return render_template("suggestions.html",
        username=session["username"],
        suggestions=entry.ai_suggestions,
        sentiment=entry.sentiment,
        emotions=entry.emotions.split(","),
        mood=entry.mood_emoji,
        timestamp=entry.timestamp,
        youtube_videos=videos
    )

@app.route("/trends")
def trends():
    if "user_id" not in session:
        return redirect("/login")

    entries = JournalEntry.query.filter_by(user_id=session["user_id"]).order_by(JournalEntry.timestamp).all()
    dates = [e.timestamp for e in entries]
    sentiments = [e.sentiment_score for e in entries]
    stresses = [e.stress_score for e in entries]
    all_emotions = []
    for entry in entries:
        all_emotions.extend([e.strip() for e in entry.emotions.split(",") if e.strip()])
    top_emotions = Counter(all_emotions).most_common(5)

    return render_template("trends.html", dates=dates, sentiments=sentiments, stresses=stresses, top_emotions=top_emotions)

@app.route("/chatbot", methods=["GET", "POST"])
def chatbot():
    if "user_id" not in session:
        return redirect("/login")

    # Load chat history (or start fresh)
    chat_log = session.get("chat_log", [])

    if request.method == "POST":
        # ✅ Safe form input (works with both "message" or "user_input")
        user_message = request.form.get("message") or request.form.get("user_input", "")

        if user_message.strip():
            chat_log.append({"role": "user", "content": user_message})

            try:
                url = "https://api.groq.com/openai/v1/chat/completions"
                headers = {
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": "application/json"
                }

                payload = {
                    "model": "llama-3.3-70b-versatile",  # ✅ Correct Groq model
                    "messages": [
                        {"role": "system", "content": "You are a compassionate, empathetic mental health assistant. Support the user with gentle responses and reflection. Never give medical advice."},
                        *chat_log
                    ],
                    "temperature": 0.7,
                    "max_tokens": 500
                }

                response = requests.post(url, headers=headers, json=payload)
                response.raise_for_status()
                data = response.json()

                # ✅ Extract assistant’s reply
                bot_reply = data["choices"][0]["message"]["content"].strip()

                chat_log.append({"role": "assistant", "content": bot_reply})

            except Exception as e:
                print("Chatbot error:", e)
                chat_log.append({"role": "assistant", "content": "⚠️ Sorry, I’m having trouble responding right now."})

            # Save chat history in session
            session["chat_log"] = chat_log

    return render_template("chatbot.html", chat_log=chat_log, username=session.get("username"))



# ---------- MAIN ----------
if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True)
