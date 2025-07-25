# 🥗🏋️‍♀️ NutriSync

**NutriSync** combines **nutrition** and **sync** to keep you aligned with your health goals — every day.  
This AI-powered app helps you log meals, track workouts, and receive personalized coaching using your body profile and goals.

Built with **Streamlit** and **LangChain**, NutriSync gives intelligent suggestions based on:

✅ Your height, weight, age, gender, and activity level  
✅ Your daily meal intake  
✅ Your logged workouts  
✅ Your goal: weight loss, gain, or maintenance

---

## 🚀 Features

- 🧠 **AI-Powered Nutrition Analysis**: Get estimated calories, protein, carbs, and fats for any meal.
- 🏋️ **Workout Analyzer**: Log workouts in plain English — the app estimates calories burned.
- 📊 **Personalized Dashboard**: See how much you’ve eaten, how much you’ve burned, and what’s left.
- 💡 **Smart Suggestions**: Receive meal ideas and recovery tips based on your day’s log.
- 🔐 **Private & Secure**: Your API key stays private using Streamlit Secrets.

---

## 📸 Preview

<img width="1904" height="878" alt="image" src="https://github.com/user-attachments/assets/d5d602b9-a701-4b3a-9557-636328e03667" />
<img width="617" height="829" alt="image" src="https://github.com/user-attachments/assets/fe4e9f02-40d9-45b9-b30b-4708317fa8fe" />
<img width="1417" height="537" alt="image" src="https://github.com/user-attachments/assets/9b0a20ea-2df3-4238-84c2-da7fb52d6337" />
<img width="901" height="975" alt="image" src="https://github.com/user-attachments/assets/03ce6686-f345-4783-805b-f5f2629296c3" />



---

## 🛠️ Tech Stack

- [Streamlit](https://streamlit.io/)
- [LangChain](https://www.langchain.com/)
- [Google Generative AI (Gemini)](https://ai.google.dev/)
- Python, Prompt Engineering, Langchain

---

## ⚙️ How to Run Locally

1. **Clone the repository**

```bash
git clone https://github.com/ananditasharmaa/NutriSync.git
cd NutriSync
Install dependencies
```
2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Create API Key**
Create a file named .streamlit/secrets.toml and add:

```toml
GOOGLE_API_KEY = "your-google-api-key"
```
4. **Run the app**

```bash
streamlit run app.py
```
