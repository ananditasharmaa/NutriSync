# 🥗🏋️‍♀️ NutriSync — Your AI Diet, Fitness & Hydration Coach

**NutriSync** blends **nutrition**, **fitness**, and **hydration tracking**, keeping you aligned with your health goals every single day.
Powered by **Streamlit**, **LangChain**, and **Google Gemini**, NutriSync helps you log meals, track workouts, monitor water intake, and receive personalized coaching based on your unique body profile.

👉 **Live App:**

### 🔗 [https://nutrisync.streamlit.app](https://nutrisync.streamlit.app)

---

## 🚀 Features

### 🧠 AI-Powered Nutrition Analysis

Log any meal in plain English — NutriSync estimates:

* Calories
* Protein (g)
* Carbs (g)
* Fats (g)

### 🏋️ Smart Workout Analyzer

Describe your workout naturally (e.g., `"30 minutes cycling"`) and get:

* Estimated calories burned
* Adjustments applied to your daily calorie target

### 💧 Hydration Tracking

* Add glasses or litres
* See daily total & progress toward hydration goals

### 📊 Personalized Dashboard

* Calories consumed
* Target vs remaining calories
* Macro distribution pie chart
* Workout calories burned
* Water intake logs

### 🧠 AI Coach Suggestions

Receive daily:

* Meal recommendations
* Workout suggestions
* Hydration and recovery tips
* Personalized insights based on BMI, goals, meals, and workouts

### 🔐 Secure by Design

Your Google API key is protected using **Streamlit Secrets Management**.

---

## 📸 Preview

<img width="1904" height="878" alt="image" src="https://github.com/user-attachments/assets/d5d602b9-a701-4b3a-9557-636328e03667" />
<img width="617" height="829" alt="image" src="https://github.com/user-attachments/assets/fe4e9f02-40d9-45b9-b30b-4708317fa8fe" />
<img width="1417" height="537" alt="image" src="https://github.com/user-attachments/assets/9b0a20ea-2df3-4238-84c2-da7fb52d6337" />
<img width="901" height="975" alt="image" src="https://github.com/user-attachments/assets/03ce6686-f345-4783-805b-f5f2629296c3" />

---

## 🛠️ Tech Stack

* **Streamlit** — UI & state management
* **LangChain** — LLM orchestration (meal/workout analysis + coaching)
* **Google Generative AI (Gemini)** — AI reasoning & text generation
* **Python**
* **Plotly & Pandas** — Dashboard and data visualization

---

## ⚙️ Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/ananditasharmaa/NutriSync.git
cd NutriSync
```

### 2. Create secrets file

Create `.streamlit/secrets.toml` and add:

```toml
GOOGLE_API_KEY = "your-google-api-key"
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Recommended:** use Python 3.10 or 3.11 to avoid compatibility issues with some LLM ecosystem packages.

### 4. Run the app

```bash
streamlit run app.py
```

---

## 🌐 Deployment

NutriSync is live at:

### 🔗 **[https://nutrisync.streamlit.app](https://nutrisync.streamlit.app)**

Deployed with Streamlit Cloud using:

* `requirements.txt` (pinned deps)
* `runtime.txt` (Python 3.10 recommended)
* Streamlit Secrets for API key management

---

## 🔧 Troubleshooting & Notes

* If you see `ModuleNotFoundError: No module named 'langchain_core.pydantic_v1'`, ensure your environment uses `pydantic==1.10.13` and compatible LangChain packages. Example `requirements.txt` excerpt:

```
langchain==0.2.11
langchain-core==0.2.22
langchain-community==0.2.11
langchain-google-genai==1.0.6
google-generativeai==0.7.1
pydantic==1.10.13
streamlit
plotly
pandas
```

* If deployment environment uses Python 3.13 or newer, add `runtime.txt` to pin a supported Python (e.g., `python-3.10.12`).

* For quick local diagnosis, run:

```bash
pip freeze
python -c "import pydantic; print(pydantic.__version__)"
```
