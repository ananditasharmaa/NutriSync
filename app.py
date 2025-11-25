import streamlit as st
import json
import re
import pandas as pd
import plotly.express as px

# ----------------------------
# 0. PAGE CONFIG (Must be first)
# ----------------------------
st.set_page_config(layout="wide", page_title="AI Diet & Fitness Coach")

# ----------------------------
# 1. IMPORTS & SETUP
# ----------------------------
try:
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_google_genai import ChatGoogleGenerativeAI
except ImportError:
    st.error("Dependencies missing. Please ensure requirements.txt is installed correctly.")
    st.stop()

# ----------------------------
# 2. CONSTANTS
# ----------------------------
GLASS_ML = 250          # 1 glass = 250 ml
WATER_GOAL_LITRES = 2.5 # Global average hydration goal

# ----------------------------
# 3. HELPER FUNCTIONS
# ----------------------------
def clean_json_response(response_text):
    """Extract JSON block from model output."""
    # Remove markdown code blocks if present (```json ... ```)
    text = re.sub(r'```json\s*', '', response_text)
    text = re.sub(r'```', '', text)
    
    match = re.search(r"\{.*\}", text, re.DOTALL)
    return match.group(0) if match else None

def calculate_tdee(gender, weight, height, age, activity_level):
    if gender == "Male":
        bmr = 10 * weight + 6.25 * height - 5 * age + 5
    else:
        bmr = 10 * weight + 6.25 * height - 5 * age - 161

    multipliers = {
        "Sedentary": 1.2,
        "Lightly Active": 1.375,
        "Moderately Active": 1.55,
        "Very Active": 1.725,
    }
    return bmr * multipliers.get(activity_level, 1.2)

def get_calorie_target(tdee, goal):
    if goal == "Weight Loss": return tdee - 500
    if goal == "Weight Gain": return tdee + 500
    return tdee

def calculate_bmi(weight_kg, height_cm):
    if height_cm <= 0: return 0
    h_m = height_cm / 100
    return weight_kg / (h_m ** 2)

def get_bmi_category(bmi):
    if bmi < 18.5: return "Underweight"
    if bmi < 24.9: return "Normal weight"
    if bmi < 29.9: return "Overweight"
    return "Obesity"

# ----------------------------
# 4. LLM & PROMPT SETUP
# ----------------------------

# Check for API Key
if "GOOGLE_API_KEY" not in st.secrets:
    st.error("Missing Google API Key. Please set `GOOGLE_API_KEY` in your Streamlit Secrets.")
    st.stop()

try:
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        api_key=st.secrets["GOOGLE_API_KEY"],
        temperature=0.7
    )
except Exception as e:
    st.error(f"Error initializing Gemini: {e}")
    st.stop()

# Prompts
prompt_meal_analyzer = ChatPromptTemplate.from_template(
    "You are a nutrition analysis expert. Analyze the following meal description "
    "and provide a reasonable estimate for its nutritional content.\n"
    "Your response MUST be ONLY a valid JSON object with the keys "
    "'calories' (number), 'protein_g' (number), 'carbs_g' (number), and 'fats_g' (number). "
    "Do not include any other text or markdown formatting.\n\n"
    "Meal: {meal_description}\n\n"
    "JSON Output:"
)

prompt_workout_analyzer = ChatPromptTemplate.from_template(
    "You are a fitness expert. Analyze the following workout and user profile to "
    "estimate calories burned.\n"
    "Your response MUST be ONLY a valid JSON object with the key 'calories_burned' (number). "
    "Do not include any other text or markdown formatting.\n\n"
    "Workout: {workout_description}\n"
    "User: {user_profile}\n\n"
    "JSON Output:"
)

prompt_daily_coach = ChatPromptTemplate.from_template(
    "You are a structured, encouraging AI Diet & Fitness Coach.\n"
    "YOUR ENTIRE RESPONSE MUST BE IN CLEAN MARKDOWN WITH HEADINGS, BULLET LISTS, "
    "AND CLEAR SPACING. KEEP EACH PARAGRAPH UNDER 3 LINES. DO NOT RETURN WALLS OF TEXT.\n\n"
    "------------------------\n"
    "### 👤 User Summary\n"
    "- **Profile:** {user_profile}\n"
    "- **Primary Goal:** {goal}\n"
    "- **BMI Category:** {bmi_category}\n"
    "- **Base Calorie Target:** {calorie_target} kcal\n"
    "- **Adjusted Calorie Target (with workouts):** {adjusted_calorie_target} kcal\n"
    "- **Calories Burned from Workouts:** {calories_burned} kcal\n"
    "- **Meals Logged:** {logged_meals_summary}\n"
    "- **Today's Intake:** {total_consumption}\n"
    "- **Water Intake:** {water_litres} L / {water_goal_litres} L goal\n"
    "------------------------\n\n"
    "### 📈 Daily Summary & Insights\n"
    "- Start with a short, motivating summary.\n"
    "- Compare intake vs adjusted calorie target (over, under, or on track).\n"
    "- Call out one key macro insight (e.g., protein high/low, carbs heavy, fats balance).\n"
    "- Briefly mention if hydration is below, meeting, or above the goal.\n\n"
    "### 🍎 Meal Recommendations\n"
    "- Suggest 2–3 specific meal or snack ideas aligned with their goal.\n"
    "- For each, explain WHY it's good (e.g., high protein for muscle, high fiber for satiety).\n"
    "- If hydration is low, prefer water-rich foods (soups, fruits, etc.) where relevant.\n\n"
    "### 🏋️‍♂️ Workout Suggestions\n"
    "- Recommend 1–2 exercise types that suit their goal and BMI category.\n"
    "- Explain the benefit of each in simple, practical language.\n\n"
    "### 💧 Hydration & Recovery Tip\n"
    "- Provide 1 hydration-focused tip (e.g., spread water across the day, add an extra glass with meals).\n"
    "- Add a short recovery tip (sleep, stretching, light movement) if workouts are logged.\n"
)

# Chains
meal_analyzer_chain = prompt_meal_analyzer | llm | StrOutputParser()
workout_analyzer_chain = prompt_workout_analyzer | llm | StrOutputParser()
daily_coach_chain = prompt_daily_coach | llm | StrOutputParser()

# ----------------------------
# 5. STATE INITIALIZATION
# ----------------------------
if "meals" not in st.session_state:
    st.session_state.meals = {
        "Breakfast": [], "Breakfast Snack": [], "Lunch": [], 
        "Evening Snack": [], "Dinner": [], "Dessert": []
    }
if "total_consumption" not in st.session_state:
    st.session_state.total_consumption = {"calories": 0.0, "protein_g": 0.0, "carbs_g": 0.0, "fats_g": 0.0}
if "workouts" not in st.session_state:
    st.session_state.workouts = []
if "calories_burned" not in st.session_state:
    st.session_state.calories_burned = 0.0
if "water_ml" not in st.session_state:
    st.session_state.water_ml = 0
if "water_logs" not in st.session_state:
    st.session_state.water_logs = []

# ----------------------------
# 6. SIDEBAR UI
# ----------------------------
st.title("Your Interactive AI Diet & Fitness Coach 🥗🏋️‍♀️")

with st.sidebar:
    st.header("👤 Your Profile")
    with st.form("profile_form"):
        age = st.number_input("Age", 15, 90, 25)
        gender = st.selectbox("Gender", ["Male", "Female"])
        weight = st.number_input("Weight (kg)", 35.0, 200.0, 65.0, step=0.5)
        height = st.number_input("Height (cm)", 130.0, 220.0, 165.0, step=0.5)
        activity_level = st.selectbox("Activity Level", ["Sedentary", "Lightly Active", "Moderately Active", "Very Active"])
        goal = st.selectbox("Primary Goal", ["Weight Loss", "Maintenance", "Weight Gain"])
        st.form_submit_button("Update Profile")

    tdee = calculate_tdee(gender, weight, height, age, activity_level)
    calorie_target = get_calorie_target(tdee, goal)
    bmi = calculate_bmi(weight, height)
    bmi_category = get_bmi_category(bmi)

    st.info(f"**BMI:** {bmi:.1f} ({bmi_category})")
    st.success(f"**Base Target:** {calorie_target:,.0f} kcal")

    st.header("🏃‍♀️ Workouts")
    workout_input = st.text_input("Describe workout:", placeholder="e.g., 30 mins running")
    if st.button("Log Workout"):
        if workout_input.strip():
            with st.spinner("Analyzing..."):
                try:
                    resp = workout_analyzer_chain.invoke({
                        "workout_description": workout_input, 
                        "user_profile": f"{gender}, {weight}kg, {age}yo"
                    })
                    data = json.loads(clean_json_response(resp))
                    burned = data.get("calories_burned", 0)
                    st.session_state.workouts.append({"description": workout_input, "calories_burned": burned})
                    st.session_state.calories_burned += burned
                    st.rerun()
                except Exception as e:
                    st.error(f"Error logging workout: {e}")

    st.header("💧 Hydration")
    water_l = st.session_state.water_ml / 1000.0
    st.metric("Water", f"{water_l:.2f} L", f"Goal: {WATER_GOAL_LITRES} L")
    col_w1, col_w2 = st.columns(2)
    if col_w1.button("+ 1 Glass"):
        st.session_state.water_ml += GLASS_ML
        st.session_state.water_logs.append({"type": "glass", "ml": GLASS_ML})
        st.rerun()
    
    add_l = col_w2.number_input("Add L", 0.0, 5.0, 0.0, 0.25, label_visibility="collapsed")
    if col_w2.button("Add"):
        if add_l > 0:
            st.session_state.water_ml += int(add_l * 1000)
            st.session_state.water_logs.append({"type": "litres", "ml": int(add_l * 1000)})
            st.rerun()

# ----------------------------
# 7. MAIN DASHBOARD
# ----------------------------
adjusted_target = calorie_target + st.session_state.calories_burned
col1, col2 = st.columns([1, 1.2], gap="large")

# Left: Meal Log
with col1:
    st.header("✍️ Log Meals")
    for m_type in st.session_state.meals.keys():
        with st.expander(f"Log {m_type}"):
            txt = st.text_area(f"Describe {m_type}", key=f"txt_{m_type}")
            if st.button(f"Add {m_type}", key=f"btn_{m_type}"):
                if txt.strip():
                    with st.spinner("Analyzing nutrition..."):
                        try:
                            resp = meal_analyzer_chain.invoke({"meal_description": txt})
                            nutr = json.loads(clean_json_response(resp))
                            st.session_state.meals[m_type].append({"description": txt, "nutrition": nutr})
                            for k in st.session_state.total_consumption:
                                st.session_state.total_consumption[k] += nutr.get(k, 0)
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error: {e}")

# Right: Stats
with col2:
    st.header("📊 Dashboard")
    consumed = st.session_state.total_consumption["calories"]
    remaining = adjusted_target - consumed
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Target", f"{adjusted_target:,.0f}")
    c2.metric("Consumed", f"{consumed:,.0f}")
    c3.metric("Remaining", f"{remaining:,.0f}", delta_color="inverse")
    
    st.progress(min(consumed / adjusted_target, 1.0) if adjusted_target > 0 else 0)

    # Chart
    macros = st.session_state.total_consumption
    macro_data = {
        "Protein": macros["protein_g"] * 4,
        "Carbs": macros["carbs_g"] * 4,
        "Fats": macros["fats_g"] * 9
    }
    if sum(macro_data.values()) > 0:
        df = pd.DataFrame(list(macro_data.items()), columns=["Macro", "Cals"])
        fig = px.pie(df, values="Cals", names="Macro", title="Macro Split", hole=0.4)
        fig.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Log meals to see data.")

    # List Log
    with st.container(border=True, height=300):
        st.subheader("Today's Log")
        has_data = False
        for mt, items in st.session_state.meals.items():
            if items:
                st.markdown(f"**{mt}**")
                for i in items:
                    n = i['nutrition']
                    st.text(f"- {i['description']} ({n['calories']:.0f} cal)")
                has_data = True
        if st.session_state.workouts:
            st.markdown("**Workouts**")
            for w in st.session_state.workouts:
                st.text(f"- {w['description']} (-{w['calories_burned']:.0f} cal)")
            has_data = True
        
        if not has_data: st.caption("No activity yet.")

# ----------------------------
# 8. AI COACH
# ----------------------------
st.divider()
if st.button("🧠 Get Coach Insights", type="primary", use_container_width=True):
    # Prepare summary text
    meals_txt = []
    for mt, items in st.session_state.meals.items():
        for i in items: meals_txt.append(f"{mt}: {i['description']}")
    
    workouts_txt = [w['description'] for w in st.session_state.workouts]
    
    if not meals_txt and not workouts_txt:
        st.warning("Log something first!")
    else:
        with st.spinner("Coach is thinking..."):
            try:
                advice = daily_coach_chain.invoke({
                    "user_profile": f"{age}yo {gender}, {weight}kg, {height}cm",
                    "goal": goal,
                    "bmi_category": bmi_category,
                    "calorie_target": f"{calorie_target:,.0f}",
                    "adjusted_calorie_target": f"{adjusted_target:,.0f}",
                    "logged_meals_summary": "; ".join(meals_txt) or "None",
                    "total_consumption": str(st.session_state.total_consumption),
                    "logged_workouts_summary": "; ".join(workouts_txt) or "None",
                    "calories_burned": str(st.session_state.calories_burned),
                    "water_litres": f"{water_l:.2f}",
                    "water_goal_litres": str(WATER_GOAL_LITRES)
                })
                st.markdown(advice)
            except Exception as e:
                st.error(f"Coach Error: {e}")