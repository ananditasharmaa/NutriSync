import streamlit as st
import json
import re
from datetime import datetime
from langchain.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser

# --- 1. SETUP: API Key, LLM, and Prompts ---

try:
    llm = GoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=st.secrets["GOOGLE_API_KEY"])
except Exception as e:
    st.error("Could not initialize the language model. Please check your API key in .streamlit/secrets.toml.")
    st.stop()

# --- Prompts ---
prompt_meal_analyzer = PromptTemplate(
    input_variables=['meal_description'],
    template=(
        "You are a nutrition analysis expert. Analyze the following meal description and provide a reasonable estimate for its nutritional content. "
        "Your response MUST be ONLY a JSON object with the keys 'calories', 'protein_g', 'carbs_g', and 'fats_g'.\n\n"
        "Meal: {meal_description}\n\n"
        "JSON Output:"
    )
)

# --- NEW: Prompt 3: For analyzing a workout ---
prompt_workout_analyzer = PromptTemplate(
    input_variables=['workout_description', 'user_profile'],
    template=(
        "You are a fitness expert. Analyze the following workout description and the user's profile to provide a reasonable estimate for calories burned. "
        "The user's profile is: {user_profile}. "
        "Your response MUST be ONLY a JSON object with the key 'calories_burned'.\n\n"
        "Workout: {workout_description}\n\n"
        "JSON Output:"
    )
)

prompt_daily_coach = PromptTemplate(
    input_variables=['user_profile', 'goal', 'calorie_target', 'logged_meals_summary', 'total_consumption', 'logged_workouts_summary', 'calories_burned'],
    template=(
        "You are an encouraging and helpful AI Diet Coach. Your goal is to provide actionable insights and suggestions based on the user's progress today. "
        "Keep your tone positive and motivating.\n\n"
        "Here is the user's data for today:\n"
        "------------------------\n"
        "User Profile: {user_profile}\n"
        "Primary Goal: {goal}\n"
        "Original Daily Calorie Target: {calorie_target} kcal\n"
        "Workouts Logged Today: {logged_workouts_summary}\n"
        "Calories Burned from Workouts: {calories_burned} kcal\n"
        "Adjusted Daily Calorie Target (Original + Burned): {adjusted_calorie_target} kcal\n"
        "Meals Logged Today: {logged_meals_summary}\n"
        "Total Consumption Today: {total_consumption}\n"
        "------------------------\n\n"
        "Based on all the information above, please provide the following in a clear, structured Markdown format:\n"
        "1.  **💡 Insight:** A brief, positive analysis of their progress. Mention their workout and compare their consumption to their *Adjusted Calorie Target*.\n"
        "2.  **🍎 Next Meal Suggestion:** Suggest a specific, healthy meal or snack suitable for their remaining calories.\n"
        "3.  **🏋️‍♂️ Recovery Tip:** A short tip related to their workout, like stretching or hydration."
    )
)

# --- Chains using LangChain Expression Language (LCEL) ---
meal_analyzer_chain = prompt_meal_analyzer | llm | StrOutputParser()
workout_analyzer_chain = prompt_workout_analyzer | llm | StrOutputParser()
daily_coach_chain = prompt_daily_coach | llm | StrOutputParser()

# --- 2. HELPER FUNCTIONS ---
def calculate_tdee(gender, weight, height, age, activity_level):
    if gender == 'Male': bmr = 10 * weight + 6.25 * height - 5 * age + 5
    else: bmr = 10 * weight + 6.25 * height - 5 * age - 161
    activity_multipliers = {'Sedentary': 1.2, 'Lightly Active': 1.375, 'Moderately Active': 1.55, 'Very Active': 1.725}
    return bmr * activity_multipliers[activity_level]

def get_calorie_target(tdee, goal):
    if goal == 'Weight Loss': return tdee - 500
    elif goal == 'Weight Gain': return tdee + 500
    else: return tdee

# --- 3. STREAMLIT UI ---
st.set_page_config(layout="wide", page_title="AI Diet & Fitness Coach")
st.title("Your Interactive AI Diet & Fitness Coach 🥗")

# --- Initialize session state ---
meal_types = ['Breakfast', 'Breakfast Snack', 'Lunch', 'Evening Snack', 'Dinner', 'Dessert']
if 'meals' not in st.session_state: st.session_state.meals = {meal_type: [] for meal_type in meal_types}
if 'total_consumption' not in st.session_state: st.session_state.total_consumption = {"calories": 0, "protein_g": 0, "carbs_g": 0, "fats_g": 0}
if 'workouts' not in st.session_state: st.session_state.workouts = []
if 'calories_burned' not in st.session_state: st.session_state.calories_burned = 0

# --- Sidebar for User Profile and Goal ---
with st.sidebar:
    st.header("👤 Your Profile & Goal")
    age = st.number_input("Age", 20, 80, 30)
    gender = st.selectbox("Gender", ["Male", "Female"])
    weight = st.number_input("Weight (kg)", 40.0, 150.0, 70.0)
    height = st.number_input("Height (cm)", 130.0, 220.0, 175.0)
    activity_level = st.selectbox("Activity Level", ['Sedentary', 'Lightly Active', 'Moderately Active', 'Very Active'])
    goal = st.selectbox("Your Primary Goal", ['Weight Loss', 'Maintenance', 'Weight Gain'])

    tdee = calculate_tdee(gender, weight, height, age, activity_level)
    calorie_target = get_calorie_target(tdee, goal)

    st.success(f"Base Calorie Target: **{calorie_target:,.0f} kcal**")

    # --- NEW: Workout Logging UI in Sidebar ---
    st.header("🏃‍♀️ Log Your Workout")
    workout_input = st.text_input("Describe your workout:", placeholder="e.g., 30 minutes of jogging")
    if st.button("Log Workout"):
        if workout_input:
            with st.spinner("Analyzing your workout..."):
                try:
                    user_profile_for_workout = f"Weight: {weight}kg, Age: {age}, Gender: {gender}"
                    response_text = workout_analyzer_chain.invoke({'workout_description': workout_input, 'user_profile': user_profile_for_workout})
                    json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
                    if json_match:
                        json_str = json_match.group(0)
                        workout_data = json.loads(json_str)
                        calories_burned = workout_data.get("calories_burned", 0)

                        workout_log = {"description": workout_input, "calories_burned": calories_burned}
                        st.session_state.workouts.append(workout_log)
                        st.session_state.calories_burned += calories_burned
                        st.success(f"Workout logged! Approx. {calories_burned} kcal burned.")
                    else:
                        st.error("Could not analyze workout.")
                except Exception as e:
                    st.error(f"An error occurred: {e}")
        else:
            st.warning("Please describe your workout.")

# --- Main App Layout ---
col1, col2 = st.columns([1, 1.2], gap="large")

with col1:
    st.header("✍️ Log Your Meals")
    tabs = st.tabs(meal_types)
    for i, tab in enumerate(tabs):
        with tab:
            meal_type = meal_types[i]
            meal_input = st.text_area(f"Describe your {meal_type.lower()}:", placeholder="e.g., A bowl of oatmeal with berries", key=f"input_{meal_type}")
            if st.button(f"Log {meal_type}", key=f"btn_{meal_type}", use_container_width=True):
                if meal_input:
                    with st.spinner(f"Analyzing your {meal_type.lower()}..."):
                        try:
                            response_text = meal_analyzer_chain.invoke({'meal_description': meal_input})
                            json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
                            if json_match:
                                json_str = json_match.group(0)
                                nutrition_data = json.loads(json_str)
                                meal_log = {"description": meal_input, "nutrition": nutrition_data}
                                st.session_state.meals[meal_type].append(meal_log)
                                # --- BUG FIX: Use the full key to get the value ---
                                for key in st.session_state.total_consumption:
                                    st.session_state.total_consumption[key] += nutrition_data.get(key, 0)
                                st.success(f"{meal_type} logged!")
                            else: st.error("Sorry, I couldn't understand that meal.")
                        except Exception as e: st.error(f"An error occurred: {e}")
                else: st.warning(f"Please describe your {meal_type.lower()}.")

with col2:
    st.header("📊 Your Daily Dashboard")
    # --- UPDATED: Dynamic calorie target including workouts ---
    adjusted_calorie_target = calorie_target + st.session_state.calories_burned
    calories_consumed = st.session_state.total_consumption['calories']
    calories_remaining = adjusted_calorie_target - calories_consumed
    progress = 0 if adjusted_calorie_target == 0 else min(calories_consumed / adjusted_calorie_target, 1.0)
    
    m_col1, m_col2, m_col3 = st.columns(3)
    m_col1.metric("Adjusted Target", f"{adjusted_calorie_target:,.0f} kcal")
    m_col2.metric("Consumed", f"{calories_consumed:,.0f} kcal")
    m_col3.metric("Remaining", f"{calories_remaining:,.0f} kcal", delta_color="inverse")
    st.progress(progress)
    st.markdown(f"**Protein:** {st.session_state.total_consumption['protein_g']:.0f}g | **Carbs:** {st.session_state.total_consumption['carbs_g']:.0f}g | **Fats:** {st.session_state.total_consumption['fats_g']:.0f}g")

    st.header("🗓️ Today's Log")
    # Create a container with a fixed height to enable scrolling
    log_container = st.container(height=300) # You can adjust the height in pixels
    with log_container:
        # --- Display meals categorized by type (now inside the container) ---
        all_meals_logged = []
        for meal_type, meals in st.session_state.meals.items():
            if meals:
                st.subheader(meal_type)
                for meal in meals:
                    st.markdown(f"- {meal['description']} *({meal['nutrition']['calories']} kcal)*")
                    all_meals_logged.append(f"{meal_type}: {meal['description']}")

        if st.session_state.workouts:
            st.subheader("Workouts")
            for workout in st.session_state.workouts:
                st.markdown(f"- {workout['description']} *({workout['calories_burned']:.0f} kcal burned)*")

        if not all_meals_logged and not st.session_state.workouts:
            st.info("Log a meal or workout to get started.")

    st.header("🧠 Your AI Coach's Advice")
    if st.button("Get My Insights & Suggestions", use_container_width=True, type="primary"):
        if not all_meals_logged:
            st.warning("Please log at least one meal to get advice.")
        else:
            with st.spinner("Your coach is thinking..."):
                user_profile_summary = f"Age: {age}, Gender: {gender}, Weight: {weight}kg"
                logged_meals_summary = "; ".join(all_meals_logged)
                logged_workouts_summary = "; ".join([w['description'] for w in st.session_state.workouts]) or "None"

                prompt_input = {
                    "user_profile": user_profile_summary, "goal": goal,
                    "calorie_target": f"{calorie_target:,.0f}",
                    "adjusted_calorie_target": f"{adjusted_calorie_target:,.0f}",
                    "logged_meals_summary": logged_meals_summary,
                    "total_consumption": f"{st.session_state.total_consumption['calories']:.0f} kcal consumed",
                    "logged_workouts_summary": logged_workouts_summary,
                    "calories_burned": f"{st.session_state.calories_burned:.0f}"
                }
                try:
                    advice = daily_coach_chain.invoke(prompt_input)
                    st.markdown(advice)
                except Exception as e:
                    st.error(f"Could not get advice. Error: {e}")