import streamlit as st
import json
import re
from datetime import datetime
from langchain.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
import plotly.express as px
import pandas as pd

# --- 1. SETUP: API Key, LLM, and Prompts ---

# It's recommended to place the LLM initialization in a try-except block
try:
    llm = GoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=st.secrets["GOOGLE_API_KEY"])
except Exception as e:
    st.error("Could not initialize the language model. Please ensure your 'GOOGLE_API_KEY' is set correctly in Streamlit's secrets.")
    st.info("For more information, visit: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management")
    st.stop()


# --- Prompts ---
# This prompt is designed to extract structured JSON data from a meal description.
prompt_meal_analyzer = PromptTemplate(
    input_variables=['meal_description'],
    template=(
        "You are a nutrition analysis expert. Analyze the following meal description and provide a reasonable estimate for its nutritional content. "
        "Your response MUST be ONLY a JSON object with the keys 'calories', 'protein_g', 'carbs_g', and 'fats_g'. Do not include any other text or formatting.\n\n"
        "Meal: {meal_description}\n\n"
        "JSON Output:"
    )
)

# This prompt analyzes a workout description along with user's physical profile for a better calorie burn estimate.
prompt_workout_analyzer = PromptTemplate(
    input_variables=['workout_description', 'user_profile'],
    template=(
        "You are a fitness expert. Analyze the following workout description and the user's profile to provide a reasonable estimate for calories burned. "
        "The user's profile is: {user_profile}. "
        "Your response MUST be ONLY a JSON object with the key 'calories_burned'. Do not include any other text or formatting.\n\n"
        "Workout: {workout_description}\n\n"
        "JSON Output:"
    )
)

# This is the main coaching prompt, updated to include BMI and provide more structured, actionable advice.
prompt_daily_coach = PromptTemplate(
    input_variables=['user_profile', 'goal', 'calorie_target', 'logged_meals_summary', 'total_consumption', 'logged_workouts_summary', 'calories_burned', 'adjusted_calorie_target', 'bmi_category'],
    template=(
        "You are an encouraging and insightful AI Diet & Fitness Coach. Your goal is to provide structured, actionable, and motivating advice based on the user's progress today. "
        "Keep your tone positive and helpful.\n\n"
        "**Today's User Data:**\n"
        "------------------------\n"
        "User Profile: {user_profile}\n"
        "Primary Goal: {goal}\n"
        "BMI Category: {bmi_category}\n"
        "Base Daily Calorie Target: {calorie_target} kcal\n"
        "Workouts Logged: {logged_workouts_summary}\n"
        "Calories Burned from Workouts: {calories_burned} kcal\n"
        "Adjusted Daily Calorie Target (Base + Burned): {adjusted_calorie_target} kcal\n"
        "Meals Logged: {logged_meals_summary}\n"
        "Total Consumption: {total_consumption}\n"
        "------------------------\n\n"
        "**Your Task:**\n"
        "Based on all the information above, please provide the following in a clear, structured Markdown format. Address the user directly.\n\n"
        "### 📈 Your Daily Summary & Insights\n"
        "* Start with a brief, positive analysis of their progress. Compare their consumption to their *Adjusted Calorie Target*.\n"
        "* Mention their workout effort and how it contributes to their goal.\n"
        "* Provide one key insight about their current eating pattern (e.g., 'Great job on protein intake!' or 'You're right on track with your calorie goal.').\n\n"
        "### 🍎 Meal Recommendations for Your Goal\n"
        "* Based on their goal, remaining calories, and BMI category ({bmi_category}), suggest **two to three specific, healthy meal or snack ideas**.\n"
        "* For each suggestion, briefly explain *why* it's a good choice for them (e.g., 'high in protein to support muscle gain', 'fiber-rich to keep you full').\n\n"
        "### 🏋️‍♂️ Workout Suggestions for Improvement\n"
        "* Based on their goal and BMI category, suggest **one or two types of exercises** they could consider adding to their routine.\n"
        "* Explain the benefit of each suggestion (e.g., 'Consider adding strength training to boost metabolism', 'Low-impact cardio like swimming could be great for joint health and calorie burn').\n\n"
        "### 💧 Recovery & Wellness Tip\n"
        "* Provide one short, actionable tip related to recovery, such as stretching, hydration, or sleep, especially if they logged a workout."
    )
)


# --- Chains using LangChain Expression Language (LCEL) ---
meal_analyzer_chain = prompt_meal_analyzer | llm | StrOutputParser()
workout_analyzer_chain = prompt_workout_analyzer | llm | StrOutputParser()
daily_coach_chain = prompt_daily_coach | llm | StrOutputParser()

# --- 2. HELPER FUNCTIONS ---
def calculate_tdee(gender, weight, height, age, activity_level):
    """Calculates Total Daily Energy Expenditure (TDEE)."""
    if gender == 'Male':
        bmr = 10 * weight + 6.25 * height - 5 * age + 5
    else: # Female
        bmr = 10 * weight + 6.25 * height - 5 * age - 161
    activity_multipliers = {'Sedentary': 1.2, 'Lightly Active': 1.375, 'Moderately Active': 1.55, 'Very Active': 1.725}
    return bmr * activity_multipliers.get(activity_level, 1.2)

def get_calorie_target(tdee, goal):
    """Calculates the daily calorie target based on the user's goal."""
    if goal == 'Weight Loss':
        return tdee - 500
    elif goal == 'Weight Gain':
        return tdee + 500
    else: # Maintenance
        return tdee

def calculate_bmi(weight_kg, height_cm):
    """Calculates Body Mass Index (BMI)."""
    if height_cm > 0:
        height_m = height_cm / 100
        return weight_kg / (height_m ** 2)
    return 0

def get_bmi_category(bmi):
    """Returns the BMI category string."""
    if bmi < 18.5:
        return "Underweight"
    elif 18.5 <= bmi < 24.9:
        return "Normal weight"
    elif 25 <= bmi < 29.9:
        return "Overweight"
    else:
        return "Obesity"

def clean_json_response(response_text):
    """Extracts a JSON object from a string, even if it's embedded in other text."""
    match = re.search(r"\{.*\}", response_text, re.DOTALL)
    if match:
        return match.group(0)
    return None

# --- 3. STREAMLIT UI ---
st.set_page_config(layout="wide", page_title="AI Diet & Fitness Coach")
st.title("Your Interactive AI Diet & Fitness Coach 🥗")

# --- Initialize session state ---
meal_types = ['Breakfast', 'Breakfast Snack', 'Lunch', 'Evening Snack', 'Dinner', 'Dessert']
if 'meals' not in st.session_state:
    st.session_state.meals = {meal_type: [] for meal_type in meal_types}
if 'total_consumption' not in st.session_state:
    st.session_state.total_consumption = {"calories": 0, "protein_g": 0, "carbs_g": 0, "fats_g": 0}
if 'workouts' not in st.session_state:
    st.session_state.workouts = []
if 'calories_burned' not in st.session_state:
    st.session_state.calories_burned = 0

# --- Sidebar for User Profile and Goal ---
with st.sidebar:
    st.header("👤 Your Profile & Goal")
    with st.form("profile_form"):
        age = st.number_input("Age", 20, 80, 30)
        gender = st.selectbox("Gender", ["Male", "Female"])
        weight = st.number_input("Weight (kg)", 40.0, 150.0, 70.0, format="%.1f")
        height = st.number_input("Height (cm)", 130.0, 220.0, 175.0, format="%.1f")
        activity_level = st.selectbox("Typical Activity Level (non-exercise)", ['Sedentary', 'Lightly Active', 'Moderately Active', 'Very Active'])
        goal = st.selectbox("Your Primary Goal", ['Weight Loss', 'Maintenance', 'Weight Gain'])
        st.form_submit_button("Update Profile")

    tdee = calculate_tdee(gender, weight, height, age, activity_level)
    calorie_target = get_calorie_target(tdee, goal)
    bmi = calculate_bmi(weight, height)
    bmi_category = get_bmi_category(bmi)

    st.info(f"**BMI:** {bmi:.1f} ({bmi_category})")
    st.success(f"**Base Calorie Target:** {calorie_target:,.0f} kcal")

    st.header("🏃‍♀️ Log Your Workout")
    workout_input = st.text_input("Describe your workout:", placeholder="e.g., 45 minutes of weightlifting")
    if st.button("Log Workout"):
        if workout_input:
            with st.spinner("Analyzing your workout..."):
                try:
                    user_profile_for_workout = f"Weight: {weight}kg, Age: {age}, Gender: {gender}"
                    response_text = workout_analyzer_chain.invoke({'workout_description': workout_input, 'user_profile': user_profile_for_workout})
                    json_str = clean_json_response(response_text)
                    if json_str:
                        workout_data = json.loads(json_str)
                        calories_burned = workout_data.get("calories_burned", 0)
                        workout_log = {"description": workout_input, "calories_burned": calories_burned}
                        st.session_state.workouts.append(workout_log)
                        st.session_state.calories_burned += calories_burned
                        st.success(f"Workout logged! Approx. {calories_burned} kcal burned.")
                        st.rerun()
                    else:
                        st.error("Could not analyze workout. The model returned an invalid format.")
                except json.JSONDecodeError:
                    st.error("Failed to parse the workout analysis. Please try again.")
                except Exception as e:
                    st.error(f"An error occurred: {e}")
        else:
            st.warning("Please describe your workout.")

# --- Main App Layout ---
col1, col2 = st.columns([1, 1.2], gap="large")

with col1:
    st.header("✍️ Log Your Meals")
    # --- UPDATED: Using st.expander for a vertical layout ---
    for meal_type in meal_types:
        with st.expander(f"Log {meal_type}"):
            meal_input = st.text_area(f"Describe your {meal_type.lower()}:", placeholder=f"e.g., A bowl of oatmeal with berries and nuts", key=f"input_{meal_type}")
            if st.button(f"Submit {meal_type}", key=f"btn_{meal_type}", use_container_width=True):
                if meal_input:
                    with st.spinner(f"Analyzing your {meal_type.lower()}..."):
                        try:
                            response_text = meal_analyzer_chain.invoke({'meal_description': meal_input})
                            json_str = clean_json_response(response_text)
                            if json_str:
                                nutrition_data = json.loads(json_str)
                                meal_log = {"description": meal_input, "nutrition": nutrition_data}
                                st.session_state.meals[meal_type].append(meal_log)
                                for key in st.session_state.total_consumption:
                                    st.session_state.total_consumption[key] += nutrition_data.get(key, 0)
                                st.success(f"{meal_type} logged!")
                                st.rerun()
                            else:
                                st.error("Sorry, I couldn't understand that meal. Please be more descriptive.")
                        except json.JSONDecodeError:
                            st.error("Failed to parse the meal analysis. Please try again with a more specific description.")
                        except Exception as e:
                            st.error(f"An error occurred: {e}")
                else:
                    st.warning(f"Please describe your {meal_type.lower()}.")


with col2:
    st.header("📊 Your Daily Dashboard")
    
    with st.container(border=True):
        adjusted_calorie_target = calorie_target + st.session_state.calories_burned
        calories_consumed = st.session_state.total_consumption['calories']
        calories_remaining = adjusted_calorie_target - calories_consumed
        progress = 0 if adjusted_calorie_target == 0 else min(calories_consumed / adjusted_calorie_target, 1.0)

        m_col1, m_col2, m_col3 = st.columns(3)
        m_col1.metric("Adjusted Target", f"{adjusted_calorie_target:,.0f} kcal")
        m_col2.metric("Consumed", f"{calories_consumed:,.0f} kcal")
        m_col3.metric("Remaining", f"{calories_remaining:,.0f} kcal", delta_color="inverse")
        st.progress(progress)

        macros = {
            'Protein (g)': st.session_state.total_consumption['protein_g'],
            'Carbs (g)': st.session_state.total_consumption['carbs_g'],
            'Fats (g)': st.session_state.total_consumption['fats_g']
        }
        
        if sum(macros.values()) > 0:
            macro_calories = {
                'Protein': macros['Protein (g)'] * 4,
                'Carbohydrates': macros['Carbs (g)'] * 4,
                'Fats': macros['Fats (g)'] * 9
            }
            
            total_calories_from_macros = sum(macro_calories.values())
            if total_calories_from_macros > 0:
                df = pd.DataFrame(list(macro_calories.items()), columns=['Nutrient', 'Calories'])
                fig = px.pie(df, values='Calories', names='Nutrient', title='Calorie Distribution from Macros',
                             hole=0.4, color_discrete_sequence=px.colors.sequential.RdBu)
                fig.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Log a meal to see your nutrient distribution.")
        else:
            st.info("Log a meal to see your nutrient distribution.")

    st.header("🗓️ Today's Log")
    with st.container(border=True):
        log_container = st.container(height=250)
        all_meals_logged = []
        with log_container:
            for meal_type, meals in st.session_state.meals.items():
                if meals:
                    st.subheader(meal_type)
                    for meal in meals:
                        st.markdown(f"- {meal['description']} *({meal['nutrition']['calories']:.0f} kcal, {meal['nutrition']['protein_g']:.0f}p, {meal['nutrition']['carbs_g']:.0f}c, {meal['nutrition']['fats_g']:.0f}f)*")
                        all_meals_logged.append(f"{meal_type}: {meal['description']}")
            
            if st.session_state.workouts:
                st.subheader("Workouts")
                for workout in st.session_state.workouts:
                    st.markdown(f"- {workout['description']} *({workout['calories_burned']:.0f} kcal burned)*")

            if not all_meals_logged and not st.session_state.workouts:
                st.info("Log a meal or workout to see your daily summary here.")

# --- AI Coach Section (Moved outside columns for better layout) ---
st.divider()
st.header("🧠 Your AI Coach's Advice")
if st.button("Get My Insights & Suggestions", use_container_width=True, type="primary"):
    # Check if there is anything to analyze
    if not all_meals_logged and not st.session_state.workouts:
        st.warning("Please log at least one meal or workout to get personalized advice.")
    else:
        with st.spinner("Your coach is thinking..."):
            # Prepare all the necessary inputs for the prompt
            user_profile_summary = f"Age: {age}, Gender: {gender}, Weight: {weight}kg, Height: {height}cm"
            logged_meals_summary = "; ".join(all_meals_logged) if all_meals_logged else "None"
            logged_workouts_summary = "; ".join([w['description'] for w in st.session_state.workouts]) or "None"
            adjusted_calorie_target = calorie_target + st.session_state.calories_burned

            prompt_input = {
                "user_profile": user_profile_summary,
                "goal": goal,
                "bmi_category": bmi_category,
                "calorie_target": f"{calorie_target:,.0f}",
                "adjusted_calorie_target": f"{adjusted_calorie_target:,.0f}",
                "logged_meals_summary": logged_meals_summary,
                "total_consumption": f"{st.session_state.total_consumption['calories']:.0f} kcal consumed ({st.session_state.total_consumption['protein_g']:.0f}g P, {st.session_state.total_consumption['carbs_g']:.0f}g C, {st.session_state.total_consumption['fats_g']:.0f}g F)",
                "logged_workouts_summary": logged_workouts_summary,
                "calories_burned": f"{st.session_state.calories_burned:.0f}"
            }
            try:
                # Invoke the chain and display the advice in a container
                advice = daily_coach_chain.invoke(prompt_input)
                with st.container(border=True):
                    st.markdown(advice)
            except Exception as e:
                st.error(f"Could not get advice from the coach. Error: {e}")
