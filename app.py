import streamlit as st
import json
import re
from datetime import datetime
from langchain.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
import plotly.express as px
import pandas as pd

# ----------------------------
# 1. CONSTANTS
# ----------------------------
GLASS_ML = 250          # 1 glass = 250 ml
WATER_GOAL_LITRES = 2.5 # Global average hydration goal


# ----------------------------
# 2. LLM SETUP
# ----------------------------
try:
    llm = GoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=st.secrets["GOOGLE_API_KEY"]
    )
except Exception:
    st.error(
        "Could not initialize the language model. Please ensure your "
        "'GOOGLE_API_KEY' is set correctly in Streamlit's secrets."
    )
    st.info(
        "For more information, visit Streamlit's secrets management docs."
    )
    st.stop()


# ----------------------------
# 3. PROMPTS
# ----------------------------

# Meal analyzer: returns macros in JSON
prompt_meal_analyzer = PromptTemplate(
    input_variables=["meal_description"],
    template=(
        "You are a nutrition analysis expert. Analyze the following meal description "
        "and provide a reasonable estimate for its nutritional content.\n"
        "Your response MUST be ONLY a JSON object with the keys "
        "'calories', 'protein_g', 'carbs_g', and 'fats_g'. "
        "Do not include any other text.\n\n"
        "Meal: {meal_description}\n\n"
        "JSON Output:"
    ),
)

# Workout analyzer: returns calories burned in JSON
prompt_workout_analyzer = PromptTemplate(
    input_variables=["workout_description", "user_profile"],
    template=(
        "You are a fitness expert. Analyze the following workout and user profile to "
        "estimate calories burned.\n"
        "Your response MUST be ONLY a JSON object with the key 'calories_burned'. "
        "Do not include any other text.\n\n"
        "Workout: {workout_description}\n"
        "User: {user_profile}\n\n"
        "JSON Output:"
    ),
)

# Daily coach prompt – now includes HYDRATION
prompt_daily_coach = PromptTemplate(
    input_variables=[
        "user_profile",
        "goal",
        "calorie_target",
        "logged_meals_summary",
        "total_consumption",
        "logged_workouts_summary",
        "calories_burned",
        "adjusted_calorie_target",
        "bmi_category",
        "water_litres",
        "water_goal_litres",
    ],
    template=(
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
    ),
)

# LangChain chains
meal_analyzer_chain = prompt_meal_analyzer | llm | StrOutputParser()
workout_analyzer_chain = prompt_workout_analyzer | llm | StrOutputParser()
daily_coach_chain = prompt_daily_coach | llm | StrOutputParser()


# ----------------------------
# 4. HELPER FUNCTIONS
# ----------------------------
def calculate_tdee(gender, weight, height, age, activity_level):
    """Calculate TDEE using Mifflin-St Jeor + activity multiplier."""
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
    """Derive calorie target from TDEE + goal."""
    if goal == "Weight Loss":
        return tdee - 500
    if goal == "Weight Gain":
        return tdee + 500
    return tdee


def calculate_bmi(weight_kg, height_cm):
    """Calculate BMI."""
    if height_cm <= 0:
        return 0
    h_m = height_cm / 100
    return weight_kg / (h_m ** 2)


def get_bmi_category(bmi):
    """Return BMI category text."""
    if bmi < 18.5:
        return "Underweight"
    if bmi < 24.9:
        return "Normal weight"
    if bmi < 29.9:
        return "Overweight"
    return "Obesity"


def clean_json_response(response_text):
    """Extract JSON block from model output."""
    match = re.search(r"\{.*\}", response_text, re.DOTALL)
    return match.group(0) if match else None


# ----------------------------
# 5. STREAMLIT SETUP & STATE
# ----------------------------
st.set_page_config(layout="wide", page_title="AI Diet, Fitness & Hydration Coach")
st.title("Your Interactive AI Diet, Fitness & Hydration Coach 🥗🏋️‍♀️💧")

meal_types = [
    "Breakfast",
    "Breakfast Snack",
    "Lunch",
    "Evening Snack",
    "Dinner",
    "Dessert",
]

# Initialize session state
if "meals" not in st.session_state:
    st.session_state.meals = {mt: [] for mt in meal_types}

if "total_consumption" not in st.session_state:
    st.session_state.total_consumption = {
        "calories": 0.0,
        "protein_g": 0.0,
        "carbs_g": 0.0,
        "fats_g": 0.0,
    }

if "workouts" not in st.session_state:
    st.session_state.workouts = []

if "calories_burned" not in st.session_state:
    st.session_state.calories_burned = 0.0

# Water tracking
if "water_ml" not in st.session_state:
    st.session_state.water_ml = 0  # total ml

if "water_logs" not in st.session_state:
    st.session_state.water_logs = []  # list of dicts: {type, ml}


# ----------------------------
# 6. SIDEBAR: PROFILE, WORKOUT, WATER
# ----------------------------
with st.sidebar:
    st.header("👤 Your Profile & Goal")

    with st.form("profile_form"):
        age = st.number_input("Age", 15, 90, 25)
        gender = st.selectbox("Gender", ["Male", "Female"])
        weight = st.number_input("Weight (kg)", 35.0, 200.0, 65.0, step=0.5)
        height = st.number_input("Height (cm)", 130.0, 220.0, 165.0, step=0.5)
        activity_level = st.selectbox(
            "Typical Activity (non-exercise)",
            ["Sedentary", "Lightly Active", "Moderately Active", "Very Active"],
        )
        goal = st.selectbox("Primary Goal", ["Weight Loss", "Maintenance", "Weight Gain"])
        st.form_submit_button("Update Profile")

    # Energy calcs
    tdee = calculate_tdee(gender, weight, height, age, activity_level)
    calorie_target = get_calorie_target(tdee, goal)
    bmi = calculate_bmi(weight, height)
    bmi_category = get_bmi_category(bmi)

    st.info(f"**BMI:** {bmi:.1f} ({bmi_category})")
    st.success(f"**Base Calorie Target:** {calorie_target:,.0f} kcal")

    st.header("🏃‍♀️ Log Your Workout")
    workout_input = st.text_input(
        "Describe your workout:",
        placeholder="e.g., 45 minutes brisk walking, or 30 minutes weightlifting",
    )

    if st.button("Log Workout"):
        if workout_input.strip():
            with st.spinner("Analyzing your workout..."):
                try:
                    user_profile_for_workout = f"Weight: {weight}kg, Age: {age}, Gender: {gender}"
                    response_text = workout_analyzer_chain.invoke(
                        {
                            "workout_description": workout_input,
                            "user_profile": user_profile_for_workout,
                        }
                    )
                    json_str = clean_json_response(response_text)
                    if json_str:
                        workout_data = json.loads(json_str)
                        calories_burned = workout_data.get("calories_burned", 0)
                        workout_log = {
                            "description": workout_input,
                            "calories_burned": calories_burned,
                        }
                        st.session_state.workouts.append(workout_log)
                        st.session_state.calories_burned += calories_burned
                        st.success(f"Workout logged! Approx. {calories_burned:.0f} kcal burned.")
                        st.rerun()
                    else:
                        st.error("Could not analyze workout. The model returned an invalid format.")
                except json.JSONDecodeError:
                    st.error("Failed to parse workout analysis. Please try again.")
                except Exception as e:
                    st.error(f"An error occurred: {e}")
        else:
            st.warning("Please describe your workout.")

    st.header("💧 Water Intake")

    water_litres = st.session_state.water_ml / 1000.0
    st.metric("Today's Water", f"{water_litres:.2f} L", f"Goal: {WATER_GOAL_LITRES:.1f} L")

    w_col1, w_col2 = st.columns(2)
    with w_col1:
        if st.button("Add 1 Glass (250 ml)"):
            st.session_state.water_ml += GLASS_ML
            st.session_state.water_logs.append({"type": "glass", "ml": GLASS_ML})
            st.rerun()

    with w_col2:
        add_litres = st.number_input(
            "Add Litres",
            min_value=0.0,
            max_value=5.0,
            value=0.0,
            step=0.25,
            key="water_litre_input",
        )
        if st.button("Log Litres"):
            if add_litres > 0:
                ml = int(add_litres * 1000)
                st.session_state.water_ml += ml
                st.session_state.water_logs.append({"type": "litres", "ml": ml})
                st.rerun()
            else:
                st.warning("Enter a value greater than 0.")


# ----------------------------
# 7. GLOBAL CALC: ADJUSTED CALORIE TARGET
# ----------------------------
adjusted_calorie_target = calorie_target + st.session_state.calories_burned


# ----------------------------
# 8. MAIN LAYOUT
# ----------------------------
col1, col2 = st.columns([1, 1.2], gap="large")

# --- LEFT: MEAL LOGGING ---
with col1:
    st.header("✍️ Log Your Meals")

    for meal_type in meal_types:
        with st.expander(f"Log {meal_type}"):
            meal_input = st.text_area(
                f"Describe your {meal_type.lower()}:",
                placeholder="e.g., 2 rotis, dal, sabzi, salad",
                key=f"input_{meal_type}",
            )
            if st.button(f"Submit {meal_type}", key=f"btn_{meal_type}", use_container_width=True):
                if meal_input.strip():
                    with st.spinner(f"Analyzing your {meal_type.lower()}..."):
                        try:
                            response_text = meal_analyzer_chain.invoke(
                                {"meal_description": meal_input}
                            )
                            json_str = clean_json_response(response_text)
                            if json_str:
                                nutrition_data = json.loads(json_str)
                                meal_log = {
                                    "description": meal_input,
                                    "nutrition": nutrition_data,
                                }
                                st.session_state.meals[meal_type].append(meal_log)

                                # Update total daily macros
                                for key in st.session_state.total_consumption:
                                    st.session_state.total_consumption[key] += nutrition_data.get(
                                        key, 0
                                    )

                                st.success(f"{meal_type} logged!")
                                st.rerun()
                            else:
                                st.error(
                                    "Sorry, I couldn't understand that meal. Please be more descriptive."
                                )
                        except json.JSONDecodeError:
                            st.error(
                                "Failed to parse the meal analysis. Please try again with a more specific description."
                            )
                        except Exception as e:
                            st.error(f"An error occurred: {e}")
                else:
                    st.warning(f"Please describe your {meal_type.lower()}.")


# --- RIGHT: DASHBOARD & LOGS ---
with col2:
    st.header("📊 Your Daily Dashboard")

    with st.container(border=True):
        calories_consumed = st.session_state.total_consumption["calories"]
        calories_remaining = adjusted_calorie_target - calories_consumed
        progress = (
            0
            if adjusted_calorie_target <= 0
            else min(calories_consumed / adjusted_calorie_target, 1.0)
        )

        m_col1, m_col2, m_col3 = st.columns(3)
        m_col1.metric("Adjusted Target", f"{adjusted_calorie_target:,.0f} kcal")
        m_col2.metric("Consumed", f"{calories_consumed:,.0f} kcal")
        m_col3.metric(
            "Remaining",
            f"{calories_remaining:,.0f} kcal",
            delta_color="inverse",
        )

        st.progress(progress)

        # Macros pie chart
        macros = st.session_state.total_consumption
        if (
            macros["protein_g"] > 0
            or macros["carbs_g"] > 0
            or macros["fats_g"] > 0
        ):
            macro_calories = {
                "Protein": macros["protein_g"] * 4,
                "Carbohydrates": macros["carbs_g"] * 4,
                "Fats": macros["fats_g"] * 9,
            }
            total_macro_cals = sum(macro_calories.values())
            if total_macro_cals > 0:
                df = pd.DataFrame(
                    list(macro_calories.items()),
                    columns=["Nutrient", "Calories"],
                )
                fig = px.pie(
                    df,
                    values="Calories",
                    names="Nutrient",
                    title="Calorie Distribution from Macros",
                    hole=0.4,
                )
                fig.update_traces(textposition="inside", textinfo="percent+label")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Log a meal to see your nutrient distribution.")

    st.header("🗓️ Today's Log")
    with st.container(border=True):
        log_container = st.container(height=250)
        all_meals_logged = []
        with log_container:
            # Meals
            for meal_type, meals in st.session_state.meals.items():
                if meals:
                    st.subheader(meal_type)
                    for meal in meals:
                        n = meal["nutrition"]
                        st.markdown(
                            f"- {meal['description']} "
                            f"*({n['calories']:.0f} kcal, "
                            f"{n['protein_g']:.0f}g P, "
                            f"{n['carbs_g']:.0f}g C, "
                            f"{n['fats_g']:.0f}g F)*"
                        )
                        all_meals_logged.append(
                            f"{meal_type}: {meal['description']}"
                        )

            # Workouts
            if st.session_state.workouts:
                st.subheader("Workouts")
                for workout in st.session_state.workouts:
                    st.markdown(
                        f"- {workout['description']} "
                        f"*({workout['calories_burned']:.0f} kcal burned)*"
                    )

            # Water logs
            if st.session_state.water_ml > 0:
                st.subheader("Water Intake")
                total_l = st.session_state.water_ml / 1000.0
                st.markdown(f"- Total: **{total_l:.2f} L**")
                if st.session_state.water_logs:
                    for w in st.session_state.water_logs[-5:]:
                        l = w["ml"] / 1000.0
                        label = "Glass" if w["type"] == "glass" else "Litres"
                        st.markdown(f"  - {label}: {l:.2f} L")

            if not all_meals_logged and not st.session_state.workouts and st.session_state.water_ml == 0:
                st.info("Log a meal, workout, or water to see your daily summary here.")


# ----------------------------
# 9. AI COACH SECTION
# ----------------------------
st.divider()
st.header("🧠 Your AI Coach's Advice")

if st.button("Get My Insights & Suggestions", use_container_width=True, type="primary"):
    # Rebuild all_meals_logged (we also did it above, but do it safely here too)
    all_meals_logged = []
    for meal_type, meals in st.session_state.meals.items():
        for meal in meals:
            all_meals_logged.append(f"{meal_type}: {meal['description']}")

    if not all_meals_logged and not st.session_state.workouts and st.session_state.water_ml == 0:
        st.warning("Please log at least one meal, workout, or some water to get personalized advice.")
    else:
        with st.spinner("Your coach is thinking..."):
            water_litres = st.session_state.water_ml / 1000.0

            prompt_input = {
                "user_profile": f"Age: {age}, Gender: {gender}, Weight: {weight}kg, Height: {height}cm",
                "goal": goal,
                "bmi_category": bmi_category,
                "calorie_target": f"{calorie_target:,.0f}",
                "adjusted_calorie_target": f"{adjusted_calorie_target:,.0f}",
                "logged_meals_summary": "; ".join(all_meals_logged) if all_meals_logged else "None",
                "total_consumption": (
                    f"{st.session_state.total_consumption['calories']:.0f} kcal "
                    f"({st.session_state.total_consumption['protein_g']:.0f}g P, "
                    f"{st.session_state.total_consumption['carbs_g']:.0f}g C, "
                    f"{st.session_state.total_consumption['fats_g']:.0f}g F)"
                ),
                "logged_workouts_summary": "; ".join(
                    [w["description"] for w in st.session_state.workouts]
                ) or "None",
                "calories_burned": f"{st.session_state.calories_burned:.0f}",
                "water_litres": f"{water_litres:.2f}",
                "water_goal_litres": f"{WATER_GOAL_LITRES:.2f}",
            }

            try:
                advice = daily_coach_chain.invoke(prompt_input)
                with st.container(border=True):
                    # Allow rich markdown rendering
                    st.markdown(advice, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Could not get advice from the coach. Error: {e}")
