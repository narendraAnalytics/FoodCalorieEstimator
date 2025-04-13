import streamlit as st
import google.generativeai as genai
from PIL import Image
from dotenv import load_dotenv
import os
import io
import time
from agno.agent import Agent
from agno.models.google import Gemini
from agno.tools.duckduckgo import DuckDuckGoTools
import matplotlib
from textwrap import dedent

# Use a non-interactive backend for Matplotlib
matplotlib.use('Agg')


# Load environment variables
load_dotenv()
api_key = os.getenv('GEMINI_API_KEY')

#load_dotenv()
#api_key = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY"))


# Check if the API key is loaded correctly
if not api_key:
    st.error("🛑 GEMINI_API_KEY not found. Please ensure it's set in your .env file or environment variables.")
    st.stop() # Stop the app if the key is missing


# Configure the Gemini API client
try:
    genai.configure(api_key=api_key)
except Exception as e:
    st.error(f"🛑 Error configuring Gemini API: {e}")
    st.stop()

# Define the input prompt for the Gemini API (Keep your original detailed prompt)
# --- Input Prompt remains the same as provided in the previous version ---
input_prompt = """
You are an expert AI nutritional consultant 🧑‍⚕️ analyzing food items and drinks from an image 📸.
Your task is to identify each food item or drink component visible, estimate its calorie count 🔢, and provide a brief nutritional overview (like estimated protein, carbs, fats, vitamins if possible). Aim to give the best, most informative response based on the image.

Please present the information clearly for each item, followed by the total estimated calories if multiple items are present. Use emojis to make it engaging!

Please also include:
1. 🕐 **Meal Time Relevance**: Based on the composition of the food, suggest the best time to consume it (e.g., breakfast, lunch, dinner, snack, avoid late night, etc.) and why.
2. 🚦 **Health Traffic Light Indicators**: For each food item, rate the sugar, salt, and saturated fat levels using the color system:
    - 🟢 Green: Healthy/low
    - 🟡 Amber: Moderate/acceptable
    - 🔴 Red: High — caution

Format your response like this:

**Detailed Breakdown:**

1.  **[Food Item Name 1]** 🍎
    * Calories: ~[Number] kcal
    * Nutrition Notes: [Brief notes, e.g., Good source of fiber, High in sugar, Estimated Protein/Carbs/Fats]
    * Meal Time Relevance: [e.g., Breakfast, Snack, etc.]
    * Health Traffic Light Indicators: [e.g., Sugar: 🟢, Salt: 🟡, Saturated Fat: 🔴]
2.  **[Food Item Name 2]** 🍕
    * Calories: ~[Number] kcal
    * Nutrition Notes: [Brief notes]
    * Meal Time Relevance: [e.g., Breakfast, Snack, etc.]
    * Health Traffic Light Indicators: [e.g., Sugar: 🟢, Salt: 🟡, Saturated Fat: 🔴]
3.  **[Food/Drink Item Name 3]** (if applicable)
    * Calories: ~[Number] kcal
    * Nutrition Notes: [Brief notes]...
    * Meal Time Relevance: [e.g., Breakfast, Snack, etc.]
    * Health Traffic Light Indicators: [e.g., Sugar: 🟢, Salt: 🟡, Saturated Fat: 🔴]


---
**Total Estimated Calories:** ~[Total Calories] kcal 📊

---
**Expert Nutritional Insights & Considerations:** 💡
[Provide a detailed nutritional summary based *specifically* on the food items identified in the image. Discuss:
* 🍽️ **Overall Meal Profile:** (e.g., Is it balanced? High in carbs/fat/protein? etc.)
* ✅ **Key Benefits:** (Mention positive nutritional aspects or uses of the main ingredients.)
* ⚠️ **Potential Considerations/Side Effects:** (e.g., Mention high sodium, sugar, saturated fat content if applicable and potential effects of overconsumption. Be factual and avoid overly strong warnings.)
* ❤️‍🩹 **Notes for Health Conditions:** (Provide general dietary considerations related to the identified foods for individuals managing conditions like high blood pressure or diabetes. For example, comment on sodium content for BP or carbohydrate/sugar content for diabetes. Suggest moderation or healthier preparation methods if relevant.)
* ℹ️ **Disclaimer:** Conclude **explicitly** with this sentence: "Note: This information is for general awareness and educational purposes only, and does not substitute professional medical or nutritional advice. Consult with a healthcare provider for personalized guidance."]

---

**🎯 Health & Weather Considerations:**
- **High Blood Pressure:** ❤️‍🩹 [If the user indicated 'Yes' for high blood pressure, analyze the meal's sodium content (low/moderate/high) and suggest specific adjustments like reducing processed items or adding potassium-rich foods found in the meal or suggested additions.]
- **High Blood Sugar:** 🩸 [If the user indicated 'Yes' for high blood sugar, analyze the meal's likely glycemic impact (low/moderate/high based on carbs/sugars) and suggest focusing on fiber, protein, or specific low-GI alternatives relevant to the meal.]
- **Weather:** ☀️🌧️❄️ [Based on the selected weather ([Summer/Rainy/Winter]), comment if the meal is suitable (e.g., hydrating/warming) and suggest minor adjustments like portion size or adding weather-appropriate sides.]

---
**✅ Personalized Suggestions:**
- Based on the Fitness Goal ([Weight Loss/Muscle Gain/etc.]): You may want to add [lean protein/greens/healthy fats] or reduce [sugar/fried components] to better support your goal.
- Based on Dietary Preference ([Keto/Low Carb/etc.]): This meal seems [compatible/partially compatible/not compatible] with a [Preference] diet because [reason]. Consider [specific adjustments like swapping rice for cauliflower rice, removing sugary sauce, etc.].
"""

# --- get_gemini_response function remains the same ---
def get_gemini_response(image_input, name="", age=None, weight=None, height=None, activity_level=None, dietary_preference=None, fitness_goal=None, tdee=None, has_bp=None, has_sugar=None, weather=None):
    """
    Sends the image and the input prompt to the Gemini API. Incorporates user context.
    (Function code is identical to the previous version provided by the user)
    """
    try:
        model = genai.GenerativeModel('gemini-1.5-flash') # Or your preferred model
        prompt_text = input_prompt
        context = ""

        # Build context string safely
        if name: context += f"The user's name is {name}. "
        if age: context += f"The user is {age} years old. "
        if weight: context += f"The user weighs {weight} kg. "
        if height: context += f"The user is {height} cm tall. "
        if activity_level: context += f"The user's activity level is {activity_level}. "
        if dietary_preference: context += f"The user's dietary preference is {dietary_preference}. "
        if fitness_goal: context += f"The user's fitness goal is {fitness_goal}. "
        if tdee: context += f"The user's estimated daily calorie needs are {tdee} kcal. "
        if has_bp: context += f"The user has high blood pressure: {has_bp}. "
        if has_sugar: context += f"The user has high blood sugar: {has_sugar}. "
        if weather: context += f"The current weather is {weather}. "

        # Prepend context to the main prompt if any context exists
        if context:
            prompt_text = f"Here is some context about the user: {context.strip()} Please use this information to tailor your response, especially the 'Health & Weather Considerations' and 'Personalized Suggestions' sections. {input_prompt}"

        # Ensure image_input is a PIL Image (Streamlit often handles this)
        if not isinstance(image_input, Image.Image):
             st.error("Invalid image input provided to Gemini.")
             return "Error: Invalid image format."

        # Generate content
        response = model.generate_content([prompt_text, image_input])

        # Check for safety ratings or blocks if necessary (optional)
        # if response.prompt_feedback.block_reason:
        #     st.warning(f"Response blocked due to: {response.prompt_feedback.block_reason}")
        #     return "Blocked response. Please try a different image or prompt."

        response_text = response.text

        # Construct a friendly greeting and intro with user details
        greeting = f"👋 Hello {name}," if name else "👋 Hello,"
        intro_details = []
        if age: intro_details.append(f"- Age: {age} years")
        if weight: intro_details.append(f"- Weight: {weight} kg")
        if height: intro_details.append(f"- Height: {height} cm")
        if activity_level: intro_details.append(f"- Activity Level: {activity_level}")
        if dietary_preference: intro_details.append(f"- Dietary Preference: {dietary_preference}")
        if fitness_goal: intro_details.append(f"- Fitness Goal: {fitness_goal}")
        if tdee: intro_details.append(f"- Estimated Daily Calorie Needs: {tdee} kcal")
        if has_bp: intro_details.append(f"- High Blood Pressure: {has_bp}")
        if has_sugar: intro_details.append(f"- High Blood Sugar: {has_sugar}")
        if weather: intro_details.append(f"- Weather: {weather}")

        intro = "\nHere is the nutritional breakdown of your meal, considering your inputs:\n"
        if intro_details:
            intro += "\n" + "\n".join(intro_details) + "\n"

        # Combine greeting, intro, and the main analysis
        # Look for the start of the detailed breakdown to ensure structure
        if "Detailed Breakdown:" in response_text:
            # Extract the part after "Detailed Breakdown:"
            breakdown_part = response_text.split("Detailed Breakdown:", 1)[1]
            # Reconstruct the full response with the greeting and intro
            full_response = f"{greeting} {intro}\n\n**Detailed Breakdown:**\n{breakdown_part.strip()}"
        else:
            # If the expected structure isn't found, return the raw response after the greeting/intro
            full_response = f"{greeting} {intro}\n\n{response_text.strip()}"

        return full_response

    except Exception as e:
        st.error(f"🚨 An error occurred while contacting the Gemini API: {e}")
        # You might want to log the full traceback here for debugging
        # import traceback
        # st.error(traceback.format_exc())
        return "Error: Could not get response from AI model."




# --- estimate_daily_calories function remains the same ---
def estimate_daily_calories(weight, height, age, activity_level):
    """Estimates Total Daily Energy Expenditure (TDEE) using Mifflin-St Jeor formula."""
    # Basic validation
    if not all([weight, height, age, activity_level]):
        return 0 # Return 0 or None if inputs are missing

    try:
        # Mifflin-St Jeor Equation for BMR (assuming male, adjust if needed or add gender input)
        # Using +5 for male, use -161 for female if gender is collected
        bmr = (10 * float(weight)) + (6.25 * float(height)) - (5 * int(age)) + 5

        # Activity multipliers
        multiplier = {"Low": 1.2, "Moderate": 1.55, "High": 1.725}
        activity_multiplier = multiplier.get(activity_level, 1.2) # Default to Low if invalid

        tdee = bmr * activity_multiplier
        return int(tdee) # Return as integer
    except (ValueError, TypeError):
        st.error("Invalid input for TDEE calculation.")
        return 0 # Return 0 or handle error as appropriate


def main():
    """
    Main function to run the Streamlit application.
    """
    # Set page configuration
    st.set_page_config(page_title="Food Calorie Estimator", page_icon="🥗", layout="centered")

    # Apply custom CSS (keep your original CSS)
    st.markdown(
        """
        <style>
            /* Your existing gradient, button, popover, and sidebar styles here */
            .stApp {
                background: linear-gradient(135deg, #e0f7fa, #c2e59d, #ffeb3b, #f9a8d4);
                background-size: 400% 400%;
                animation: gradient 15s ease infinite;
            }
            @keyframes gradient {
                0% { background-position: 0% 50%; }
                50% { background-position: 100% 50%; }
                100% { background-position: 0% 50%; }
            }
            .title { color: #2e7d32; }
            .header { color: #66bb6a; }
            .subheader{ color: #9ccc65; }
            .stButton > button {
                color: #ffffff; background-color: #81c784; border: none;
                padding: 10px 20px; text-align: center; text-decoration: none;
                display: inline-block; font-size: 16px; margin: 4px 8px;
                cursor: pointer; border-radius: 8px; transition: background-color 0.3s ease;
            }
            .stButton > button:hover { background-color: #66bb6a; }
            .stButton > button:active { background-color: #4caf50; }
            .stPopover > button { /* Style for popover button if needed */ }
            .stSidebar {
                background: linear-gradient(to bottom right, #e0f7fa, #f1f8e9, #fff3e0, #fce4ec);
                background-size: 300% 300%;
                animation: gradient 20s ease infinite;
            }
            /* Add specific style for the 'Additional Info' button if desired */
            .stButton.additional-info-button > button {
                 background-color: #4fc3f7; /* Light blue */
            }
            .stButton.additional-info-button > button:hover {
                 background-color: #29b6f6; /* Slightly darker blue */
            }
            .stButton.additional-info-button > button:active {
                 background-color: #03a9f4; /* Darker blue */
            }
             /* Add specific style for the 'Creative Advice' button */
            .stButton.creative-advice-button > button {
                 background-color: #ffb74d; /* Orange */
            }
            .stButton.creative-advice-button > button:hover {
                 background-color: #ffa726; /* Slightly darker orange */
            }
            .stButton.creative-advice-button > button:active {
                 background-color: #ff9800; /* Darker orange */
            }
        </style>
        """,
        unsafe_allow_html=True
    )

    # --- Initialize Session State ---
    if 'calorie_info' not in st.session_state:
        st.session_state.calorie_info = None
    if 'image_processed' not in st.session_state:
        st.session_state.image_processed = False # Flag to track if analysis was done
    if 'additional_info' not in st.session_state:
        st.session_state.additional_info = None # Store web search results
    if 'creative_advice' not in st.session_state:
        st.session_state.creative_advice = None

  
  # --- Sidebar Elements ---
    with st.sidebar: # Group sidebar elements
        st.header("⚙️ Health & Fitness Inputs")
        st.subheader("Personalize Your Analysis")

        # User inputs (using keys for state)
        age = st.number_input("Age (in years)", min_value=10, max_value=100, value=25, key="user_age")
        weight = st.number_input("Weight (in kg)", min_value=30.0, max_value=200.0, value=70.0, format="%.1f", key="user_weight")
        height = st.number_input("Height (in cm)", min_value=100.0, max_value=250.0, value=170.0, format="%.1f", key="user_height")
        activity_level = st.selectbox("Activity Level", ["Low", "Moderate", "High"], index=1, key="user_activity")
        dietary_preference = st.selectbox("Dietary Preference", ["Balanced", "Keto", "Vegetarian", "Low Carb", "Vegan"], key="user_diet")
        fitness_goal = st.selectbox("Fitness Goal", ["Weight Loss", "Muscle Gain", "Maintenance", "Endurance", "Flexibility"], key="user_goal")
        has_bp = st.radio("High Blood Pressure?", ["No", "Yes"], key="user_bp", horizontal=True)
        has_sugar = st.radio("High Blood Sugar/Diabetes?", ["No", "Yes"], key="user_sugar", horizontal=True)
        weather = st.selectbox("Current Weather", ["Summer", "Rainy", "Winter", "Moderate"], key="user_weather")

        # Estimate and display TDEE
        tdee = estimate_daily_calories(weight, height, age, activity_level)
        if tdee > 0:
            st.markdown(f"### 🔥 Est. Daily Needs (TDEE):")
            st.markdown(f"<p style='font-size: 24px; font-weight: bold; color: #2e7d32;'>{tdee} kcal</p>", unsafe_allow_html=True)
        else:
            st.warning("Provide age, weight, height for TDEE.")


    # --- Main Page UI Elements ---
    st.title("🥗 Food Calorie Estimator 📸")
    st.write("Upload an image of your meal, get nutritional insights, web context, and creative advice!")

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], key="file_uploader")

    # --- Processing Logic (Handles New Uploads) ---
    if uploaded_file is not None:
        # This block now *only* handles the initial processing of a new file
        try:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image.", use_container_width=True)

            # Add validation checks before allowing analysis
            is_valid = True
            validation_messages = []

            # Check name
            name = st.text_input("What's your name?", key="user_name_main")
            if not name:
                is_valid = False
                validation_messages.append("Please enter your name")

            # Check sidebar inputs
            if not st.session_state.user_age:
                is_valid = False
                validation_messages.append("Please enter your age")
            
            if not st.session_state.user_weight:
                is_valid = False
                validation_messages.append("Please enter your weight")
                
            if not st.session_state.user_height:
                is_valid = False
                validation_messages.append("Please enter your height")
                
            if not st.session_state.user_activity:
                is_valid = False
                validation_messages.append("Please select your activity level")
                
            if not st.session_state.user_diet:
                is_valid = False
                validation_messages.append("Please select your dietary preference")
                
            if not st.session_state.user_goal:
                is_valid = False
                validation_messages.append("Please select your fitness goal")

            # Display validation messages if any
            if validation_messages:
                st.warning("Please complete the following required fields:")
                for msg in validation_messages:
                    st.markdown(f"- {msg}")

            # Only show and enable the analyze button if all inputs are valid
            if is_valid:
                if st.button("Analyze Image for Nutritional Information 🍽️", key="analyze_button"):
                    with st.spinner("🔍 Analyzing the image... Please wait.",show_time=True):
                        analysis_result = get_gemini_response(
                            image, name, st.session_state.user_age,
                            st.session_state.user_weight, st.session_state.user_height,
                            st.session_state.user_activity, st.session_state.user_diet,
                            st.session_state.user_goal, tdee, st.session_state.user_bp,
                            st.session_state.user_sugar, st.session_state.user_weather
                        )
                        st.session_state.calorie_info = analysis_result
                        st.session_state.image_processed = True
                        st.session_state.additional_info = None
                        st.session_state.creative_advice = None
                        st.rerun()
            else:
                st.button("Analyze Image for Nutritional Information 🍽️", 
                         key="analyze_button_disabled", 
                         disabled=True)

        except Exception as e:
            st.error(f"🖼️ Error loading or processing image: {e}")
            # Reset state on error
            st.session_state.calorie_info = None
            st.session_state.image_processed = False
            st.session_state.additional_info = None
            st.session_state.creative_advice = None


    # --- Display Results Area (Depends *only* on Session State) ---
    # MOVED OUTSIDE: This section is no longer inside 'if uploaded_file is not None'
    if st.session_state.image_processed and st.session_state.calorie_info:
        st.divider() # Add a visual separator

        # --- Display Nutritional Analysis ---
        st.subheader("🔬 Nutritional Analysis:")
        with st.container(height=600): # Use container for scrollable results
             st.markdown(st.session_state.calorie_info) # Display stored analysis
  
        # --- Additional Info Section ---
        st.divider()
        # 🌐 Section: Get More Context (Web Search)
        st.subheader("🌐 Get More Context (Web Search)")

        if st.button("Search Web Based on Analysis", key="web_search_button"):
            with st.spinner("🤖 Searching the web...",show_time=True):
                try:
                    dietary_planner = Agent(
                        model=Gemini(id="gemini-2.5-pro-exp-03-25", api_key=api_key),
                        description=dedent("""\
                            Creates personalized dietary plans based on user input.
                            Generates customized workout routines based on fitness goals.
                            Combines diet and workout plans into a holistic health strategy.            
                            Expert nutritionist and dietary advisor specializing in personalized meal planning
                            and evidence-based nutritional recommendations."""),
                        instructions=dedent("""\
                            "Generate a diet plan with breakfast, lunch, dinner, and snacks.",
                            "Consider dietary preferences like Keto, Vegetarian, or Low Carb.",
                            "Ensure proper hydration and electrolyte balance.",
                            "Provide nutritional breakdown including macronutrients and vitamins.",
                            "Suggest meal preparation tips for easy implementation.",
                            "If necessary, search the web using DuckDuckGo for additional information.",
                            "Create a workout plan including warm-ups, main exercises, and cool-downs.",
                            "Adjust workouts based on fitness level: Beginner, Intermediate, Advanced.",
                            "Consider weight loss, muscle gain, endurance, or flexibility goals.",
                            "Provide safety tips and injury prevention advice.",
                            "Suggest progress tracking methods for motivation.",
                            "Merge personalized diet and fitness plans for a comprehensive approach, use tables if possible.",
                            "Ensure alignment between diet and exercise for optimal results.",
                            "Suggest lifestyle tips for motivation and consistency.",
                            "Provide realistic, real-time nutritional advice tailored to the user's data with engaging emojis, including suggestions for meal modifications, portion control, and healthy eating practices."
                            "Recommend specific yoga asanas based on user's fitness level and health conditions.",
                            "Include optimal timing for yoga practice (morning/evening) with duration.",
                            "List 3-4 specific yogasanas with their benefits and duration.",
                            "Suggest meditation techniques aligned with user's lifestyle and goals.",
                            """),
                        expected_output=dedent("""\
                            Prepare the output so that it captures the user data from st.session_state.calorie_info.
                            Return additional information with clear bullet points, emojis in the headings (including yoga and meditation tips), and comprehensive advice merging diet, workout recommendations, and realistic nutritional advice tailored to the user's data with emoji-enhanced suggestions.
                            Include a dedicated '🧘 Yoga & Meditation Corner' section with:
                            - Best time to practice (morning/evening with specific timing)
                            - 3-4 specific yogasanas names with their benefits
                            - Duration for each asana (in minutes)
                            - Total session duration
                            - Meditation technique with timing
                            - Breathing exercises (pranayama) if applicable
                            Provide comprehensive advice merging diet, workout recommendations, and realistic nutritional advice 
                            tailored to the user's data with emoji-enhanced suggestions.
                            """),
                        tools=[DuckDuckGoTools()],
                        show_tool_calls=False,
                        markdown=True
                    )

                    # Perform agent web search
                    web_result = dietary_planner.run(st.session_state.calorie_info)

                    # Store results
                    st.session_state.additional_info = {
                        "raw": web_result.content,
                        "structured": web_result.content
                    }

                except Exception as err:
                    st.error(f"❌ Web search failed: {err}")


        # ✅ Display Web Search Results if available
        if st.session_state.additional_info:
            with st.expander("📄 Raw Web Search Result"):
                st.markdown(st.session_state.additional_info["raw"], unsafe_allow_html=True)
            
            # Extract user name from greeting line (if available)
            name_line = next((line for line in st.session_state.calorie_info.splitlines() if "👋 Hello" in line), None)
            user_label = name_line.replace("👋 Hello", "").strip(",") if name_line else "User"

            st.markdown("### 📌 Personalized Summary Based on Your Analysis")
            structured_output = st.session_state.additional_info["structured"]
            structured_output = "\n".join(
                line for line in structured_output.splitlines()
                if "User Data (from" not in line
            )

            # Now display the cleaned version
            st.markdown(structured_output, unsafe_allow_html=True)            
    

        # Display additional static web insights if needed
        if st.session_state.additional_info and "calorie_info" in st.session_state:
            formatted_info = f"""
            ---

            ### 📝 AI-Powered Recommendations:
            - ✅ Choose low-sodium and low-sugar options where possible.
            - 🥗 Add fiber-rich veggies (spinach, broccoli, kale) for gut health and blood sugar support.
            - 💧 Stay hydrated: aim for 2-3 liters per day, especially in summer.
            - 🧘 Incorporate movement (e.g., yoga, walks, light strength training) aligned with your fitness goal.
            - 📏 Control portion sizes — especially carbs and sauces — for better calorie management.
            - 🔁 Revisit your plan every 1–2 weeks and tweak based on your progress and energy levels.

            - 🧘 **Yoga & Meditation Tips:**
            - 🧘‍♀️ Practice a short yoga flow to increase flexibility and reduce stress.
            - 🕉️ Consider meditation to improve mindfulness and balance.
            - 🌅 Try morning stretches to energize your day and improve circulation.

            ---
            """
            st.markdown(formatted_info, unsafe_allow_html=True)

        

        # Add a footer (optional)
        st.markdown("---")
        st.caption("Built with Streamlit & Google Gemini. AI estimations are approximate. Consult professionals for precise advice.")

        # Add minimal styling
        st.markdown("""
            <style>
            .stSelectbox {
                margin-top: 1rem;
            }
            </style>
            """, unsafe_allow_html=True)

# Entry point of the script
if __name__ == "__main__":
    main()
