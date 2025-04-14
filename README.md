# Food Calorie Estimator with AI 🥗

A Streamlit-based web application that uses Google's Gemini AI to analyze food images and provide detailed nutritional information, calorie estimates, and health insights.

## 🌟 Features

- **Image Analysis**: Upload food images for AI-powered nutritional analysis
- **Detailed Breakdown**: Get comprehensive information about each food item including:
  - Calorie estimates
  - Nutritional content (protein, carbs, fats, vitamins)
  - Health traffic light indicators
  - Meal time recommendations
- **Personalized Insights**: Customized analysis based on:
  - User's health conditions (blood pressure, blood sugar)
  - Dietary preferences
  - Fitness goals
  - Weather conditions
- **Web Search Integration**: Get additional context about foods using DuckDuckGo search


## 🚀 Getting Started

### Prerequisites

```bash
- Python 3.8+
- pip (Python package manager)
```

### Installation

1. Clone the repository:
```bash
git clone [your-repository-url]
cd IMAGEVISION
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
Create a `.env` file in the project root with:
```
GEMINI_API_KEY=your_api_key_here
```

### Running the Application

```bash
streamlit run testapp.py
```

## 📝 Usage

1. Launch the application using the command above
2. Upload a food image using the file uploader
3. Fill in optional personal information:
   - Name
   - Age
   - Weight
   - Height
   - Activity Level
   - Dietary Preferences
   - Fitness Goals
4. View the detailed nutritional analysis
5. Optionally search for additional web context

## 🛠️ Technical Components

- **Frontend**: Streamlit
- **AI Model**: Google Gemini
- **Image Processing**: PIL (Python Imaging Library)
- **Web Search**: DuckDuckGo API


## 📋 Dependencies

Main dependencies include:
- streamlit
- google.generativeai
- Pillow
- python-dotenv
- agno


## ⚠️ Important Notes

- Ensure you have a valid Google Gemini API key
- The application requires internet connectivity
- Nutritional analysis is for educational purposes only

## 🔒 Privacy

- No images or personal data are stored permanently
- All processing is done in real-time
- API keys should be kept secure and never shared



## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

## 👥 Authors

NarendraKumar

## Project Link
https://bit.ly/FoodCaloriesEstimator

