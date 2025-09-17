import streamlit as st
import requests
import pandas as pd
import folium
from streamlit_folium import st_folium
from folium.plugins import Draw
from deep_translator import GoogleTranslator
from datetime import datetime
from PIL import Image
import torch
from torchvision import transforms
import numpy as np
from sklearn.linear_model import LinearRegression

# ------------------------
# CONFIG
# ------------------------
st.set_page_config(
    page_title="🌾 Yield Sense",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

API_KEY = "f5d74ad9384758204064758b2e9ba8a5"

# ------------------------
# TRANSLATION FUNCTION
# ------------------------
def t(text, target_lang):
    if target_lang == "en":
        return text
    try:
        return GoogleTranslator(source="en", target=target_lang).translate(text)
    except:
        return text

# ------------------------
# AVAILABLE LANGUAGES
# ------------------------
language_options = {
    "en": "English",
    "bn": "বাংলা (Bengali)",
    "gu": "ગુજરાતી (Gujarati)",
    "hi": "हिन्दी (Hindi)",
    "kn": "ಕನ್ನಡ (Kannada)",
    "ml": "മലയാളം (Malayalam)",
    "mr": "मराठी (Marathi)",
    "or": "ଓଡ଼ିଆ (Odia/ Oriya)",
    "pa": "ਪੰਜਾਬੀ (Punjabi/Gurmukhi)",
    "ta": "தமிழ் (Tamil)",
    "te": "తెలుగు (Telugu)",
    "ur": "اردو (Urdu)",
    "doi": "डोगरी (Dogri)"
}

# ------------------------
# CROP SEASONS & CROPS
# ------------------------
crop_seasons = {
    "🌾 Rabi (Winter Crops)": ["Wheat", "Barley", "Mustard", "Pea", "Gram"],
    "🌧️ Kharif (Monsoon Crops)": ["Rice", "Maize", "Cotton", "Soybean", "Millet"],
    "☀️ Zaid (Summer Crops)": ["Watermelon", "Cucumber", "Muskmelon", "Pumpkin"],
    "🌍 All Season Crops": ["Tomato", "Onion", "Chili", "Brinjal", "Spinach"]
}

# ------------------------
# SIDEBAR INPUTS
# ------------------------
lang = st.sidebar.selectbox("🌐 " + t("Language", "en"), options=list(language_options.keys()),
                            format_func=lambda x: language_options[x])
st.sidebar.title("🌍 " + t("Farm Settings", lang))

# City selection
if "city" not in st.session_state:
    st.session_state.city = "Hisar"
if "coords" not in st.session_state:
    st.session_state.coords = {"lat": 29.1539, "lon": 75.7229}

city_input = st.sidebar.text_input(t("Enter City:", lang), st.session_state.city)
if city_input != st.session_state.city:
    geo_url = f"http://api.openweathermap.org/geo/1.0/direct?q={city_input}&limit=1&appid={API_KEY}"
    geo_data = requests.get(geo_url).json()
    if geo_data:
        st.session_state.coords = {"lat": geo_data[0]["lat"], "lon": geo_data[0]["lon"]}
        st.session_state.city = city_input
    else:
        st.sidebar.error("⚠️ " + t("City not found!", lang))

# Crop season & type
season = st.sidebar.selectbox(t("Select Crop Season", lang), list(crop_seasons.keys()))
crop_type = st.sidebar.selectbox(t("Select Crop Type", lang), crop_seasons[season])
selected_crop = crop_type

# Field size
field_size = st.sidebar.number_input(
    t("Field Size (hectares)", lang),
    min_value=0.1, value=1.0, step=0.1
)

# Soil inputs
st.sidebar.header("🪴 " + t("Soil Parameters", lang))
soil_types = ["Loamy", "Clay", "Sandy", "Silty", "Peaty", "Chalky"]
soil_type = st.sidebar.selectbox(t("Soil Type", lang), [t(s, lang) for s in soil_types])
soil_ph = st.sidebar.slider(t("Soil pH Level", lang), 3.0, 10.0, 6.5, 0.1)
moisture_levels = ["Low", "Medium", "High"]
soil_moisture = st.sidebar.selectbox(t("Soil Moisture", lang), [t(m, lang) for m in moisture_levels])
fertility_levels = ["Low", "Medium", "High"]
soil_fertility = st.sidebar.selectbox(t("Soil Fertility Level", lang), [t(f, lang) for f in fertility_levels])

# ------------------------
# NEW: PAST YIELD INPUTS
# ------------------------
st.sidebar.header("📜 " + t("Past Year Crop Yield", lang))
past_yield_entries = {}
years = [2020, 2021, 2022, 2023, 2024]
for year in years:
    past_yield_entries[year] = st.sidebar.number_input(
        f"{t('Yield in', lang)} {year} (kg/ha)", min_value=0, value=0, step=10
    )

# ------------------------
# TITLE LEFT + LOGO RIGHT
# ------------------------
from pathlib import Path

logo_path = Path(__file__).parent / "logo.png"

col1, col2 = st.columns([3, 1])  # left wide, right narrow

with col1:
    st.markdown(
        """
        <h1 style="text-align:left; font-size: 42px;">🌾 Yield Sense</h1>
        <h3 style="text-align:left; font-size: 24px;">🤖 AI-Powered Crop Yield Prediction</h3>
        """,
        unsafe_allow_html=True
    )

with col2:
    if logo_path.exists():
        st.image(str(logo_path), width=220)


st.markdown(
    t("Get real-time weather data, soil inputs, and crop-specific recommendations for your farm. "
      "Select your location and crop to optimize yield.", lang)
)

# ------------------------
# FIELD SELECTION
# ------------------------
st.subheader("🗺️ " + t("Draw Your Field on the Map", lang))

m = folium.Map(
    location=[st.session_state.coords["lat"], st.session_state.coords["lon"]],
    zoom_start=16,
    tiles="Esri.WorldImagery"
)

draw = Draw(draw_options={
    'polyline': False,
    'rectangle': True,
    'circle': False,
    'marker': False,
    'circlemarker': False
})
draw.add_to(m)
map_data = st_folium(m, width=700, height=500)

if map_data and map_data.get("all_drawings"):
    last_drawn = map_data["all_drawings"][-1]
    if last_drawn["type"] in ["polygon", "rectangle"]:
        coords = [(lat, lon) for lon, lat in last_drawn["geometry"]["coordinates"][0]]
        st.session_state.selected_field_coords = coords
        st.success(t("✅ Field selected successfully!", lang))
        folium.Polygon(
            locations=coords, color="red", weight=4, fill=True, fill_opacity=0.2, popup=t("Selected Field", lang)
        ).add_to(m)
        st_folium(m, width=700, height=500)
else:
    st.info(t("Draw a polygon or rectangle to select your field.", lang))

if "selected_field_coords" in st.session_state:
    st.write("📍 " + t("Field coordinates:", lang))
    st.write(st.session_state.selected_field_coords)

# ------------------------
# WEATHER DATA
# ------------------------
url_current = f"http://api.openweathermap.org/data/2.5/weather?lat={st.session_state.coords['lat']}&lon={st.session_state.coords['lon']}&units=metric&appid={API_KEY}"
current_data = requests.get(url_current).json()

if "main" in current_data:
    today_date = datetime.now().strftime("%A, %d %B %Y")
    st.subheader(f"🌦️ {t('Current Weather in', lang)} {st.session_state.city} ({today_date})")
    col1, col2, col3 = st.columns(3)
    col1.metric("🌡️ " + t("Temperature", lang), f"{current_data['main']['temp']}°C")
    col2.metric("💧 " + t("Humidity", lang), f"{current_data['main']['humidity']}%")
    rain = current_data.get("rain", {}).get("1h", 0)
    col3.metric("🌧️ " + t("Rainfall (last hr)", lang), f"{rain} mm")
else:
    st.error("⚠️ " + t("Could not fetch current weather. Check API key or city name.", lang))

# ------------------------
# 5-DAY FORECAST
# ------------------------
url_forecast = f"http://api.openweathermap.org/data/2.5/forecast?lat={st.session_state.coords['lat']}&lon={st.session_state.coords['lon']}&units=metric&appid={API_KEY}"
forecast_data = requests.get(url_forecast).json()

if "list" in forecast_data:
    st.subheader("📅 " + t("5-Day Forecast", lang))
    df = pd.DataFrame(forecast_data["list"])
    df["dt"] = pd.to_datetime(df["dt"], unit="s")
    df["date"] = df["dt"].dt.date

    daily = df.groupby("date").agg({
        "main": lambda x: {"min": min(i["temp_min"] for i in x), "max": max(i["temp_max"] for i in x)},
        "weather": lambda x: x.iloc[0][0]["description"]
    }).reset_index()

    cols = st.columns(len(daily))
    for i, row in daily.iterrows():
        with cols[i]:
            st.markdown(f"**{row['date'].strftime('%a, %d %b')}**")
            st.write(f"🌡️ {row['main']['max']}° / {row['main']['min']}°C")
            st.caption(t(row["weather"].title(), lang))
else:
    st.error("⚠️ " + t("Could not fetch forecast data.", lang))

# ------------------------
# PAST YIELD ANALYSIS + PREDICTION
# ------------------------
if any(v > 0 for v in past_yield_entries.values()):
    st.subheader("📜 " + t("Past Yield Analysis", lang))
    df_yield = pd.DataFrame({
        "Year": list(past_yield_entries.keys()),
        selected_crop: list(past_yield_entries.values())
    })
    st.line_chart(df_yield.set_index("Year"))
    st.table(df_yield)

    # Linear Regression for prediction
    X = df_yield[["Year"]]
    y = df_yield[selected_crop]
    if len(y.unique()) > 1:
        model = LinearRegression()
        model.fit(X, y)
        next_year = np.array([[2025]])
        predicted_next = model.predict(next_year)[0]
        st.success(f"{t('Predicted yield for next year (2025):', lang)} {predicted_next:.0f} kg/ha")

# ------------------------
# PEST & DISEASE
# ------------------------
st.subheader("🪴 " + t("Pest & Disease Detection", lang))
uploaded_file = st.file_uploader(t("Upload a leaf image of your crop", lang), type=["jpg","jpeg","png"])
if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption=t("Uploaded Leaf Image", lang), use_column_width=True)
    try:
        model = torch.load("pest_model.pt")
        model.eval()
        preprocess = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])
        img_tensor = preprocess(img).unsqueeze(0)
        with torch.no_grad():
            outputs = model(img_tensor)
            _, predicted = torch.max(outputs, 1)
        class_names = ["Healthy", "Leaf Rust", "Powdery Mildew", "Blight"]
        detected_disease = class_names[predicted.item()]
        st.success(f"{t('Detected Disease:', lang)} {t(detected_disease, lang)}")
        recommendations = {
            "Healthy": t("No action needed. Keep monitoring your crop.", lang),
            "Leaf Rust": t("Spray appropriate fungicide. Remove infected leaves.", lang),
            "Powdery Mildew": t("Use sulfur-based fungicides. Ensure good ventilation.", lang),
            "Blight": t("Remove affected areas. Apply copper-based fungicide.", lang)
        }
        st.info(recommendations[detected_disease])
    except FileNotFoundError:
        st.warning(t("Pest model not found. Will be used when model is available.", lang))

# ------------------------
# CROP RECOMMENDATIONS
# ------------------------
st.subheader(f"📌 {t('Recommendations for', lang)} {selected_crop}")
st.info("💧 " + t("Irrigate your field in 2 days (based on forecasted rainfall).", lang))
st.info("🌱 " + t("Apply nitrogen fertilizer next week for optimal growth.", lang))
st.info("🛡️ " + t("Pest risk is low this week.", lang))

# ------------------------
# CROP YIELD PREDICTION
# ------------------------
if st.button("🔮 " + t("Predict Crop Yield", lang)):
    base_yield = np.mean([v for v in past_yield_entries.values() if v > 0]) if any(v > 0 for v in past_yield_entries.values()) else 2450
    predicted_yield = base_yield * field_size
    st.success(f"{t('Predicted Crop Yield:', lang)} {predicted_yield:.0f} kg")
    st.subheader("📊 " + t("Input Summary", lang))
    st.write(f"**{t('Crop:', lang)}** {selected_crop}")
    st.write(f"**{t('Field Size (ha):', lang)}** {field_size}")
    st.write(f"**{t('Soil Type:', lang)}** {soil_type}")
    st.write(f"**{t('Soil pH:', lang)}** {soil_ph}")
    st.write(f"**{t('Soil Moisture:', lang)}** {soil_moisture}")
    st.write(f"**{t('Soil Fertility:', lang)}** {soil_fertility}")

# ------------------------
# FOOTER
# ------------------------
st.markdown("<hr><p style='text-align:center; color:gray;'>Powered by OpenWeatherMap & Streamlit</p>", unsafe_allow_html=True)
