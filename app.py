from flask import Flask, render_template, request, redirect, send_from_directory, url_for, jsonify
import numpy as np
import json
import uuid
import tensorflow as tf
from googletrans import Translator
import requests
import os
import openai
import gdown

app = Flask(__name__)
MODEL_PATH = "models/plant_disease_recog_model_pwp.keras"
MODEL_ID = "1Ond7UzrNOfdAXWedjlZr2sDXYU6MRBuj"  # <-- Replace with your actual Google Drive file ID
# Auto-download the model if it doesn't exist
if not os.path.exists(MODEL_PATH):
    print("Model not found locally. Downloading from Google Drive...")
    os.makedirs("models", exist_ok=True)
    gdown.download(f"https://drive.google.com/uc?id={MODEL_ID}", MODEL_PATH, quiet=False)
    print("Model downloaded successfully.")
model = tf.keras.models.load_model(MODEL_PATH)
label = ['Apple___Apple_scab',
 'Apple___Black_rot',
 'Apple___Cedar_apple_rust',
 'Apple___healthy',
 'Background_without_leaves',
 'Blueberry___healthy',
 'Cherry___Powdery_mildew',
 'Cherry___healthy',
 'Corn___Cercospora_leaf_spot Gray_leaf_spot',
 'Corn___Common_rust',
 'Corn___Northern_Leaf_Blight',
 'Corn___healthy',
 'Grape___Black_rot',
 'Grape___Esca_(Black_Measles)',
 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
 'Grape___healthy',
 'Orange___Haunglongbing_(Citrus_greening)',
 'Peach___Bacterial_spot',
 'Peach___healthy',
 'Pepper,_bell___Bacterial_spot',
 'Pepper,_bell___healthy',
 'Potato___Early_blight',
 'Potato___Late_blight',
 'Potato___healthy',
 'Raspberry___healthy',
 'Soybean___healthy',
 'Squash___Powdery_mildew',
 'Strawberry___Leaf_scorch',
 'Strawberry___healthy',
 'Tomato___Bacterial_spot',
 'Tomato___Early_blight',
 'Tomato___Late_blight',
 'Tomato___Leaf_Mold',
 'Tomato___Septoria_leaf_spot',
 'Tomato___Spider_mites Two-spotted_spider_mite',
 'Tomato___Target_Spot',
 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
 'Tomato___Tomato_mosaic_virus',
 'Tomato___healthy']
translator = Translator()

# Load disease data
with open("plant_disease.json", 'r', encoding='utf-8') as file:
    plant_disease = json.load(file)

# Weather API Config
WEATHER_API_KEY = "51cb54ef1621a9440dd6db30789954f6"
WEATHER_API_URL = "https://api.openweathermap.org/data/2.5/weather"

# OpenAI Config - Replace with your actual key

openai.api_key = os.getenv("OPENAI_API_KEY")
# openai.api_key = OPENAI_API_KEY
try:
    test_response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=5
    )
    print("API is working! Response:", test_response)
except Exception as e:
    print("API Error:", str(e))

# Supported languages
LANGUAGES = {
    'en': 'English',
    'bn': 'বাংলা'
}

@app.route('/uploadimages/<path:filename>')
def uploaded_images(filename):
    return send_from_directory('./uploadimages', filename)

@app.route('/', methods=['GET'])
def home():
    lang = request.args.get('lang', 'en')
    return render_template('home.html', lang=lang, languages=LANGUAGES)

def extract_features(image):
    image = tf.keras.utils.load_img(image, target_size=(160, 160))
    feature = tf.keras.utils.img_to_array(image)
    feature = np.array([feature])
    return feature

def model_predict(image):
    img = extract_features(image)
    prediction = model.predict(img)[0]
    predicted_index = np.argmax(prediction)
    predicted_label = plant_disease[predicted_index]
    
    prediction_info = {
        'name': predicted_label['name'],
        'cause': predicted_label['cause'],
        'cure': predicted_label['cure'],
        'confidence': round(float(prediction[predicted_index]) * 100, 2)
    }
    return prediction_info

@app.route('/upload/', methods=['POST', 'GET'])
def uploadimage():
    if request.method == "POST":
        image = request.files['img']
        lang = request.form.get('lang', 'en')
        
        temp_name = f"uploadimages/temp_{uuid.uuid4().hex}"
        image_path = f'{temp_name}_{image.filename}'
        image.save(image_path)
        
        prediction_info = model_predict(image_path)
        
        # Translate if needed
        if lang == 'bn':
            try:
                prediction_info['name'] = translator.translate(prediction_info['name'], src='en', dest='bn').text
                prediction_info['cause'] = translator.translate(prediction_info['cause'], src='en', dest='bn').text
                prediction_info['cure'] = translator.translate(prediction_info['cure'], src='en', dest='bn').text
            except:
                pass  # Fallback to English if translation fails
        
        return render_template(
            'home.html',
            result=True,
            imagepath=f'/{image_path}',
            prediction=prediction_info,
            confidence=prediction_info['confidence'],
            lang=lang,
            languages=LANGUAGES
        )
    else:
        return redirect('/')

@app.route('/get_weather', methods=['GET'])
def get_weather():
    lang = request.args.get('lang', 'en')
    city = request.args.get('city', 'Dhaka')
    
    try:
        params = {
            'q': city,
            'appid': WEATHER_API_KEY,
            'units': 'metric'
        }
        response = requests.get(WEATHER_API_URL, params=params)
        data = response.json()
        
        weather_info = {
            'temp': data['main']['temp'],
            'humidity': data['main']['humidity'],
            'description': data['weather'][0]['description'],
            'icon': data['weather'][0]['icon']
        }
        
        # Generate farmer advice based on weather
        advice = generate_farmer_advice(data, lang)
        
        if lang == 'bn':
            try:
                weather_info['description'] = translator.translate(weather_info['description'], src='en', dest='bn').text
                advice = translator.translate(advice, src='en', dest='bn').text
            except:
                pass
        
        return jsonify({
            'success': True,
            'weather': weather_info,
            'advice': advice
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

def generate_farmer_advice(weather_data, lang='en'):
    temp = weather_data['main']['temp']
    humidity = weather_data['main']['humidity']
    weather_main = weather_data['weather'][0]['main']

    advice = []

    # English advice
    if lang == 'en':
        if temp > 30:
            advice.append("High temperatures detected. Consider watering plants in early morning or late afternoon.")
        elif temp < 15:
            advice.append("Low temperatures detected. Protect sensitive plants with covers or move them indoors if possible.")

        if humidity > 80:
            advice.append("High humidity detected. Watch for fungal diseases and ensure proper ventilation.")
        elif humidity < 40:
            advice.append("Low humidity detected. Increase watering frequency and consider mulching to retain soil moisture.")

        if weather_main == 'Rain':
            advice.append("Rain expected. Avoid watering and postpone fertilizer application.")
        elif weather_main == 'Clear':
            advice.append("Sunny weather expected. Ideal for most crops but monitor for drought stress.")

        if not advice:
            advice.append("Weather conditions are favorable for most crops. Monitor plants regularly.")

    # Bangla advice
    elif lang == 'bn':
        if temp > 30:
            advice.append("উচ্চ তাপমাত্রা সনাক্ত হয়েছে। গাছপালা সকাল বা সন্ধ্যায় পানি দিন।")
        elif temp < 15:
            advice.append("নিম্ন তাপমাত্রা সনাক্ত হয়েছে। সংবেদনশীল গাছপালা ঢেকে রাখুন বা ভিতরে নিয়ে যান।")

        if humidity > 80:
            advice.append("উচ্চ আর্দ্রতা সনাক্ত হয়েছে। ছত্রাক রোগের জন্য সতর্ক থাকুন এবং পর্যাপ্ত বায়ুচলাচল নিশ্চিত করুন।")
        elif humidity < 40:
            advice.append("নিম্ন আর্দ্রতা সনাক্ত হয়েছে। পানি দেওয়ার পরিমাণ বাড়ান এবং মাটি আর্দ্র রাখতে মালচ ব্যবহার করুন।")

        if weather_main == 'Rain':
            advice.append("বৃষ্টি হওয়ার সম্ভাবনা রয়েছে। পানি দেওয়া এবং সার প্রয়োগ এড়িয়ে চলুন।")
        elif weather_main == 'Clear':
            advice.append("রোদেলা আবহাওয়া প্রত্যাশিত। বেশিরভাগ ফসলের জন্য উপযুক্ত তবে খরার লক্ষণ পর্যবেক্ষণ করুন।")

        if not advice:
            advice.append("আবহাওয়া বেশিরভাগ ফসলের জন্য অনুকূল। নিয়মিত গাছ পর্যবেক্ষণ করুন।")

    return " ".join(advice)


@app.route('/chatbot', methods=['POST'])
def chatbot():
    try:
        user_message = request.json.get('message', '').strip()
        lang = request.json.get('lang', 'en')
        
        if not user_message:
            return jsonify({'success': False, 'error': 'Empty message'})
        
        # Prepare context for GPT-3.5
        context = (
            "You are AgroAid, a helpful agricultural assistant for Bangladeshi farmers. "
            "You specialize in plant diseases, farming techniques, and weather impacts. "
            "Keep responses concise (1-2 paragraphs) and practical. "
            "Respond in the same language as the user's question."
        )
        
        # Call GPT-3.5 API
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": context},
                {"role": "user", "content": user_message}
            ],
            temperature=0.7,
            max_tokens=300
        )
        
        # Get the response
        bot_response = response['choices'][0]['message']['content']
        
        return jsonify({
            'success': True, 
            'response': bot_response
        })
        
    except Exception as e:
        print(f"Chatbot error: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e),
            'response': "I'm having trouble responding right now. Please try again later."
        })

if __name__ == "__main__":
    os.makedirs('uploadimages', exist_ok=True)
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))