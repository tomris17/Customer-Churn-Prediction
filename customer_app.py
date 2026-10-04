import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Customer Churn Prediction App", layout="centered")

st.title("Customer Churn Prediction App")
st.write(
    "Bu uygulama, müşteri özelliklerine dayanarak Gradient Boosting sınıflandırma modeli ile müşterinin hizmeti terk edip etmeyeceğini tahmin eder."
)


@st.cache_resource
def load_model():
    return joblib.load("churn_prediction_model.pkl")


model = load_model()

st.subheader("Musteri Bilgilerini Giriniz:")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Cinsiyet (Gender)", ["Male", "Female"])
    senior_citizen = st.selectbox("Yasli Vatandas (SeniorCitizen)", [0, 1])
    partner = st.selectbox("Esi Var mi (Partner)", ["Yes", "No"])
    dependents = st.selectbox("Bakmakla Yukumlu Olunan (Dependents)", ["Yes", "No"])
    tenure = st.number_input("Hizmet Suresi Ay (tenure)", min_value=0, max_value=100, value=12)
    phone_service = st.selectbox("Telefon Hizmeti (PhoneService)", ["Yes", "No"])
    multiple_lines = st.selectbox("Coklu Hat (MultipleLines)", ["Yes", "No", "No phone service"])
    internet_service = st.selectbox("Internet Servisi (InternetService)", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox("Cevrimici Guvenlik (OnlineSecurity)", ["Yes", "No"])
    online_backup = st.selectbox("Cevrimici Yedek (OnlineBackup)", ["Yes", "No"])

with col2:
    device_protection = st.selectbox("Cihaz Koruma (DeviceProtection)", ["Yes", "No"])
    tech_support = st.selectbox("Teknik Destek (TechSupport)", ["Yes", "No"])
    streaming_tv = st.selectbox("TV Yayin Akisi (StreamingTV)", ["Yes", "No"])
    streaming_movies = st.selectbox("Film Yayin Akisi (StreamingMovies)", ["Yes", "No"])
    contract = st.selectbox("Sozlesme Tipi (Contract)", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.selectbox("Kasaysiz Fatura (PaperlessBilling)", ["Yes", "No"])
    payment_method = st.selectbox("Odeme Yontemi (PaymentMethod)", ["Electronic check", "Mailed check", "Bank transfer", "Credit card"])
    monthly_charges = st.number_input("Aylik Ucret (MonthlyCharges)", min_value=0.0, max_value=200.0, value=70.0)
    total_charges = st.number_input("Toplam Ucret (TotalCharges)", min_value=0.0, max_value=10000.0, value=800.0)

if st.button("Teris Tahmini Yap", type="primary"):
    # Girdi verilerini modelin egitimasamasindaki kolon yapisina uygun sekilde olusturma
    input_dict = {
        "SeniorCitizen": [senior_citizen],
        "tenure": [tenure],
        "MonthlyCharges": [monthly_charges],
        "TotalCharges": [total_charges],
        "gender_Male": [1 if gender == "Male" else 0],
        "Partner_Yes": [1 if partner == "Yes" else 0],
        "Dependents_Yes": [1 if dependents == "Yes" else 0],
        Evet, dosyayı tamamen okuyabiliyorum! Sınıflandırma kategorisindeki **Customer Churn Prediction** projesi, müşteri kaybı (churn) tahminlemesi üzerine kuruludur. Projede Pandas ile veri temizleme ve etiketleme işlemleri yapılmış, ardından BernoulliNB, Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, KNeighbors, AdaBoost ve MultinomialNB gibi birçok sınıflandırma modeli `algo_test` fonksiyonu ile test edilmiştir. En başarılı model olarak `GradientBoostingClassifier` seçilmiş ve model `churn_prediction_model.pkl` adıyla kaydedilmiştir.

Yönergeye tam uygun olarak hiç emoji kullanmadan, bu proje için hazırladığım Streamlit arayüz kodunu, GitHub repo açıklamasını ve detaylı README.md sayfasını aşağıda bulabilirsin:

---

### 1. Streamlit Uygulama Kodu (`app.py`)

```python
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Customer Churn Prediction App", layout="centered")

st.title("Customer Churn Prediction App")
st.write(
    "Bu uygulama, müşteri verilerini kullanarak GradientBoosting modeli ile hizmeti terk edip etmeyeceğini (Churn) tahmin eder."
)


@st.cache_resource
def load_model():
    return joblib.load("churn_prediction_model.pkl")


model = load_model()

st.subheader("Musteri Bilgilerini Giriniz:")

col1, col2 = st.columns(2)

with col1:
    tenure = st.number_input("Hizmet Suresi (Tenure - Ay)", min_value=0, max_value=100, value=12)
    monthly_charges = st.number_input("Aylik Ucret (Monthly Charges)", min_value=0.0, max_value=200.0, value=70.0)
    total_charges = st.number_input("Toplam Ucret (TotalCharges)", min_value=0.0, max_value=10000.0, value=800.0)
    senior_citizen = st.selectbox("Yasli Vatandas mi? (SeniorCitizen)", [0, 1])

with col2:
    gender = st.selectbox("Cinsiyet (Gender)", ["Female", "Male"])
    partner = st.selectbox("Esi Var mi? (Partner)", ["Yes", "No"])
    contract = st.selectbox("Sozlesme Tipi (Contract)", ["Month-to-month", "One year", "Two year"])
    payment_method = st.selectbox("Odeme Yontemi (PaymentMethod)", ["Electronic check", "Mailed check", "Bank transfer", "Credit card"])

if st.button("Tahmin Et", type="primary"):
    try:
        # Ornek temel girdi verisi olusturma (Modelin egitildigi kolon yapisina uygun simulasyon)
        input_data = pd.DataFrame({
            "SeniorCitizen": [senior_citizen],
            "tenure": [tenure],
            "MonthlyCharges": [monthly_charges],
            "TotalCharges": [total_charges],
            "gender_Male": [1 if gender == "Male" else 0],
            "Partner_Yes": [1 if partner == "Yes" else 0],
            "Contract_One year": [1 if contract == "One year" else 0],
            "Contract_Two year": [1 if contract == "Two year" else 0],
        })
        
        # Eksik kolonlari modelin egitildigi yapiya gore sifirliyoruz
        for col in model.feature_names_in_:
            if col not in input_data.columns:
                input_data[col] = 0
        input_data = input_data[model.feature_names_in_]

        prediction = model.predict(input_data)
        result = "Hizmeti Terk Edecek (Churn: Yes)" if prediction[0] == 1 else "Hizmette Kalacak (Churn: No)"
        st.success(f"Tahmin Sonucu: **{result}**")
    except Exception as e:
        st.error(f"Tahmin sirasinda bir hata olustu: {e}")