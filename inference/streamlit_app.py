# =============================================================================
# 1. IMPOR LIBRARY
# =============================================================================
import streamlit as st
import pandas as pd
import numpy as np
import re
import string
from nltk.corpus import stopwords
import nltk
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from collections import Counter
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import pickle
from openai import OpenAI
from PIL import Image

# =============================================================================
# 2. KONFIGURASI TERPUSAT
# =============================================================================
CONFIG = {
    "model_path": "model.h5",
    "tokenizer_path": "tokenizer.pickle",
    "nnya_exceptions_path": "nnya_exceptions.pkl",
    "logo_path": "logo1.png",
    "placeholder_image_path": "illustration.svg",
    "max_length": 120,
    "padding_type": 'post',
    "trunc_type": 'post',
    "llm_model": "llama3.2:latest",
    "llm_base_url": "http://localhost:11434"
}

LABEL_DESCRIPTIONS = {
    0: "Tidak Ada Cyberbullying", 1: "Tingkat Keparahan Rendah",
    2: "Tingkat Keparahan Sedang", 3: "Tingkat Keparahan Tinggi"
}
LABEL_UI_DETAILS = {
    0: {"icon": "✅", "color": "green",
        "header_style": "background-color: #e8f5e9; border-left: 5px solid #4caf50; padding: 10px; border-radius: 5px;"},
    1: {"icon": "ℹ️", "color": "blue",
        "header_style": "background-color: #e3f2fd; border-left: 5px solid #2196f3; padding: 10px; border-radius: 5px;"},
    2: {"icon": "⚠️", "color": "orange",
        "header_style": "background-color: #fff3e0; border-left: 5px solid #ff9800; padding: 10px; border-radius: 5px;"},
    3: {"icon": "🚨", "color": "red",
        "header_style": "background-color: #ffebee; border-left: 5px solid #f44336; padding: 10px; border-radius: 5px;"}
}


# =============================================================================
# 3. FUNGSI PEMUATAN RESOURCE (DICACHE)
# =============================================================================

@st.cache_resource
def load_all_resources():
    """
    Memuat semua resource yang mahal (model, tokenizer, stemmer, dll.) sekali saja.
    Mengembalikan dictionary berisi semua resource yang telah dimuat.
    """
    print("Memulai inisialisasi SEMUA resource (cached)...")

    # Inisialisasi Klien LLM
    try:
        llm_client = OpenAI(base_url=CONFIG["llm_base_url"], api_key='ollama')
        llm_available = True
        print("Klien LLM berhasil diinisialisasi.")
    except Exception as e:
        llm_client = None
        llm_available = False
        print(f"Fitur LLM tidak aktif: {e}")

    # Pemuatan Model dan Tokenizer
    model, tokenizer, model_success, tokenizer_success = None, None, False, False
    try:
        model = load_model(CONFIG["model_path"])
        model_success = True
        print(f"Model '{CONFIG['model_path']}' berhasil dimuat.")
    except Exception as e:
        print(f"GAGAL memuat model: {e}")
    try:
        with open(CONFIG["tokenizer_path"], 'rb') as f:
            tokenizer = pickle.load(f)
        tokenizer_success = True
        print(f"Tokenizer '{CONFIG['tokenizer_path']}' berhasil dimuat.")
    except Exception as e:
        print(f"GAGAL memuat tokenizer: {e}")

    # Setup Preprocessing Resources
    try:
        stopwords.words('indonesian')
    except LookupError:
        nltk.download('stopwords')

    factory = StemmerFactory()
    stemmer = factory.create_stemmer()

    try:
        with open(CONFIG["nnya_exceptions_path"], 'rb') as f:
            nnya_exceptions = pickle.load(f)
        nnya_setup_success = True
    except FileNotFoundError:
        print(
            f"File '{CONFIG['nnya_exceptions_path']}' tidak ditemukan. 'kata_baku_berulang_plus_nnya' mungkin tidak lengkap.")
        nnya_exceptions = set()  # Fallback
        nnya_setup_success = False

    kata_baku_berulang_final = {
        "allah", "nggak", "saat", "tinggal", "ngga", "alloh", "bukannya", "maaf", "uud", "tinggi", "omongannya",
        "nunggu", "tunggu",
        "sesungguhnya", "hingga", "ucapannya", "dajjal", "astaghfirullah", "sehingga", "menjelekkan", "meninggal",
        "sll", "menunjukkan", "panggung", "kerjaan",
        "kenyataan", "sungguh", "bangga", "panggil", "muhammadiyah", "ttp", "nggk", "kekuasaan", "menggonggong", "sllu",
        "melanggar", "cangkemmu", "kanggo", "menunggu", "dipanggil", "pertanggung", "menggulingkan", "pikirannya",
        "perkataan", "menganggap", "suul", "keadaan", "saatnya", "muhammad", "engga", "anggota", "kelakukannya",
        "bloon", "dianggap", "kerjaannya", "manfaatnya", "dll", "diindonesia", "jelekkan", "tanggung", "alhamdulillah",
    }
    kata_baku_plus_nnya = kata_baku_berulang_final.copy()
    kata_baku_plus_nnya.update(nnya_exceptions)

    norm_dict = {"amin": "", "yg": "yang", "rais": "", "mbah": "kakek", "sengkuni": "licik", "gak": "tidak",
                 "gk": "tidak", "amien": "", "tobat": "taubat", "sdh": "sudah", "ga": "tidak", "quot": "kutipan",
                 "org": "orang", "tdk": "tidak", "mu": "kamu", "wes": "sudah", "wong": "orang", "tak": "tidak",
                 "mpr": "", "gusdur": "", "allah": "", "lah": "", "tau": "tahu", "dah": "sudah", "bpk": "bapak",
                 "lu": "kamu", "opo": "apa", "jd": "jadi", "aki": "kakek", "tengil": "menyebalkan", "lo": "kamu",
                 "tp": "tapi", "wis": "sudah", "klo": "kalau", "to": "", "tuwek": "tua", "yo": "iya", "d": "",
                 "plongo": "bingung", "kalo": "kalau", "ora": "tidak", "g": "tidak", "iki": "ini", "gus": "", "dur": "",
                 "mbok": "ibu", "pk": "bapak", "ra": "tidak", "pa": "bapak", "plonga": "bingung", "nggak": "tidak",
                 "bener": "benar", "ki": "ini", "jgn": "jangan", "udh": "sudah", "ae": "aja", "ko": "kok", "dr": "dari",
                 "pikun": "lupa", "p": "", "ni": "ini", "km": "kamu", "mbh": "kakek", "sampean": "kamu", "is": "",
                 "ngaca": "kaca", "asu": "anjing", "dgn": "dengan", "sih": "", "men": "", "sing": "yang", "wae": "saja",
                 "jdi": "jadi", "tuek": "tua", "pinter": "pintar", "rakus": "serakah", "amp": "", "alloh": "",
                 "dg": "dengan", "gitu": "begitu", "kek": "seperti", "inilah": "ini lah", "se": "", "kowe": "kamu",
                 "bin": "", "dirimu": "diri kamu", "inget": "ingat", "pret": "bohong", "istighfar": "",
                 "gini": "begini", "modar": "meninggal", "prabowo": "", "sepuh": "tua", "e": "", "banget": "sangat",
                 "islam": "", "waras": "sehat", "koyo": "seperti", "tuo": "tua", "lg": "lagi", "mulutmu": "mulut kamu",
                 "krn": "karena", "dn": "dan", "jg": "juga", "nih": "ini", "cangkem": "mulut", "tu": "itu",
                 "karna": "karena", "iku": "itu", "uda": "sudah", "prof": "profesor", "dadi": "jadi",
                 "glandangan": "gelandangan", "eling": "ingat", "kmu": "kamu", "edan": "gila", "cangkeme": "mulut",
                 "sy": "saya", "n": "", "istigfar": "", "cangkemu": "mulut", "utk": "untuk", "koe": "kamu",
                 "blm": "belum", "klu": "kalau", "seng": "yang", "joko": "", "ngga": "tidak", "nyinyir": "ngomong",
                 "msh": "masih", "liat": "lihat", "sm": "sama", "odgj": "gila", "mulyono": "", "jokowi": "",
                 "alhamdulillah": ""}

    kata_penting = {"kamu", "dia", "aku", "ini", "itu", "sangat", "sekali", "sih", "banget"}
    custom_sw_list = set(stopwords.words('indonesian')) - kata_penting

    protected_words = {"bodoh", "goblok", "tolol", "jelek", "buruk", "busuk", "kotor", "kebodohan", "ketolohan",
                       "kegoblokan", "kejelekan", "keburukan", "kebusukan", "pembodohan", "penjelekan", "penghinaan",
                       "menyebalkan", "menjijikkan", "memalukan", "mengecewakan", "mengganggu", "menyakitkan",
                       "menghina", "merendahkan", "memfitnah", "mencemooh", "membenci", "memarahi", "menghujat",
                       "mengolok", "menyerang", "dibenci", "dihina", "dimarahi", "dicemooh", "difitnah", "terburuk",
                       "terbodoh", "tergoblok", "terjelek", "terjijik", "terkutuk", "terjahat", "paling", "gelandangan",
                       "pengemis", "sampah", "bangkai", "comberan", "kotoran"}

    print("SEMUA resource selesai diinisialisasi.")

    return {
        "model": model, "tokenizer": tokenizer, "stemmer": stemmer, "llm_client": llm_client,
        "norm": norm_dict, "custom_stopwords": custom_sw_list,
        "protected_words": protected_words, "kata_baku_plus_nnya": kata_baku_plus_nnya,
        "status": {
            "model_ok": model_success, "tokenizer_ok": tokenizer_success, "llm_ok": llm_available,
            "nnya_ok": nnya_setup_success
        }
    }


# =============================================================================
# 4. FUNGSI LOGIKA INTI (PREPROCESSING, PREDIKSI, LLM)
# =============================================================================

def cleaningText(text, exceptions_list):
    text_lower = str(text).lower()
    text_lower = re.sub(r'<br\s*/?>', ' ', text_lower)
    text_lower = re.sub(r'http\S+|www\S+|<a.*?>|</a>', '', text_lower)
    text_lower = re.sub(r'@\w+|#\w+', '', text_lower)
    text_lower = re.sub(r'[^\x00-\x7F]+', ' ', text_lower)
    text_lower = re.sub(r'\d+', '', text_lower)
    text_lower = re.sub(f"[{re.escape(string.punctuation)}]", ' ', text_lower)
    text_lower = ' '.join(text_lower.split())
    tokens = text_lower.split()
    normalized_tokens = []
    for token in tokens:
        if token in exceptions_list:
            normalized_tokens.append(token)
        else:
            token = re.sub(r'(.)\1+', r'\1', token)
            normalized_tokens.append(token)
    return ' '.join(normalized_tokens)


def normalisasi(text, norm_dict):
    text_normalized = str(text)
    for word, replacement in norm_dict.items():
        text_normalized = re.sub(r'\b' + re.escape(word) + r'\b', replacement, text_normalized)
    text_normalized = ' '.join(text_normalized.split())
    return text_normalized


def remove_stopwords_cyberbullying(text, custom_sw_list):
    words = str(text).split()
    filtered_words = [word for word in words if word not in custom_sw_list]
    return ' '.join(filtered_words)


def selective_stemming(text, stemmer_instance, protected_words_list):
    words = str(text).split()
    result = []
    for word in words:
        if word.lower() in protected_words_list:
            result.append(word)
        else:
            result.append(stemmer_instance.stem(word))
    return ' '.join(result)


def is_text_valid_for_inference(text_to_check):
    if not isinstance(text_to_check, str): return False
    if not text_to_check.strip(): return False
    if not re.search(r'[a-zA-Z]{2,}', text_to_check): return False
    return True


def preprocess_text(raw_text, resources):
    if not isinstance(raw_text, str): return ""
    text = raw_text
    text = cleaningText(text, resources["kata_baku_plus_nnya"])
    if not is_text_valid_for_inference(text): return ""
    text = normalisasi(text, resources["norm"])
    if not is_text_valid_for_inference(text): return ""
    text = remove_stopwords_cyberbullying(text, resources["custom_stopwords"])
    if not is_text_valid_for_inference(text): return ""
    text = selective_stemming(text, resources["stemmer"], resources["protected_words"])
    if not is_text_valid_for_inference(text): return ""
    return text

@st.cache_data
def get_llm_feedback(prediction_index, _llm_client):  # <--- PERUBAHAN DI SINI
    """
    Menghasilkan prompt dan memanggil LLM berdasarkan prediksi.
    Argumen _llm_client diabaikan oleh cache Streamlit.
    """
    if not _llm_client: # <--- GUNAKAN _llm_client DI DALAM FUNGSI
        return "Layanan Asisten AI tidak tersedia saat ini. Periksa apakah Ollama atau LM Studio sudah berjalan."

    prompt = ""
    system_role = "Anda adalah Asisten AI yang positif dan suportif bernama 'AURA' (Asisten Untuk Ruang Aman)."

    if prediction_index == 0:
        prompt = "Sebuah teks baru saja dianalisis dan teridentifikasi tidak mengandung perundungan. Berikan pujian singkat atas komunikasi yang positif dan berikan 1-2 tips umum untuk terus menjaga interaksi online tetap sehat dan positif. Jaga agar respons singkat dan memotivasi."
    elif prediction_index == 1:
        system_role = "Anda adalah Asisten AI yang bijaksana dan empatik bernama 'AURA'."
        prompt = "Sebuah teks dianalisis dan terdeteksi mengandung potensi perundungan tingkat rendah, seperti sarkasme yang bisa menyinggung atau ejekan halus. Tanpa perlu tahu teks aslinya, jelaskan secara umum mengapa komunikasi semacam ini kadang bisa disalahpahami dan berikan satu tips untuk memastikan candaan atau kritik diterima dengan baik. Fokus pada kesadaran diri dan empati."
    elif prediction_index == 2:
        system_role = "Anda adalah Asisten AI yang peduli dan bertanggung jawab bernama 'AURA'."
        prompt = "Sebuah teks dianalisis dan terdeteksi mengandung potensi perundungan tingkat sedang, seperti penggunaan kata-kata kasar atau serangan personal. Tanpa perlu tahu teks aslinya, berikan nasihat edukatif. Jelaskan secara umum dampak negatif dari bahasa semacam itu. Kemudian, berikan 1-2 saran praktis untuk refleksi diri sebelum mengirim pesan, seperti 'berpikir sejenak' atau 'memeriksa ulang nada tulisan'. Tujuannya adalah mendorong refleksi, bukan menghakimi."
    elif prediction_index == 3:
        system_role = "Anda adalah Asisten AI yang sangat peduli terhadap keamanan online bernama 'AURA'."
        prompt = "Sebuah teks baru saja dianalisis dan terdeteksi mengandung konten berbahaya atau perundungan tingkat tinggi, seperti ancaman atau ujaran kebencian serius. Tanpa perlu tahu teks aslinya, tugas Anda adalah memberikan peringatan yang serius dan fokus pada keamanan. Jelaskan secara umum bahaya dari komunikasi semacam itu. Sarankan dengan tegas untuk tidak mengirim pesan tersebut dan pertimbangkan untuk berbicara dengan seseorang yang dipercaya jika sedang merasa sangat marah. Prioritaskan de-eskalasi dan keamanan."
    else:
        return "Tidak ada saran yang tersedia untuk prediksi ini."

    try:
        response = _llm_client.chat.completions.create(model=CONFIG["llm_model"], # <--- GUNAKAN _llm_client
                                                      messages=[{"role": "system", "content": system_role},
                                                                {"role": "user", "content": prompt}], temperature=0.7,
                                                      max_tokens=1000)
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error saat menghubungi LLM: {e}")
        return "Maaf, terjadi kesalahan saat mencoba menghubungi Asisten AI. Pastikan layanan LLM (Ollama/LM Studio) Anda aktif dan dapat diakses."


def run_prediction_pipeline(raw_text, resources):
    processed_text = preprocess_text(raw_text, resources)

    if not processed_text:
        probabilities = np.zeros(len(LABEL_DESCRIPTIONS))
        probabilities[0] = 1.0
    else:
        sequence = resources["tokenizer"].texts_to_sequences([processed_text])
        padded = pad_sequences(sequence, maxlen=CONFIG["max_length"], padding=CONFIG["padding_type"],
                               truncating=CONFIG["trunc_type"])
        probabilities = resources["model"].predict(padded)[0]

    pred_index = np.argmax(probabilities)
    pred_label = LABEL_DESCRIPTIONS.get(pred_index, "Tidak Diketahui")

    llm_feedback = get_llm_feedback(pred_index, resources["llm_client"])

    return {"prediction": pred_label, "probabilities": probabilities, "processed_text": processed_text,
            "llm_feedback": llm_feedback}


# =============================================================================
# 5. FUNGSI UNTUK MERENDER UI
# =============================================================================

def render_sidebar(resources):
    st.sidebar.header("Status Model & Resource")
    status = resources.get("status", {})
    if status.get("model_ok") and status.get("tokenizer_ok"):
        st.sidebar.success("Model Deteksi Siap.", icon="✅")
    else:
        st.sidebar.error("Model Deteksi Gagal Dimuat.", icon="❌")

    if status.get("llm_ok"):
        st.sidebar.success("Asisten AI 'AURA' Aktif.", icon="✨")
    else:
        st.sidebar.warning("Asisten AI 'AURA' Tidak Aktif.", icon="🔌")

    if not status.get("nnya_ok"):
        st.sidebar.warning(
            f"File '{CONFIG['nnya_exceptions_path']}' tidak ditemukan. Preprocessing mungkin kurang akurat.")

    st.sidebar.header("Navigasi")
    st.sidebar.info("Halaman 'About' dan 'Contact' bisa ditambahkan di sini pada aplikasi multi-halaman.")


def render_results(result_data):
    probabilities = result_data["probabilities"]
    prediction = result_data["prediction"]
    llm_feedback = result_data["llm_feedback"]

    pred_label_index = np.argmax(probabilities)
    confidence_score = probabilities[pred_label_index]
    ui_detail = LABEL_UI_DETAILS.get(pred_label_index, {"icon": "❓", "color": "gray"})

    metric_cols = st.columns(2)
    metric_cols[0].metric(label="Prediksi", value=prediction)
    metric_cols[1].metric(label="Tingkat Keyakinan", value=f"{confidence_score:.2%}")
    st.markdown("---")

    st.markdown("##### ✨ Masukan dari Asisten AI 'AURA'")
    feedback_style = ui_detail.get("header_style", "")
    st.markdown(f"<div style='{feedback_style}'>{llm_feedback}</div>", unsafe_allow_html=True)

    with st.expander("Lihat Rincian Analisis"):
        st.write("**Teks Setelah Preprocessing:**")
        st.text(result_data["processed_text"] if result_data["processed_text"] else "(Tidak ada teks valid)")
        st.write("**Probabilitas per Kelas:**")
        prob_df = pd.DataFrame({'Kelas': LABEL_DESCRIPTIONS.values(), 'Probabilitas': probabilities})
        st.bar_chart(prob_df.set_index('Kelas'))


# =============================================================================
# 6. APLIKASI UTAMA
# =============================================================================

def main():
    st.set_page_config(page_title="SendShield - Deteksi Cyberbullying", layout="wide", initial_sidebar_state="auto")

    resources = load_all_resources()

    header_cols = st.columns([1, 4])
    try:
        header_cols[0].image(Image.open(CONFIG["logo_path"]), width=150)
    except FileNotFoundError:
        header_cols[0].markdown("## 🛡️ SendShield")

    render_sidebar(resources)

    st.title("Cyberbullying Detection")
    st.markdown("Analisis teks untuk mendeteksi potensi perundungan siber secara real-time.")
    st.markdown("---")

    if 'prediction_result' not in st.session_state:
        st.session_state.prediction_result = None

    main_cols = st.columns(2)

    with main_cols[0]:
        st.subheader("Masukkan Teks Anda")
        user_input = st.text_area(
            "Teks untuk dianalisis:", height=250, key="user_text_input",
            placeholder="Contoh: Kamu hebat sekali! Terima kasih atas bantuannya kemarin."
        )
        analyze_button = st.button("Analisis Teks", type="primary", use_container_width=True,
                                   disabled=not (
                                               resources["status"]["model_ok"] and resources["status"]["tokenizer_ok"]))

    if analyze_button:
        if user_input.strip():
            with st.spinner('Menganalisis teks...'):
                st.session_state.prediction_result = run_prediction_pipeline(user_input, resources)
        else:
            st.warning("Input teks tidak boleh kosong.", icon="✍️")
            st.session_state.prediction_result = None

    with main_cols[1]:
        st.subheader("Hasil Analisis")
        if st.session_state.prediction_result:
            render_results(st.session_state.prediction_result)
        else:
            try:
                st.image(CONFIG["placeholder_image_path"], caption="Menunggu analisis...")
            except FileNotFoundError:
                st.info("Hasil akan ditampilkan di sini.")


if __name__ == "__main__":
    main()