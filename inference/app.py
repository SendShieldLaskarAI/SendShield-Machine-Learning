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
import time

PATH_TO_DATAFULL_CSV = 'datafull.csv'
MODEL_PATH = 'model.h5'
TOKENIZER_PATH = 'tokenizer.pickle'


def create_nnya_exception_list_global(df_input):
    all_nnya_words = []
    if 'text' not in df_input.columns or df_input['text'].empty:
        return set(), Counter()

    for text_val in df_input['text'].dropna().astype(str).values:
        words = text_val.lower().split()
        for word in words:
            clean_word = re.sub(r'[^\w]', '', word)
            if clean_word.endswith('nnya'):
                all_nnya_words.append(clean_word)
    nnya_counter = Counter(all_nnya_words)
    nnya_exception_set = set(word for word, freq in nnya_counter.items() if freq >= 1)
    return nnya_exception_set, nnya_counter


def filter_valid_text_global(df, column):
    if column not in df.columns:
        return pd.DataFrame(columns=df.columns)
    df = df[df[column].notna()]
    df = df[df[column].str.strip() != '']
    if not df.empty:
        df = df[df[column].str.contains(r'[a-zA-Z]{2,}', na=False)]
    return df

@st.cache_resource
def load_all_preprocessing_resources_cached():
    print("Memulai inisialisasi SEMUA resource preprocessing (cached)...")
    nltk_download_attempted = False
    try:
        stopwords.words('indonesian')
    except LookupError:
        print("NLTK stopwords 'indonesian' tidak ditemukan, mencoba mengunduh...")
        nltk.download('stopwords')
        nltk_download_attempted = True

    factory_local = StemmerFactory()
    local_stemmer = factory_local.create_stemmer()

    local_nnya_exceptions = set()
    nnya_setup_success = False
    nnya_count_val = 0
    try:
        data_for_nnya_setup = pd.read_csv(PATH_TO_DATAFULL_CSV)
        data_for_nnya_setup.drop_duplicates(inplace=True)
        data_for_nnya_setup = filter_valid_text_global(data_for_nnya_setup, 'text')
        if not data_for_nnya_setup.empty:
            local_nnya_exceptions, _ = create_nnya_exception_list_global(data_for_nnya_setup)
            nnya_count_val = len(local_nnya_exceptions)
            nnya_setup_success = True
            print(f"Berhasil meregenerasi {nnya_count_val} 'nnya_exceptions' dari '{PATH_TO_DATAFULL_CSV}'.")
        else:
            print(
                f"Data kosong setelah filter untuk setup 'nnya_exceptions' dari '{PATH_TO_DATAFULL_CSV}'. Menggunakan placeholder.")
            local_nnya_exceptions.update({"sebenarnya", "kenyataannya", "harusnya"})  # Placeholder minimal
            nnya_count_val = len(local_nnya_exceptions)

    except FileNotFoundError:
        print(f"File '{PATH_TO_DATAFULL_CSV}' tidak ditemukan. Menggunakan 'nnya_exceptions' placeholder.")
        local_nnya_exceptions.update({"sebenarnya", "kenyataannya", "harusnya"})
        nnya_count_val = len(local_nnya_exceptions)
    except Exception as e:
        print(f"Error saat regenerasi 'nnya_exceptions': {e}. Menggunakan placeholder.")
        local_nnya_exceptions.update({"sebenarnya", "kenyataannya", "harusnya"})
        nnya_count_val = len(local_nnya_exceptions)
    local_kata_baku_berulang_final = {
        "allah", "nggak", "saat", "tinggal", "ngga", "alloh", "bukannya", "maaf", "uud", "tinggi", "omongannya",
        "nunggu", "tunggu",
        "sesungguhnya", "hingga", "ucapannya", "dajjal", "astaghfirullah", "sehingga", "menjelekkan", "meninggal",
        "sll", "menunjukkan", "panggung", "kerjaan",
        "kenyataan", "sungguh", "bangga", "panggil", "muhammadiyah", "ttp", "nggk", "kekuasaan", "menggonggong", "sllu",
        "melanggar", "cangkemmu", "kanggo", "menunggu", "dipanggil", "pertanggung", "menggulingkan", "pikirannya",
        "perkataan", "menganggap", "suul", "keadaan", "saatnya", "muhammad", "engga", "anggota", "kelakukannya",
        "bloon", "dianggap", "kerjaannya", "manfaatnya", "dll", "diindonesia", "jelekkan", "tanggung", "alhamdulillah",
    }
    local_kata_baku_plus_nnya = local_kata_baku_berulang_final.copy()
    for word in local_nnya_exceptions:
        local_kata_baku_plus_nnya.add(word)
    local_norm = {"amin": "", "yg": "yang", "rais": "", "mbah": "kakek", "sengkuni": "licik", "gak": "tidak",
                  "gk": "tidak", "amien": "", "tobat": "taubat", "sdh": "sudah",
                  "ga": "tidak", "quot": "kutipan", "org": "orang", "tdk": "tidak", "mu": "kamu", "wes": "sudah",
                  "wong": "orang", "tak": "tidak", "mpr": "", "gusdur": "", "allah": "",
                  "lah": "", "tau": "tahu", "dah": "sudah", "bpk": "bapak", "lu": "kamu", "opo": "apa", "jd": "jadi",
                  "aki": "kakek", "tengil": "menyebalkan", "lo": "kamu",
                  "tp": "tapi", "wis": "sudah", "klo": "kalau", "to": "", "tuwek": "tua", "yo": "iya", "d": "",
                  "plongo": "bingung", "kalo": "kalau", "ora": "tidak",
                  "g": "tidak", "iki": "ini", "gus": "", "dur": "", "mbok": "ibu", "pk": "bapak", "ra": "tidak",
                  "pa": "bapak", "plonga": "bingung",
                  "nggak": "tidak", "bener": "benar", "ki": "ini", "jgn": "jangan", "udh": "sudah", "ae": "aja",
                  "ko": "kok", "dr": "dari", "pikun": "lupa", "p": "",
                  "ni": "ini", "km": "kamu", "mbh": "kakek", "sampean": "kamu", "is": "", "ngaca": "kaca",
                  "asu": "anjing", "dgn": "dengan", "sih": "", "men": "", "sing": "yang",
                  "wae": "saja", "jdi": "jadi", "tuek": "tua", "pinter": "pintar", "rakus": "serakah", "amp": "",
                  "alloh": "", "dg": "dengan", "gitu": "begitu", "kek": "seperti",
                  "inilah": "ini lah", "se": "", "kowe": "kamu", "bin": "", "dirimu": "diri kamu", "inget": "ingat",
                  "pret": "bohong", "istighfar": "",
                  "gini": "begini", "modar": "meninggal", "prabowo": "", "sepuh": "tua", "e": "", "banget": "sangat",
                  "islam": "", "waras": "sehat",
                  "koyo": "seperti", "tuo": "tua", "lg": "lagi", "mulutmu": "mulut kamu", "krn": "karena", "dn": "dan",
                  "jg": "juga", "nih": "ini", "cangkem": "mulut",
                  "tu": "itu", "karna": "karena", "iku": "itu", "uda": "sudah", "prof": "profesor", "dadi": "jadi",
                  "glandangan": "gelandangan", "eling": "ingat",
                  "kmu": "kamu", "edan": "gila", "cangkeme": "mulut", "sy": "saya", "n": "", "istigfar": "",
                  "cangkemu": "mulut", "utk": "untuk", "koe": "kamu", "blm": "belum",
                  "klu": "kalau", "seng": "yang", "joko": "", "ngga": "tidak", "nyinyir": "ngomong", "msh": "masih",
                  "liat": "lihat", "sm": "sama", "odgj": "gila", "mulyono": "", "jokowi": "", "alhamdulillah": ""
                  }
    nltk_stopwords_local = set(stopwords.words('indonesian'))
    kata_penting_local = {"kamu", "dia", "aku", "ini", "itu", "sangat", "sekali", "sih", "banget"}
    local_custom_stopwords = nltk_stopwords_local - kata_penting_local
    local_cyberbullying_protected_words = {
        "bodoh", "goblok", "tolol", "jelek", "buruk", "busuk", "kotor",
        "kebodohan", "ketolohan", "kegoblokan", "kejelekan", "keburukan",
        "kebusukan", "pembodohan", "penjelekan", "penghinaan",
        "menyebalkan", "menjijikkan", "memalukan", "mengecewakan",
        "mengganggu", "menyakitkan", "menghina", "merendahkan",
        "memfitnah", "mencemooh",
        "membenci", "memarahi", "menghujat", "mengolok", "menyerang",
        "dibenci", "dihina", "dimarahi", "dicemooh", "difitnah",
        "terburuk", "terbodoh", "tergoblok", "terjelek", "terjijik",
        "terkutuk", "terjahat", "paling",
        "gelandangan", "pengemis", "sampah", "bangkai", "comberan",
        "kotoran"
    }

    print("SEMUA resource preprocessing selesai diinisialisasi.")
    return {
        "stemmer": local_stemmer,
        "kata_baku_berulang_plus_nnya": local_kata_baku_plus_nnya,
        "norm": local_norm,
        "custom_stopwords": local_custom_stopwords,
        "cyberbullying_protected_words": local_cyberbullying_protected_words,
        "nltk_download_attempted": nltk_download_attempted,
        "nnya_setup_success": nnya_setup_success,
        "nnya_count": nnya_count_val,
        "setup_ok": True
    }

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

def preprocess_text_for_inference(raw_text_input, prep_resources):
    if not isinstance(raw_text_input, str): return ""
    current_text = raw_text_input

    current_text = cleaningText(current_text, prep_resources["kata_baku_berulang_plus_nnya"])
    if not is_text_valid_for_inference(current_text): return ""

    current_text = normalisasi(current_text, prep_resources["norm"])
    if not is_text_valid_for_inference(current_text): return ""

    current_text = remove_stopwords_cyberbullying(current_text, prep_resources["custom_stopwords"])
    if not is_text_valid_for_inference(current_text): return ""

    current_text = selective_stemming(current_text, prep_resources["stemmer"],
                                      prep_resources["cyberbullying_protected_words"])
    if not is_text_valid_for_inference(current_text): return ""

    return current_text


@st.cache_resource
def load_model_and_tokenizer_cached(model_path, tokenizer_path):
    loaded_model = None
    loaded_tokenizer = None
    model_load_success = False
    tokenizer_load_success = False
    try:
        loaded_model = load_model(model_path)
        print(f"Model '{model_path}' berhasil dimuat.")
        model_load_success = True
    except Exception as e:
        print(f"GAGAL memuat model dari {model_path}: {e}")
    try:
        with open(tokenizer_path, 'rb') as handle:
            loaded_tokenizer = pickle.load(handle)
        print(f"Tokenizer '{tokenizer_path}' berhasil dimuat.")
        tokenizer_load_success = True
    except Exception as e:
        print(f"GAGAL memuat tokenizer dari {tokenizer_path}: {e}")
    return loaded_model, loaded_tokenizer, model_load_success, tokenizer_load_success


MAX_LENGTH = 120
PADDING_TYPE = 'post'
TRUNC_TYPE = 'post'
LABEL_DESCRIPTIONS = {
    0: "Tidak ada Cyberbullying",
    1: "Tingkat Keparahan Rendah",
    2: "Tingkat Keparahan Sedang",
    3: "Tingkat Keparahan Tinggi"
}
LABEL_UI_DETAILS = {
    0: {"icon": "✅", "color": "green"},
    1: {"icon": "ℹ️", "color": "blue"},
    2: {"icon": "⚠️", "color": "orange"},
    3: {"icon": "🚨", "color": "red"}
}


def predict_new_text_streamlit(raw_text, current_model, current_tokenizer,
                               max_len, padding_t, trunc_t, label_desc_map,
                               prep_resources_arg):
    if current_model is None or current_tokenizer is None:
        return "Model atau Tokenizer belum dimuat.", None, raw_text, "(Error: Model/Tokenizer tidak tersedia)"

    original_text_for_display = raw_text
    if not isinstance(raw_text, str) or not raw_text.strip():
        num_classes = len(label_desc_map)
        default_probs = np.zeros(num_classes)
        if 0 in label_desc_map: default_probs[0] = 1.0
        return label_desc_map.get(0,
                                  'Tidak Diketahui'), default_probs, original_text_for_display, "(Input tidak valid/kosong)"

    processed_text = preprocess_text_for_inference(raw_text, prep_resources_arg)

    if not processed_text:
        num_classes = len(label_desc_map)
        default_probs = np.zeros(num_classes)
        if 0 in label_desc_map: default_probs[0] = 1.0
        return label_desc_map.get(0,
                                  'Tidak Diketahui'), default_probs, original_text_for_display, "(Teks kosong setelah preprocessing)"

    sequence = current_tokenizer.texts_to_sequences([processed_text])
    padded_sequence = pad_sequences(sequence, maxlen=max_len, padding=padding_t, truncating=trunc_t)

    predictions_prob = current_model.predict(padded_sequence)
    predicted_class_index = np.argmax(predictions_prob[0])
    predicted_label_desc = label_desc_map.get(predicted_class_index,
                                              f"Indeks Kelas Tidak Diketahui: {predicted_class_index}")

    return predicted_label_desc, predictions_prob[0], original_text_for_display, processed_text

st.set_page_config(page_title="Deteksi Cyberbullying", layout="wide", initial_sidebar_state="expanded")
all_prep_resources = None
model_loaded = None
tokenizer_loaded = None

with st.spinner("Mempersiapkan resource aplikasi... Ini mungkin perlu beberapa saat pada pemuatan pertama."):
    all_prep_resources = load_all_preprocessing_resources_cached()
    model_loaded, tokenizer_loaded, model_success, tokenizer_success = load_model_and_tokenizer_cached(MODEL_PATH,
                                                                                                       TOKENIZER_PATH)
if all_prep_resources and all_prep_resources.get("setup_ok"):
    st.sidebar.success("Resource preprocessing siap.")
    if all_prep_resources.get("nltk_download_attempted"):
        st.sidebar.info("NLTK stopwords telah diperiksa/diunduh.")
    if not all_prep_resources.get("nnya_setup_success"):
        st.sidebar.warning(
            f"Setup 'nnya_exceptions' menggunakan placeholder ({all_prep_resources.get('nnya_count')} entri). File '{PATH_TO_DATAFULL_CSV}' mungkin tidak ditemukan atau error.")
    else:
        st.sidebar.info(f"'nnya_exceptions' ({all_prep_resources.get('nnya_count')} entri) berhasil diregenerasi.")

else:
    st.sidebar.error("Gagal memuat resource preprocessing!")

if model_success:
    st.sidebar.success(f"Model '{MODEL_PATH}' dimuat.")
else:
    st.sidebar.error(f"Model '{MODEL_PATH}' gagal dimuat!")

if tokenizer_success:
    st.sidebar.success(f"Tokenizer '{TOKENIZER_PATH}' dimuat.")
else:
    st.sidebar.error(f"Tokenizer '{TOKENIZER_PATH}' gagal dimuat!")

st.title("🔍 Aplikasi Deteksi Cyberbullying")
st.markdown("""
Aplikasi ini menggunakan model Machine Learning untuk mendeteksi potensi cyberbullying dalam teks berbahasa Indonesia 
dan mengklasifikasikannya berdasarkan tingkat keparahannya. Masukkan teks di bawah ini dan klik "Analisis Teks".
""")

st.sidebar.header("Tentang Aplikasi")
st.sidebar.info("""
- **Model**: BiLSTM dilatih pada data teks berbahasa Indonesia.
- **Preprocessing**: Normalisasi, stemming, stopword removal, dll.
- **Output**: Prediksi tingkat keparahan dan probabilitasnya.
""")
st.sidebar.markdown("---")
st.sidebar.markdown(
    f"*Pastikan file model (`{MODEL_PATH}`), tokenizer (`{TOKENIZER_PATH}`), dan data awal (`{PATH_TO_DATAFULL_CSV}`) berada di lokasi yang benar.*")

col1, col2 = st.columns([2, 1.5])

with col1:
    st.subheader("📝 Masukkan Teks untuk Dianalisis:")
    user_input = st.text_area("Ketik atau tempel teks di sini...", height=200, key="user_text_input",
                              placeholder="Contoh: Kamu jelek sekali dan tidak berguna!")

    analyze_button = st.button("Analisis Teks", type="primary", use_container_width=True)

with col2:
    st.subheader("📊 Hasil Analisis:")
    if analyze_button:
        if model_loaded and tokenizer_loaded and all_prep_resources and all_prep_resources.get("setup_ok"):
            if user_input.strip():
                with st.spinner('Sedang menganalisis teks...'):
                    prediction, probabilities, original_display, processed_display = predict_new_text_streamlit(
                        user_input,
                        model_loaded,
                        tokenizer_loaded,
                        MAX_LENGTH,
                        PADDING_TYPE,
                        TRUNC_TYPE,
                        LABEL_DESCRIPTIONS,
                        all_prep_resources
                    )

                st.markdown(f"**Teks Asli (Input):**")
                st.text_area("", value=original_display, height=100, disabled=True, key="original_text_display")
                st.markdown(f"**Teks Setelah Preprocessing:**")
                st.text_area("",
                             value=processed_display if processed_display else "(Tidak ada teks valid setelah preprocessing)",
                             height=100, disabled=True, key="processed_text_display")
                st.markdown("---")

                pred_label_index = -1
                if probabilities is not None:
                    pred_label_index = np.argmax(probabilities)

                ui_detail = LABEL_UI_DETAILS.get(pred_label_index, {"icon": "❓", "color": "gray"})

                st.markdown(f"**Prediksi Tingkat Keparahan:**")
                st.markdown(f"<h3 style='color:{ui_detail['color']};'>{ui_detail['icon']} {prediction}</h3>",
                            unsafe_allow_html=True)

                if probabilities is not None:
                    st.markdown("**Probabilitas per Kelas:**")
                    prob_df_data = {
                        'Kelas': [LABEL_DESCRIPTIONS.get(i, f"Kelas {i}") for i in range(len(probabilities))],
                        'Probabilitas': probabilities
                    }
                    prob_df = pd.DataFrame(prob_df_data)
                    st.bar_chart(prob_df.set_index('Kelas'))
                else:
                    st.markdown("Tidak dapat menampilkan probabilitas.")
            else:
                st.warning("Input teks tidak boleh kosong. Silakan masukkan teks untuk dianalisis.", icon="✍️")
        else:
            st.error(
                "Model, Tokenizer, atau Resource Preprocessing belum siap. Periksa pesan error di sidebar atau konsol.",
                icon="⚙️")
    else:
        st.info("Hasil analisis akan muncul di sini setelah Anda memasukkan teks dan mengklik tombol 'Analisis Teks'.",
                icon="💡")

st.markdown("---")
st.caption("Aplikasi Streamlit untuk Deteksi Cyberbullying")