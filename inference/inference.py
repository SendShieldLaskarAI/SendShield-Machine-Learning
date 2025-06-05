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

try:
    stopwords.words('indonesian')
except LookupError:
    print("Mengunduh NLTK stopwords untuk 'indonesian'...")
    nltk.download('stopwords')

print("Menginisialisasi Sastrawi Stemmer...")
factory = StemmerFactory()
stemmer = factory.create_stemmer()
print("Sastrawi Stemmer siap.")

def create_nnya_exception_list(df_input):
    all_nnya_words = []
    for text_val in df_input['text'].values:
        words = str(text_val).lower().split()
        for word in words:
            clean_word = re.sub(r'[^\w]', '', word)
            if clean_word.endswith('nnya'):
                all_nnya_words.append(clean_word)
    nnya_counter = Counter(all_nnya_words)
    nnya_exception_set = set(word for word, freq in nnya_counter.items() if freq >= 1)
    return nnya_exception_set, nnya_counter

def filter_valid_text(df, column):
    df = df[df[column].notna()]
    df = df[df[column].str.strip() != '']
    df = df[df[column].str.contains(r'[a-zA-Z]{2,}', na=False)]
    return df

print("Mempersiapkan 'nnya_exceptions'...")
PATH_TO_DATAFULL_CSV = 'datafull.csv'
try:
    data_for_nnya_setup = pd.read_csv(PATH_TO_DATAFULL_CSV)
    data_for_nnya_setup.drop_duplicates(inplace=True)
    data_for_nnya_setup = filter_valid_text(data_for_nnya_setup, 'text')
    nnya_exceptions, _ = create_nnya_exception_list(data_for_nnya_setup)
    print(f"Berhasil meregenerasi {len(nnya_exceptions)} 'nnya_exceptions' dari '{PATH_TO_DATAFULL_CSV}'.")
except FileNotFoundError:
    print(f"⚠PERINGATAN: '{PATH_TO_DATAFULL_CSV}' tidak ditemukan.")
    print("   Menggunakan daftar 'nnya_exceptions' placeholder. Hasil inferensi mungkin tidak sepenuhnya akurat.")
    nnya_exceptions = {"sebenarnya", "kenyataannya", "harusnya", "omongannya", "ucapannya", "pikirannya", "perkataan", "keadaannya", "saatnya", "kelakuannya", "kerjaannya", "manfaatnya"}
except Exception as e:
    print(f"⚠️ Error saat mencoba meregenerasi nnya_exceptions dari CSV: {e}")
    print("   Menggunakan daftar 'nnya_exceptions' placeholder. Hasil inferensi mungkin tidak sepenuhnya akurat.")
    nnya_exceptions = {"sebenarnya", "kenyataannya", "harusnya", "omongannya", "ucapannya", "pikirannya", "perkataan", "keadaannya", "saatnya", "kelakuannya", "kerjaannya", "manfaatnya"}

kata_baku_berulang_final = {
    "allah", "nggak", "saat", "tinggal", "ngga", "alloh", "bukannya", "maaf", "uud", "tinggi", "omongannya", "nunggu", "tunggu",
    "sesungguhnya", "hingga", "ucapannya", "dajjal", "astaghfirullah", "sehingga", "menjelekkan", "meninggal", "sll", "menunjukkan", "panggung", "kerjaan",
    "kenyataan", "sungguh", "bangga", "panggil", "muhammadiyah", "ttp", "nggk", "kekuasaan", "menggonggong", "sllu", "melanggar", "cangkemmu", "kanggo", "menunggu", "dipanggil", "pertanggung", "menggulingkan", "pikirannya", "perkataan", "menganggap", "suul", "keadaan", "saatnya", "muhammad", "engga", "anggota", "kelakukannya", "bloon", "dianggap", "kerjaannya", "manfaatnya", "dll", "diindonesia", "jelekkan", "tanggung", "alhamdulillah",
}
kata_baku_berulang_plus_nnya = kata_baku_berulang_final.copy()
for word in nnya_exceptions:
    kata_baku_berulang_plus_nnya.add(word)

norm = {"amin":"", "yg":"yang","rais" : "", "mbah":"kakek", "sengkuni":"licik", "gak":"tidak","gk":"tidak","amien":"", "tobat":"taubat", "sdh":"sudah",
        "ga":"tidak","quot":"kutipan","org":"orang","tdk":"tidak","mu":"kamu","wes":"sudah","wong":"orang","tak":"tidak","mpr":"","gusdur":"","allah":"",
       "lah":"","tau":"tahu","dah":"sudah","bpk":"bapak","lu":"kamu","opo":"apa","jd":"jadi","aki":"kakek","tengil":"menyebalkan","lo":"kamu",
       "tp":"tapi","wis":"sudah","klo":"kalau","to":"","tuwek":"tua","yo":"iya","d":"","plongo":"bingung","kalo":"kalau","ora":"tidak",
       "g":"tidak","iki":"ini","gus":"","dur":"","mbok":"ibu","pk":"bapak","ra":"tidak","pa":"bapak","plonga":"bingung",
       "nggak":"tidak","bener":"benar","ki":"ini","jgn":"jangan","udh":"sudah","ae":"aja","ko":"kok","dr":"dari","pikun":"lupa","p":"",
       "ni":"ini","km":"kamu","mbh":"kakek","sampean":"kamu","is":"","ngaca":"kaca","asu":"anjing","dgn":"dengan","sih":"","men":"","sing":"yang",
       "wae":"saja","jdi":"jadi","tuek":"tua","pinter":"pintar","rakus":"serakah","amp":"","alloh":"","dg":"dengan","gitu":"begitu","kek":"seperti",
       "inilah":"ini lah","se":"","kowe":"kamu","bin":"","dirimu":"diri kamu","inget":"ingat","pret":"bohong","istighfar":"",
       "gini":"begini","modar":"meninggal","prabowo":"","sepuh":"tua","e":"","banget":"sangat","islam":"","waras":"sehat",
       "koyo":"seperti","tuo":"tua","lg":"lagi","mulutmu":"mulut kamu","krn":"karena","dn":"dan","jg":"juga","nih":"ini","cangkem":"mulut",
       "tu":"itu","karna":"karena","iku":"itu","uda":"sudah","prof":"profesor","dadi":"jadi","glandangan":"gelandangan","eling":"ingat",
       "kmu":"kamu","edan":"gila","cangkeme":"mulut","sy":"saya","n":"","istigfar":"","cangkemu":"mulut","utk":"untuk","koe":"kamu","blm":"belum",
       "klu":"kalau","seng":"yang","joko":"","ngga":"tidak","nyinyir":"ngomong","msh":"masih","liat":"lihat","sm":"sama","odgj":"gila","mulyono":"","jokowi":"","alhamdulillah":""
       }

nltk_stopwords = set(stopwords.words('indonesian'))
kata_penting = {"kamu", "dia", "aku", "ini", "itu", "sangat", "sekali", "sih", "banget"}
custom_stopwords = nltk_stopwords - kata_penting
cyberbullying_protected_words = {
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
def cleaningText(text, exceptions):
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
    if token in exceptions:
      normalized_tokens.append(token)
    else:
      token = re.sub(r'(.)\1+', r'\1', token)
      normalized_tokens.append(token)
  return ' '.join(normalized_tokens)
def normalisasi(text):
  text_normalized = str(text)
  for word, replacement in norm.items():
    text_normalized = re.sub(r'\b' + re.escape(word) + r'\b', replacement, text_normalized)
  text_normalized = ' '.join(text_normalized.split())
  return text_normalized

def remove_stopwords_cyberbullying(text):
  words = str(text).split()
  filtered_words = [word for word in words if word not in custom_stopwords]
  return ' '.join(filtered_words)

def selective_stemming(text):
  words = str(text).split()
  result = []
  for word in words:
    if word.lower() in cyberbullying_protected_words:
      result.append(word)
    else:
      result.append(stemmer.stem(word))
  return ' '.join(result)

def is_text_valid_for_inference(text_to_check):
    if not isinstance(text_to_check, str): return False
    if not text_to_check.strip(): return False
    if not re.search(r'[a-zA-Z]{2,}', text_to_check): return False
    return True

def preprocess_text_for_inference(raw_text_input):
    if not isinstance(raw_text_input, str): return ""
    current_text = raw_text_input
    current_text = cleaningText(current_text, kata_baku_berulang_plus_nnya)
    if not is_text_valid_for_inference(current_text): return ""
    current_text = normalisasi(current_text)
    if not is_text_valid_for_inference(current_text): return ""
    current_text = remove_stopwords_cyberbullying(current_text)
    if not is_text_valid_for_inference(current_text): return ""
    current_text = selective_stemming(current_text)
    if not is_text_valid_for_inference(current_text): return ""
    return current_text

MAX_LENGTH = 120
PADDING_TYPE = 'post'
TRUNC_TYPE = 'post'
LABEL_DESCRIPTIONS = {
    0: "Tidak ada Cyberbullying",
    1: "Tingkat Keparahan Rendah",
    2: "Tingkat Keparahan Sedang",
    3: "Tingkat Keparahan Tinggi"
}
MODEL_PATH = 'model.h5'
TOKENIZER_PATH = 'tokenizer.pickle'
model = load_model(MODEL_PATH)
with open(TOKENIZER_PATH, 'rb') as handle:
    tokenizer = pickle.load(handle)

def predict_new_text(raw_text, loaded_model, loaded_tokenizer,
                     max_len, padding_t, trunc_t, label_desc_map):
    if loaded_model is None or loaded_tokenizer is None:
        return "Model atau Tokenizer belum dimuat dengan benar.", None
    if not isinstance(raw_text, str) or not raw_text.strip():
        num_classes = len(label_desc_map)
        default_probs = np.zeros(num_classes)
        if 0 in label_desc_map : default_probs[0] = 1.0
        return f"Input: '{str(raw_text)[:50]}...' -> (Input tidak valid/kosong) -> {label_desc_map.get(0, 'Tidak Diketahui')}", default_probs
    processed_text = preprocess_text_for_inference(raw_text)
    if not processed_text:
        num_classes = len(label_desc_map)
        default_probs = np.zeros(num_classes)
        if 0 in label_desc_map : default_probs[0] = 1.0
        return f"Input: '{str(raw_text)[:50]}...' -> (Teks kosong setelah preprocessing) -> {label_desc_map.get(0, 'Tidak Diketahui')}", default_probs
    sequence = loaded_tokenizer.texts_to_sequences([processed_text])
    padded_sequence = pad_sequences(sequence, maxlen=max_len, padding=padding_t, truncating=trunc_t)
    predictions_prob = loaded_model.predict(padded_sequence)
    predicted_class_index = np.argmax(predictions_prob[0])
    predicted_label_desc = label_desc_map.get(predicted_class_index, f"Indeks Kelas Tidak Diketahui: {predicted_class_index}")
    return f"Input: '{str(raw_text)[:70]}...' \n   -> Hasil Preprocessing: '{processed_text[:70]}...' \n   -> Prediksi: {predicted_label_desc}", predictions_prob[0]

if model is not None and tokenizer is not None:
    print("Ketik teks yang ingin Anda analisis.")
    print("Ketik 'exit', 'quit', atau 'keluar' untuk mengakhiri.")
    while True:
        try:
            user_input_text = input("\nMasukkan teks: ")
            if user_input_text.lower() in ['exit', 'quit', 'keluar']:
                print("Mengakhiri sesi inferensi.")
                break
            if not user_input_text.strip(): # Jika pengguna hanya menekan Enter
                print("Input kosong, silakan masukkan teks atau ketik 'exit' untuk keluar.")
                continue

            hasil_prediksi_teks, probabilitas = predict_new_text(
                user_input_text,
                model,
                tokenizer,
                MAX_LENGTH,
                PADDING_TYPE,
                TRUNC_TYPE,
                LABEL_DESCRIPTIONS
            )
            print(hasil_prediksi_teks)
            if probabilitas is not None:
                prob_str = ", ".join([f"Kelas {i}={p:.3f}" for i, p in enumerate(probabilitas)])
                print(f"   Probabilitas: [{prob_str}]")
            print("-" * 50)
        except KeyboardInterrupt: # Menangani jika pengguna menekan Ctrl+C
            print("\nSesi inferensi dihentikan oleh pengguna (Ctrl+C).")
            break
        except Exception as e: # Menangani error tak terduga lainnya
            print(f"Terjadi error saat pemrosesan: {e}")
            print("Silakan coba lagi atau ketik 'exit' untuk keluar.")
else:
    print("🛑 Tidak dapat menjalankan mode inferensi interaktif karena model atau tokenizer gagal dimuat.")

