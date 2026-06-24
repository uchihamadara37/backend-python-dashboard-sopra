from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib
import os
from groq import Groq
from dotenv import load_dotenv
load_dotenv()


app = Flask(__name__)
# Mengizinkan semua origin untuk fase development
CORS(app) 

TIME_STEPS = 7

# Load Data Transaksi untuk mengambil historis 7 hari terakhir
try:
    df_pelanggan = pd.read_csv("data_pelanggan.csv")
    df_produk = pd.read_csv("data_produk.csv")
    df_transaksi = pd.read_csv("data_transaksi.csv")
    
    df_pelanggan['terakhir_beli'] = pd.to_datetime(df_pelanggan['terakhir_beli'])
    df_transaksi['tanggal'] = pd.to_datetime(df_transaksi['tanggal'])
    
    print("✅ Data transaksi berhasil dimuat!")
except Exception as e:
    print(f"⚠️ Gagal memuat data transaksi: {e}")
    df_transaksi = pd.DataFrame()



loaded_models = {}
loaded_scalers = {}


# integrasi groq llm
SOPRA_INFO = """
Anda adalah asisten virtual resmi untuk perusahaan SOPRA Solusi-Pack. 
Tugas Anda adalah menjawab pertanyaan pelanggan dengan ramah, profesional, dan ringkas.

Gambaran Umum Perusahaan
- Nama Perusahaan yaitu PT Solusi Prima Packaging (SOPRA)

Bidang:
- Manufaktur kemasan plastik
- Packaging solutions
- PET bottle manufacturing
- Food packaging
- Cosmetic packaging
- Pharmaceutical packaging

Lokasi Produksi:
- Bekasi, Jawa Barat
- Pasuruan, Jawa Timur

Perusahaan juga terafiliasi dengan PT Trass Anugrah Makmur, yang berada dalam grup usaha yang sama untuk meningkatkan kapasitas dan kualitas produksi
SOPRA berkembang sebagai produsen kemasan plastik yang menargetkan pasar B2B (Business-to-Business)
Melayani:
- Industri makanan
- Industri minuman
- Industri farmasi
- Industri kosmetik
- Personal care
- Home care
- Retail tradisional
- Perusahaan manufaktur menengah dan besar

SOPRA bukan hanya menjual kemasan siap pakai, tetapi juga menyediakan:
- Custom bottle design
- Custom mold development
- Printing pada kemasan
- Konsultasi desain kemasan

Aturan Tambahan:
- Jika pengguna bertanya tentang stok atau harga yang spesifik, arahkan mereka untuk menghubungi tim sales.
- Jika pengguna bertangan tentang hal lain diluar konteks ini, jawab saja saya kurang memiliki informasi akurat mengenai hal tersebut, silakan hubungi tim kami atau kunjungi pusat informasi lain, saya adalah asisten untuk pertanyaan seputar SOPRA Solusi-Pack".
- Jawaban harus selalu relevan dengan konteks perusahaan dan tidak boleh menyimpang ke topik lain. Jangan pernah mengarang jawaban diluar informasi yang diberikan di atas.
"""
client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

def format_tanggal_indonesia(dt_obj):
    # Jika datanya kosong (NaT / NaN), kembalikan apa adanya
    if pd.isnull(dt_obj):
        return None
        
    hari_map = {
        0: 'Senin', 1: 'Selasa', 2: 'Rabu', 3: 'Kamis', 
        4: 'Jumat', 5: 'Sabtu', 6: 'Minggu'
    }
    
    bulan_map = {
        1: 'Januari', 2: 'Februari', 3: 'Maret', 4: 'April', 
        5: 'Mei', 6: 'Juni', 7: 'Juli', 8: 'Agustus', 
        9: 'September', 10: 'Oktober', 11: 'November', 12: 'Desember'
    }
    
    nama_hari = hari_map[dt_obj.weekday()]
    nama_bulan = bulan_map[dt_obj.month]
    
    # Format akhir: "Hari, DD Bulan YYYY"
    return f"{nama_hari}, {dt_obj.day} {nama_bulan} {dt_obj.year}"













# GET ALL DATA ENDPOINTS
@app.route('/api/customers', methods=['GET'])
def get_all_customers():
    try:
        df_temp = df_pelanggan.copy()
        df_temp['terakhir_beli'] = df_temp['terakhir_beli'].apply(format_tanggal_indonesia)
        # Mengubah DataFrame pelanggan menjadi list of dictionaries (JSON Array)
        data_json = df_temp.to_dict(orient='records')
        return jsonify({
            "status": "success",
            "total_data": len(data_json),
            "data": data_json
        }), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/api/products', methods=['GET'])
def get_all_products():
    try:
        data_json = df_produk.to_dict(orient='records')
        return jsonify({
            "status": "success",
            "total_data": len(data_json),
            "data": data_json
        }), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/api/transactions', methods=['GET'])
def get_all_transactions():
    try:
        # Salin dataframe agar tidak merubah data asli saat formatting tanggal
        df_temp = df_transaksi.copy()
        # Mengubah format datetime kembali menjadi string berformat YYYY-MM-DD agar rapi di JSON
        df_temp['tanggal'] = df_temp['tanggal'].dt.strftime('%Y-%m-%d')
        
        data_json = df_temp.to_dict(orient='records')
        return jsonify({
            "status": "success",
            "total_data": len(data_json),
            "data": data_json
        }), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500















# =====================================================================
# PREDIKSI STOK (LSTM)
# =====================================================================
@app.route('/api/predict-stock', methods=['GET'])
def predict_stock():
    # Contoh request: http://localhost:5000/api/predict-stock?product_id=PROD-01
    product_id = request.args.get('product_id')
    
    if not product_id:
        return jsonify({"status": "error", "message": "Parameter 'product_id' wajib diisi"}), 400
    
    try:
        # cek model
        if product_id not in loaded_models:
            model_path = f"model_lstm/lstm_{product_id}.keras"
            scaler_path = f"model_lstm/scaler_{product_id}.pkl"
            
            # Cek apakah file fisiknya ada di dalam folder
            if not os.path.exists(model_path) or not os.path.exists(scaler_path):
                return jsonify({
                    "status": "error", 
                    "message": f"Model untuk produk '{product_id}' tidak ditemukan. Pastikan model telah dilatih dan disimpan di folder 'model_lstm'."
                }), 404
                
            # Jika ada, load model & scaler, lalu simpan ke dictionary cache
            loaded_models[product_id] = load_model(model_path)
            loaded_scalers[product_id] = joblib.load(scaler_path)
            print(f"✅ Model {product_id} berhasil dimuat ke memori.")
            
        # Ambil model & scaler dari memori
        lstm_model = loaded_models[product_id]
        scaler = loaded_scalers[product_id]    
        
        # Ambil data spesifik produk unggulan
        df_prod = df_transaksi[df_transaksi['product_id'] == product_id].copy()
        if df_prod.empty:
            return jsonify({"status": "error", "message": f"Tidak ada data transaksi historis untuk '{product_id}'"}), 404
        
        daily_sales = df_prod.groupby('tanggal')['qty'].sum().reset_index()
        daily_sales = daily_sales.set_index('tanggal').asfreq('D', fill_value=0)
        
        TIME_STEPS = 14
        if len(daily_sales) < TIME_STEPS:
            return jsonify({"status": "error", "message": f"Data historis kurang dari {TIME_STEPS} hari."}), 400
        
        # =====================================================================
        # Analisis stock minimum dan lead time untuk rekomendasi stok aman
        # Asumsi parameter operasional pengiriman dari gudang SOPRA ke Reseller:
        LEAD_TIME_AVG = 3   # Rata-rata pengiriman butuh waktu 3 hari
        LEAD_TIME_MAX = 5   # Jika kurir overload/terlambat, maksimal 5 hari
        
        # Ekstrak data penjualan harian riil
        data_penjualan = daily_sales['qty'].values
        
        avg_daily_sales = float(np.mean(data_penjualan))
        max_daily_sales = float(np.max(data_penjualan))
        
        # Hitung Stok Minimum Keamanan (Safety Stock)
        safety_stock = (max_daily_sales * LEAD_TIME_MAX) - (avg_daily_sales * LEAD_TIME_AVG)
        safety_stock = int(np.ceil(max(0, safety_stock))) # Pembulatan ke atas karena fisik barang
        
        # Hitung Titik Pesan Ulang (Reorder Point)
        reorder_point = (avg_daily_sales * LEAD_TIME_AVG) + safety_stock
        reorder_point = int(np.ceil(reorder_point))
        
        # prediksi lstm untuk minggu depan ==================================
        # Ambil 14 hari terakhir
        raw_data = daily_sales['qty'].values.reshape(-1, 1)
        last_14_days = raw_data[-TIME_STEPS:]
        
        # Normalisasi menggunakan scaler yang sama saat training
        scaled_input = scaler.transform(last_14_days)
        
        reshaped_input = scaled_input.reshape(1, TIME_STEPS, 1)
        prediksi_scaled = lstm_model.predict(reshaped_input, verbose=0)
        prediksi_scaled_reshaped = prediksi_scaled.reshape(-1, 1)
        prediksi_asli_array = scaler.inverse_transform(prediksi_scaled_reshaped)
        
        # Ekstrak ke dalam list dasar Python, bulatkan, dan amankan (tidak boleh minus)
        list_prediksi_harian = [max(0, int(np.round(val[0]))) for val in prediksi_asli_array]
        
        # Kalkulasi total seminggu
        total_forecast_seminggu = sum(list_prediksi_harian)
        
        # reshaped_input = scaled_input.reshape(1, TIME_STEPS, 1)
        
        # # Lakukan peramalan
        # prediksi_scaled = lstm_model.predict(reshaped_input, verbose=0)
        # prediksi_asli = scaler.inverse_transform(prediksi_scaled)
        # hasil_prediksi = int(np.round(prediksi_asli[0][0]))
        # hasil_prediksi = max(0, hasil_prediksi)
        
        # Kembalikan response JSON 
        return jsonify({
            "status": "success",
            "product_id": product_id,
            "forecast_harian_minggu_depan": list_prediksi_harian,
            "forecast_total_minggu_depan": total_forecast_seminggu,
            "rekomendasi_reseller": {
                "stok_minimum_aman (safety_stock)": safety_stock,
                "titik_pesan_ulang (reorder_point)": reorder_point,
                "catatan_operasional": {
                    "rata_rata_penjualan_harian": round(avg_daily_sales, 2),
                    "asumsi_waktu_tunggu_gudang_hari": LEAD_TIME_AVG
                }
            }
        }), 200

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500



















# =====================================================================
# INTEGRASI LLM (Tanya Jawab SOPRA)
# =====================================================================
@app.route('/api/chat', methods=['POST'])
def chat_with_llm():
    data = request.json
    user_message = data.get("message", "")
    
    if not user_message:
        return jsonify({"error": "Pesan tidak boleh kosong"}), 400
    
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": SOPRA_INFO,
            },
            {
                "role": "user",
                "content": user_message,
            },
        ],
        model="openai/gpt-oss-120b",
        temperature=0.5,        # Mengatur kreativitas (0.5 cukup aman agar tidak halusinasi)
        max_tokens=500,         # Membatasi panjang jawaban agar tidak boros
    )
    ai_reply = chat_completion.choices[0].message.content
    
    return jsonify({
            "status": "success",
            "reply": ai_reply
        }), 200



















# =====================================================================
# Rekomendasi Produk (Asosiasi Produk)
# =====================================================================
@app.route('/api/best-product', methods=['GET'])
def get_best_product():
    try:
        threshold = request.args.get('threshold', default=5, type=int)
        
        # BEST SELLERS (5 Produk Paling Laris)
        # Kelompokkan berdasarkan ID produk dan jumlahkan qty
        top_sales = df_transaksi.groupby('product_id')['qty'].sum().sort_values(ascending=False).head(threshold)
        print(top_sales.head())
        
        best_sellers_list = []
        for pid, qty in top_sales.items():
            # Cari informasi detail produk dari df_produk
            prod_info = df_produk[df_produk['product_id'] == pid].to_dict(orient='records')
            if prod_info:
                prod_detail = prod_info[0]
                prod_detail['total_terjual'] = int(qty)
                best_sellers_list.append(prod_detail)

        # Kembalikan response JSON
        return jsonify({
            "status": "success",
            "best_sellers": best_sellers_list,
        }), 200

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500























@app.route('/api/recomendations', methods=['GET'])
def get_recomendations():
    try:
        # Ambil parameter dari query string
        target_product = request.args.get('target_product')
        # FREQUENTLY BOUGHT TOGETHER (Berdasarkan Hari) karena berdasarkan order_id ternyata tidak ada yang sama
        bought_together_list = []
        
        # Ambil kombinasi unik (tanggal + product_id) agar tidak menghitung qty dobel
        df_unique_daily = df_transaksi[['tanggal', 'product_id']].drop_duplicates()
        
        # Lakukan "Self-Join" berdasarkan tanggal. 
        # Ini akan memasangkan semua produk yang laku di hari yang sama
        pairs = pd.merge(
            df_unique_daily, 
            df_unique_daily, 
            on='tanggal'
        )
        
        # Buang pasangan produk yang sama (misal: Produk A dipasangkan dengan Produk A)
        pairs = pairs[pairs['product_id_x'] != pairs['product_id_y']]
        
        if target_product:
            # Jika user meminta rekomendasi untuk produk tertentu
            target_pairs = pairs[pairs['product_id_x'] == target_product]
            
            # Hitung produk apa (y) yang paling sering muncul bersama target (x)
            top_pairs = target_pairs.groupby('product_id_y').size().sort_values(ascending=False).head(3)
            
            for pid, count in top_pairs.items():
                prod_info = df_produk[df_produk['product_id'] == pid].to_dict(orient='records')
                if prod_info:
                    prod_detail = prod_info[0]
                    prod_detail['frekuensi_hari_bersamaan'] = int(count)
                    bought_together_list.append(prod_detail)
        else:
            # Jika tidak ada target spesifik, kembalikan 3 pasangan produk paling top secara global
            pair_counts = pairs.groupby(['product_id_x', 'product_id_y']).size().sort_values(ascending=False).reset_index(name='count')
            
            seen_pairs = set()
            for _, row in pair_counts.iterrows():
                p1, p2, count = row['product_id_x'], row['product_id_y'], row['count']
                
                # Urutkan ID agar pasangan A-B dan B-A dianggap sama, menghindari duplikat di JSON
                pair_hash = tuple(sorted([p1, p2]))
                if pair_hash not in seen_pairs:
                    seen_pairs.add(pair_hash)
                    
                    # Ambil nama produk untuk p1 dan p2 (mencegah error jika produk tidak ditemukan)
                    p1_match = df_produk[df_produk['product_id'] == p1]
                    p2_match = df_produk[df_produk['product_id'] == p2]
                    
                    p1_name = p1_match['nama_produk'].values[0] if not p1_match.empty else p1
                    p2_name = p2_match['nama_produk'].values[0] if not p2_match.empty else p2
                    
                    bought_together_list.append({
                        "produk_1": p1,
                        "nama_produk_1": p1_name,
                        "produk_2": p2,
                        "nama_produk_2": p2_name,
                        "frekuensi_hari_bersamaan": int(count)
                    })
                    if len(bought_together_list) >= 3:
                        break
                    
        # Kembalikan response JSON
        return jsonify({
            "status": "success",
            "target_product_id": target_product,
            "bought_together": bought_together_list
        }), 200

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
























STRATEGI_PROMOSI = {
    "Pelanggan VIP": [ 
        "Akses prioritas untuk produk kemasan baru SOPRA",
        "Diskon khusus pengiriman volume besar (Grosir)",
        "Dukungan Account Manager Khusus (B2B Priority)"
    ],
    "Pelanggan Aktif": [
        "Voucher potongan ongkir untuk pembelian bulan ini",
        "Penawaran bundle (Cross-selling) seperti Kardus + Lakban",
        "Undangan bergabung ke Program Poin Reward"
    ],
    "Berisiko Hilang": [
        "Kirim email 'We Miss You' dengan diskon spesial 20%",
        "Kirimkan survei singkat evaluasi kepuasan pelanggan",
        "Rekomendasikan ulang produk Best-Seller yang pernah dibeli"
    ],
    "Pelanggan Baru": [
        "Kupon Welcome 10% untuk transaksi kedua",
        "Kirimkan e-katalog kemasan ramah lingkungan SOPRA",
        "Follow-up WhatsApp H+3 menanyakan kualitas pengiriman"
    ]
}

def generate_segmentation_data():
    """Helper function untuk menghitung segmentasi semua pelanggan"""
    # Hitung Recency (Terakhir Beli) & Frequency (Jumlah Transaksi) dari data transaksi
    rfm = df_transaksi.groupby('customer_id').agg(
        terakhir_beli=('tanggal', 'max'),
        frekuensi=('qty', 'count')  # Menghitung jumlah kali transaksi
    ).reset_index()
    
    # Gabungkan profil pelanggan dengan data RFM
    # Menggunakan how='left' agar pelanggan yang belum pernah transaksi tetap masuk
    df_segmen = pd.merge(
        df_pelanggan, 
        rfm, 
        on='customer_id', 
        how='left'
    )
    
    # Referensi waktu saat ini untuk menghitung jarak hari
    current_date = df_transaksi['tanggal'].max() + pd.Timedelta(days=1)
    hasil_segmentasi = []
    
    for _, row in df_segmen.iterrows():
        segmen_nama = "Pelanggan Baru"
        
        if pd.notnull(row['frekuensi']) and row['frekuensi'] > 0:
            hari_sejak_beli = (current_date - row['terakhir_beli_x']).days
            
            # Logika Aturan Bisnis (Business Rules)
            if row['frekuensi'] >= 5 and hari_sejak_beli <= 30:
                segmen_nama = "Pelanggan VIP"
            elif hari_sejak_beli > 30:
                segmen_nama = "Berisiko Hilang"
            else:
                segmen_nama = "Pelanggan Aktif"
                
        # Format tanggal untuk response JSON agar rapi
        tgl_terakhir = format_tanggal_indonesia(row['terakhir_beli_x']) if pd.notnull(row['terakhir_beli_x']) else "Belum ada transaksi"
        
        # Susun dictionary untuk setiap pelanggan
        customer_data = {
            "customer_id": row['customer_id'],
            "nama_pelanggan": row.get('nama_pelanggan', 'Tanpa Nama'),
            "total_frekuensi_belanja": int(row['frekuensi']) if pd.notnull(row['frekuensi']) else 0,
            "terakhir_beli": tgl_terakhir,
            "segmen": segmen_nama,
            "strategi_promosi": STRATEGI_PROMOSI[segmen_nama] # Memasukkan Array Strategi
        }
        hasil_segmentasi.append(customer_data)
        
    return hasil_segmentasi


# =====================================================================
# GET ALL CUSTOMER SEGMENTS
# =====================================================================
@app.route('/api/segments', methods=['GET'])
def get_all_segments():
    try:
        semua_segmen = generate_segmentation_data()
        
        # Opsional: Hitung statistik ringkas untuk dashboard
        statistik = {
            "Pelanggan VIP": sum(1 for c in semua_segmen if c["segmen"] == "Pelanggan VIP"),
            "Pelanggan Aktif": sum(1 for c in semua_segmen if c["segmen"] == "Pelanggan Aktif"),
            "Berisiko Hilang": sum(1 for c in semua_segmen if c["segmen"] == "Berisiko Hilang"),
            "Pelanggan Baru": sum(1 for c in semua_segmen if c["segmen"] == "Pelanggan Baru")
        }
        
        return jsonify({
            "status": "success",
            "ringkasan_populasi": statistik,
            "total_data": len(semua_segmen),
            "data": semua_segmen
        }), 200
        
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# =====================================================================
# ENDPOINT 2: GET ONE CUSTOMER SEGMENT (By Customer ID)
# =====================================================================
@app.route('/api/segments/<customer_id>', methods=['GET'])
def get_one_segment(customer_id):
    try:
        semua_segmen = generate_segmentation_data()
        
        # Cari satu pelanggan secara spesifik dari list
        pelanggan_ditemukan = next((cust for cust in semua_segmen if cust["customer_id"] == customer_id), None)
        
        if pelanggan_ditemukan:
            return jsonify({
                "status": "success",
                "data": pelanggan_ditemukan
            }), 200
        else:
            return jsonify({
                "status": "error", 
                "message": f"Pelanggan dengan ID '{customer_id}' tidak ditemukan"
            }), 404
            
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500













# Jalankan server
if __name__ == '__main__':
    # Mode debug ON mempermudah pencarian error saat development
    app.run(debug=True, port=5000)