import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import folium
from streamlit_folium import folium_static
import os

# -----------------------------------------------------------------------------------
# Deskripsi:
# Dashboard interaktif berbasis Streamlit untuk analisis data kualitas udara dari
# beberapa stasiun di Tionghoa. Fitur:
# - Peta geospasial konsentrasi PM2.5 menggunakan Folium.
# - Tab visualisasi: Statistik Stasiun, Tren & Hubungan, dan Perbandingan Antar Stasiun.
# - Filter data berdasarkan stasiun dan rentang tanggal.
# -----------------------------------------------------------------------------------

# -----------------------------------------------------------------------------------
# Fungsi-fungsi dengan caching untuk optimisasi
# -----------------------------------------------------------------------------------
@st.cache_data
def load_data(filepath):
    data = pd.read_csv(filepath)
    data.rename(columns={'datetime': 'tanggal', 'station': 'stasiun'}, inplace=True)
    data['tanggal'] = pd.to_datetime(data['tanggal'])
    return data

@st.cache_data
def filter_data(data, start_date, end_date):
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    return data[(data['tanggal'] >= start_date) & (data['tanggal'] <= end_date)]

@st.cache_data
def get_sample(data, n=1000):
    return data.sample(n, random_state=42)

@st.cache_data
def get_stations_filtered(data, start_date, end_date):
    filtered = filter_data(data, start_date, end_date)
    stations_filtered = {st: group for st, group in filtered.groupby('stasiun')}
    return filtered, stations_filtered

@st.cache_data
def precompute_metrics(data):
    """Lakukan pre-kalkulasi pada dataset penuh untuk tiap stasiun.
       Hasil pre-kalkulasi hanya digunakan apabila rentang tanggal tidak diubah."""
    stats_dict = {}
    for station in data['stasiun'].unique():
        station_data = data[data['stasiun'] == station]
        stats_dict[station] = {
            'avg_pm25': station_data['PM2.5'].mean(),
            'avg_pm10': station_data['PM10'].mean(),
            'descriptive': pd.concat([
                station_data['PM2.5'].describe().rename('PM2.5'),
                station_data['PM10'].describe().rename('PM10')
            ], axis=1),
            'pollutants': station_data[['NO2', 'SO2', 'CO', 'O3']].mean()
        }
    return stats_dict

# -----------------------------------------------------------------------------------
# 1. Pemuatan Data dan Persiapan
# -----------------------------------------------------------------------------------
data = load_data('./dashboard/PSRA_Data_SemuaStasiun.csv')

# Menambahkan koordinat stasiun
coords = {
    'Aotizhongxin': (25.982070730427132, 117.45514627560887),
    'Changping': (40.220773648721526, 116.22458346928182),
    'Dingling': (40.29608173026575, 116.22351026646444),
    'Dongsi': (39.93205741673973, 116.43419736174938),
    'Guanyuan': (32.44985723335085, 105.87975278857984),
    'Gucheng': (26.86786447702828, 100.27761232290194),
    'Huairou': (40.30923966919258, 116.66964440378989),
    'Nongzhanguan': (39.93562032913985, 116.46753727846819),
    'Shunyi': (40.15491661940465, 116.54192610220052),
    'Tiantan': (39.89078817042267, 116.39886221016597),
    'Wanliu': (39.99950871850554, 116.25689305717239),
    'Wanshouxigong': (39.909282001378656, 116.26337498325312)
}
data['latitude'] = data['stasiun'].map(lambda x: coords[x][0])
data['longitude'] = data['stasiun'].map(lambda x: coords[x][1])

# Lakukan pre-kalkulasi untuk data penuh dan simpan di session_state (hanya sekali saat startup)
if 'precomputed_metrics' not in st.session_state:
    st.session_state['precomputed_metrics'] = precompute_metrics(data)

# -----------------------------------------------------------------------------------
# 2. Sidebar: Pilihan Filter Data
# -----------------------------------------------------------------------------------
cols = st.sidebar.columns([1, 2, 1])
cols[1].image("./images/station.png", width=110)

st.sidebar.title("Kualitas Udara di Beberapa Stasiun di Tionghoa")
selected_station = st.sidebar.selectbox("Pilih Stasiun", data['stasiun'].unique())
tanggal_mulai = st.sidebar.date_input("Tanggal Mulai", value=data['tanggal'].min())
tanggal_akhir = st.sidebar.date_input("Tanggal Akhir", value=data['tanggal'].max())

# Tentukan apakah pengguna menggunakan rentang tanggal penuh
full_range = (tanggal_mulai == data['tanggal'].min().date()) and (tanggal_akhir == data['tanggal'].max().date())

# Filter data dan buat dictionary per stasiun (menggunakan caching)
date_filtered, stations_filtered = get_stations_filtered(data, tanggal_mulai, tanggal_akhir)
selected_data = stations_filtered.get(selected_station, pd.DataFrame()).copy()

st.sidebar.write("Peringatan! hanya dapat bekerja jika venv dieksekusi di root!")
st.sidebar.write("Dibuat oleh: Alif Nurhidayat\nEmail: alifnurhidayatwork@gmail.com")

# -----------------------------------------------------------------------------------
# 3. Peta Geospasial Konsentrasi PM2.5
# -----------------------------------------------------------------------------------
st.subheader("Peta Lokasi Stasiun")
center_lat, center_lon = coords[selected_station]
m = folium.Map(location=[center_lat, center_lon], zoom_start=10)
for stasiun_name, group in date_filtered.groupby('stasiun'):
    avg_pm25 = group['PM2.5'].mean()
    lat, lon = coords[stasiun_name]
    color = 'orange' if stasiun_name == selected_station else 'red'
    folium.CircleMarker(
        location=[lat, lon],
        radius=avg_pm25 / 10,
        popup=f"{stasiun_name}: {avg_pm25:.2f} µg/m³",
        color=color,
        fill=True,
        fill_color=color
    ).add_to(m)
folium_static(m)

# -----------------------------------------------------------------------------------
# 4. Tab Visualisasi
# -----------------------------------------------------------------------------------
tabs = st.tabs(["Statistik Stasiun", "Tren & Hubungan", "Perbandingan Antar Stasiun"])

# ===============================================================================
# Tab 1: Statistik Stasiun
# ===============================================================================
with tabs[0]:
    st.subheader(f"Statistik untuk Stasiun: {selected_station}")
    
    # Gunakan pre-kalkulasi bila rentang tanggal penuh, jika tidak hitung ulang dari filter
    if full_range:
        pre_stats = st.session_state['precomputed_metrics'][selected_station]
        avg_pm = pd.Series({'PM2.5': pre_stats['avg_pm25'], 'PM10': pre_stats['avg_pm10']})
        stats_pm = pre_stats['descriptive']
        avg_pollutants = pre_stats['pollutants']
    else:
        avg_pm = selected_data[['PM2.5', 'PM10']].mean()
        stats_pm = pd.concat([
            selected_data['PM2.5'].describe().rename('PM2.5'),
            selected_data['PM10'].describe().rename('PM10')
        ], axis=1)
        avg_pollutants = selected_data[['NO2', 'SO2', 'CO', 'O3']].mean()
    
    # Pie Chart PM2.5 & PM10
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Proporsi Rata-rata PM2.5 & PM10**")
        fig, ax = plt.subplots()
        ax.pie(avg_pm, labels=avg_pm.index, autopct='%1.1f%%', startangle=90)
        ax.axis('equal')
        plt.tight_layout()
        st.pyplot(fig)
    with col2:
        st.write("**Proporsi Rata-rata Polutan Lainnya**")
        fig, ax = plt.subplots()
        ax.pie(avg_pollutants, labels=avg_pollutants.index, autopct='%1.1f%%', startangle=90)
        ax.axis('equal')
        plt.tight_layout()
        st.pyplot(fig)
    
    st.write("**Statistik Deskriptif PM2.5 & PM10**")
    st.dataframe(stats_pm)
    
    stats_pollutants = selected_data[['NO2', 'SO2', 'CO', 'O3']].describe().T
    
    # Facet Plot tren polutan
    st.subheader("Facet Plot Tren Polutan")
    pollutants_all = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3']
    df_melt = selected_data[['tanggal'] + pollutants_all].melt(id_vars='tanggal', var_name='Pollutant', value_name='Concentration')
    g = sns.FacetGrid(df_melt, col="Pollutant", col_wrap=3, height=3, sharex=True, sharey=False)
    g.map(sns.lineplot, "tanggal", "Concentration")
    g.set_axis_labels("Tanggal", "Konsentrasi")
    for ax in g.axes.flat:
        ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    st.pyplot(g.fig)
    
    # Visualisasi hubungan meteorologi dan polutan (dengan sample data)
    st.subheader("Visualisasi Hubungan Meteorologi dan Polutan")
    sample_df = get_sample(data)
    cols_corr = ['TEMP', 'PRES', 'DEWP', 'WSPM', 'NO2', 'SO2', 'CO']
    pairgrid = sns.pairplot(sample_df[cols_corr])
    pairgrid.fig.suptitle('Pairplot: Faktor Meteorologi vs Polutan (Sample 1000 Baris)', y=1.02)
    for ax in pairgrid.axes.flatten():
        ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    st.pyplot(pairgrid.fig)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    corr_matrix = sample_df[cols_corr].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
    ax.set_title('Matriks Korelasi: Meteorologi vs Polutan')
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    st.pyplot(fig)
    
    st.write("**Statistik Deskriptif Polutan Lainnya**")
    st.dataframe(stats_pollutants)
    
    # Distribusi polutan berdasarkan jenis hari
    st.subheader("Pilihan Distribusi Polutan")
    pollutant = st.selectbox("Pilih Polutan", ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3'])
    plot_type = st.selectbox("Pilih Jenis Plot", 
                             ['Box Plot', 'Bar Plot', 'Line Plot', 'Heatmap', 'Pola Musiman', 'Pola Harian'])
    
    # Tambahkan kolom jenis hari (Hari Kerja vs Akhir Pekan)
    selected_data['hari'] = selected_data['tanggal'].dt.dayofweek
    selected_data['jenis_hari'] = selected_data['hari'].apply(lambda x: 'Akhir Pekan' if x >= 5 else 'Hari Kerja')
    
    if plot_type == 'Box Plot':
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.boxplot(x='jenis_hari', y=pollutant, data=selected_data, 
                    palette={'Hari Kerja': 'lightblue', 'Akhir Pekan': 'orange'}, ax=ax)
        ax.set_xlabel("Jenis Hari")
        ax.set_ylabel(f"Konsentrasi {pollutant} (µg/m³)")
        ax.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
    elif plot_type == 'Bar Plot':
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(x='jenis_hari', y=pollutant, data=selected_data, 
                    palette={'Hari Kerja': 'lightblue', 'Akhir Pekan': 'orange'}, ax=ax)
        ax.set_xlabel("Jenis Hari")
        ax.set_ylabel(f"Rata-rata {pollutant} (µg/m³)")
        ax.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
    elif plot_type == 'Line Plot':
        st.subheader(f"Tren Konsentrasi {pollutant} Seiring Waktu")
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.lineplot(x='tanggal', y=pollutant, hue='jenis_hari', data=selected_data, 
                     palette={'Hari Kerja': 'lightblue', 'Akhir Pekan': 'orange'}, ax=ax)
        ax.set_xlabel("Tanggal")
        ax.set_ylabel(f"Konsentrasi {pollutant} (µg/m³)")
        ax.legend(title="Jenis Hari")
        ax.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
    elif plot_type == 'Heatmap':
        st.subheader(f"Heatmap Rata-rata {pollutant} per Jam")
        selected_data['jam'] = selected_data['tanggal'].dt.hour
        heatmap_data = selected_data.pivot_table(values=pollutant, index='jam', columns='jenis_hari', aggfunc='mean')
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(heatmap_data, cmap='YlOrRd', annot=True, fmt='.1f', ax=ax)
        ax.set_xlabel("Jenis Hari")
        ax.set_ylabel("Jam")
        ax.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
    elif plot_type == 'Pola Musiman':
        st.subheader(f"Pola Musiman Konsentrasi {pollutant}")
        selected_data['bulan'] = selected_data['tanggal'].dt.month
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.lineplot(x='bulan', y=pollutant, data=selected_data, estimator='mean', ci=None, ax=ax)
        ax.set_xlabel("Bulan")
        ax.set_ylabel(f"Rata-rata {pollutant} (µg/m³)")
        ax.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
    elif plot_type == 'Pola Harian':
        st.subheader(f"Pola Harian Konsentrasi {pollutant}")
        selected_data['jam'] = selected_data['tanggal'].dt.hour
        heatmap_data = selected_data.pivot_table(values=pollutant, index='jam', aggfunc='mean')
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(heatmap_data, cmap='YlOrRd', annot=True, fmt='.1f', ax=ax)
        ax.set_xlabel(f"Konsentrasi Rata-rata {pollutant} (µg/m³)")
        ax.set_ylabel("Jam")
        ax.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        st.pyplot(fig)

# ===============================================================================
# Tab 2: Tren & Hubungan
# ===============================================================================
with tabs[1]:
    option = st.selectbox("Pilih jenis visualisasi", 
                          ["PM2.5", "PM10", "PM2.5 & PM10", "Dampak Meteorologi vs Polutan Lainnya"])
    if option == "PM2.5":
        st.subheader("Tren Konsentrasi PM2.5")
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.lineplot(x='tanggal', y='PM2.5', data=selected_data, ax=ax, label='PM2.5', color='blue')
        ax.set_xlabel("Tanggal")
        ax.set_ylabel("Konsentrasi PM2.5 (µg/m³)")
        ax.legend()
        ax.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
        
        st.subheader("Distribusi Konsentrasi PM2.5")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(selected_data['PM2.5'].dropna(), bins=30, kde=True, ax=ax, color='blue')
        ax.set_xlabel("Konsentrasi PM2.5 (µg/m³)")
        ax.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
        
        st.subheader("Hubungan PM2.5 dengan Variabel Meteorologi")
        met_vars = ['TEMP', 'DEWP', 'PRES', 'WSPM']
        fig, axes = plt.subplots(1, len(met_vars), figsize=(4*len(met_vars), 4))
        for i, var in enumerate(met_vars):
            sns.scatterplot(x=var, y='PM2.5', data=selected_data, ax=axes[i], color='blue')
            axes[i].set_title(f"{var} vs PM2.5")
            axes[i].tick_params(axis='x', rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
    elif option == "PM10":
        st.subheader("Tren Konsentrasi PM10")
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.lineplot(x='tanggal', y='PM10', data=selected_data, ax=ax, label='PM10', color='green')
        ax.set_xlabel("Tanggal")
        ax.set_ylabel("Konsentrasi PM10 (µg/m³)")
        ax.legend()
        ax.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
        
        st.subheader("Distribusi Konsentrasi PM10")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(selected_data['PM10'].dropna(), bins=30, kde=True, ax=ax, color='green')
        ax.set_xlabel("Konsentrasi PM10 (µg/m³)")
        ax.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
        
        st.subheader("Hubungan PM10 dengan Variabel Meteorologi")
        met_vars = ['TEMP', 'DEWP', 'PRES', 'WSPM']
        fig, axes = plt.subplots(1, len(met_vars), figsize=(4*len(met_vars), 4))
        for i, var in enumerate(met_vars):
            sns.scatterplot(x=var, y='PM10', data=selected_data, ax=axes[i], color='green')
            axes[i].set_title(f"{var} vs PM10")
            axes[i].tick_params(axis='x', rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
    elif option == "PM2.5 & PM10":
        st.subheader("Tren Konsentrasi PM2.5 dan PM10")
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.lineplot(x='tanggal', y='PM2.5', data=selected_data, ax=ax, label='PM2.5', color='blue')
        sns.lineplot(x='tanggal', y='PM10', data=selected_data, ax=ax, label='PM10', color='green')
        ax.set_xlabel("Tanggal")
        ax.set_ylabel("Konsentrasi (µg/m³)")
        ax.legend()
        ax.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
        
        st.subheader("Distribusi Konsentrasi PM2.5 dan PM10")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(selected_data['PM2.5'].dropna(), bins=30, kde=True, ax=ax, color='blue', label='PM2.5', alpha=0.5)
        sns.histplot(selected_data['PM10'].dropna(), bins=30, kde=True, ax=ax, color='green', label='PM10', alpha=0.5)
        ax.set_xlabel("Konsentrasi (µg/m³)")
        ax.legend()
        ax.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
        
        st.subheader("Hubungan Variabel Meteorologi dengan PM2.5 dan PM10")
        met_vars = ['TEMP', 'DEWP', 'PRES', 'WSPM']
        fig, axes = plt.subplots(2, len(met_vars), figsize=(4*len(met_vars), 8), sharex=False, sharey=False)
        for i, pol in enumerate(['PM2.5', 'PM10']):
            for j, var in enumerate(met_vars):
                color = 'blue' if pol == 'PM2.5' else 'green'
                sns.scatterplot(x=var, y=pol, data=selected_data, ax=axes[i, j], color=color)
                axes[i, j].set_title(f"{var} vs {pol}")
                axes[i, j].tick_params(axis='x', rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
    elif option == "Dampak Meteorologi vs Polutan Lainnya":
        st.subheader("Hubungan Variabel Meteorologi vs Polutan Lainnya")
        met_vars = ['TEMP', 'DEWP', 'PRES', 'WSPM']
        pollutants_other = ['NO2', 'SO2', 'CO', 'O3']
        fig, axes = plt.subplots(len(met_vars), len(pollutants_other), figsize=(4*len(pollutants_other), 4*len(met_vars)))
        for i, met in enumerate(met_vars):
            for j, pol in enumerate(pollutants_other):
                sns.scatterplot(x=met, y=pol, data=selected_data, ax=axes[i, j])
                axes[i, j].set_title(f"{met} vs {pol}")
                axes[i, j].tick_params(axis='x', rotation=45)
        plt.tight_layout()
        st.pyplot(fig)

# ===============================================================================
# Tab 3: Perbandingan Antar Stasiun
# ===============================================================================
with tabs[2]:
    compare_option = st.selectbox("Pilih variabel untuk perbandingan antar stasiun", 
                                  ["PM2.5", "PM10", "Meteorologi", "Polutan Lainnya", "Keduanya (PM & Lainnya)"])
    
    st.write("**Tren Harian Rata-rata (Line Plot)**")
    if compare_option in ["PM2.5", "PM10", "Keduanya (PM & Lainnya)"]:
        if compare_option == "PM2.5":
            daily_avg = date_filtered.groupby(['stasiun', 'tanggal'])['PM2.5'].mean().reset_index()
            ylabel = "Rata-rata PM2.5 (µg/m³)"
        elif compare_option == "PM10":
            daily_avg = date_filtered.groupby(['stasiun', 'tanggal'])['PM10'].mean().reset_index()
            ylabel = "Rata-rata PM10 (µg/m³)"
        else:
            daily_avg = date_filtered.groupby(['stasiun', 'tanggal'])['PM2.5'].mean().reset_index()
            ylabel = "Rata-rata PM2.5 (µg/m³) (PM10 tidak ditampilkan)"
            
        fig, ax = plt.subplots(figsize=(10, 5))
        for stn in daily_avg['stasiun'].unique():
            color = 'orange' if stn == selected_station else 'lightblue'
            subset = daily_avg[daily_avg['stasiun'] == stn]
            ax.plot(subset['tanggal'], subset.iloc[:, 2], label=stn, color=color, alpha=0.8)
        ax.set_xlabel("Tanggal")
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
    
    # Palet warna untuk perbandingan antar stasiun
    palette = {stn: ('orange' if stn == selected_station else 'lightblue') 
               for stn in date_filtered['stasiun'].unique()}
    
    if compare_option in ["PM2.5", "PM10", "Keduanya (PM & Lainnya)"]:
        if compare_option in ["PM2.5", "Keduanya (PM & Lainnya)"]:
            st.write("**Distribusi PM2.5**")
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.boxplot(x='stasiun', y='PM2.5', data=date_filtered, palette=palette, ax=ax)
            ax.tick_params(axis='x', rotation=45)
            ax.set_xlabel("Stasiun")
            ax.set_ylabel("Konsentrasi PM2.5 (µg/m³)")
            plt.tight_layout()
            st.pyplot(fig)
        if compare_option in ["PM10", "Keduanya (PM & Lainnya)"]:
            st.write("**Distribusi PM10**")
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.boxplot(x='stasiun', y='PM10', data=date_filtered, palette=palette, ax=ax)
            ax.tick_params(axis='x', rotation=45)
            ax.set_xlabel("Stasiun")
            ax.set_ylabel("Konsentrasi PM10 (µg/m³)")
            plt.tight_layout()
            st.pyplot(fig)
    
    if compare_option in ["Meteorologi", "Keduanya (PM & Lainnya)"]:
        st.subheader("Perbandingan Meteorologi Antar Stasiun")
        met_vars = ['TEMP', 'DEWP', 'PRES', 'WSPM']
        fig, axes = plt.subplots(1, len(met_vars), figsize=(5*len(met_vars), 5))
        for i, var in enumerate(met_vars):
            sns.boxplot(x='stasiun', y=var, data=date_filtered, palette=palette, ax=axes[i])
            axes[i].set_title(f"{var} per Stasiun")
            axes[i].tick_params(axis='x', rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
    
    if compare_option in ["Polutan Lainnya", "Keduanya (PM & Lainnya)"]:
        st.subheader("Perbandingan Polutan Lainnya Antar Stasiun")
        pollutants_other = ['NO2', 'SO2', 'CO', 'O3']
        fig, axes = plt.subplots(1, len(pollutants_other), figsize=(5*len(pollutants_other), 5))
        for i, pol in enumerate(pollutants_other):
            sns.boxplot(x='stasiun', y=pol, data=date_filtered, palette=palette, ax=axes[i])
            axes[i].set_title(f"{pol} per Stasiun")
            axes[i].tick_params(axis='x', rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
