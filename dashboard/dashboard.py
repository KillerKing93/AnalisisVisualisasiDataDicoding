import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import folium
from streamlit_folium import st_folium
import os

# -----------------------------------------------------------------------------------
# Deskripsi:
# Dashboard interaktif berbasis Streamlit untuk menganalisis data kualitas udara
# dari beberapa stasiun di Tionghoa. Fitur yang tersedia:
# - Peta geospasial konsentrasi PM2.5 dengan Folium.
# - Visualisasi lengkap yang meliputi:
#    a. Statistik Stasiun
#    b. Trend & Hubungan (termasuk trend bulanan, harian, dan hubungan dengan faktor meteorologi)
#    c. Perbandingan Antar Stasiun
#    d. Distribusi Waktu (per jam, hari dalam seminggu, dan perbandingan hari kerja vs. akhir pekan)
# -----------------------------------------------------------------------------------

# -----------------------------------------------------------------------------------
# 1. Pemuatan Data dan Persiapan
# -----------------------------------------------------------------------------------
@st.cache_data
def load_data():
    """Memuat dan mempersiapkan data dari file CSV."""
    data = pd.read_csv('./dashboard/PSRA_Data_SemuaStasiun.csv')
    data.rename(columns={'datetime': 'tanggal', 'station': 'stasiun'}, inplace=True)
    data['tanggal'] = pd.to_datetime(data['tanggal'])
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
    return data

data = load_data()

# Pisahkan data per stasiun ke dalam dictionary (digunakan untuk filter awal)
station_data_dict = {station: group for station, group in data.groupby('stasiun')}

# -----------------------------------------------------------------------------------
# Fungsi Pre-Aggregasi Data
# -----------------------------------------------------------------------------------
@st.cache_data
def aggregate_station(filtered_df):
    """Menghitung rata-rata PM2.5 per stasiun beserta koordinat (untuk peta)."""
    return filtered_df.groupby('stasiun').agg({
        'PM2.5': 'mean',
        'latitude': 'first',
        'longitude': 'first'
    }).reset_index()

@st.cache_data
def aggregate_daily(filtered_df):
    """Menghitung rata-rata harian PM2.5 dan PM10 per stasiun (untuk perbandingan antar stasiun)."""
    return filtered_df.groupby(['stasiun', 'tanggal']).agg({
        'PM2.5': 'mean',
        'PM10': 'mean'
    }).reset_index()

# -----------------------------------------------------------------------------------
# 2. Sidebar: Pilihan Filter Data
# -----------------------------------------------------------------------------------
cols = st.sidebar.columns([1, 2, 1])
cols[1].image("./images/station.png", width=110)

st.sidebar.title("Kualitas Udara di Beberapa Stasiun di Tionghoa")
selected_station = st.sidebar.selectbox("Pilih Stasiun", data['stasiun'].unique())
tanggal_mulai = st.sidebar.date_input("Tanggal Mulai", value=data['tanggal'].min())
tanggal_akhir = st.sidebar.date_input("Tanggal Akhir", value=data['tanggal'].max())

# Ambil data stasiun yang dipilih (sebelum filtering tanggal)
selected_station_data = station_data_dict[selected_station]

# Filter data berdasarkan tanggal untuk stasiun yang dipilih
filtered_data = selected_station_data[
    (selected_station_data['tanggal'] >= pd.to_datetime(tanggal_mulai)) &
    (selected_station_data['tanggal'] <= pd.to_datetime(tanggal_akhir))
].copy()

# Filter data berdasarkan tanggal (untuk perbandingan antar stasiun & visualisasi seluruh data)
date_filtered = data[
    (data['tanggal'] >= pd.to_datetime(tanggal_mulai)) &
    (data['tanggal'] <= pd.to_datetime(tanggal_akhir))
]

# Pre-aggregate data untuk peta dan perbandingan
station_avg = aggregate_station(date_filtered)
daily_avg_all = aggregate_daily(date_filtered)

st.sidebar.write("Peringatan! Hanya dapat bekerja jika venv dieksekusi di root!")
st.sidebar.write("Dibuat oleh: Alif Nurhidayat\nEmail: alifnurhidayatwork@gmail.com")

# -----------------------------------------------------------------------------------
# 3. Peta Geospasial Konsentrasi PM2.5
# -----------------------------------------------------------------------------------
st.subheader("Peta Lokasi Stasiun")
# Mendapatkan koordinat pusat dari stasiun yang dipilih
center_lat, center_lon = station_data_dict[selected_station][['latitude', 'longitude']].iloc[0]
m = folium.Map(location=[center_lat, center_lon], zoom_start=10)

# Gunakan data teragregasi untuk setiap stasiun
for idx, row in station_avg.iterrows():
    stasiun_name = row['stasiun']
    avg_pm25 = row['PM2.5']
    lat, lon = row['latitude'], row['longitude']
    color = 'orange' if stasiun_name == selected_station else 'red'
    folium.CircleMarker(
        location=[lat, lon],
        radius=avg_pm25 / 10,
        popup=f"{stasiun_name}: {avg_pm25:.2f} µg/m³",
        color=color,
        fill=True,
        fill_color=color
    ).add_to(m)
st_folium(m, width=700, height=500)

# -----------------------------------------------------------------------------------
# 4. Tab Visualisasi
# -----------------------------------------------------------------------------------
tabs = st.tabs(["Statistik Stasiun", "Tren & Hubungan", "Perbandingan Antar Stasiun", "Distribusi Waktu"])

# ===============================================================================
# Tab 1: Statistik Stasiun
# ===============================================================================
with tabs[0]:
    st.subheader(f"Statistik untuk Stasiun: {selected_station}")
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Proporsi Rata-rata PM2.5 & PM10**")
        avg_pm = filtered_data[['PM2.5', 'PM10']].mean()
        fig, ax = plt.subplots()
        ax.pie(avg_pm, labels=avg_pm.index, autopct='%1.1f%%', startangle=90)
        ax.axis('equal')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    
    with col2:
        st.write("**Proporsi Rata-rata Polutan Lainnya**")
        pollutants = ['NO2', 'SO2', 'CO', 'O3']
        avg_pollutants = filtered_data[pollutants].mean()
        fig, ax = plt.subplots()
        ax.pie(avg_pollutants, labels=avg_pollutants.index, autopct='%1.1f%%', startangle=90)
        ax.axis('equal')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    
    stats_pm = pd.concat([
        filtered_data['PM2.5'].describe().rename('PM2.5'),
        filtered_data['PM10'].describe().rename('PM10')
    ], axis=1)
    
    st.subheader("Heatmap Korelasi Polutan")
    pollutants_all = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3']
    corr = filtered_data[pollutants_all].corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    
    st.write("**Statistik Deskriptif PM2.5 & PM10**")
    st.dataframe(stats_pm)
    
    stats_pollutants = filtered_data[pollutants].describe().T
    
    st.subheader("Facet Plot Tren Polutan")
    df_melt = filtered_data[['tanggal'] + pollutants_all].melt(id_vars='tanggal', var_name='Pollutant', value_name='Concentration')
    g = sns.FacetGrid(df_melt, col="Pollutant", col_wrap=3, height=3, sharex=True, sharey=False)
    g.map(sns.lineplot, "tanggal", "Concentration")
    g.set_axis_labels("Tanggal", "Konsentrasi")
    for ax in g.axes.flat:
        ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    st.pyplot(g.fig)
    plt.close(g.fig)
    
    st.write("**Statistik Deskriptif Polutan Lainnya**")
    st.dataframe(stats_pollutants)

# ===============================================================================
# Tab 2: Tren & Hubungan
# ===============================================================================
with tabs[1]:
    option = st.selectbox("Pilih jenis visualisasi", 
                          ["Trend Bulanan", "Tren Konsentrasi PM2.5 & Hubungan dengan Variabel Meteorologi", "Tren Konsentrasi PM10 & Hubungan dengan Variabel Meteorologi", "Tren Konsentrasi PM2.5 & PM10 Beserta Hubungan dengan Variabel Meteorologi", "Dampak Meteorologi vs Polutan Lainnya"])
    
    if option == "Trend Bulanan":
        st.subheader("Trend Bulanan Konsentrasi PM2.5 dan PM10 (Semua Stasiun)")
        df_temp = date_filtered.copy()
        df_temp.set_index('tanggal', inplace=True)
        trend_pm = df_temp.groupby('stasiun')[['PM2.5', 'PM10']].resample('ME').mean().reset_index()
        
        # Plot trend bulanan PM2.5
        fig, ax = plt.subplots(figsize=(14, 6))
        sns.lineplot(data=trend_pm, x='tanggal', y='PM2.5', hue='stasiun', marker='o', ax=ax)
        ax.set_title('Trend Rata-rata Bulanan PM2.5 per Stasiun')
        ax.set_xlabel('Bulan')
        ax.set_ylabel('Konsentrasi PM2.5 (µg/m³)')
        ax.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        
        # Plot trend bulanan PM10
        fig, ax = plt.subplots(figsize=(14, 6))
        sns.lineplot(data=trend_pm, x='tanggal', y='PM10', hue='stasiun', marker='o', ax=ax)
        ax.set_title('Trend Rata-rata Bulanan PM10 per Stasiun')
        ax.set_xlabel('Bulan')
        ax.set_ylabel('Konsentrasi PM10 (µg/m³)')
        ax.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    
    elif option == "Tren Konsentrasi PM2.5 & Hubungan dengan Variabel Meteorologi":
        st.subheader("Tren Konsentrasi PM2.5")
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.lineplot(x='tanggal', y='PM2.5', data=filtered_data, ax=ax, label='PM2.5', color='blue')
        ax.set_xlabel("Tanggal")
        ax.set_ylabel("Konsentrasi PM2.5 (µg/m³)")
        ax.legend(loc='upper right')
        ax.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        
        st.subheader("Distribusi Konsentrasi PM2.5")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(filtered_data['PM2.5'], bins=30, kde=True, ax=ax, color='blue')
        ax.set_xlabel("Konsentrasi PM2.5 (µg/m³)")
        ax.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        
        st.subheader("Hubungan PM2.5 dengan Variabel Meteorologi")
        met_vars = ['TEMP', 'DEWP', 'PRES', 'WSPM']
        n_cols = len(met_vars)
        fig, axes = plt.subplots(1, n_cols, figsize=(4*n_cols, 4))
        for i, var in enumerate(met_vars):
            sns.scatterplot(x=var, y='PM2.5', data=filtered_data, ax=axes[i], color='blue')
            axes[i].set_title(f"{var} vs PM2.5")
            axes[i].tick_params(axis='x', rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    
    elif option == "Tren Konsentrasi PM10 & Hubungan dengan Variabel Meteorologi":
        st.subheader("Tren Konsentrasi PM10")
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.lineplot(x='tanggal', y='PM10', data=filtered_data, ax=ax, label='PM10', color='green')
        ax.set_xlabel("Tanggal")
        ax.set_ylabel("Konsentrasi PM10 (µg/m³)")
        ax.legend(loc='upper right')
        ax.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        
        st.subheader("Distribusi Konsentrasi PM10")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(filtered_data['PM10'], bins=30, kde=True, ax=ax, color='green')
        ax.set_xlabel("Konsentrasi PM10 (µg/m³)")
        ax.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        
        st.subheader("Hubungan PM10 dengan Variabel Meteorologi")
        met_vars = ['TEMP', 'DEWP', 'PRES', 'WSPM']
        n_cols = len(met_vars)
        fig, axes = plt.subplots(1, n_cols, figsize=(4*n_cols, 4))
        for i, var in enumerate(met_vars):
            sns.scatterplot(x=var, y='PM10', data=filtered_data, ax=axes[i], color='green')
            axes[i].set_title(f"{var} vs PM10")
            axes[i].tick_params(axis='x', rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    
    elif option == "Tren Konsentrasi PM2.5 & PM10 Beserta Hubungan dengan Variabel Meteorologi":
        st.subheader("Tren Konsentrasi PM2.5 dan PM10")
        daily_avg = daily_avg_all[['stasiun', 'tanggal', 'PM2.5']]
        ylabel = "Rata-rata PM2.5 (µg/m³) (PM10 tidak ditampilkan)"
            
        fig, ax = plt.subplots(figsize=(10, 5))
        for stn in daily_avg['stasiun'].unique():
            color = 'orange' if stn == selected_station else 'lightblue'
            subset = daily_avg[daily_avg['stasiun'] == stn]
            ax.plot(subset['tanggal'], subset['PM2.5'], label=stn, color=color, alpha=0.8)
        ax.set_xlabel("Tanggal")
        ax.set_ylabel(ylabel)
        ax.legend(loc='upper right')
        ax.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        
        st.subheader("Distribusi Konsentrasi PM2.5 dan PM10")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(filtered_data['PM2.5'], bins=30, kde=True, ax=ax, color='blue', label='PM2.5', alpha=0.5)
        sns.histplot(filtered_data['PM10'], bins=30, kde=True, ax=ax, color='green', label='PM10', alpha=0.5)
        ax.set_xlabel("Konsentrasi (µg/m³)")
        ax.legend(loc='upper right')
        ax.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        
        st.subheader("Hubungan Variabel Meteorologi dengan PM2.5 dan PM10")
        met_vars = ['TEMP', 'DEWP', 'PRES', 'WSPM']
        n_cols = len(met_vars)
        fig, axes = plt.subplots(2, n_cols, figsize=(4*n_cols, 8))
        for i, pol in enumerate(['PM2.5', 'PM10']):
            for j, var in enumerate(met_vars):
                color = 'blue' if pol == 'PM2.5' else 'green'
                sns.scatterplot(x=var, y=pol, data=filtered_data, ax=axes[i, j], color=color)
                axes[i, j].set_title(f"{var} vs {pol}")
                axes[i, j].tick_params(axis='x', rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    
    elif option == "Dampak Meteorologi vs Polutan Lainnya":
        st.subheader("Hubungan Variabel Meteorologi vs Polutan Lainnya")
        met_vars = ['TEMP', 'DEWP', 'PRES', 'WSPM']
        pollutants_other = ['NO2', 'SO2', 'CO', 'O3']
        n_rows = len(met_vars)
        n_cols = len(pollutants_other)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
        for i, met in enumerate(met_vars):
            for j, pol in enumerate(pollutants_other):
                sns.scatterplot(x=met, y=pol, data=filtered_data, ax=axes[i, j])
                axes[i, j].set_title(f"{met} vs {pol}")
                axes[i, j].tick_params(axis='x', rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

# ===============================================================================
# Tab 3: Perbandingan Antar Stasiun
# ===============================================================================
with tabs[2]:
    compare_option = st.selectbox("Pilih variabel untuk perbandingan antar stasiun", 
                                  ["PM2.5", "PM10", "Meteorologi", "Polutan Lainnya", "Keduanya (PM & Lainnya)"])
    
    st.write("**Tren Harian Rata-rata (Line Plot)**")
    if compare_option in ["PM2.5", "PM10", "Keduanya (PM & Lainnya)"]:
        if compare_option == "PM2.5":
            daily_avg = daily_avg_all[['stasiun', 'tanggal', 'PM2.5']]
            ylabel = "Rata-rata PM2.5 (µg/m³)"
        elif compare_option == "PM10":
            daily_avg = daily_avg_all[['stasiun', 'tanggal', 'PM10']]
            ylabel = "Rata-rata PM10 (µg/m³)"
        else:
            daily_avg = daily_avg_all[['stasiun', 'tanggal', 'PM2.5']]
            ylabel = "Rata-rata PM2.5 (µg/m³) (PM10 tidak ditampilkan)"
            
        fig, ax = plt.subplots(figsize=(10, 5))
        for stn in daily_avg['stasiun'].unique():
            color = 'orange' if stn == selected_station else 'lightblue'
            subset = daily_avg[daily_avg['stasiun'] == stn]
            ax.plot(subset['tanggal'], subset.iloc[:, 2], label=stn, color=color, alpha=0.8)
        ax.set_xlabel("Tanggal")
        ax.set_ylabel(ylabel)
        ax.legend(loc='upper right')
        ax.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    
    palette = {stn: ('orange' if stn == selected_station else 'lightblue') 
               for stn in date_filtered['stasiun'].unique()}
    
    if compare_option in ["PM2.5", "PM10", "Keduanya (PM & Lainnya)"]:
        if compare_option in ["PM2.5", "Keduanya (PM & Lainnya)"]:
            st.write("**Distribusi PM2.5**")
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.boxplot(x='stasiun', y='PM2.5', hue='stasiun', data=date_filtered, palette=palette, ax=ax, legend=False)
            ax.tick_params(axis='x', rotation=45)
            ax.set_xlabel("Stasiun")
            ax.set_ylabel("Konsentrasi PM2.5 (µg/m³)")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        if compare_option in ["PM10", "Keduanya (PM & Lainnya)"]:
            st.write("**Distribusi PM10**")
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.boxplot(x='stasiun', y='PM10', hue='stasiun', data=date_filtered, palette=palette, ax=ax, legend=False)
            ax.tick_params(axis='x', rotation=45)
            ax.set_xlabel("Stasiun")
            ax.set_ylabel("Konsentrasi PM10 (µg/m³)")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
    
    if compare_option in ["Meteorologi", "Keduanya (PM & Lainnya)"]:
        st.subheader("Perbandingan Meteorologi Antar Stasiun")
        met_vars = ['TEMP', 'DEWP', 'PRES', 'WSPM']
        fig, axes = plt.subplots(1, len(met_vars), figsize=(5*len(met_vars), 5))
        for i, var in enumerate(met_vars):
            sns.boxplot(x='stasiun', y=var, hue='stasiun', data=date_filtered, palette=palette, ax=axes[i], legend=False)
            axes[i].set_title(f"{var} per Stasiun")
            axes[i].tick_params(axis='x', rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    
    if compare_option in ["Polutan Lainnya", "Keduanya (PM & Lainnya)"]:
        st.subheader("Perbandingan Polutan Lainnya Antar Stasiun")
        pollutants_other = ['NO2', 'SO2', 'CO', 'O3']
        fig, axes = plt.subplots(1, len(pollutants_other), figsize=(5*len(pollutants_other), 5))
        for i, pol in enumerate(pollutants_other):
            sns.boxplot(x='stasiun', y=pol, hue='stasiun', data=date_filtered, palette=palette, ax=axes[i], legend=False)
            axes[i].set_title(f"{pol} per Stasiun")
            axes[i].tick_params(axis='x', rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

# ===============================================================================
# Tab 4: Distribusi Waktu
# ===============================================================================
with tabs[3]:
    st.subheader("Distribusi Berdasarkan Waktu")
    
    # Pilih variabel yang ingin divisualisasikan
    variable_choice = st.selectbox(
        "Pilih Variabel",
        ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3', 'TEMP', 'PRES', 'DEWP', 'WSPM']
    )
    
    # Pastikan kolom waktu sudah ada pada filtered_data
    filtered_data['hour'] = filtered_data['tanggal'].dt.hour
    filtered_data['day_of_week'] = filtered_data['tanggal'].dt.day_name()
    filtered_data['hari'] = filtered_data['tanggal'].dt.dayofweek
    filtered_data['jenis_hari'] = filtered_data['hari'].apply(lambda x: 'Akhir Pekan' if x >= 5 else 'Hari Kerja')
    
    # 1. Boxplot variabel per Jam
    st.write(f"**Boxplot {variable_choice} per Jam**")
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.boxplot(data=filtered_data, x='hour', y=variable_choice, ax=ax, palette='Set3')
    ax.set_xlabel("Jam")
    ax.set_ylabel(variable_choice)
    ax.set_title(f"Distribusi {variable_choice} per Jam")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    
    # 2. Boxplot variabel per Hari dalam Seminggu
    st.write(f"**Boxplot {variable_choice} per Hari (Day of Week)**")
    order_days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.boxplot(data=filtered_data, x='day_of_week', y=variable_choice, order=order_days, ax=ax, palette='Set2')
    ax.set_xlabel("Hari")
    ax.set_ylabel(variable_choice)
    ax.set_title(f"Distribusi {variable_choice} per Hari dalam Seminggu")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    
    # 3. Boxplot perbandingan: Hari Kerja vs Akhir Pekan
    st.write(f"**Boxplot {variable_choice}: Hari Kerja vs Akhir Pekan**")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=filtered_data, x='jenis_hari', y=variable_choice, 
                palette={'Hari Kerja': 'lightblue', 'Akhir Pekan': 'orange'}, ax=ax)
    ax.set_xlabel("Jenis Hari")
    ax.set_ylabel(variable_choice)
    ax.set_title(f"Perbandingan {variable_choice}: Hari Kerja vs Akhir Pekan")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    
    # 4. Bar Chart: Median nilai variabel berdasarkan Hari Kerja vs Akhir Pekan
    st.write(f"**Bar Chart Median {variable_choice}: Hari Kerja vs Akhir Pekan**")
    median_var = filtered_data.groupby('jenis_hari')[variable_choice].median().reset_index()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(data=median_var, x='jenis_hari', y=variable_choice, 
                palette={'Hari Kerja': 'lightblue', 'Akhir Pekan': 'orange'}, ax=ax)
    ax.set_xlabel("Jenis Hari")
    ax.set_ylabel(f"Median {variable_choice}")
    ax.set_title(f"Median {variable_choice}: Hari Kerja vs Akhir Pekan")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
