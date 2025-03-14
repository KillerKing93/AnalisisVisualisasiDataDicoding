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
# dari beberapa stasiun di Tionghoa, dirancang agar mudah dipahami oleh audiens non-teknis.
# Fitur yang tersedia:
# - Peta geospasial konsentrasi PM2.5 dengan Folium.
# - Visualisasi lengkap yang meliputi:
#    a. Statistik Stasiun
#    b. Tren & Hubungan (tren bulanan, harian, hubungan dengan faktor meteorologi, dan pairplot)
#    c. Perbandingan Antar Stasiun (menggunakan bar chart)
#    d. Distribusi Waktu (menggunakan line chart dan bar chart)
# Warna: Oranye untuk stasiun yang dipilih, Light Blue untuk stasiun lainnya.
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
center_lat, center_lon = station_data_dict[selected_station][['latitude', 'longitude']].iloc[0]
m = folium.Map(location=[center_lat, center_lon], zoom_start=10)

# Plot stasiun lain terlebih dahulu (light blue)
for idx, row in station_avg[station_avg['stasiun'] != selected_station].iterrows():
    stasiun_name = row['stasiun']
    avg_pm25 = row['PM2.5']
    lat, lon = row['latitude'], row['longitude']
    folium.CircleMarker(
        location=[lat, lon],
        radius=avg_pm25 / 10,
        popup=f"{stasiun_name}: {avg_pm25:.2f} µg/m³",
        color='lightblue',
        fill=True,
        fill_color='lightblue'
    ).add_to(m)

# Plot stasiun yang dipilih terakhir agar di atas (oranye)
selected_row = station_avg[station_avg['stasiun'] == selected_station].iloc[0]
folium.CircleMarker(
    location=[selected_row['latitude'], selected_row['longitude']],
    radius=selected_row['PM2.5'] / 10,
    popup=f"{selected_row['stasiun']}: {selected_row['PM2.5']:.2f} µg/m³",
    color='orange',
    fill=True,
    fill_color='orange'
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
        ax.pie(avg_pm, labels=avg_pm.index, autopct='%1.1f%%', startangle=90, colors=['blue', 'green'])
        ax.axis('equal')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    
    with col2:
        st.write("**Proporsi Rata-rata Polutan Lainnya**")
        pollutants = ['NO2', 'SO2', 'CO', 'O3']
        avg_pollutants = filtered_data[pollutants].mean()
        fig, ax = plt.subplots()
        ax.pie(avg_pollutants, labels=avg_pollutants.index, autopct='%1.1f%%', startangle=90, 
               colors=['lightblue', 'lightgreen', 'orange', 'red'])
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
    
    st.subheader("Tren Polutan")
    df_melt = filtered_data[['tanggal'] + pollutants_all].melt(id_vars='tanggal', var_name='Pollutant', value_name='Concentration')
    g = sns.FacetGrid(df_melt, col="Pollutant", col_wrap=3, height=3, sharex=True, sharey=False)
    g.map(sns.lineplot, "tanggal", "Concentration", color='lightblue')
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
                          ["Tren Bulanan", "Tren Konsentrasi PM2.5 & Hubungan dengan Variabel Meteorologi", 
                           "Tren Konsentrasi PM10 & Hubungan dengan Variabel Meteorologi", 
                           "Tren Konsentrasi PM2.5 & PM10 Beserta Hubungan dengan Variabel Meteorologi", 
                           "Dampak Meteorologi vs Polutan Lainnya", "Pairplot Polutan & Meteorologi"])
    
    if option == "Tren Bulanan":
        st.subheader("Tren Bulanan Konsentrasi PM2.5 dan PM10 (Semua Stasiun)")
        df_temp = date_filtered.copy()
        df_temp.set_index('tanggal', inplace=True)
        trend_pm = df_temp.groupby('stasiun')[['PM2.5', 'PM10']].resample('ME').mean().reset_index()
        
        # Plot PM2.5
        fig, ax = plt.subplots(figsize=(14, 6))
        for stn in trend_pm['stasiun'].unique():
            if stn != selected_station:
                subset = trend_pm[trend_pm['stasiun'] == stn]
                ax.plot(subset['tanggal'], subset['PM2.5'], label=stn, color='lightblue', marker='o', alpha=0.8)
        subset_selected = trend_pm[trend_pm['stasiun'] == selected_station]
        ax.plot(subset_selected['tanggal'], subset_selected['PM2.5'], label=selected_station, 
                color='orange', marker='o', alpha=0.8)
        ax.set_title('Tren Rata-rata Bulanan PM2.5 per Stasiun')
        ax.set_xlabel('Bulan')
        ax.set_ylabel('Konsentrasi PM2.5 (µg/m³)')
        ax.tick_params(axis='x', rotation=45)
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        
        # Plot PM10
        fig, ax = plt.subplots(figsize=(14, 6))
        for stn in trend_pm['stasiun'].unique():
            if stn != selected_station:
                subset = trend_pm[trend_pm['stasiun'] == stn]
                ax.plot(subset['tanggal'], subset['PM10'], label=stn, color='lightblue', marker='o', alpha=0.8)
        subset_selected = trend_pm[trend_pm['stasiun'] == selected_station]
        ax.plot(subset_selected['tanggal'], subset_selected['PM10'], label=selected_station, 
                color='orange', marker='o', alpha=0.8)
        ax.set_title('Tren Rata-rata Bulanan PM10 per Stasiun')
        ax.set_xlabel('Bulan')
        ax.set_ylabel('Konsentrasi PM10 (µg/m³)')
        ax.tick_params(axis='x', rotation=45)
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    
    elif option == "Pairplot Polutan & Meteorologi":
        st.subheader("Pairplot: Hubungan Polutan dan Variabel Meteorologi")
        pair_vars = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3', 'TEMP', 'PRES', 'DEWP', 'WSPM']
        sampled_data = filtered_data[pair_vars].sample(n=1000, random_state=42)
        sns.set(style="ticks")
        pair_plot = sns.pairplot(sampled_data, diag_kind="kde", plot_kws={'color': 'orange'})
        plt.tight_layout()
        st.pyplot(pair_plot.figure)
        plt.close(pair_plot.figure)
    
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
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.lineplot(x='tanggal', y='PM2.5', data=filtered_data, ax=ax, label='PM2.5', color='blue')
        sns.lineplot(x='tanggal', y='PM10', data=filtered_data, ax=ax, label='PM10', color='green')
        ax.set_xlabel("Tanggal")
        ax.set_ylabel("Konsentrasi (µg/m³)")
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
                sns.scatterplot(x=met, y=pol, data=filtered_data, ax=axes[i, j], color='lightblue')
                axes[i, j].set_title(f"{met} vs {pol}")
                axes[i, j].tick_params(axis='x', rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

# ===============================================================================
# Tab 3: Perbandingan Antar Stasiun (Menggunakan Bar Chart)
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
            if stn != selected_station:
                subset = daily_avg[daily_avg['stasiun'] == stn]
                ax.plot(subset['tanggal'], subset.iloc[:, 2], label=stn, color='lightblue', alpha=0.8)
        subset_selected = daily_avg[daily_avg['stasiun'] == selected_station]
        ax.plot(subset_selected['tanggal'], subset_selected.iloc[:, 2], label=selected_station, 
                color='orange', alpha=0.8)
        ax.set_xlabel("Tanggal")
        ax.set_ylabel(ylabel)
        ax.tick_params(axis='x', rotation=45)
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    
    # Rata-rata PM2.5 per Stasiun
    if compare_option in ["PM2.5", "Keduanya (PM & Lainnya)"]:
        st.write("**Rata-rata PM2.5 per Stasiun**")
        avg_pm25 = date_filtered.groupby('stasiun')['PM2.5'].mean().reset_index()
        # Buat list warna: oranye untuk stasiun yang dipilih, lightblue untuk yang lain
        colors = ['orange' if stn == selected_station else 'lightblue' for stn in avg_pm25['stasiun']]
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(x='stasiun', y='PM2.5', data=avg_pm25, ax=ax, palette=colors)
        ax.set_xlabel("Stasiun")
        ax.set_ylabel("Rata-rata PM2.5 (µg/m³)")
        ax.set_title("Rata-rata PM2.5 per Stasiun")
        ax.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    
    # Rata-rata PM10 per Stasiun
    if compare_option in ["PM10", "Keduanya (PM & Lainnya)"]:
        st.write("**Rata-rata PM10 per Stasiun**")
        avg_pm10 = date_filtered.groupby('stasiun')['PM10'].mean().reset_index()
        # Buat list warna: oranye untuk stasiun yang dipilih, lightblue untuk yang lain
        colors = ['orange' if stn == selected_station else 'lightblue' for stn in avg_pm10['stasiun']]
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(x='stasiun', y='PM10', data=avg_pm10, ax=ax, palette=colors)
        ax.set_xlabel("Stasiun")
        ax.set_ylabel("Rata-rata PM10 (µg/m³)")
        ax.set_title("Rata-rata PM10 per Stasiun")
        ax.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    
    # Rata-rata Variabel Meteorologi per Stasiun
    if compare_option in ["Meteorologi", "Keduanya (PM & Lainnya)"]:
        st.subheader("Rata-rata Variabel Meteorologi per Stasiun")
        met_vars = ['TEMP', 'DEWP', 'PRES', 'WSPM']
        for var in met_vars:
            st.write(f"**Rata-rata {var} per Stasiun**")
            avg_met = date_filtered.groupby('stasiun')[var].mean().reset_index()
            # Buat list warna: oranye untuk stasiun yang dipilih, lightblue untuk yang lain
            colors = ['orange' if stn == selected_station else 'lightblue' for stn in avg_met['stasiun']]
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.barplot(x='stasiun', y=var, data=avg_met, ax=ax, palette=colors)
            ax.set_xlabel("Stasiun")
            ax.set_ylabel(f"Rata-rata {var}")
            ax.set_title(f"Rata-rata {var} per Stasiun")
            ax.tick_params(axis='x', rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
    
    # Rata-rata Polutan Lainnya per Stasiun
    if compare_option in ["Polutan Lainnya", "Keduanya (PM & Lainnya)"]:
        st.subheader("Rata-rata Polutan Lainnya per Stasiun")
        pollutants_other = ['NO2', 'SO2', 'CO', 'O3']
        for pol in pollutants_other:
            st.write(f"**Rata-rata {pol} per Stasiun**")
            avg_pol = date_filtered.groupby('stasiun')[pol].mean().reset_index()
            # Buat list warna: oranye untuk stasiun yang dipilih, lightblue untuk yang lain
            colors = ['orange' if stn == selected_station else 'lightblue' for stn in avg_pol['stasiun']]
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.barplot(x='stasiun', y=pol, data=avg_pol, ax=ax, palette=colors)
            ax.set_xlabel("Stasiun")
            ax.set_ylabel(f"Rata-rata {pol}")
            ax.set_title(f"Rata-rata {pol} per Stasiun")
            ax.tick_params(axis='x', rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

# ===============================================================================
# Tab 4: Distribusi Waktu (Menggunakan Line Chart dan Bar Chart)
# ===============================================================================
with tabs[3]:
    st.subheader("Distribusi Berdasarkan Waktu")
    
    variable_choice = st.selectbox(
        "Pilih Variabel",
        ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3', 'TEMP', 'PRES', 'DEWP', 'WSPM']
    )
    
    filtered_data['hour'] = filtered_data['tanggal'].dt.hour
    filtered_data['day_of_week'] = filtered_data['tanggal'].dt.day_name()
    filtered_data['hari'] = filtered_data['tanggal'].dt.dayofweek
    filtered_data['jenis_hari'] = filtered_data['hari'].apply(lambda x: 'Akhir Pekan' if x >= 5 else 'Hari Kerja')
    
    # 1. Line Chart: Tren Rata-rata per Jam
    st.write(f"**Tren Rata-rata {variable_choice} per Jam**")
    hourly_avg = filtered_data.groupby('hour')[variable_choice].mean().reset_index()
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(x='hour', y=variable_choice, data=hourly_avg, ax=ax, color='lightblue')
    ax.set_xlabel("Jam")
    ax.set_ylabel(f"Rata-rata {variable_choice}")
    ax.set_title(f"Tren Rata-rata {variable_choice} per Jam")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    
    # 2. Bar Chart: Rata-rata per Hari dalam Seminggu
    st.write(f"**Rata-rata {variable_choice} per Hari dalam Seminggu**")
    order_days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    daily_avg = filtered_data.groupby('day_of_week')[variable_choice].mean().reindex(order_days).reset_index()
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x='day_of_week', y=variable_choice, data=daily_avg, ax=ax, color='lightblue')
    ax.set_xlabel("Hari")
    ax.set_ylabel(f"Rata-rata {variable_choice}")
    ax.set_title(f"Rata-rata {variable_choice} per Hari dalam Seminggu")
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    
    # 3. Bar Chart: Rata-rata Hari Kerja vs Akhir Pekan
    st.write(f"**Rata-rata {variable_choice}: Hari Kerja vs Akhir Pekan**")
    workday_avg = filtered_data.groupby('jenis_hari')[variable_choice].mean().reset_index()
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x='jenis_hari', y=variable_choice, data=workday_avg, 
                palette={'Hari Kerja': 'lightblue', 'Akhir Pekan': 'orange'}, ax=ax)
    ax.set_xlabel("Jenis Hari")
    ax.set_ylabel(f"Rata-rata {variable_choice}")
    ax.set_title(f"Rata-rata {variable_choice}: Hari Kerja vs Akhir Pekan")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)