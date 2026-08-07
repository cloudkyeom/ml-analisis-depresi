import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def EDA(df):
    dataset = "Depression_Student_Dataset.csv"
    df = pd.read_csv(dataset)
    
    os.makedirs('output', exist_ok=True)
    # Cek struktur data
    print("Shape data:", df.shape)
    print("\nKolom yang tersedia:")
    print(df.columns.tolist())
    print("\ndata:")
    print(df.head())

    # CEK MISSING VALUES & TIPE DATA
    print("\nInfo dataset:")
    print(df.info())

    print("\nJumlah missing value per kolom:")
    print(df.isnull().sum())
    df = df.dropna()  # hapus baris dengan missing value
    # EXPLORATORY DATA ANALYSIS (EDA)
    plt.figure(figsize=(5,4))
    bars = sns.countplot(data=df, x='Depression', hue='Depression', palette=['skyblue', 'pink'], legend=False)
    plt.title("Distribusi Depresi pada Mahasiswa")
    for container in bars.containers:
        bars.bar_label(container)
    plt.show()

    # Cek korelasi antar fitur numerik
    plt.figure(figsize=(10,8))
    sns.heatmap(df.select_dtypes(include=[np.number]).corr(), annot=True, cmap='coolwarm')
    plt.title("Korelasi Antar Fitur Numerik")
    plt.show()

    # Academic Pressure vs Depression
    plt.figure(figsize=(6,4))
    bp = sns.boxplot(data=df, x='Depression', y='Academic Pressure', hue='Depression',
                    palette=['skyblue', 'pink'], legend=False)
    colors = ['black', 'black']
    categories = df['Depression'].unique()

    for i, category in enumerate(categories):
        data = df[df['Depression'] == category]['Academic Pressure']

        q1 = data.quantile(0.25)
        median = data.median()
        q3 = data.quantile(0.75)
        minimum = data.min()
        maximum = data.max()

        stats = {
            minimum: 'Min',
            q1: 'Q1',
            median: 'Median',
            q3: 'Q3',
            maximum: 'Max'
        }

        for value, label in stats.items():
            bp.text(i + 0.35, value, f'{label}: {value:.2f}',
                    horizontalalignment='left',
                    verticalalignment='center',
                    fontsize=8,
                    color=colors[i])

    plt.xlim(-0.5, len(categories) - 0.2)
    plt.title('Academic Pressure vs Depression')
    plt.show()
    plt.tight_layout()
    plt.savefig('output/academic_pressure_vs_depresi.png')
    plt.close()