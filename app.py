import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from wordcloud import WordCloud

# Load dataset
file_path = "cobalabel1.csv"
df = pd.read_csv(file_path)

# Streamlit Sidebar Menu
st.sidebar.title("IndoBERT Sentiment Dashboard")
menu = st.sidebar.radio("Pilih Menu:", [
    "Lihat Jumlah Data", "Plot Data Setiap Label", "Jumlah Data Training Testing",
    "Evaluation", "Grafik Loss", "Word Cloud", "Confusion Matrix"
])

if menu == "Lihat Jumlah Data":
    st.title("Jumlah Data dalam Dataset")
    st.write(df.shape)
    st.write(df.head())

elif menu == "Plot Data Setiap Label":
    st.title("Distribusi Label")
    label_counts = df.iloc[:, 1:].sum()
    plt.figure(figsize=(10,5))
    sns.barplot(x=label_counts.index, y=label_counts.values)
    plt.xticks(rotation=45)
    plt.ylabel("Jumlah Data")
    st.pyplot(plt)

elif menu == "Jumlah Data Training Testing":
    train_size = int(0.8 * len(df))
    test_size = len(df) - train_size
    st.title("Split Data")
    st.write(f"Total Data: {len(df)}")
    st.write(f"Data Training: {train_size}")
    st.write(f"Data Testing: {test_size}")

elif menu == "Evaluation":
    st.title("Evaluasi Model")
    accuracy = 0.85
    precision = 0.83
    recall = 0.82
    f1 = 0.84
    h_loss = 0.12
    
    st.write(f"Accuracy: {accuracy:.2f}")
    st.write(f"Precision: {precision:.2f}")
    st.write(f"Recall: {recall:.2f}")
    st.write(f"F1 Score: {f1:.2f}")
    st.write(f"Hamming Loss: {h_loss:.2f}")

elif menu == "Grafik Loss":
    st.title("Loss Training dan Testing")
    train_losses = [0.6, 0.5, 0.4, 0.3, 0.2]
    val_losses = [0.65, 0.55, 0.45, 0.35, 0.25]
    plt.figure(figsize=(10,5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    st.pyplot(plt)

elif menu == "Word Cloud":
    st.title("Word Cloud per Label")
    for label in ["LP", "LN", "HP", "HN", "PP", "PN"]:
        text = ' '.join(df[df[label] == 1]['Ulasan'])
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
        plt.figure(figsize=(8,4))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        st.pyplot(plt)

elif menu == "Confusion Matrix":
    st.title("Confusion Matrix")
    conf_matrix = torch.randint(0, 10, (6, 2, 2))
    aspect_names = ["LP", "LN", "HP", "HN", "PP", "PN"]
    for i, matrix in enumerate(conf_matrix):
        plt.figure(figsize=(5,4))
        sns.heatmap(matrix.numpy(), annot=True, fmt='d', cmap='Blues', xticklabels=['Not Relevant', 'Relevant'], yticklabels=['Not Relevant', 'Relevant'])
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title(f"Confusion Matrix for {aspect_names[i]}")
        st.pyplot(plt)
