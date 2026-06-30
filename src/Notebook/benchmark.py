import time
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras import layers, Model, Input

# Nomes exatos das classes para o relatório
CLASSES_NOMES = [
    "Carcinoma Basocelular (Maligno)", "Dermatofibroma (Benigno)", 
    "Lesao Vascular (Benigno)", "Melanoma (Maligno)", 
    "Nevo Melanocitico (Benigno)", "Queratose Actinica (Pre-Maligno)", 
    "Queratose Benigna (Benigno)"
]

# --- 1. FUNÇÕES DE CONSTRUÇÃO DAS ARQUITETURAS ---

def construir_mobilenetv2():
    inputs = Input(shape=(224, 224, 3))
    base_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights=None)
    x = base_model(inputs)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.6)(x)
    outputs = layers.Dense(7, activation='softmax')(x)
    return Model(inputs=inputs, outputs=outputs)

def construir_efficientnetb0():
    inputs = Input(shape=(224, 224, 3))
    # EfficientNetB0 nativa do Keras
    base_model = tf.keras.applications.EfficientNetB0(input_shape=(224, 224, 3), include_top=False, weights=None)
    x = base_model(inputs)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.6)(x)
    outputs = layers.Dense(7, activation='softmax')(x)
    return Model(inputs=inputs, outputs=outputs)

# --- 2. FUNÇÃO AUXILIAR DE PREDIÇÃO COM THRESHOLD (0.35) ---

def predizer_com_threshold(probabilidades, threshold=0.35):
    predicoes_finais = []
    idx_carcinoma, idx_melanoma = 0, 3
    
    for probs in probabilidades:
        if probs[idx_melanoma] > threshold:
            predicoes_finais.append(idx_melanoma)
        elif probs[idx_carcinoma] > threshold:
            predicoes_finais.append(idx_carcinoma)
        else:
            predicoes_finais.append(np.argmax(probs))
    return np.array(predicoes_finais)

# --- 3. PIPELINE DE AVALIAÇÃO DO BENCHMARK ---

def rodar_benchmark(nome_modelo, modelo_instanciado, caminho_pesos, dataset_validacao):
    print(f"\n🚀 Avaliando {nome_modelo}...")
    
    # Carrega os pesos salvos do respectivo modelo
    modelo_instanciado.load_weights(caminho_pesos)
    
    # Extrair os labels reais (y_true)
    y_true = np.concatenate([y for x, y in dataset_validacao], axis=0)
    
    # Medir tempo de inferência (Latência)
    inicio = time.time()
    probabilidades_brutas = modelo_instanciado.predict(dataset_validacao, verbose=0)
    tempo_total = time.time() - inicio
    
    latencia_por_imagem = (tempo_total / len(y_true)) * 1000 # em milissegundos
    
    # Aplicar o threshold de 0.35 calibrado
    y_pred = predizer_com_threshold(probabilidades_brutas, threshold=0.35)
    
    # Extrair Métricas do Classification Report do Scikit-Learn
    report = classification_report(y_true, y_pred, target_names=CLASSES_NOMES, output_dict=True)
    
    # Focar especificamente no calcanhar de Aquiles: Recall do Melanoma
    recall_melanoma = report["Melanoma (Maligno)"]["recall"]
    acuracia_geral = report["accuracy"]
    
    print(f"📊 Acurácia Geral ({nome_modelo}): {acuracia_geral*100:.2f}%")
    print(f"🩺 Recall Melanoma ({nome_modelo}): {recall_melanoma*100:.2f}%")
    print(f"⚡ Latência Média por Imagem: {latencia_por_imagem:.2f} ms")
    
    return {
        "Modelo": nome_modelo,
        "Acurácia Geral": acuracia_geral,
        "Recall Melanoma": recall_melanoma,
        "Latência Média (ms)": latencia_por_imagem,
        "cm": confusion_matrix(y_true, y_pred)
    }

# --- 4. EXECUÇÃO DO BENCHMARK ---

# Certifique-se de que os nomes dos arquivos de pesos estão corretos
resultados = []

# Teste 1: MobileNetV2
model_mobilenet = construir_mobilenetv2()
res_mobilenet = rodar_benchmark("MobileNetV2", model_mobilenet, "pesos_skin_cancer.weights.h5", val_dataset)
resultados.append(res_mobilenet)

# Teste 2: EfficientNetB0
model_efficientnet = construir_efficientnetb0()
res_efficientnet = rodar_benchmark("EfficientNetB0", model_efficientnet, "pesos_efficientnet.weights.h5", val_dataset)
resultados.append(res_efficientnet)

# --- 5. PLOT COMPARATIVO DAS MATRIZES DE CONFUSÃO ---

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Benchmark: MobileNetV2 vs EfficientNetB0 (Threshold = 0.35)", fontsize=16, fontweight='bold')

for i, res in enumerate(resultados):
    sns.heatmap(res["cm"], annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=[c[:14] for c in CLASSES_NOMES], # Abrevia os nomes para caber no gráfico
                yticklabels=[c[:14] for c in CLASSES_NOMES], ax=axes[i])
    axes[i].set_title(f"{res['Modelo']} (Acc: {res['Acurácia Geral']*100:.1f}% | Recall Mel: {res['Recall Melanoma']*100:.1f}%)", fontsize=12, fontweight='bold')
    axes[i].set_xlabel("Classe Predita")
    axes[i].set_xticklabels([c[:14] for c in CLASSES_NOMES], rotation=45, ha='right')
    if i == 0: axes[i].set_ylabel("Classe Real")

plt.tight_layout()
plt.show()