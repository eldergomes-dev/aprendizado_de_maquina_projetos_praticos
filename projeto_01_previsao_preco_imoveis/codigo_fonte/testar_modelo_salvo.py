import os
import joblib
import pandas as pd

# 1. Localiza o modelo .joblib
diretorio_atual = os.path.dirname(os.path.abspath(__file__))
caminho_modelo = os.path.join(diretorio_atual, "..", "modelos", "modelo_previsao_imoveis.joblib")

# 2. Carrega o modelo treinado
print("Carregando o modelo salvo...")
modelo = joblib.load(caminho_modelo)
print("Modelo carregado com sucesso!\n")

# 3. Dados do novo imóvel com os nomes EXATOS das colunas usadas no treino
novo_imovel = pd.DataFrame([{
    'tamanho_metros_quadrados': 85,
    'quantidade_quartos': 2,
    'distancia_centro_quilometros': 5
}])

# 4. Realizar a previsão
preco_estimado = modelo.predict(novo_imovel)

print("--- PREVISÃO DE PREÇO ---")
print(f"Valor estimado do imóvel: R$ {preco_estimado[0]:,.2f}")