import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import numpy as np

# Carregar os dados
data = pd.read_excel('onlinefoods.xlsx', sheet_name='onlinefoods')

# Aplicar LabelEncoder para converter 'Feedback' em valores numéricos
label_encoder = LabelEncoder()
data['Feedback'] = label_encoder.fit_transform(data['Feedback'])

# Selecionar as variáveis independentes (features) e a variável dependente (target)
X = data[['Age', 'Gender', 'Marital Status', 'Occupation', 'Monthly Income', 'Educational Qualifications', 'Family size', 'latitude', 'longitude', 'Pin code']]
y = data['Feedback']

# Convertendo variáveis categóricas em variáveis dummy, se necessário
X = pd.get_dummies(X)

# Dividindo o conjunto de dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Criando o modelo de regressão linear
model = LinearRegression()

# Treinando o modelo nos dados de treinamento
model.fit(X_train, y_train)

# Fazendo previsões nos dados de teste
predictions = model.predict(X_test)

# Calculando o erro quadrático médio (RMSE)
rmse = mean_squared_error(y_test, predictions, squared=False)
print("RMSE:", rmse)

# Coeficientes da regressão
print("Coeficientes da Regressão:")
coefficients = pd.DataFrame({'Feature': X.columns, 'Coefficient': model.coef_})
print(coefficients)

# Intercepto da regressão
print("Intercepto da Regressão:", model.intercept_)


# Gráfico de Dispersão de Valores Reais vs. Valores Previstos
plt.figure(figsize=(8, 6))
plt.scatter(y_test, predictions, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Valores Reais')
plt.ylabel('Valores Previstos')
plt.title('Valores Reais vs. Valores Previstos')
#plt.show()
plt.savefig('grafico_dispersao.png')

# Gráfico de Resíduos
residuals = y_test - predictions

plt.figure(figsize=(8, 6))
plt.scatter(predictions, residuals, color='green')
plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
plt.xlabel('Valores Previstos')
plt.ylabel('Resíduos')
plt.title('Gráfico de Resíduos')
#plt.show()
plt.savefig('grafico_residuos.png')

# Gráfico de Importância das Variáveis
# Obtendo os coeficientes e os nomes das variáveis
coefficients = model.coef_
feature_names = X.columns

# Ordenando os coeficientes por valor absoluto
sorted_indices = np.argsort(np.abs(coefficients))
sorted_coefficients = coefficients[sorted_indices]
sorted_feature_names = feature_names[sorted_indices]

plt.figure(figsize=(25, 8))
plt.barh(sorted_feature_names, sorted_coefficients)
plt.xlabel('Coeficiente')
plt.title('Importância das Variáveis')
#plt.show()
plt.savefig('grafico_coeficientes.png')

# Gráfico de Linha da Regressão Linear
plt.figure(figsize=(8, 6))
plt.scatter(X_test.iloc[:, 0], y_test, color='blue', label='Valores Reais')
plt.plot(X_test.iloc[:, 0], predictions, color='red', linewidth=2, label='Regressão Linear')
plt.xlabel('Variável Independente')
plt.ylabel('Feedback')
plt.title('Regressão Linear')
plt.legend()
#plt.show()
plt.savefig('grafico_regressao_linear.png')