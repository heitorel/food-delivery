import os
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_excel('metade_teste.xlsx', sheet_name='Plan1')

# Calculando média, moda e mediana de 'Age'
media_age = df['Age'].mean()
moda_age = df['Age'].mode()  
mediana_age = df['Age'].median()

print("Média de Age:", media_age)
print("Moda de Age:", moda_age)
print("Mediana de Age:", mediana_age)

# Calculando média, moda e mediana de 'Family Size'
media_Family_Size = df['Family size'].mean()
moda_Family_Size = df['Family size'].mode()  
mediana_Family_Size = df['Family size'].median()

print("Média de Family Size:", media_Family_Size)
print("Moda de Family Size:", moda_Family_Size)
print("Mediana de Family Size:", mediana_Family_Size)

# Gráfico de barra para quantidade por Gender
gender_counts = df['Gender'].value_counts()
plt.figure(figsize=(8, 6))
gender_counts.plot(kind='bar', color='skyblue')
plt.title('Quantidade por Gênero')
plt.xlabel('Gênero')
plt.ylabel('Quantidade')
plt.tick_params(axis='x', rotation=0)
for i, count in enumerate(gender_counts):
    plt.text(i, count + 0.1, str(count), ha='center', va='bottom')
plt.savefig('grafico_gender.png')
#plt.show()

# Gráfico de barra para quantidade por Marital Status
Marital_Status_counts = df['Marital Status'].value_counts()
plt.figure(figsize=(8, 6))
Marital_Status_counts.plot(kind='bar', color='skyblue')
plt.title('Quantidade por Marital Status')
plt.xlabel('Marital Status')
plt.ylabel('Quantidade')
plt.tick_params(axis='x', rotation=0)
for i, count in enumerate(Marital_Status_counts):
    plt.text(i, count + 0.1, str(count), ha='center', va='bottom')
plt.savefig('grafico_marital_status.png')
#plt.show()

# Gráfico de barra para quantidade por Occupation
Occupation_counts = df['Occupation'].value_counts()
plt.figure(figsize=(8, 6))
Occupation_counts.plot(kind='bar', color='skyblue')
plt.title('Quantidade por Occupation')
plt.xlabel('Occupation')
plt.ylabel('Quantidade')
plt.tick_params(axis='x', rotation=0)
for i, count in enumerate(Occupation_counts):
    plt.text(i, count + 0.1, str(count), ha='center', va='bottom')
plt.savefig('grafico_Occupation.png')
#plt.show()

# Gráfico de barra para quantidade por Monthly Income
Monthly_Income_counts = df['Monthly Income'].value_counts()
plt.figure(figsize=(8, 6))
Monthly_Income_counts.plot(kind='bar', color='skyblue')
plt.title('Quantidade por Monthly Income')
plt.xlabel('Monthly Income')
plt.ylabel('Quantidade')
plt.tick_params(axis='x', rotation=0)
for i, count in enumerate(Monthly_Income_counts):
    plt.text(i, count + 0.1, str(count), ha='center', va='bottom')
plt.savefig('grafico_Monthly_Income.png')
#plt.show()

# Gráfico de barra para quantidade por Educational Qualifications
Educational_Qualifications_counts = df['Educational Qualifications'].value_counts()
plt.figure(figsize=(8, 6))
Educational_Qualifications_counts.plot(kind='bar', color='skyblue')
plt.title('Quantidade por Educational Qualifications')
plt.xlabel('Educational Qualifications')
plt.ylabel('Quantidade')
plt.tick_params(axis='x', rotation=0)
for i, count in enumerate(Educational_Qualifications_counts):
    plt.text(i, count + 0.1, str(count), ha='center', va='bottom')
plt.savefig('grafico_Educational_Qualifications.png')
#plt.show()

# Agrupe os dados por 'Gender' e 'Monthly Income' e calcule as contagens
income_gender_counts = df.groupby(['Gender', 'Monthly Income']).size().unstack().fillna(0)
ax = income_gender_counts.plot(kind='bar', figsize=(10, 6))
plt.title('Quantidade de Monthly Income por Gender')
plt.xlabel('Monthly Income')
plt.ylabel('Quantidade')
plt.tick_params(axis='x', rotation=0)
plt.legend(title='Income x Gender')
for p in ax.patches:
    ax.annotate(str(int(p.get_height())), (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, 10), textcoords='offset points')
plt.savefig('grafico_income_gender.png')
plt.show()

# Agrupe os dados por 'Gender' e 'Feedback' e calcule as contagens
feedback_gender_counts = df.groupby(['Gender', 'Feedback']).size().unstack().fillna(0)
ax = feedback_gender_counts.plot(kind='bar', figsize=(10, 6))
plt.title('Quantidade de Feedback por Gender')
plt.xlabel('Feedback')
plt.ylabel('Quantidade')
plt.tick_params(axis='x', rotation=0)
plt.legend(title='Feedback x Gender')
for p in ax.patches:
    ax.annotate(str(int(p.get_height())), (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, 10), textcoords='offset points')
plt.savefig('grafico_feedback_gender.png')
plt.show()