import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.tree import export_graphviz
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import io
import graphviz

# Passo 1: Interface para upload de arquivo
st.title("Análise de Random Forest")
st.write("Faça upload do arquivo CSV para análise")

uploaded_file = st.file_uploader("Escolha um arquivo CSV", type="csv")

if uploaded_file is not None:
    try:
        # Ler o conteúdo do arquivo como texto
        content = uploaded_file.read().decode('utf-8-sig').strip()  # Adicione -sig para lidar com BOM

        # Remover aspas do conteúdo
        content = content.replace('"', '')

        # Tentar detectar o delimitador
        if content.count(';') > content.count(','):
            sep = ';'
        else:
            sep = ','

        # Usar StringIO para passar o conteúdo para pandas
        data = pd.read_csv(io.StringIO(content), sep=sep, engine='python')

        # Verifique se o DataFrame foi preenchido corretamente
        if data.empty:
            st.error("Não foi possível ler o arquivo CSV.")
        else:
            # Remover espaços em branco das colunas
            data.columns = data.columns.str.strip()

            # Exibir os dados e seus tipos
            st.write("Primeiras linhas do arquivo:")
            st.write(data.head())

            # Passo 3: Seleção de features e target
            st.subheader("Selecione as Features e o Target")
            columns = data.columns.tolist()

            # Seleciona o target
            target_column = st.selectbox("Selecione a coluna alvo (target):", columns)
            
            # Seleciona as features
            features_columns = st.multiselect("Selecione as colunas de features:", columns, default=columns)

            # Remover a coluna target das features selecionadas
            if target_column in features_columns:
                features_columns.remove(target_column)

            # Verificação se as colunas foram selecionadas
            if not features_columns:
                st.error("Por favor, selecione ao menos uma coluna para usar como feature.")
            elif target_column is None:
                st.error("Por favor, selecione uma coluna como target.")
            else:
                # Seção para ajuste de hiperparâmetros
                st.sidebar.subheader("Ajuste de Hiperparâmetros")
                n_estimators = st.sidebar.slider("Número de Árvores (n_estimators)", min_value=1, max_value=100, value=10)
                max_depth = st.sidebar.slider("Profundidade Máxima (max_depth)", min_value=1, max_value=20, value=5)
                min_samples_split = st.sidebar.slider("Mínimo de Amostras para Dividir (min_samples_split)", min_value=2, max_value=20, value=2)
                max_leaf_nodes = st.sidebar.slider("Máximo de Folhas (max_leaf_nodes)", min_value=2, max_value=100, value=20)
                min_samples_leaf = st.sidebar.slider("Mínimo de Amostras em uma Folha (min_samples_leaf)", min_value=1, max_value=20, value=1)

                # Ajustar test_size e random_state
                test_size = st.sidebar.slider("Proporção do Conjunto de Teste", 0.1, 0.9, 0.2, 0.05)
                random_state = st.sidebar.number_input("Valor de Random State", min_value=0, value=42, step=1)
                train_size = 1 - test_size  # Calcular train_size como 1 - test_size

                # Passo 4: Treinamento do modelo
                features = data[features_columns]
                target = data[target_column].astype(str)  # Converter para string

                # Verificação para evitar erros na divisão dos dados
                if features.empty or target.empty:
                    st.error("As features ou a coluna alvo estão vazias. Verifique sua seleção.")
                else:
                    # Convertendo a coluna alvo para valores numéricos
                    le = LabelEncoder()
                    target = le.fit_transform(target)

                    # Convertendo as features para numéricas
                    features = features.apply(pd.to_numeric, errors='coerce')

                    # Tratar valores infinitos e muito grandes
                    features.replace([float('inf'), float('-inf')], pd.NA, inplace=True)  # Substitui infinitos por NaN
                    features = features.fillna(features.mean())  # Substitui NaN pela média das colunas

                    # Divisão dos dados
                    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=test_size, train_size=train_size, random_state=random_state)

                    # Treinamento do modelo
                    model = RandomForestClassifier(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        min_samples_split=min_samples_split,
                        max_leaf_nodes=max_leaf_nodes,
                        min_samples_leaf=min_samples_leaf,
                        random_state=random_state
                    )
                    model.fit(X_train, y_train)
                    predictions = model.predict(X_test)

                    # Passo 5: Exibição dos resultados
                    st.subheader("Resultados da Classificação")

                    # Botões para visualizar resultados
                    if st.button("Mostrar Relatório de Classificação"):
                        st.subheader("Relatório de Classificação")
                        st.text(classification_report(y_test, predictions))

                    if st.button("Mostrar Matriz de Confusão"):
                        st.subheader("Matriz de Confusão")
                        cm = confusion_matrix(y_test, predictions)
                        fig, ax = plt.subplots()
                        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
                        st.pyplot(fig)

                    if st.button("Mostrar Importância das Variáveis"):
                        st.subheader("Importância das Variáveis")
                        feature_importance = model.feature_importances_
                        feature_df = pd.DataFrame({
                            "Feature": features_columns,
                            "Importance": feature_importance
                        }).sort_values(by="Importance", ascending=False)

                        # Gráfico de importância das variáveis
                        fig2, ax2 = plt.subplots()
                        sns.barplot(x="Importance", y="Feature", data=feature_df, ax=ax2)
                        st.pyplot(fig2)

                        st.write("Tabela de Importância das Variáveis:")
                        st.write(feature_df)

                    # Exibir uma árvore do Random Forest
                    st.subheader("Visualização de Árvore")
                    # Selecionar a primeira árvore do modelo
                    estimator = model.estimators_[0]
                    
                    # Exportar a árvore
                    dot_data = export_graphviz(estimator, out_file=None, 
                                                feature_names=features_columns,
                                                class_names=[str(cls) for cls in le.classes_],  # Converter para string
                                                filled=True, rounded=True, 
                                                special_characters=True)  
                    graph = graphviz.Source(dot_data)  
                    st.graphviz_chart(graph)

    except Exception as e:
        st.error(f"Erro ao processar o arquivo: {e}")
