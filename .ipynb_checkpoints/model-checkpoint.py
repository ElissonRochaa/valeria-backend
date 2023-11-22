import os
import pickle
import numpy as np
import pandas as pd
import lime.lime_tabular

class Model:

    def __init__(self):
        #Saídas formatadas do modelo para visualização no front.
        self.outputs = {
            "CHIKUNGUNYA": "Chikungunya",
            "DENGUE": "Dengue",
            "OUTRAS_DOENCAS": "Inconclusivo"
        }

        # Inputs categóricos(binários) do modelo.
        self.categorical_labels = {
            "FEBRE": "Febre", "MIALGIA": "Mialgia", "CEFALEIA": "Cefaleia", "EXANTEMA": "Exantema", "NAUSEA": "Náusea", "DOR_COSTAS": "Dor nas costas",
            "CONJUNTVIT": "Conjuntivite", "ARTRITE": "Artrite", "ARTRALGIA": "Artralgia", "PETEQUIA_N": "Petéquias", "DOR_RETRO": "Dor Retroorbital",
            "DIABETES": "Diabetes", "HIPERTENSA": "Hipertensão"
        }

        # Inputs numéricos do modelo
        self.numerical_labels = {
            "DIAS": "Período dos sintomas"
        }

        self.labels = dict(self.categorical_labels, **self.numerical_labels)
        
        self.load_model()
        self.load_explainer()
        
    def load_model(self):
        with open("gradient_model.pkl", "rb") as f:
            self.model = pickle.load(f)
            print("model carregado com sucesso")
    
    def load_explainer(self):
        path_database = "database.csv"
        database = pd.read_csv(path_database, sep=';', usecols=self.labels.keys())

        self.explainer = lime.lime_tabular.LimeTabularExplainer(
            database.to_numpy(),
            feature_names=self.labels.keys(),
            class_names=self.outputs.keys(),
            categorical_features=[count for count, value in enumerate(self.categorical_labels)],
            categorical_names=self.categorical_labels,
            kernel_width=3,
            verbose=False
        )
    
    def read_json(self, dados):
        days = dados.get('quant_dias')
        fever = dados.get('febre')
        myalgia = dados.get('mialgia')
        headache = dados.get('cefaleia')
        rash = dados.get('exantema')
        nausea = dados.get('nausea')
        backPain = dados.get('dor_nas_costas')
        conjunctivitis = dados.get('conjuntivite')
        arthritis = dados.get('artrite')
        arthralgia = dados.get('artralgia')
        petechia = dados.get('petequias')
        eyaPain = dados.get('dor_retroorbital')
        diabetes = dados.get('diabetes')
        hypertension = dados.get('hipertensao')
        
        return [[
            fever,
            myalgia,
            headache,
            rash,
            nausea,
            backPain,
            conjunctivitis,
            arthritis,
            arthralgia,
            petechia,
            eyaPain,
            diabetes,
            hypertension,
            days
        ]]
    
    def predict(self, dados):
        print("Entrou no predict")
        print(dados)
        data = self.read_json(dados)
        self.classification = self.model.predict(data)[0]
        self.classification_proba = self.model.predict_proba(data)[0]
        print(self.classification)
        print(self.classification_proba)
        
        exp_df = self.explainer_function(dados)
        
        return self.outputs[self.classification], self.classification_proba, exp_df

    def explainer_function(self, dados):
        exp = self.explainer.explain_instance(
            np.array(self.read_json(dados)[0]),
            self.model.predict_proba,
            num_features=14,
            top_labels=3
        )

        # For para saber a posição da saída, infelizmente não consegui fazer isso de uma forma mais elegante.
        for count, value in enumerate(self.outputs.keys()):
            if value == self.classification:
                pos_label = count
                break

        # As saídas do explainer são inseridas em um dict para que possam ser convertidas em um dataframe posteriormente.
        exp_dict = sorted(dict(exp.as_map()[pos_label]).items())

        exp_df = pd.DataFrame(
            exp_dict,
            columns=["key", "Valor"],
            index=self.labels.values()
        )

        # A coluna de resultado contém as informações do paciente, a saída do as_map() do explainer está na mesma ordem da entrada dos atributos, e consequentemente o método getRecord() da classe também esta na mesma ordem, não sendo necessário ordenar antes de unificar.
        exp_df["Resposta do Paciente"] = np.array(self.read_json(dados)[0])

        # Necessário converter o tipo da coluna para poder modificar o valor livremente. Para uma melhor visualização, as colunas boolenasa foram convertidas para um resultado de "Sim" ou "Não".
        exp_df["Resposta do Paciente"] = exp_df["Resposta do Paciente"].astype(str)
        for attribute in self.categorical_labels.values():
            exp_df.loc[(exp_df.index == attribute) & (exp_df["Resposta do Paciente"] == "0"), "Resposta do Paciente"] = "Não"
            exp_df.loc[(exp_df.index == attribute) & (exp_df["Resposta do Paciente"] == "1"), "Resposta do Paciente"] = "Sim"

        # Para uma melhor visualização, o valor do peso foi multiplicado por 100.
        exp_df["Valor"] = exp_df["Valor"].apply(lambda x: x * 100)

        exp_df = exp_df.sort_values(by=["Valor"], ascending=False)
        
        exp_df = exp_df.reset_index()
        
        #exp_pos = exp_df[exp_df["Valor"] > 0].sort_values(by=["Valor"], ascending=False)
        #exp_neg = exp_df[exp_df["Valor"] < 0].sort_values(by=["Valor"], ascending=True)

        # Pegar apenas o resutlado do paciente
        #exp_pos = exp_pos[["Resposta do Paciente"]]
        #exp_neg = exp_neg[["Resposta do Paciente"]]
        
        #categories = exp_df.index.to_list()
        #categories = [*categories, categories[0]]

        #values_to_plot = (exp_df["Valor"] * 10).to_list()
        #values_to_plot = [*values_to_plot, values_to_plot[0]]
        exp_df = exp_df[["index","Resposta do Paciente"]]
        print(exp_df)
        
        return exp_df
        