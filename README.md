## Installation

```bash
pip install -r requirements.txt
```


---

| Fonctionnalité            | Description                                               |
|--------------------------|-----------------------------------------------------------|
| **But**                  | Classifier les avis de films en positif ou négatif       |
| **Dataset**              | Dataset IMDB (reviews de films, labels binaires 0/1)     |
| **Modèle**               | Réseau de neurones Keras avec Embedding + LSTM           |
| **Interface**            | Application Tkinter type chatbot pour tester facilement  |
| **Utilisation**          | Saisir un avis dans la fenêtre, obtenir la prédiction    |
| **Prérequis**            | Python, TensorFlow, Keras, Tkinter (inclus dans Python)   |
| **Commandes**            | `python deploiement.py` pour lancer l’application        |
| **Fonctionnalités futures** | Support multilingue, prise en compte avis neutres     |

---




## Exemple d’utilisation




*Exemple 1* : ( lang ENG ) 

<img width="454" height="309" alt="posi_Eng" src="https://github.com/user-attachments/assets/28432736-b3bb-4003-8386-2c61677517c0" />


*Exemple 2* : (lang FRA ) 

<img width="453" height="305" alt="nga_posi" src="https://github.com/user-attachments/assets/2eeaf7ca-769c-4c50-bb4e-a2cb83da0649" />




## Améliorations & Limites

| Aspects               | Description                                                                                   |
|-----------------------|-----------------------------------------------------------------------------------------------|
| **Améliorations possibles** | - Ajouter la prise en charge des avis neutres (classe 3)                                   |
|                       | - Utiliser un modèle plus complexe (BERT, Transformer) pour améliorer la précision            |
|                       | - Enrichir la prétraitement des textes (lemmatisation, correction orthographique)             |
|                       | - Support multilingue (actuellement limité au vocabulaire anglais IMDB)                       |
|                       | - Interface utilisateur plus riche (ajout d’historique, export des conversations, etc.)      |
|                       | - Intégration d’un système de feedback pour améliorer le modèle en continu                    |
| **Limites actuelles**  | - Classification binaire uniquement (positif/négatif), pas d’avis neutres                     |
|                       | - Dépendance au dataset IMDB, ce qui peut limiter la généralisation sur d’autres types d’avis |
|                       | - Tokenisation et encodage basiques, ne gérant pas bien les synonymes ou expressions complexes|
|                       | - Interface Tkinter simple, non responsive ni adaptée aux mobiles                             |
|                       | - Modèle entraîné uniquement sur des données textuelles, sans prise en compte du contexte     |
