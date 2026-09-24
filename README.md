# 🛒 Retail Customer Purchase Prediction

<p align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0f2027,50:203a43,100:2c5364&height=220&section=header&text=Retail%20Purchase%20Prediction&fontSize=42&fontColor=ffffff&animation=fadeIn&fontAlignY=38"/>

</p>

<p align="center">

### 🤖 Machine Learning Based Customer Purchase Prediction System

Predict whether an online customer is likely to make a purchase based on their website activity and customer behavior.

<br>

<img src="https://img.shields.io/badge/Python-3.x-blue?style=for-the-badge&logo=python"/>
<img src="https://img.shields.io/badge/Pandas-Data%20Analysis-150458?style=for-the-badge&logo=pandas"/>
<img src="https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-F7931E?style=for-the-badge&logo=scikit-learn"/>
<img src="https://img.shields.io/badge/Streamlit-Web%20App-FF4B4B?style=for-the-badge&logo=streamlit"/>
<img src="https://img.shields.io/badge/Status-Working-success?style=for-the-badge"/>

</p>

---

## 🚀 About The Project

**Retail Customer Purchase Prediction** is a Machine Learning project designed to predict whether a customer is likely to purchase a product during an online shopping session.

The system uses customer interaction data such as:

- 🌐 Pages visited
- ⏱️ Time spent on website
- 👀 Products viewed
- 🛒 Cart activity
- 🎁 Discount usage
- 📱 Device type
- 🔎 Traffic source
- 🛍️ Previous purchases
- 🔢 Session count

The project includes a complete Machine Learning workflow from **data cleaning to model prediction and a Streamlit web interface**.

---

## 🎯 Problem Statement

E-commerce websites collect a large amount of information about customer behavior.

The objective of this project is to use that information to build a classification model that predicts:

> **Will the customer purchase a product or not?**

The prediction can help businesses understand customer behavior and identify customers who are more likely to purchase.

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 🧹 Data Cleaning | Handles missing values and duplicate rows |
| 🔄 Data Preprocessing | Converts categorical data into numerical features |
| 🤖 ML Models | Logistic Regression and Random Forest |
| 📊 Model Evaluation | Accuracy, Precision, Recall and F1 Score |
| 🔮 Prediction | Predicts customer purchase behaviour |
| 📈 Probability | Shows purchase probability |
| 🖥️ Web App | Interactive Streamlit interface |
| 💾 Model Saving | Trained models saved using Joblib |

---

## 🧠 Machine Learning Workflow

```text
              ┌─────────────────────┐
              │     Customer Data   │
              └──────────┬──────────┘
                         ↓
              ┌─────────────────────┐
              │    Data Cleaning    │
              └──────────┬──────────┘
                         ↓
              ┌─────────────────────┐
              │ Feature Preparation │
              └──────────┬──────────┘
                         ↓
              ┌─────────────────────┐
              │  Categorical        │
              │     Encoding        │
              └──────────┬──────────┘
                         ↓
              ┌─────────────────────┐
              │   Train / Test      │
              │       Split         │
              └──────────┬──────────┘
                         ↓
              ┌─────────────────────┐
              │   ML Classification │
              │       Models        │
              └──────────┬──────────┘
                         ↓
          ┌──────────────┴──────────────┐
          ↓                             ↓
 ┌─────────────────┐           ┌─────────────────┐
 │ Logistic        │           │ Random Forest   │
 │ Regression      │           │                 │
 └────────┬────────┘           └────────┬────────┘
          │                             │
          └──────────────┬──────────────┘
                         ↓
              ┌─────────────────────┐
              │ Model Evaluation    │
              └──────────┬──────────┘
                         ↓
              ┌─────────────────────┐
              │ Purchase Prediction │
              └──────────┬──────────┘
                         ↓
              ┌─────────────────────┐
              │   Streamlit App     │
              └─────────────────────┘
