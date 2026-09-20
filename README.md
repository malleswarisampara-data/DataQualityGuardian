# 🛡️ AI Data Quality Guardian

### AI-Powered Data Quality, Anomaly Detection & ETL Platform

AI Data Quality Guardian is an intelligent data quality platform built with **Python and Streamlit** that helps identify, analyze, clean, and monitor data-quality problems automatically.

The system performs dataset profiling, missing-value detection, duplicate detection, data validation, anomaly detection, automated cleaning, ETL processing, quality scoring, and interactive reporting through a user-friendly web interface.

## 🚀 Live Demo

👉 **[Open AI Data Quality Guardian](https://malleswarisampara-data-dataqualityguardian-app-d9ogoy.streamlit.app/)**

## 📌 Project Overview

Poor-quality data can lead to incorrect analysis, unreliable business decisions, and inefficient data pipelines.

AI Data Quality Guardian provides an automated solution for detecting and improving data quality before the data is used for analytics or further processing.

The platform allows users to upload datasets and automatically analyze their structure and quality. It identifies potential issues, applies cleaning techniques, detects unusual records, and presents the results through an interactive dashboard.

## 🎯 Objectives

- Detect data-quality issues automatically
- Identify missing and duplicate records
- Validate important fields such as email addresses
- Detect unusual or anomalous data
- Automatically clean datasets
- Perform basic ETL processing
- Generate a data-quality score
- Provide interactive visualizations
- Generate downloadable reports
- Store pipeline execution information using SQLite

## ✨ Key Features

### 📂 1. Data Ingestion

Upload datasets directly through the web application.

Supported formats:

- CSV
- Excel (`.xlsx`)
- Excel (`.xls`)

### 🔍 2. Automated Data Profiling

The system analyzes the uploaded dataset and provides information such as:

- Number of rows
- Number of columns
- Data types
- Missing values
- Duplicate records
- Unique values
- Dataset structure

### 🤖 3. AI-Based Quality Detection

The platform automatically identifies potential data-quality problems, including:

- Missing values
- Duplicate records
- Invalid email addresses
- Unusual numeric values
- Potential anomalies
- Inconsistent data

### 🚨 4. Anomaly Detection

The project uses **Isolation Forest** from Scikit-learn to identify unusual records in numerical data.

This helps detect values that may require further investigation.

### 🧹 5. Automated Data Cleaning

The ETL pipeline performs several cleaning operations:

- Removes completely empty rows
- Removes duplicate records
- Removes unnecessary whitespace
- Handles missing numerical values
- Handles missing categorical values
- Standardizes column names

### 🔄 6. ETL Pipeline

The application follows a simplified ETL workflow:

Extract
   ↓
Transform
   ↓
Validate
   ↓
Analyze
   ↓
Load / Export

## 👩‍💻 Author

**Durga Malleswari Sampara**

Computer Science Engineering Student

**Project:** AI Data Quality Guardian

## 🪪License
This project is developed for educational, learning, and portfolio purposes.
