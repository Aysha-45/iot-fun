# IoT Final Project  

## 📌 Project Overview  
This project focuses on an IoT-based data collection, analysis, and visualization system using **InfluxDB, Docker, and machine learning** for predictive insights. It integrates **sensor data acquisition, cloud storage, and data analytics** to monitor and analyze real-time information effectively.  

## 🔧 Technologies Used  
- **Hardware:** Arduino (`PROJECT 4.ino`)  
- **Database:** InfluxDB  
- **Containerization:** Docker & Docker Compose  
- **Data Processing:** Python (`Data_Analysis.py`, `model_train.py`)  
- **Messaging System:** MQTT (`consumer.py`)  

## 📂 Project Structure  
```
📁 IoT-FinalProject  
├── 📜 PROJECT 4.ino → Microcontroller firmware for sensor data acquisition  
├── 📜 consumer.py → MQTT consumer for data ingestion into InfluxDB  
├── 📜 docker-compose.yml → Dockerized setup for InfluxDB & related services  
├── 📜 model_train.py → Machine learning model for predictive analytics  
├── 📜 Data_Analysis.py → Data processing and visualization scripts  
├── 📜 .env → Configuration file for environment variables  
```

## 🚀 Features  
✅ **Real-time IoT Data Streaming via MQTT**  
✅ **Efficient Data Storage using InfluxDB**  
✅ **Predictive Analytics using ML models**  
✅ **Scalable Deployment with Docker**  

## 📊 Future Enhancements  
- 🚀 Implement advanced ML models for better predictive accuracy  
- 🌐 Develop a web dashboard for real-time monitoring  
- 💾 Optimize data storage for large-scale deployments  
