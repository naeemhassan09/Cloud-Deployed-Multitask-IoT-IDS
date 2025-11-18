# 📡 Cloud-Deployed Multitask Deep Learning for IoT Device & Intrusion Detection

### A CNN–Transformer multitask model deployed as an AWS ECS FastAPI microservice.

---

# 👤 Author & Contact

**Author:**  
**Naeem ul Hassan**  
Strategic Engineering Manager | MSc Artificial Intelligence Student  
Dublin, Ireland  
Email: **naeemhassan09@gmail.com**  
Phone: **+353 87 031 1061**
       **+92 336 6622999**

**Supervisor:**  
**Dr. Syed Mustufa**  
Lecturer & Research Supervisor  
Dublin Business School  
Email: **syed.mustufa@dbs.ie**

---

## 📘 Project Summary

This repository contains the source code, data pipeline, training environment, baseline models, and deployment scripts for a **multitask CNN–Transformer framework** designed for **IoT device identification** and **network intrusion detection** using the **CIC IoT-IDAD 2024** dataset.  
The system is implemented as a **containerised FastAPI microservice**, deployed on **AWS ECS Fargate**, with CI/CD automation and optional monitoring through Prometheus and Grafana.

This project is developed as part of the **MSc in Artificial Intelligence** at **Dublin Business School**, supervised by **Dr. Syed Mustufa**, with a strong emphasis on reproducibility, performance benchmarking, and cloud-based deployment.

---

## 🚀 Features
- Multitask CNN–Transformer architecture  
- Two prediction heads:
  - IoT device identification  
  - Intrusion/attack classification  
- Baseline models: XGBoost, BiLSTM, TabNet  
- DVC-tracked data pipeline  
- FastAPI microservice  
- Dockerised deployment  
- AWS ECS Fargate infrastructure  
- CI/CD pipeline (GitHub Actions → ECR → ECS)  
- Optional monitoring with Prometheus/Grafana  

---

## 📛 Badges

![Docker](https://img.shields.io/badge/Docker-ready-blue)
![AWS ECS](https://img.shields.io/badge/AWS-ECS-orange)
![Python](https://img.shields.io/badge/Python-3.11-blue)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

---

## 📂 Repository Structure

```
Cloud-Deployed-Multitask-IoT-IDS/
├── data/
├── src/
│   ├── data/
│   ├── models/
│   ├── training/
│   ├── api/
│   └── utils/
├── configs/
├── experiments/
├── deployment/
├── monitoring/
├── notebooks/
├── docs/
├── admin/
└── README.md
```

---

## 🔧 Installation

### Clone repo
```
git clone https://github.com/naeemhassan09/Cloud-Deployed-Multitask-IoT-IDS.git
cd Cloud-Deployed-Multitask-IoT-IDS
```

### Install dependencies
```
pip install -r requirements.txt
```

### Setup DVC
```
dvc pull
```

---

## 🧹 Data Pipeline

Run full ETL pipeline:

```
dvc repro
```

---

## 🧠 Training

### Baselines
```
python -m src.training.train_baselines --model xgboost
python -m src.training.train_baselines --model bilstm
python -m src.training.train_baselines --model tabnet
```

### Multitask CNN–Transformer
```
python -m src.training.train_multitask \
  --config configs/model_multitask.yaml \
  --training configs/training_multitask.yaml
```

---

## 🖥 Run API

```
uvicorn src.api.main:app --reload
```

Docs at:

```
http://localhost:8000/docs
```

---

## 🐳 Docker

```
docker build -t iot-ids-api -f deployment/Dockerfile.api .
docker run -p 8000:8000 iot-ids-api
```

---

## ☁️ AWS ECS Deployment

Push container:
```
deployment/scripts/build_and_push_ecr.sh
```

Redeploy ECS service:
```
aws ecs update-service --cluster <cluster> --service <service> --force-new-deployment
```

---

## 📈 Monitoring

### Prometheus
Exposes:
```
/metrics
```

Grafana dashboards included in:
```
monitoring/grafana_dashboards/
```

---

## 📘 Documentation

- `docs/architecture_diagram.drawio`
- `docs/methodology_diagram.png`
- `docs/api_openapi_schema.json`
- `docs/results_tables.md`

---



# 📚 About the Author

I work across AI, Deep Learning, IoT Security, and Cloud DevOps.  
My expertise spans full ML lifecycle — ETL → model → API → AWS deployment.

## Core Capabilities

### Machine Learning & AI
- CNNs, Transformers, LSTM/BiLSTM, TabNet  
- Multitask learning  
- IoT device fingerprinting  
- Network intrusion detection  
- ROC/PR analysis & evaluation  

### Cloud & DevOps
- AWS ECS, ECR, Lambda, CloudFront, WAF, RDS  
- Docker & container orchestration  
- GitHub Actions CI/CD  
- Monitoring: Prometheus, Grafana, CloudWatch  

### Software Engineering
- Python, FastAPI, Node.js, NestJS  
- Scalable microservices  
- Logging, observability, cloud automation  

---

# 📜 License — MIT

```
MIT License

Copyright (c) 2025 ...

Permission is hereby granted, free of charge, to any person obtaining a copy
...
(Full MIT License Text)
```
