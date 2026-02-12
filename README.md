# AUTO-MAINT: A Serverless Automated MLOps Framework

[cite_start]**AUTO-MAINT** is a fully automated, cloud-based platform designed to democratize **Predictive Maintenance (PdM)** for Micro, Small, and Medium Enterprises (MSMEs)[cite: 31, 32]. [cite_start]By leveraging a serverless microservices architecture on AWS, the framework simplifies the implementation of Machine Learning (ML) models, significantly reducing the technical expertise and capital investment traditionally required for industrial equipment monitoring[cite: 32, 92].

![Auto-Maint Block Diagram](images/Automaint_block_diag.png)

## 🚀 Key Features
* [cite_start]**Serverless MLOps Pipeline**: Fully automated data cleaning, feature engineering, model training, and deployment utilizing AWS Lambda and EC2[cite: 170, 351].
* [cite_start]**Real-Time Predictions**: Employs MQTT streams for near-instantaneous Remaining Useful Life (RUL) estimation with latency under 1 second[cite: 171, 439].
* [cite_start]**High Accuracy**: Features the **GLD Stack** (GRU, LSTM, and Dense layers), achieving an $R^2$ score of over 0.99 across industrial datasets[cite: 379, 502].
* [cite_start]**Cost-Effective**: Designed to operate on a minimal budget; the entire experimentation phase cost less than **$50**[cite: 499].
* [cite_start]**Dual-Mode Interface**: A Flask-based web dashboard supports both novice users through automated templates and advanced users via custom code uploads[cite: 169, 505].

---

## 🏗️ System Architecture
[cite_start]The platform utilizes a microservices paradigm to ensure independent scalability for each stage of the predictive maintenance workflow[cite: 173, 351]:

1. [cite_start]**Data Ingestion**: Users upload machine sensor datasets to **AWS S3** via the web interface[cite: 176, 356].
2. [cite_start]**Automated Cleaning**: An **AWS Lambda** function removes noise, handles missing values (dropping columns with >50% missing data), and standardizes terminology[cite: 187, 188].
3. [cite_start]**Feature Engineering**: **EC2 instances** generate lagged features, calculate anomaly scores using Isolation Forest, and perform KMeans clustering[cite: 204, 205, 208, 212].
4. [cite_start]**Model Training**: Supports templates including LSTM, XGBoost, Random Forest, and custom stacked ensembles, with automated hyperparameter tuning via Optuna[cite: 229, 311].
5. [cite_start]**Live Inference**: **AWS IoT Core** manages unique MQTT topics for real-time sensor data streaming and RUL dashboard updates[cite: 371, 372].

![Auto-Maint Platform Architecture](images/Automaint_architecture_diag.png)

---

## 📊 Model Performance & Efficiency
[cite_start]Our research validated the framework using the NASA Turbofan Jet Engine and Lithium-ion Battery datasets[cite: 455, 464].

### **Prediction Accuracy**
[cite_start]The **GLD Stack** consistently outperformed traditional regression and ensemble methods[cite: 466, 469].

| Model Template | $R^{2}$ Score (NASA FD001) | MAE (NASA FD001) | MSE (NASA FD001) |
| :--- | :--- | :--- | :--- |
| **GLD Stack** | **0.99011** | **4.736** | **42.455** |
| LSTM | 0.79472 | 20.290 | 581.144 |
| XGBoost | 0.56987 | 32.871 | 1984.818 |
| SVR | 0.55765 | 32.406 | 2041.204 |


![R2 Score Comparison Graph](images/R2_SCORE_graph.png)

### **Training Efficiency**
[cite_start]Training times remain computationally feasible even for complex architectures[cite: 487, 488].

| Model Template | Training Time (20k rows) |
| :--- | :--- |
| **LSTM (1 iteration)** | **5 minutes** |
| LightGBM (50 iterations) | 6 minutes |
| Stacking Ensemble | 16 minutes |
| GLD Stack (27 permutations) | 270 minutes |



---

## 🛠️ Tech Stack
* [cite_start]**Cloud Infrastructure**: AWS (Lambda, S3, EC2, IoT Core) [cite: 170, 351]
* [cite_start]**Web Framework**: Python, Flask [cite: 97, 167]
* [cite_start]**Database**: Firebase / Firestore [cite: 97, 351]
* [cite_start]**Machine Learning**: TensorFlow/Keras (LSTMs, GRUs), Scikit-Learn, Optuna [cite: 311, 472]
* [cite_start]**Communication**: MQTT Protocol [cite: 97, 343]

---

## 📖 Citation
If you use this code or framework in your research, please cite our paper:

> Acharya V., Saketh V.N., Zaki A. et al. **A serverless automated MLOps framework for scalable industrial predictive maintenance**. *Discov Internet Things* (2026). https://doi.org/10.1007/s43926-026-00291-0

---

## 🤝 Contributors
* [cite_start]**Vadiraja Acharya** – Original Concept & Guidance [cite: 529]
* **V. [cite_start]Naga Saketh** – Architecture Design & ML Implementation [cite: 529, 530]
* [cite_start]**Adnan Zaki** – Frontend User Interface [cite: 530]
* **M. [cite_start]Manas Gowda** – AWS Deployment & Firebase Setup [cite: 530, 531]
* [cite_start]**Naitik Jain** – Machine Learning & Literature Survey [cite: 531]
* [cite_start]**Prasad B. Honnavalli** – Project Management & Funding [cite: 531, 532]
