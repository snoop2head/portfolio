# snoop2head's portfolio

**Capture Questions, Answer with Code**

- Name: Young Jin Ahn
- Department: Economics Major and Statistics Minor at Yonsei University
- Email: young_ahn@yonsei.ac.kr
- Blog: https://snoop2head.github.io/

----

## 🏆 Competition Awards

|            Host / Platform            |                         Topic / Task                         |          Result          |                          Repository                          |
| :-----------------------------------: | :----------------------------------------------------------: | :----------------------: | :----------------------------------------------------------: |
| National IT Industry Promotion Agency | [Machine Reading Compehension](https://aichallenge.or.kr/competition/detail/1/task/5/taskInfo) | 🥈 2nd Place<br />(2/26)  | [<img width="18" alt="image" src="./images/GitHub-Mark/PNG/GitHub-Mark-Light-64px.png">MRC_Baseline](https://github.com/QuoQA-NLP/MRC_Baseline) |
|        Ministry of Statistics         | [Korean Standard Industry Classification](https://data.kostat.go.kr/sbchome/bbs/boardList.do?boardId=SBCSBBS_000000025000&curMenuNo=OPT_09_02_00_0) | 🎖 7th Place<br />(7/311) |                              -                               |
|                 Dacon                 | [KLUE benchmark Natural Language Inference](https://dacon.io/competitions/official/235875/overview/description) | 🥇 1st Place<br />(1/468) | [<img width="18" alt="image" src="./images/GitHub-Mark/PNG/GitHub-Mark-Light-64px.png">KLUE NLI](https://dacon.io/competitions/official/235875/codeshare/4589?page=1&dtype=recent) |
|           Dacon & AI Frenz            | [Python Code Clone Detection](https://dacon.io/competitions/official/235900/overview/description) | 🥉 3rd Place<br />(3/337) | [<img width="18" alt="image" src="./images/GitHub-Mark/PNG/GitHub-Mark-Light-64px.png">CloneDetection](https://github.com/sangHa0411/CloneDetection) |
|          Dacon & CCEI Korea           | [Stock Price Forecast on KOSPI & KOSDAQ](https://dacon.io/competitions/official/235857/overview/description) | 🎖 6th Place<br />(6/205) | [<img width="18" alt="image" src="./images/GitHub-Mark/PNG/GitHub-Mark-Light-64px.png">elastic-stock-prediction](https://github.com/snoop2head/elastic-stock-prediction) |

**Dacon is Kaggle alike competitive data science and deep learning platform in Korea.

## 🛠 Multimodal Projects

### [<img width="18" alt="image" src="./images/GitHub-Mark/PNG/GitHub-Mark-Light-64px.png"> KoDALLE: Text to Fashion](https://github.com/KR-HappyFace/KoDALLE)

[<img width="700" alt="image" src="https://github.com/KR-HappyFace/KoDALLE/raw/main/assets/README/image-20211227151557604.png">](https://github.com/KR-HappyFace/KoDALLE)

**Generating dress outfit images based on given input text** | [📄 Presentation](https://github.com/KR-HappyFace/KoDALLE/blob/main/README.pdf)

- **Created training pipeline from VQGAN through DALLE**
- **Maintained versions of 1 million pairs image-caption dataset.**
- Trained VQGAN and DALLE model from the scratch.
- Established live demo for the KoDALLE on Huggingface Space via FastAPI.

---

## 💬 Natural Language Processing Projects

### [<img width="18" alt="image" src="./images/GitHub-Mark/PNG/GitHub-Mark-Light-64px.png"> Deep Encoder Shallow Decoder](https://github.com/snoop2head/Deep-Encoder-Shallow-Decoder)

**Huggingface implementation for the paper "Deep Encoder, Shallow Decoder: Reevaluating Non-autoregressive Machine Translation"** | [📄 Translation Output](https://docs.google.com/spreadsheets/d/1IqEuRuEpphPEX3ni1m0EwqYuOU4E4t4-jC6uullpJhE/edit#gid=204599913)

- Composed custom dataset, trainer, inference code in pytorch.
- Trained and hosted encoder-decoder transformers model using huggingface.

### [<img width="18" alt="image" src="./images/GitHub-Mark/PNG/GitHub-Mark-Light-64px.png"> KLUE-RBERT](https://github.com/snoop2head/KLUE-RBERT)

**Extracting relations between subject and object entity in KLUE Benchmark dataset** | [✍️ Blog Post](https://snoop2head.github.io/Relation-Extraction-Code/)

- Finetuned RoBERTa model according to RBERT structure in pytorch.
- Applied stratified k-fold cross validation for the custom trainer.

### [<img width="18" alt="image" src="./images/GitHub-Mark/PNG/GitHub-Mark-Light-64px.png"> Conditional Generation with KoGPT](https://github.com/snoop2head/KoGPT-Joong-2)

**Sentence generation with given emotion conditions** | [🤗 Huggingface Demo](https://huggingface.co/spaces/snoop2head/KoGPT-Conditional-Generation)

- Finetuned KoGPT-Trinity with conditional emotion labels.
- Maintained huggingface hosted model and live demo.

### [<img width="18" alt="image" src="./images/GitHub-Mark/PNG/GitHub-Mark-Light-64px.png"> Emotional Instagram Posts(글스타그램) Dataset](https://github.com/Keracorn/geulstagram)

**Created Emotional Instagram Posts(글스타그램) dataset** | [📄 Presentation](https://github.com/Keracorn/geulstagram/blob/master/README.pdf)

- Managed version control for the project Github Repository.
- Converted Korean texts on image file into text file using Google Cloud Vision API.

### [<img width="18" alt="image" src="./images/GitHub-Mark/PNG/GitHub-Mark-Light-64px.png"> Machine Reading Comprehension in Naver Boostcamp](https://snoop2head.github.io/Custom-MRC-Reader/)

**Retrieved and extracted answers from wikipedia texts for given question** | [✍️ Blog Post](https://snoop2head.github.io/Custom-MRC-Reader/)

- Attached bidirectional LSTM layers to the backbone transformers model to extract answers.
- Divided benchmark into start token prediction accuracy and end token prediction accuracy.

### [<img width="18" alt="image" src="./images/GitHub-Mark/PNG/GitHub-Mark-Light-64px.png"> Mathpresso Corporation Joint Project](https://github.com/snoop2head/Mathpresso_Classification)

**Corporate joint project for mathematics problems classification task** | [📄 Presentation](https://github.com/snoop2head/Mathpresso_Classification/blob/main/YBIGTA_%EB%A7%A4%EC%93%B0%ED%94%84%EB%A0%88%EC%86%8C_%EB%AA%BD%EB%8D%B0%EC%9D%B4%ED%81%AC_Final.pdf)

- Preprocessed Korean mathematics problems dataset based on EDA.
- Maintained version of preprocessing module.

---

## 👀 Computer Vision Projects

### [<img width="18" alt="image" src="./images/GitHub-Mark/PNG/GitHub-Mark-Light-64px.png"> Realtime Turtleneck Prevention](https://github.com/snoop2head/ml_classification_tutorial)

**Real-time desk posture detection through webcam** | [📷 Demo Video](https://www.youtube.com/watch?v=6z_TJaj71io&t=459s)

- Created real-time detection window using opencv-python.
- Converted image dataset into Yaw/Pitch/Roll numerical dataset using RetinaFace model.
- Trained and optimized random forest classification model with precision rate of 93%.

### [<img width="18" alt="image" src="./images/GitHub-Mark/PNG/GitHub-Mark-Light-64px.png"> ELimNet](https://github.com/snoop2head/ELimNet)

**Elimination based Lightweight Neural Net with Pretrained Weights** | [📄 Presentation](https://github.com/snoop2head/ELimNet/blob/main/README.pdf)

- Constructed lightweight CNN model with less than 1M #params by removing top layers from pretrained CNN models.
- Assessed on Trash Annotations in Context(TACO) Dataset sampled for 6 classes with 20,851 images.
- Compared metrics accross VGG11, MobileNetV3 and EfficientNetB0.

### [<img width="18" alt="image" src="./images/GitHub-Mark/PNG/GitHub-Mark-Light-64px.png"> Face Mask, Age, Gender Classification in Naver Boostcamp](https://github.com/boostcampaitech2/image-classification-level1-23)

**Identifying 18 classes from given images: Age Range(3 classes), Biological Sex(2 classes), Face Mask(3 classes)** | [✍️ Blog Post](https://snoop2head.github.io/Mask-Age-Gender-Classification-Competition/)

- Optimized combination of backbone models, losses and optimizers.
- Created additional dataset with labels(age, sex, mask) to resolve class imbalance.
- Cropped facial characteristics with MTCNN and RetinaFace to reduce noise in the image.

---

## 🕸 Web Projects

### [<img width="18" alt="image" src="./images/GitHub-Mark/PNG/GitHub-Mark-Light-64px.png"> Exchange Program Overview Website](https://github.com/snoop2head/yonsei-exchange-program)

**Overview for student life in foreign universities** | [✈️ Website Demo](https://yonsei-exchange.netlify.app/)

- **6000 Pageviews within 6 Months**
- **4 minutes+ of Average Retention Time**

<img height="300" width="200" alt="image" src="./images/yonsei_exchange1.png"><img height="300" width="250" alt="image" src="./images/yonsei_exchange2.png">

- Collected and preprocessed 11200 text review data from the Yonsei website using pandas.
- Visualized department distribution and weather information using matplotlib.
- Sentiment analysis on satisfaction level for foreign universities with pretrained BERT model.
- Clustered universities with provided curriculum with K-means clustering.
- Hosted reports on universities using Gatsby.js, GraphQL, and Netlify.

### [<img width="18" alt="image" src="./images/GitHub-Mark/PNG/GitHub-Mark-Light-64px.png"> fitcuration website](https://github.com/snoop2head/fitcuration-django)

**Search-based exercise retrieval web service** | [📷 Demo Video](https://youtu.be/kef0CxzMANo?t=38)

- Built retrieval algorithm based on search keyword using TF-IDF.
- Deployed website using Docker, AWS RDS, AWS S3, AWS EBS
- Constructed backend using Django, Django ORM & PostgreSQL.
- Composed client-side using Sass, Tailwind, HTML5.

<img width="160" alt="image" src="./images/fit_1_home_1.jpg"><img width="160" alt="image" src="./images/fit_2_search_1.jpg"><img width="150" alt="image" src="./images/fit_5_category_2.jpg"><img width="160" alt="image" src="./images/fit_4_user.jpg">

---

## 💰 Quantitative Finance Projects

### [<img width="18" alt="image" src="./images/GitHub-Mark/PNG/GitHub-Mark-Light-64px.png"> Stock Price Prediction Competition @DACON](https://github.com/snoop2head/elastic-stock-prediction)

**Top 5% in Dacon's _Stock Price Prediction Competition_** | [✍️ Blog Post](https://snoop2head.github.io/Dacon-Stock-Price-Competition/)

- Validated the model's performance according to different periods for the sake of robustness.
- Applied cross validation by using ElasticNetCV model.
- Completed the model's inference for the evaluation period.
- Tested ARIMA, RandomforestRegressor and ElasticNetCV.

### [<img width="18" alt="image" src="./images/GitHub-Mark/PNG/GitHub-Mark-Light-64px.png"> Forecasting Federal Rate with Lasso Regression Model](https://github.com/snoop2head/Federal-Rate-Prediction)

**Federal Rate Prediction for the next FOMC Meeting**

- Wrangled quantitative dataset with Finance Data Reader.
- Yielded metrics and compared candidate regression models for the adaquate fit.
- Hyperparameter optimization for the candidate models.

### [<img width="18" alt="image" src="./images/GitHub-Mark/PNG/GitHub-Mark-Light-64px.png"> Korean Spinoff Event Tracker](https://github.com/snoop2head/spinoff_hunter_kor)

**Get financial data of public companies involved in spinoff events on Google Spreadsheet** | [🧩 Dataset Demo](https://docs.google.com/spreadsheets/d/1chJ2NKHVc0gKjsMaQI1UHEPxdjneV1ZWaTGHseQvxP4/edit?usp=sharing)

- Wrangled finance dataset which are displayed on Google Sheets

---

## Opensource Contributions

### [<img width="18" alt="image" src="./images/GitHub-Mark/PNG/GitHub-Mark-Light-64px.png"> docker/docker.github.io](https://github.com/docker/docker.github.io)

**Updated PostgreSQL initialization for "Quickstart: dockerizing django" documentation** | [🐳 Pull Request](https://github.com/docker/docker.github.io/pull/10624)

- Duration: March 2020 ~ April 2020
- Skills
  - Backend: Django, Django ORM & PostgreSQL
  - Deployment: Docker, docker-compose

### [<img width="18" alt="image" src="./images/GitHub-Mark/PNG/GitHub-Mark-Light-64px.png"> NVlabs/stylegan2-ada-pytorch](https://github.com/NVlabs/stylegan2-ada-pytorch)

**Fixed torch version comparison fallback error for source repo of NVIDIA Research** | [✍️ Pull Request](https://github.com/NVlabs/stylegan2-ada-pytorch/pull/197)

- Duration: November 2020
- Skills: torch, torchvision

---

## ETC

### [<img width="18" alt="image" src="./images/GitHub-Mark/PNG/GitHub-Mark-Light-64px.png"> Indigo](https://github.com/snoop2head/indigo)

**Don't miss concerts for your favorite artists with KakaoTalk Chatbot** | [📷 Demo Video](https://www.youtube.com/watch?v=uIOWqumaOD4)

- Created API server for KakaoTalk chatbot with Flask, Pymongo and MongoDB.
- Deployed the API server on AWS EC2.
- Visualized concert schedules on user's Google Calendar.
- Created / Updated events in Google Calendar.

### [<img width="18" alt="image" src="./images/GitHub-Mark/PNG/GitHub-Mark-Light-64px.png"> Covid19 Confirmed Cases Prediction](https://github.com/Rank23/COVID19)

**Predict the spread of COVID-19 in early stage after its entrance to country.**

- Fixed existing errors on Github Repository.
- Wrote footnotes in both English and Korean.
- ±5% accuracy for one-day prediction.
- ±10% accuracy for 30-day prediction.

---

## Skillsets

**Data Analysis**

- Data Analysis Library: pandas, numpy
- Deep Learning: pytorch, transformers
- Machine Learning: scikit-learn, gensim

**Backend**

- Python / Django - Django ORM, CRUD, OAuth
- Python / FastAPI(uvicorn) - CRUD API
- Python / Flask - CRUD API

**Client**

- HTML / Pug.js
- CSS / Sass, Tailwind, Bulma
- JavaScript / ES6

**Deployment**

- Docker, docker-compose
- AWS EC2, Google Cloud App Engine
- AWS S3, RDS (PostgreSQL)
- AWS Elastic Beanstalk, CodePipeline;

### Courses and Lectures

- MATHEMATICAL STATISTICS 1 (A+)
- STATISTICAL METHOD (A+)
- LINEAR ALGEBRA (B+)
- LINEAR REGRESSION (B+)
- R AND PYTHON PROGRAMMING (A+)
- TIME SERIES ANALYSIS (A+)
- FINANCIAL ECONOMICS (B+)
- SOCIAL INFORMATICS (A+)
- INTRODUCTION TO STATISTICS (A0)
