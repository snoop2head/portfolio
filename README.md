# snoop2head's portfolio

**Capture Questions, Answer with Code**

- Name: Young Jin Ahn
- Email: young_ahn@yonsei.ac.kr
- Blog: https://snoop2head.github.io/

---

## 📖 Education

### Yonsei University

**Bachelor of Economics (Major) and Applied Statistics (Minor)**

- INTRODUCTION TO STATISTICS (A0)
- STATISTICAL METHOD (A+)
- CALCULUS (TBD)
- LINEAR ALGEBRA (B+)
- MATHEMATICAL STATISTICS 1 (A+)
- LINEAR REGRESSION (B+)
- R AND PYTHON PROGRAMMING (A+)
- DATA STRUCTURE (TBD)
- SPECIAL PROBLEMS IN COMPUTING (A0)
- SOCIAL INFORMATICS (A+)
- TIME SERIES ANALYSIS (A+)


## 🏆 Competition Awards

|            Host / Platform            |                         Topic / Task                         |          Result          |                          Repository                          | Year |
| :-----------------------------------: | :----------------------------------------------------------: | :----------------------: | :----------------------------------------------------------: | :----: |
| National IT Industry<br />Promotion Agency | [Machine Reading Compehension](https://aichallenge.or.kr/competition/detail/1/task/5/taskInfo) | 🥈 2nd <br />(2/26)  | [<img width="18" alt="image" src="./images/GitHub-Mark/PNG/GitHub-Mark-Light-64px.png"> MRC_Baseline](https://github.com/QuoQA-NLP/MRC_Baseline) | 2022 |
|        Ministry of Statistics         | [Korean Standard Industry Classification](https://data.kostat.go.kr/sbchome/bbs/boardList.do?boardId=SBCSBBS_000000025000&curMenuNo=OPT_09_02_00_0) | 🎖 7th <br />(7/311) |                              -                               | 2022 |
|                 Dacon                 | [KLUE benchmark Natural Language Inference](https://dacon.io/competitions/official/235875/overview/description) | 🥇 1st <br />(1/468) | [🌐 KLUE NLI](https://dacon.io/competitions/official/235875/codeshare/4589) | 2022 |
|           Dacon & AI Frenz            | [Python Code Clone Detection](https://dacon.io/competitions/official/235900/overview/description) | 🥉 3rd <br />(3/337) | [<img width="18" alt="image" src="./images/GitHub-Mark/PNG/GitHub-Mark-Light-64px.png"> CloneDetection](https://github.com/sangHa0411/CloneDetection) | 2022 |
|          Dacon & CCEI Korea           | [Stock Price Forecast on KOSPI & KOSDAQ](https://dacon.io/competitions/official/235857/overview/description) | 🎖 6th <br />(6/205) | [<img width="18" alt="image" src="./images/GitHub-Mark/PNG/GitHub-Mark-Light-64px.png"> elastic-stock-prediction](https://github.com/snoop2head/elastic-stock-prediction) | 2021 |

**Dacon is Kaggle alike competitive data science and deep learning platform in Korea.

---

## 🛠 Multimodal Projects

### [<img width="18" alt="image" src="./images/GitHub-Mark/PNG/GitHub-Mark-Light-64px.png"> KoDALLE: Text to Fashion (2021)](https://github.com/KR-HappyFace/KoDALLE)

[<img width="700" alt="image" src="https://github.com/KR-HappyFace/KoDALLE/raw/main/assets/README/image-20211227151557604.png">](https://github.com/KR-HappyFace/KoDALLE)

**Generating dress outfit images based on given input text** | [📄 Presentation](https://github.com/KR-HappyFace/KoDALLE/blob/main/README.pdf)

- **Created training pipeline from VQGAN through DALLE**
- **Maintained versions of 1 million pairs image-caption dataset.**
- Trained VQGAN and DALLE model from the scratch.
- Established live demo for the KoDALLE on Huggingface Space via FastAPI.


## 🔐 Deep Learning Security Projects

### [<img width="18" alt="image" src="./images/GitHub-Mark/PNG/GitHub-Mark-Light-64px.png"> Language Model Memorization (2022)](https://github.com/yonsei-cysec/Language_Model_Memorization)

**Implementation of Carlini et al(2020) [Extracting Training Data from Large Language Models](https://arxiv.org/abs/2012.07805)**

- Accelerated inference speed with parallel Multi-GPU usage.
- Ruled out 'low-quality repeated generations' appeared in the paper with repetition penalty and with ngram restriction.


### [<img width="18" alt="image" src="./images/GitHub-Mark/PNG/GitHub-Mark-Light-64px.png"> Membership Inference Attack (2022)](https://github.com/yonsei-cysec/Membership_Inference_Attack)

**Implementation of Shokri et al(2016) [Membership Inference Attacks Against Machine Learning Models](https://arxiv.org/pdf/1610.05820.pdf)**

- Modified shadow models' training methodology to prevent overfitting.
- Referenced [Carlini et al(2021)](https://arxiv.org/abs/2112.03570) to conduct further research on different types of models and metrics.
- Reproduced attack metrics as the following.

| MIA Attack Metrics | Accuracy | Precision | Recall | F1 Score |
| :----------------: | :------: | :-------: | :----: | :------: |
|      CIFAR10       |  0.8376  |  0.8087   | 0.8834 |  0.8444  |
|      CIFAR100      |  0.9746  |  0.9627   | 0.9875 |  0.9749  |


## 💬 Natural Language Processing Projects

### [<img width="18" alt="image" src="./images/GitHub-Mark/PNG/GitHub-Mark-Light-64px.png"> KoQuillBot (2022)](https://github.com/QuoQA-NLP/KoQuillBot) & [<img width="18" alt="image" src="./images/GitHub-Mark/PNG/GitHub-Mark-Light-64px.png"> T5 Translation (2022)](https://github.com/QuoQA-NLP/T5_Translation)

**Paraphrasing tool with round trip translation utilizing T5 Machine Translation.** | [🤗 KoQuillBot Demo](https://huggingface.co/spaces/QuoQA-NLP/KoQuillBot) & [🤗 Translator Demo](https://huggingface.co/spaces/QuoQA-NLP/QuoQaGo)

|                  | BLEU Score |                      Translation Result                      |
| :--------------: | :--------: | :----------------------------------------------------------: |
| Korean ➡️ English |   45.15    | [🔗 Inference Result](https://huggingface.co/datasets/QuoQA-NLP/KE-T5-Ko2En-Base-Inference-Result) |
| English ➡️ Korean |     -      |                              -                               |

### [<img width="18" alt="image" src="./images/GitHub-Mark/PNG/GitHub-Mark-Light-64px.png"> Deep Encoder Shallow Decoder (2022)](https://github.com/snoop2head/Deep-Encoder-Shallow-Decoder)

**Implementation of Kasai et al(2020) [Deep Encoder, Shallow Decoder: Reevaluating Non-autoregressive Machine Translation](https://arxiv.org/abs/2006.10369)** | [📄 Translation Output](https://docs.google.com/spreadsheets/d/1IqEuRuEpphPEX3ni1m0EwqYuOU4E4t4-jC6uullpJhE/edit#gid=204599913)

- Composed custom dataset, trainer, inference code in pytorch and huggingface.
- Trained and hosted encoder-decoder transformers model using huggingface.

|                  | BLEU Score |                      Translation Result                      |
| :----------------: | :--------: | :----------------------------------------------------------: |
| Korean ➡️ English |   35.82    | [🔗 Inference Result](https://docs.google.com/spreadsheets/d/1IqEuRuEpphPEX3ni1m0EwqYuOU4E4t4-jC6uullpJhE/edit#gid=204599913) |
| English ➡️ Korean |     -      |                              -                               |

### [<img width="18" alt="image" src="./images/GitHub-Mark/PNG/GitHub-Mark-Light-64px.png"> KLUE-RBERT (2022)](https://github.com/snoop2head/KLUE-RBERT)

**Extracting relations between subject and object entity in KLUE Benchmark dataset** | [✍️ Blog Post](https://snoop2head.github.io/Relation-Extraction-Code/)

- Finetuned RoBERTa model according to RBERT structure in pytorch.
- Applied stratified k-fold cross validation for the custom trainer.

### [<img width="18" alt="image" src="./images/GitHub-Mark/PNG/GitHub-Mark-Light-64px.png"> Conditional Generation with KoGPT (2021)](https://github.com/snoop2head/KoGPT-Joong-2)

**Sentence generation with given emotion conditions** | [🤗 Huggingface Demo](https://huggingface.co/spaces/snoop2head/KoGPT-Conditional-Generation)

- Finetuned KoGPT-Trinity with conditional emotion labels.
- Maintained huggingface hosted model and live demo.

### [<img width="18" alt="image" src="./images/GitHub-Mark/PNG/GitHub-Mark-Light-64px.png"> Machine Reading Comprehension in Naver Boostcamp (2021)](https://snoop2head.github.io/Custom-MRC-Reader/)

**Retrieved and extracted answers from wikipedia texts for given question** | [✍️ Blog Post](https://snoop2head.github.io/Custom-MRC-Reader/)

- Attached bidirectional LSTM layers to the backbone transformers model to extract answers.
- Divided benchmark into start token prediction accuracy and end token prediction accuracy.

### [<img width="18" alt="image" src="./images/GitHub-Mark/PNG/GitHub-Mark-Light-64px.png"> Mathpresso Corporation Joint Project (2020)](https://github.com/snoop2head/Mathpresso_Classification)

**Corporate joint project for mathematics problems classification task** | [📄 Presentation](https://github.com/snoop2head/Mathpresso_Classification/blob/main/YBIGTA_%EB%A7%A4%EC%93%B0%ED%94%84%EB%A0%88%EC%86%8C_%EB%AA%BD%EB%8D%B0%EC%9D%B4%ED%81%AC_Final.pdf)

- Preprocessed Korean mathematics problems dataset based on EDA.
- Maintained version of preprocessing module.


### [<img width="18" alt="image" src="./images/GitHub-Mark/PNG/GitHub-Mark-Light-64px.png"> Constructing Emotional Instagram Posts Dataset (2019)](https://github.com/Keracorn/geulstagram)

**Created Emotional Instagram Posts(글스타그램) dataset** | [📄 Presentation](https://github.com/Keracorn/geulstagram/blob/master/README.pdf)

- Managed version control for the project Github Repository.
- Converted Korean texts on image file into text file using Google Cloud Vision API.


## 👀 Computer Vision Projects

### [<img width="18" alt="image" src="./images/GitHub-Mark/PNG/GitHub-Mark-Light-64px.png"> ElimNet (2021)](https://github.com/snoop2head/ELimNet)

**Elimination based Lightweight Neural Net with Pretrained Weights** | [📄 Presentation](https://github.com/snoop2head/ELimNet/blob/main/README.pdf)

- Constructed lightweight CNN model with less than 1M #params by removing top layers from pretrained CNN models.
- Assessed on Trash Annotations in Context(TACO) Dataset sampled for 6 classes with 20,851 images.
- Compared metrics accross VGG11, MobileNetV3 and EfficientNetB0.

### [<img width="18" alt="image" src="./images/GitHub-Mark/PNG/GitHub-Mark-Light-64px.png"> Face Mask, Age, Gender Classification in Naver Boostcamp (2021)](https://github.com/boostcampaitech2/image-classification-level1-23)

**Identifying 18 classes from given images: Age Range(3 classes), Biological Sex(2 classes), Face Mask(3 classes)** | [✍️ Blog Post](https://snoop2head.github.io/Mask-Age-Gender-Classification-Competition/)

- Optimized combination of backbone models, losses and optimizers.
- Created additional dataset with labels(age, sex, mask) to resolve class imbalance.
- Cropped facial characteristics with MTCNN and RetinaFace to reduce noise in the image.


### [<img width="18" alt="image" src="./images/GitHub-Mark/PNG/GitHub-Mark-Light-64px.png"> Realtime Desktop Posture Classification (2020)](https://github.com/snoop2head/ml_classification_tutorial)

**Real-time desk posture classification through webcam** | [📷 Demo Video](https://www.youtube.com/watch?v=6z_TJaj71io&t=459s)

- Created real-time detection window using opencv-python.
- Converted image dataset into Yaw/Pitch/Roll numerical dataset using RetinaFace model.
- Trained and optimized random forest classification model with precision rate of 93%.


## 🕸 Web Projects

### [<img width="18" alt="image" src="./images/GitHub-Mark/PNG/GitHub-Mark-Light-64px.png"> Exchange Program Overview Website (2020)](https://github.com/snoop2head/yonsei-exchange-program)

**Overview for student life in foreign universities** | [✈️ Website Demo](https://yonsei-exchange.netlify.app/)

- **3400 Visitors within a year (2021.07 ~ 2022.07)**
- **22000 Pageviews within a year (2021.07 ~ 2022.07)**
- **3 minutes+ of Average Retention Time**

<img height="300" width="200" alt="image" src="./images/yonsei_exchange1.png"><img height="300" width="250" alt="image" src="./images/yonsei_exchange2.png">

- Collected and preprocessed 11200 text review data from the Yonsei website using pandas.
- Visualized department distribution and weather information using matplotlib.
- Sentiment analysis on satisfaction level for foreign universities with pretrained BERT model.
- Clustered universities with provided curriculum with K-means clustering.
- Hosted reports on universities using Gatsby.js, GraphQL, and Netlify.

### [<img width="18" alt="image" src="./images/GitHub-Mark/PNG/GitHub-Mark-Light-64px.png"> fitcuration website (2020)](https://github.com/snoop2head/fitcuration-django)

**Search-based exercise retrieval web service** | [📷 Demo Video](https://youtu.be/kef0CxzMANo?t=38)

- Built retrieval algorithm based on search keyword using TF-IDF.
- Deployed website using Docker, AWS RDS, AWS S3, AWS EBS
- Constructed backend using Django, Django ORM & PostgreSQL.
- Composed client-side using Sass, Tailwind, HTML5.

<img width="160" alt="image" src="./images/fit_1_home_1.jpg"><img width="160" alt="image" src="./images/fit_2_search_1.jpg"><img width="150" alt="image" src="./images/fit_5_category_2.jpg"><img width="160" alt="image" src="./images/fit_4_user.jpg">


## 💰 Quantitative Finance Projects

### [<img width="18" alt="image" src="./images/GitHub-Mark/PNG/GitHub-Mark-Light-64px.png"> Forecasting Federal Rate with Lasso Regression Model (2022)](https://github.com/snoop2head/Federal-Rate-Prediction)

**Federal Rate Prediction for the next FOMC Meeting**

- Wrangled quantitative dataset with Finance Data Reader.
- Yielded metrics and compared candidate regression models for the adaquate fit.
- Hyperparameter optimization for the candidate models.

### [<img width="18" alt="image" src="./images/GitHub-Mark/PNG/GitHub-Mark-Light-64px.png"> Korean Spinoff Event Tracker (2020)](https://github.com/snoop2head/spinoff_hunter_kor)

**Get financial data of public companies involved in spinoff events on Google Spreadsheet** | [🧩 Dataset Demo](https://docs.google.com/spreadsheets/d/1chJ2NKHVc0gKjsMaQI1UHEPxdjneV1ZWaTGHseQvxP4/edit?usp=sharing)

- Wrangled finance dataset which are displayed on Google Sheets

---

## 🏷 Opensource Contributions

### [<img width="18" alt="image" src="./images/GitHub-Mark/PNG/GitHub-Mark-Light-64px.png"> NVlabs/stylegan2-ada-pytorch (2021)](https://github.com/NVlabs/stylegan2-ada-pytorch)

**Fixed torch version comparison fallback error for source repo of NVIDIA Research** | [✍️ Pull Request](https://github.com/NVlabs/stylegan2-ada-pytorch/pull/197)

- Skills: torch, torchvision

### [<img width="18" alt="image" src="./images/GitHub-Mark/PNG/GitHub-Mark-Light-64px.png"> docker/docker.github.io (2020)](https://github.com/docker/docker.github.io)

**Updated PostgreSQL initialization for "Quickstart: dockerizing django" documentation** | [🐳 Pull Request](https://github.com/docker/docker.github.io/pull/10624)

- Skills: Docker, docker-compose, Django


## 🗄 ETCs


### [<img width="18" alt="image" src="./images/GitHub-Mark/PNG/GitHub-Mark-Light-64px.png"> Covid19 Confirmed Cases Prediction (2020)](https://github.com/Rank23/COVID19)

**Predict the spread of COVID-19 in early stage after its entrance to country.**

- Fixed existing errors on Github Repository.
- Wrote footnotes in both English and Korean.
- ±5% accuracy for one-day prediction.
- ±10% accuracy for 30-day prediction.

### [<img width="18" alt="image" src="./images/GitHub-Mark/PNG/GitHub-Mark-Light-64px.png"> Indigo (2019)](https://github.com/snoop2head/indigo)

**Don't miss concerts for your favorite artists with KakaoTalk Chatbot** | [📷 Demo Video](https://www.youtube.com/watch?v=uIOWqumaOD4)

- Created API server for KakaoTalk chatbot with Flask, Pymongo and MongoDB.
- Deployed the API server on AWS EC2.
- Visualized concert schedules on user's Google Calendar.
- Created / Updated events in Google Calendar.


---

## 🛠 Skillsets

**Data Analysis and Machine Learning**

- Data Analysis Library: pandas, numpy
- Deep Learning: pytorch, transformers
- Machine Learning: scikit-learn, gensim, xgboost

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
