제 강점은 필요한 지식을 빠르게 배워서 프로젝트에 적용하는 것입니다. 두 가지 프로젝트를 소개하고 싶습니다.

[fitcuration.site]
    Docker를 배운지 일주일 만에 Official Docs 컨트리뷰터가 되는 경험입니다. 
    유저가 키워드로 검색을 하면, 운동을 추천해주는 Django 웹사이트를 만들 때였습니다. 이때 검색 기능을 구현하기 위해서는 형태소 분석기 konlpy를 웹서버와 함께 돌려야만 했습니다. 하지만 형태소 분석기가 python의 라이브러리와 Java를 모두 필요로 했기에, 서버에 환경설정을 하는 일이 어려웠습니다. 이를 해결하기 위해서 Docker image로 해당 환경을 일주일 동안 만들었고, 검색 기능을 AWS EBS 서버에 배포했습니다.
    무엇보다 Docker Official Docs의 버그를 해결하면서 Docker 컨트리뷰터가 됐습니다. 기존에는 Docker Compose로 PostgreSQL DB의 environment arguments를 설정하지 않고 build했습니다. 즉 PostgreSQL DB의 environment arguments와 Django ORM에서 전달받는 arguments와 차이가 났고, 이로 인해서 에러가 발생했습니다. 따라서 Docker Compose에서 DB의 environment arguments를 수정했고, PR은 merge됐습니다.

프로젝트에서 맡은 업무
• Building Recommendation Algorithm
• Fullstack Web Development

사용한 도구
• Natural Language Processing: scikit-learn, gensim, konlpy, soynlp
• Backend: Django, Django ORM & PostgreSQL
• Client: Sass, Tailwind, HTML5
• Deployment: Docker, AWS RDS, AWS S3, AWS EBS

[yonsei.exchange]
    해외 유학 생활을 정리한 NLP 프로젝트를 지난 8월에 공개했습니다. 한 달 동안 유저 900명이 웹사이트에 방문했습니다. 
    사용자들과 많은 의논을 했습니다. 유학을 준비하는 학생들에게 어떤 정보가 필요한지, 웹사이트에 어떤 기능이 필요한지 물어봤습니다. 유저들이 필요하다면, 구현을 처음 시도하는 업무라도 도전했습니다.
    공부 잘하는 순위가 아니라 생활 만족도 순위를 보고 싶어 해서, BERT로 후기 만족도를 도출해야 했습니다. 유저들이 350만 자나 되는 교환 후기를 전부 읽기 힘들다고 해서, 이를 textrank로 요약했습니다. 전공에 적합한 학교를 궁금해해서 단과대와 종합대를 군집화했습니다. 세계 각국의 대학 정보를 모바일에서 볼 수 있어야 했기에 Gatsby.js로 웹사이트를 만들었습니다.

프로젝트에서 맡은 업무
• Sentiment Analysis for Foreign Universities
• Clustering Universities with Departments
• Summarizing Reviews
• Wrangling & Text Data Preprocessing
• Fullstack Web Development

사용한 도구
• Sentiment Analysis: BERT & transformers
• Clustering: scikit-learn, matplotlib
• Summarizing: textrankr, konlpy, soynlp
• Wrangling: pandas, numpy, statistic
• Web: Gatsby.js, GraphQL, Netlify