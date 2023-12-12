# Exploring First-Stage IR Approaches on the CORD-19 dataset
Leonidas Kaldanis, Foteini Papadopoulou, Büsra Yilmaz\
Radboud University, Nijmegen, Netherlands

This project is part of the Information Retrieval master course for AI and Data Science.

## Description
Our project is motivated by the Information Retrieval community’s interest in aiding healthcare professionals and focuses on building a first-stage retrieval system using the CORD-19 dataset. We apply traditional retrieval models(TF-IDF, BM25, Language Model with Dirichlet smoothing) and a neural IR model (Deep Impact) to experiment with different variants of topics. Additionally, we explore the Doc2Query— approach during indexing, measuring its impact on the evaluation metrics and query search runtime. The results revealed that the traditional retrieval models using the standard indexing remain competitive in the TREC-COVID challenge, showing almost the same performance as the advanced neural approaches in evaluation metrics and execution time, with the TF-IDF outperforming. Moreover, the findings suggest that the choice of query variants plays a crucial role, with the description being the best choice in this context. 

## Directories
- `avg_query_time_csv_img` contains the generated images and the csv file for the calculated average query execution time
- `code/python` contains the implementation in plain python.
    - `indexing.py` and `retrieval.py` files have the main code used in our experiments.
- `code/notebook` contains the implementation in jupyter notebook
- `docker_images` contains the docker images needed to run the experiments
- `indexes` contains all the generated indexes from the experiments
- `retrieval_query_time_logs` contains all the logs from all the retreival methods so as to extract the average execution time
### Run it locally
In order to run it locally, a docker container is needed with the libraries installed.
Run the following command in the docker-images folder in order to build and run the container
```docker-compose up --build -d```
If you want to access the container:
``` 
docker ps #get the container id
docker exec -it <container-id> bash 
```
