#base image
FROM python:3.10.12-slim-buster

#work directory
WORKDIR /app

#copy dependencies to work directory
COPY api/requirements.txt .

#install dependencies
RUN pip install -U pip && pip install -r requirements.txt

#copy local directory to cotainer directory
COPY api/ ./api

#copy model
COPY model/ridge_model.pkl ./model/ridge_model.pkl

#copy file for run app
COPY initializer.sh .

#change permits to run into container
RUN chmod +x initializer.sh

#port
EXPOSE 8000

#establish script for execute app
ENTRYPOINT ["./initializer.sh"]