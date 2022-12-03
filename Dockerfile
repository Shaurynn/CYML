# Dockerfile

FROM python:3.10-slim

EXPOSE 8080

ENV APP_HOME /app
WORKDIR $APP_HOME
COPY . ./

# Install production dependencies.
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

#RUN python fslinstaller.py -d /usr/fsl
#ENV FSLDIR=/usr/fsl/
#ENV PATH=$PATH:$FSLDIR
#ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$FSLDIR
#ENV FSLOUTPUTTYPE=NIFTI_GZ

CMD streamlit run --server.port 8080 app.py

# Run the web service on container startup. Here we use the gunicorn
# webserver, with one worker process and 8 threads.
# For environments with multiple CPU cores, increase the number of workers
# to be equal to the cores available.
# Timeout is set to 0 to disable the timeouts of the workers to allow Cloud Run to handle instance scaling.
#CMD streamlit run --server.port 8080 --server.enableCORS false main.py
