gcloud builds submit --tag gcr.io/ds1016/streamlit-project --timeout 3600 --project=ds1016

gcloud run deploy --image gcr.io/ds1016/streamlit-project --platform managed --project=ds1016 --allow-unauthenticated
