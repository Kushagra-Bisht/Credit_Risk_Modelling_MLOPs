FROM python:3.9

WORKDIR /app

# Copy the content of your local 'streamlit' folder to the '/app' directory inside the container
COPY streamlit/ /app/

# Copy the 'label_encoder.pkl' model to the appropriate directory inside the container
COPY model/label_encoder.pkl /app/models/label_encoder.pkl

# Install the dependencies specified in the 'requirements.txt'
RUN pip install -r requirements.txt

# Expose port 8501 for the Streamlit app
EXPOSE 8501

# Ensure Streamlit binds to 0.0.0.0 and uses port 8501
CMD ["streamlit", "run", "app.py"]

