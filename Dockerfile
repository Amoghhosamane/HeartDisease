FROM python:3.10

# Set working directory
WORKDIR /app
COPY requirements.txt .
# Install dependencies
RUN pip install -r requirements.txt

COPY . .

# Expose default port
EXPOSE 8501

# Run Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
