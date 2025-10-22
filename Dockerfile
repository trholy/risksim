FROM python:3.10-slim

WORKDIR /app

# Create data folder
RUN mkdir data

# Copy Streamlit app
COPY streamlit-app/app.py streamlit-app/requirements.txt /app/

# Install Streamlit app dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy your Python package source code
COPY src /app/src/
COPY pyproject.toml setup.py LICENSE README.md /app/

# Install your package
RUN pip install --no-cache-dir .

# Expose Streamlit port
EXPOSE 8501

# Start Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
