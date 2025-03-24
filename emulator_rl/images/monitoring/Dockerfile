FROM python:3.8.0-slim

RUN apt-get install -y tzdata
ENV TZ=Europe/Berlin

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir pandas

RUN pip install --no-cache-dir --force-reinstall sqlalchemy==1.4.46
RUN pip install --no-cache-dir mysql-connector-python

RUN pip install streamlit==1.25.0
RUN pip install plotly
RUN pip install matplotlib

WORKDIR /monitoring

EXPOSE 8501

ENTRYPOINT ["streamlit", "run", "app.py"]
