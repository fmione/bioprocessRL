FROM python:3.11-slim

ENV TZ=Europe/Berlin

RUN pip install tensorboard==2.19.0

EXPOSE 6006

CMD ["tensorboard", "--logdir=training_dag/ppo_logs", "--port=6006", "--host=0.0.0.0"]
