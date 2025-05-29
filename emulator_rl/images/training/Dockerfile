FROM python:3.11-slim

ENV TZ=Europe/Berlin

RUN pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cpu
RUN pip install stable_baselines3==2.6.0
RUN pip install scipy==1.15.1
RUN pip install tensorboard==2.19.0

