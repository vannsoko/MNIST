FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# Installation des dépendances si nécessaire
# RUN pip install pandas matplotlib ...

COPY train.py .

# Création du dossier data pour éviter les erreurs de permission
RUN mkdir ./data

CMD ["python", "train.py"]