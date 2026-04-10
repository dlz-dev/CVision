# définir l'image de python qu'on veut utiliser (python 3.12 en version légère)
FROM python:3.12-slim

# définir le dossier de travail dans le conteneur (app)
WORKDIR /app

# copie d'abord le requirements (optimisation du cache Docker)
COPY requirements.txt .

# installation des dépendances
RUN pip install --no-cache-dir -r requirements.txt

# copie tout le reste du projet dans le conteneur
COPY .. .

# expose le port 8000 pour l'API
EXPOSE 8000

# commande lancée au démarrage du conteneur
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]