import os
import heroku3

# Connectez-vous à l'API Heroku
heroku_conn = heroku3.from_key(os.environ.get('HRKU-7bbf880c-febb-45c0-b3c6-366ed6ca91e2'))

# Obtenez l'application par son nom
app = heroku_conn.apps()['harenabackend']

# Obtenez les variables d'environnement
config_vars = app.config()

# Écrire dans un fichier .env
with open('.env', 'w') as f:
    for key, value in config_vars.items():
        f.write(f"{key}={value}\n")