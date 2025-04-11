import requests
import json
import webbrowser
import time
import http.server
import socketserver
import threading
import urllib.parse
from datetime import datetime

# Configuration Bridge API
CLIENT_ID = "sandbox_id_16d8bcba6b1341b786189b8b5f42670e"  # Remplacez par votre client_id
CLIENT_SECRET = "sandbox_secret_nk3Nok2l7sNOjQH8tzsSnakQrN6aNsqCcQkHdnev2HGsbJtf59uKkujwTzhYORQM"  # Remplacez par votre client_secret
BRIDGE_VERSION = "2025-01-15"  # Version mise à jour
BASE_URL = "https://api.bridgeapi.io/v3"
REDIRECT_URI = "http://localhost:3000/callback"  # URL de redirection après authentification

# Variable globale pour stocker le callback
callback_code = None
callback_received = False

class CallbackHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        global callback_code, callback_received
        
        # Extraire le code du callback URL
        query_components = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
        
        if "code" in query_components:
            callback_code = query_components["code"][0]
            callback_received = True
            
            # Répondre avec une page HTML simple
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write(bytes("""
            <html>
            <head><title>Authentification réussie</title></head>
            <body>
                <h1>Authentification réussie!</h1>
                <p>Vous pouvez maintenant fermer cette fenêtre et retourner à l'application.</p>
            </body>
            </html>
            """, "utf-8"))
        else:
            self.send_response(400)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write(bytes("Erreur: Paramètre 'code' manquant", "utf-8"))

def start_callback_server():
    """Démarre un serveur HTTP pour recevoir le callback de redirection"""
    port = int(REDIRECT_URI.split(':')[2].split('/')[0])
    handler = CallbackHandler
    
    httpd = socketserver.TCPServer(("", port), handler)
    
    # Démarrer le serveur dans un thread séparé
    server_thread = threading.Thread(target=httpd.serve_forever)
    server_thread.daemon = True
    server_thread.start()
    
    return httpd

def make_api_request(endpoint, method="GET", data=None, access_token=None):
    """Fonction pour effectuer des requêtes à l'API Bridge"""
    url = f"{BASE_URL}/{endpoint}"
    headers = {
        "Bridge-Version": BRIDGE_VERSION,
        "Client-Id": CLIENT_ID,
        "Client-Secret": CLIENT_SECRET,
        "Content-Type": "application/json",
        "accept": "application/json"
    }
    
    # Ajouter le token d'authentification si fourni
    if access_token:
        headers["Authorization"] = f"Bearer {access_token}"
    
    try:
        if method == "GET":
            response = requests.get(url, headers=headers)
        elif method == "POST":
            response = requests.post(url, headers=headers, json=data if data else {})
        
        if response.status_code in [200, 201]:
            return response.json()
        else:
            print(f"Erreur: {response.status_code}")
            print(response.text)
            return None
    except Exception as e:
        print(f"Exception lors de la requête: {e}")
        return None

def list_banks():
    """Liste les banques disponibles via Bridge API"""
    print("Récupération de la liste des banques...")
    providers = make_api_request("aggregation/providers")
    
    if providers and "resources" in providers:
        print(f"Nombre de banques disponibles: {len(providers['resources'])}")
        # Affiche les 5 premières banques à titre d'exemple
        for provider in providers["resources"][:5]:
            print(f"- {provider['name']} (ID: {provider['id']})")
        return providers["resources"]
    return []

def create_user(external_user_id=None):
    """Crée un nouvel utilisateur dans Bridge API"""
    print("\nCréation d'un nouvel utilisateur...")
    
    # Selon la nouvelle doc, aucun champ n'est obligatoire pour créer un utilisateur
    # Mais on peut ajouter un external_user_id pour le suivre dans notre système
    data = {}
    if external_user_id:
        data["external_user_id"] = external_user_id
    
    user = make_api_request("aggregation/users", method="POST", data=data)
    
    if user and "uuid" in user:
        print(f"Utilisateur créé avec l'UUID: {user['uuid']}")
        if external_user_id:
            print(f"External User ID: {external_user_id}")
        return user
    return None

def authenticate_user(user_uuid):
    """Authentifie un utilisateur et obtient un token d'accès"""
    print(f"\nAuthentification de l'utilisateur {user_uuid}...")
    
    data = {
        "user_uuid": user_uuid
    }
    
    auth_response = make_api_request("aggregation/authorization/token", method="POST", data=data)
    
    if auth_response and "access_token" in auth_response:
        print(f"Token d'accès obtenu, valide jusqu'à {auth_response['expires_at']}")
        return auth_response["access_token"]
    return None

def create_connect_session(access_token, user_email=None):
    """Crée une session Connect pour lier un utilisateur à sa banque"""
    print("\nCréation d'une session Connect...")
    
    # Selon la nouvelle doc, le corps peut être vide ou contenir email
    data = {}
    if user_email:
        data["user_email"] = user_email
    
    connect_response = make_api_request("aggregation/connect-sessions", 
                                        method="POST", 
                                        data=data, 
                                        access_token=access_token)
    
    if connect_response and "url" in connect_response:
        print(f"Session Connect créée avec succès, ID: {connect_response['id']}")
        return connect_response
    return None

def get_user_items(access_token):
    """Récupère les items (connexions bancaires) d'un utilisateur"""
    print(f"\nRécupération des items pour l'utilisateur...")
    
    items = make_api_request(f"aggregation/items", access_token=access_token)
    
    if items and "resources" in items:
        print(f"Nombre d'items: {len(items['resources'])}")
        for item in items.get("resources", []):
            status_info = f"Status: {item['status']} ({item.get('status_code_info', 'N/A')})"
            print(f"- Item ID: {item['id']}, {status_info}")
            if 'last_successful_refresh' in item:
                print(f"  Dernière synchro réussie: {item['last_successful_refresh']}")
        return items["resources"]
    return []

def get_user_accounts(access_token):
    """Récupère les comptes bancaires de l'utilisateur"""
    print(f"\nRécupération des comptes utilisateur...")
    
    accounts = make_api_request("aggregation/accounts", access_token=access_token)
    
    if accounts and "resources" in accounts:
        print(f"Nombre de comptes: {len(accounts['resources'])}")
        for account in accounts.get("resources", []):
            print(f"- {account['name']} (ID: {account['id']}, Solde: {account['balance']} {account['currency_code']})")
            print(f"  Type: {account['type']}, Mise à jour: {account['updated_at']}")
            if 'iban' in account:
                print(f"  IBAN: {account['iban']}")
        return accounts["resources"]
    return []

def get_account_transactions(access_token, account_id=None, limit=10, since=None):
    """Récupère les transactions d'un compte"""
    print(f"\nRécupération des transactions...")
    
    endpoint = f"aggregation/transactions?limit={limit}"
    
    # Ajouter le paramètre account_id si fourni
    if account_id:
        endpoint += f"&account_id={account_id}"
    
    # Ajouter le paramètre since si fourni
    if since:
        endpoint += f"&since={since}"
    
    transactions = make_api_request(endpoint, access_token=access_token)
    
    if transactions and "resources" in transactions:
        print(f"Nombre de transactions: {len(transactions['resources'])}")
        for transaction in transactions.get("resources", [])[:5]:  # Affiche les 5 premières
            date_str = transaction['date']
            desc = transaction['clean_description']
            amount = transaction['amount']
            curr = transaction['currency_code']
            op_type = transaction['operation_type']
            updated_at = transaction['updated_at']
            
            print(f"- {desc} ({date_str}: {amount} {curr}, Type: {op_type})")
            print(f"  Dernière mise à jour: {updated_at}")
            
        # Vérifier s'il y a plus de données (pagination)
        if "pagination" in transactions and transactions["pagination"].get("next_uri"):
            print(f"  Plus de transactions disponibles, utilisez la pagination pour les récupérer.")
            
        return transactions["resources"]
    return []

def main():
    print("=== Test de synchronisation Bridge API pour Harena ===")
    
    # Démarrer le serveur de callback
    httpd = start_callback_server()
    print(f"Serveur de callback démarré sur {REDIRECT_URI}")
    
    try:
        # 1. Créer un utilisateur avec un ID externe (pratique pour Harena)
        external_user_id = f"harena_test_{int(time.time())}"
        user = create_user(external_user_id)
        if not user:
            print("Impossible de créer un utilisateur. Vérifiez vos credentials.")
            return
        
        # 2. Authentifier l'utilisateur
        access_token = authenticate_user(user["uuid"])
        if not access_token:
            print("Échec de l'authentification utilisateur.")
            return
        
        # 3. Créer une session Connect
        # L'email est optionnel selon la documentation
        user_email = f"test-user-{int(time.time())}@harena.app"
        connect_session = create_connect_session(access_token, user_email)
        if not connect_session:
            print("Impossible de créer une session Connect.")
            return
        
        # 4. Ouvrir le navigateur pour l'authentification bancaire
        print("\nOuverture du navigateur pour l'authentification bancaire...")
        webbrowser.open(connect_session["url"])
        
        # 5. Attendre que l'utilisateur termine le processus d'authentification
        print("\nAttente de la complétion du processus Connect...\n")
        print("Veuillez suivre les instructions dans le navigateur.")
        print("Utilisez le 'Demo Bank' pour tester facilement.")
        
        # Attendre la réception du callback (timeout après 5 minutes)
        timeout = time.time() + 300  # 5 minutes en secondes
        while not callback_received and time.time() < timeout:
            time.sleep(1)
        
        if not callback_received:
            print("\nDélai d'attente dépassé. L'authentification n'a pas été complétée.")
            return
            
        print("\nCallback reçu. Authentification bancaire terminée.")
        
        # 6. Attendre un peu pour laisser le temps à Bridge de synchroniser les données
        print("\nAttente de 10 secondes pour la synchronisation initiale...")
        time.sleep(10)
        
        # 7. Récupérer les items (connexions bancaires)
        items = get_user_items(access_token)
        if not items:
            print("Aucun item trouvé. L'authentification bancaire n'a peut-être pas réussi.")
            print("Vous pouvez réessayer l'authentification ou vérifier le statut manuellement.")
        
        # 8. Récupérer les comptes
        accounts = get_user_accounts(access_token)
        if not accounts:
            print("Aucun compte trouvé ou synchronisation non terminée.")
            print("Attendez quelques minutes et vérifiez à nouveau.")
            return
        
        # 9. Récupérer les transactions
        # D'abord toutes les transactions sans filtre
        transactions = get_account_transactions(access_token)
        
        # Ensuite, si on a des comptes et des transactions, montrer un exemple avec filtrage
        if accounts and transactions:
            print("\n=== Exemple de récupération ciblée ===")
            
            # Récupérer les transactions d'un compte spécifique
            account_id = accounts[0]["id"]
            print(f"Transactions pour le compte {account_id} ({accounts[0]['name']}):")
            get_account_transactions(access_token, account_id=account_id, limit=5)
            
            # Simuler une récupération incrémentielle
            latest_update = transactions[0]["updated_at"]
            print(f"\nExemple de récupération incrémentielle (depuis {latest_update}):")
            get_account_transactions(access_token, since=latest_update)
        
        print("\n=== Test terminé ===")
        print(f"ID utilisateur Bridge: {user['uuid']}")
        print(f"ID utilisateur externe: {external_user_id}")
        print("\nRecommandations pour l'implémentation dans Harena:")
        print("1. Associez le UUID Bridge à chaque utilisateur Harena")
        print("2. Configurez des webhooks pour les notifications de mise à jour")
        print("3. Utilisez 'since' pour les mises à jour incrémentielles")
        print("4. Traitez correctement les types de transactions (notamment deferred_debit_card)")
        print("5. Surveillez les statuts des items pour guider vos utilisateurs")
        
    finally:
        # Arrêter le serveur de callback
        httpd.shutdown()
        print("Serveur de callback arrêté")

if __name__ == "__main__":
    main()