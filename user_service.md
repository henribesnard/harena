Documentation des fonctionnalités développées - Module User Service d'Harena
# Présentation générale
Le module User Service d'Harena constitue le socle fondamental de l'application, responsable de la gestion complète des utilisateurs et de leur intégration avec l'API Bridge pour l'accès aux données bancaires. Ce document détaille les fonctionnalités mises en œuvre à travers chaque endpoint exposé.
Fonctionnalités de gestion des utilisateurs
# Création de compte utilisateur
Endpoint: POST /api/v1/users/register
Cette fonctionnalité permet la création d'un nouveau compte utilisateur dans le système Harena. Lors de l'inscription, le service:

Vérifie l'unicité de l'adresse email dans la base de données
Valide la conformité du mot de passe et sa confirmation
Sécurise le mot de passe via hachage bcrypt
Crée un profil utilisateur de base avec préférences par défaut
Initialise automatiquement une connexion avec Bridge API (création d'identifiant externe)

Le processus d'inscription est optimisé pour une expérience utilisateur fluide tout en maintenant un niveau élevé de sécurité.

# Authentification et gestion de session
Endpoint: POST /api/v1/users/auth/login
Le système d'authentification implémente le standard OAuth2 avec:

Vérification sécurisée des identifiants utilisateur
Génération de tokens JWT pour l'authentification des requêtes ultérieures
Configuration des délais d'expiration des tokens (8 jours par défaut)
Vérification de l'état actif du compte utilisateur

Ce mécanisme d'authentification robuste assure la sécurité des échanges tout en offrant une expérience utilisateur transparente.

# Consultation du profil utilisateur
Endpoint: GET /api/v1/users/me
Cette fonctionnalité permet à l'utilisateur authentifié de:

Récupérer ses informations personnelles (email, nom, prénom)
Consulter ses préférences système
Vérifier l'état de ses connexions bancaires

La protection de cet endpoint via authentification JWT garantit que seul l'utilisateur légitime peut accéder à ses informations.

# Modification du profil utilisateur
Endpoint: PUT /api/v1/users/me
L'utilisateur peut mettre à jour ses informations personnelles:

Modification de l'adresse email
Changement du nom et/ou prénom
Mise à jour du mot de passe (avec hachage automatique)

Les validations appliquées garantissent l'intégrité des données tout en offrant la flexibilité nécessaire.
Intégration avec Bridge API


# Gestion des connexions Bridge
Endpoint: GET /api/v1/users/bridge/connection
Cette fonctionnalité essentielle:

Récupère les informations de connexion Bridge de l'utilisateur
Crée automatiquement une connexion Bridge si inexistante
Établit le lien entre l'identité Harena et l'identité Bridge de l'utilisateur

Cette synchronisation transparente entre les deux systèmes constitue la base de l'expérience conversationnelle financière.


# Gestion des tokens d'authentification Bridge
Endpoint: POST /api/v1/users/bridge/token
Ce service sophistiqué gère:

La génération de tokens d'accès pour l'API Bridge
Le refresh automatique des tokens expirés
Le stockage sécurisé des tokens dans la base de données
La mise à disposition des tokens pour les autres services d'Harena

Cette gestion optimisée des tokens permet d'accéder aux données bancaires sans exposer les identifiants sensibles.


# Création de sessions de connexion bancaire
Endpoint: POST /api/v1/users/bridge/connect-session
Fonctionnalité centrale pour l'onboarding utilisateur:

Génération d'URL de session Bridge sécurisée
Support optionnel des URLs de callback pour le retour après authentification
Intégration transparente avec le processus d'authentification forte des banques
Support du protocole PSD2 pour l'accès sécurisé aux données bancaires

Cette fonctionnalité permet aux utilisateurs de connecter facilement leurs comptes bancaires à Harena tout en respectant les normes réglementaires.
Architecture technique
Le module User Service s'appuie sur une architecture robuste avec:

Modèle de données relationnel: Schéma optimisé pour les utilisateurs, préférences et connexions Bridge
Validation des données: Utilisation de Pydantic pour garantir l'intégrité des données
Migrations de base de données: Gestion des évolutions de schéma via Alembic
Sécurité renforcée: Hachage des mots de passe, tokens JWT, protocole HTTPS

Perspectives d'évolution
Le module User Service est conçu pour supporter les évolutions futures:

Intégration d'authentification multi-facteurs
Support de connexions OAuth avec réseaux sociaux
Fonctionnalités avancées de gestion de profil
Système de permissions et rôles utilisateurs

Ces fonctionnalités pourront être intégrées de manière modulaire grâce à l'architecture extensible mise en place.

Cette documentation technique résume les fonctionnalités développées et offre une vision claire des capacités actuelles du module User Service d'Harena, pierre angulaire de l'assistant financier conversationnel.