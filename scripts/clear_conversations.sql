-- Script SQL pour supprimer toutes les conversations
-- A exécuter directement dans PostgreSQL

-- Désactiver temporairement les triggers (optionnel)
BEGIN;

-- Supprimer d'abord tous les tours de conversation (contrainte FK)
DELETE FROM conversation_turns;

-- Ensuite supprimer toutes les conversations
DELETE FROM conversations;

-- Vérifier que tout est supprimé
SELECT
    (SELECT COUNT(*) FROM conversations) as nb_conversations,
    (SELECT COUNT(*) FROM conversation_turns) as nb_turns;

-- Valider les changements
COMMIT;

-- Pour annuler les changements au lieu de les valider, utilisez:
-- ROLLBACK;
