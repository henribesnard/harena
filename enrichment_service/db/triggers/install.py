"""
Installation et gestion des triggers PostgreSQL pour les mises à jour vectorielles.

Ce module contient les scripts SQL pour créer les triggers et fonctions PostgreSQL
nécessaires à la synchronisation automatique avec Qdrant.
"""

import logging
from sqlalchemy import text
from sqlalchemy.orm import Session
from typing import List, Dict, Any

from enrichment_service.core.logging import get_contextual_logger

logger = logging.getLogger(__name__)

# Définition des canaux de notification
NOTIFICATION_CHANNELS = [
    "transaction_changes",
    "account_changes", 
    "item_changes",
    "stock_changes"
]

def install_triggers(db: Session) -> Dict[str, Any]:
    """
    Installe tous les triggers nécessaires dans PostgreSQL.
    
    Args:
        db: Session de base de données
        
    Returns:
        Dict: Résultat de l'installation
    """
    ctx_logger = get_contextual_logger(__name__, enrichment_type="trigger_install")
    ctx_logger.info("Installation des triggers PostgreSQL pour les mises à jour vectorielles")
    
    result = {
        "status": "pending",
        "triggers_installed": [],
        "functions_created": [],
        "errors": []
    }
    
    try:
        # 1. Créer les fonctions de notification
        ctx_logger.info("Création des fonctions de notification")
        
        # Fonction pour les transactions
        create_transaction_function = """
        CREATE OR REPLACE FUNCTION notify_transaction_change()
        RETURNS trigger AS $$
        DECLARE
            payload TEXT;
        BEGIN
            IF (TG_OP = 'DELETE') THEN
                payload = json_build_object(
                    'id', OLD.id,
                    'bridge_transaction_id', OLD.bridge_transaction_id,
                    'operation', TG_OP,
                    'user_id', OLD.user_id,
                    'account_id', OLD.account_id,
                    'timestamp', extract(epoch from now())
                );
                PERFORM pg_notify('transaction_changes', payload);
                RETURN OLD;
            ELSE
                payload = json_build_object(
                    'id', NEW.id,
                    'bridge_transaction_id', NEW.bridge_transaction_id,
                    'operation', TG_OP,
                    'user_id', NEW.user_id,
                    'account_id', NEW.account_id,
                    'timestamp', extract(epoch from now())
                );
                PERFORM pg_notify('transaction_changes', payload);
                RETURN NEW;
            END IF;
        END;
        $$ LANGUAGE plpgsql;
        """
        
        db.execute(text(create_transaction_function))
        result["functions_created"].append("notify_transaction_change")
        
        # Fonction pour les comptes
        create_account_function = """
        CREATE OR REPLACE FUNCTION notify_account_change()
        RETURNS trigger AS $$
        DECLARE
            payload TEXT;
        BEGIN
            IF (TG_OP = 'DELETE') THEN
                payload = json_build_object(
                    'id', OLD.id,
                    'bridge_account_id', OLD.bridge_account_id,
                    'operation', TG_OP,
                    'item_id', OLD.item_id,
                    'timestamp', extract(epoch from now())
                );
                PERFORM pg_notify('account_changes', payload);
                RETURN OLD;
            ELSE
                payload = json_build_object(
                    'id', NEW.id,
                    'bridge_account_id', NEW.bridge_account_id,
                    'operation', TG_OP,
                    'item_id', NEW.item_id,
                    'timestamp', extract(epoch from now())
                );
                PERFORM pg_notify('account_changes', payload);
                RETURN NEW;
            END IF;
        END;
        $$ LANGUAGE plpgsql;
        """
        
        db.execute(text(create_account_function))
        result["functions_created"].append("notify_account_change")
        
        # Fonction pour les items
        create_item_function = """
        CREATE OR REPLACE FUNCTION notify_item_change()
        RETURNS trigger AS $$
        DECLARE
            payload TEXT;
        BEGIN
            IF (TG_OP = 'DELETE') THEN
                payload = json_build_object(
                    'id', OLD.id,
                    'bridge_item_id', OLD.bridge_item_id,
                    'operation', TG_OP,
                    'user_id', OLD.user_id,
                    'timestamp', extract(epoch from now())
                );
                PERFORM pg_notify('item_changes', payload);
                RETURN OLD;
            ELSE
                payload = json_build_object(
                    'id', NEW.id,
                    'bridge_item_id', NEW.bridge_item_id,
                    'operation', TG_OP,
                    'user_id', NEW.user_id,
                    'status', NEW.status,
                    'timestamp', extract(epoch from now())
                );
                PERFORM pg_notify('item_changes', payload);
                RETURN NEW;
            END IF;
        END;
        $$ LANGUAGE plpgsql;
        """
        
        db.execute(text(create_item_function))
        result["functions_created"].append("notify_item_change")
        
        # Fonction pour les stocks
        create_stock_function = """
        CREATE OR REPLACE FUNCTION notify_stock_change()
        RETURNS trigger AS $$
        DECLARE
            payload TEXT;
        BEGIN
            IF (TG_OP = 'DELETE') THEN
                payload = json_build_object(
                    'id', OLD.id,
                    'bridge_stock_id', OLD.bridge_stock_id,
                    'operation', TG_OP,
                    'user_id', OLD.user_id,
                    'account_id', OLD.account_id,
                    'timestamp', extract(epoch from now())
                );
                PERFORM pg_notify('stock_changes', payload);
                RETURN OLD;
            ELSE
                payload = json_build_object(
                    'id', NEW.id,
                    'bridge_stock_id', NEW.bridge_stock_id,
                    'operation', TG_OP,
                    'user_id', NEW.user_id,
                    'account_id', NEW.account_id,
                    'timestamp', extract(epoch from now())
                );
                PERFORM pg_notify('stock_changes', payload);
                RETURN NEW;
            END IF;
        END;
        $$ LANGUAGE plpgsql;
        """
        
        db.execute(text(create_stock_function))
        result["functions_created"].append("notify_stock_change")
        
        # 2. Créer les triggers
        ctx_logger.info("Création des triggers")
        
        # Trigger pour les transactions
        create_transaction_trigger = """
        DROP TRIGGER IF EXISTS transaction_change_trigger ON raw_transactions;
        CREATE TRIGGER transaction_change_trigger
        AFTER INSERT OR UPDATE OR DELETE ON raw_transactions
        FOR EACH ROW EXECUTE FUNCTION notify_transaction_change();
        """
        
        db.execute(text(create_transaction_trigger))
        result["triggers_installed"].append("transaction_change_trigger")
        
        # Trigger pour les comptes
        create_account_trigger = """
        DROP TRIGGER IF EXISTS account_change_trigger ON sync_accounts;
        CREATE TRIGGER account_change_trigger
        AFTER INSERT OR UPDATE OR DELETE ON sync_accounts
        FOR EACH ROW EXECUTE FUNCTION notify_account_change();
        """
        
        db.execute(text(create_account_trigger))
        result["triggers_installed"].append("account_change_trigger")
        
        # Trigger pour les items
        create_item_trigger = """
        DROP TRIGGER IF EXISTS item_change_trigger ON sync_items;
        CREATE TRIGGER item_change_trigger
        AFTER INSERT OR UPDATE OR DELETE ON sync_items
        FOR EACH ROW EXECUTE FUNCTION notify_item_change();
        """
        
        db.execute(text(create_item_trigger))
        result["triggers_installed"].append("item_change_trigger")
        
        # Trigger pour les stocks
        create_stock_trigger = """
        DROP TRIGGER IF EXISTS stock_change_trigger ON raw_stocks;
        CREATE TRIGGER stock_change_trigger
        AFTER INSERT OR UPDATE OR DELETE ON raw_stocks
        FOR EACH ROW EXECUTE FUNCTION notify_stock_change();
        """
        
        db.execute(text(create_stock_trigger))
        result["triggers_installed"].append("stock_change_trigger")
        
        # 3. Commit toutes les modifications
        db.commit()
        
        result["status"] = "success"
        ctx_logger.info(f"Triggers PostgreSQL installés avec succès: {len(result['triggers_installed'])} triggers, {len(result['functions_created'])} fonctions")
        
    except Exception as e:
        db.rollback()
        error_msg = f"Erreur lors de l'installation des triggers: {str(e)}"
        ctx_logger.error(error_msg, exc_info=True)
        result["status"] = "error"
        result["errors"].append(error_msg)
    
    return result

def remove_triggers(db: Session) -> Dict[str, Any]:
    """
    Supprime tous les triggers et fonctions créés.
    
    Args:
        db: Session de base de données
        
    Returns:
        Dict: Résultat de la suppression
    """
    ctx_logger = get_contextual_logger(__name__, enrichment_type="trigger_removal")
    ctx_logger.info("Suppression des triggers PostgreSQL")
    
    result = {
        "status": "pending",
        "triggers_removed": [],
        "functions_removed": [],
        "errors": []
    }
    
    try:
        # Supprimer les triggers
        triggers_to_remove = [
            ("transaction_change_trigger", "raw_transactions"),
            ("account_change_trigger", "sync_accounts"),
            ("item_change_trigger", "sync_items"),
            ("stock_change_trigger", "raw_stocks")
        ]
        
        for trigger_name, table_name in triggers_to_remove:
            try:
                drop_trigger_sql = f"DROP TRIGGER IF EXISTS {trigger_name} ON {table_name};"
                db.execute(text(drop_trigger_sql))
                result["triggers_removed"].append(trigger_name)
                ctx_logger.debug(f"Trigger supprimé: {trigger_name}")
            except Exception as e:
                error_msg = f"Erreur lors de la suppression du trigger {trigger_name}: {str(e)}"
                result["errors"].append(error_msg)
                ctx_logger.warning(error_msg)
        
        # Supprimer les fonctions
        functions_to_remove = [
            "notify_transaction_change",
            "notify_account_change",
            "notify_item_change", 
            "notify_stock_change"
        ]
        
        for function_name in functions_to_remove:
            try:
                drop_function_sql = f"DROP FUNCTION IF EXISTS {function_name}();"
                db.execute(text(drop_function_sql))
                result["functions_removed"].append(function_name)
                ctx_logger.debug(f"Fonction supprimée: {function_name}")
            except Exception as e:
                error_msg = f"Erreur lors de la suppression de la fonction {function_name}: {str(e)}"
                result["errors"].append(error_msg)
                ctx_logger.warning(error_msg)
        
        # Commit les modifications
        db.commit()
        
        result["status"] = "success" if not result["errors"] else "partial"
        ctx_logger.info(f"Suppression terminée: {len(result['triggers_removed'])} triggers, {len(result['functions_removed'])} fonctions")
        
    except Exception as e:
        db.rollback()
        error_msg = f"Erreur générale lors de la suppression des triggers: {str(e)}"
        ctx_logger.error(error_msg, exc_info=True)
        result["status"] = "error"
        result["errors"].append(error_msg)
    
    return result

def check_triggers_installed(db: Session) -> Dict[str, Any]:
    """
    Vérifie si les triggers sont correctement installés.
    
    Args:
        db: Session de base de données
        
    Returns:
        Dict: État des triggers
    """
    ctx_logger = get_contextual_logger(__name__, enrichment_type="trigger_check")
    
    result = {
        "triggers_status": {},
        "functions_status": {},
        "all_installed": False
    }
    
    try:
        # Vérifier les triggers
        triggers_to_check = [
            ("transaction_change_trigger", "raw_transactions"),
            ("account_change_trigger", "sync_accounts"),
            ("item_change_trigger", "sync_items"),
            ("stock_change_trigger", "raw_stocks")
        ]
        
        for trigger_name, table_name in triggers_to_check:
            check_trigger_sql = """
            SELECT COUNT(*) 
            FROM information_schema.triggers 
            WHERE trigger_name = :trigger_name 
            AND event_object_table = :table_name
            """
            
            count = db.execute(
                text(check_trigger_sql),
                {"trigger_name": trigger_name, "table_name": table_name}
            ).scalar()
            
            result["triggers_status"][trigger_name] = count > 0
        
        # Vérifier les fonctions
        functions_to_check = [
            "notify_transaction_change",
            "notify_account_change", 
            "notify_item_change",
            "notify_stock_change"
        ]
        
        for function_name in functions_to_check:
            check_function_sql = """
            SELECT COUNT(*) 
            FROM information_schema.routines 
            WHERE routine_name = :function_name 
            AND routine_type = 'FUNCTION'
            """
            
            count = db.execute(
                text(check_function_sql),
                {"function_name": function_name}
            ).scalar()
            
            result["functions_status"][function_name] = count > 0
        
        # Déterminer si tout est installé
        all_triggers_installed = all(result["triggers_status"].values())
        all_functions_installed = all(result["functions_status"].values())
        result["all_installed"] = all_triggers_installed and all_functions_installed
        
        ctx_logger.info(f"Vérification des triggers terminée - Installés: {result['all_installed']}")
        
    except Exception as e:
        ctx_logger.error(f"Erreur lors de la vérification des triggers: {str(e)}", exc_info=True)
        result["error"] = str(e)
    
    return result

def get_trigger_test_queries() -> List[str]:
    """
    Retourne des requêtes de test pour vérifier le fonctionnement des triggers.
    
    Returns:
        List[str]: Liste des requêtes SQL de test
    """
    return [
        # Test simple d'insertion de transaction (pour dev/test uniquement)
        """
        -- Test trigger transaction (NE PAS UTILISER EN PRODUCTION)
        -- INSERT INTO raw_transactions (bridge_transaction_id, account_id, user_id, amount, date, clean_description)
        -- VALUES (999999999, 1, 1, 10.0, NOW(), 'Test trigger transaction');
        """,
        
        # Vérification des listeners actifs
        """
        SELECT 
            pid,
            application_name,
            query
        FROM pg_stat_activity 
        WHERE state = 'idle in transaction'
        AND query LIKE '%LISTEN%';
        """,
        
        # Vérification des notifications en attente
        """
        SELECT 
            channel,
            payload
        FROM pg_notification_queue
        ORDER BY notification_id;
        """
    ]

def install_notification_helper_functions(db: Session) -> Dict[str, Any]:
    """
    Installe des fonctions helper pour la gestion des notifications.
    
    Args:
        db: Session de base de données
        
    Returns:
        Dict: Résultat de l'installation
    """
    ctx_logger = get_contextual_logger(__name__, enrichment_type="helper_install")
    ctx_logger.info("Installation des fonctions helper pour les notifications")
    
    result = {
        "status": "pending",
        "functions_created": [],
        "errors": []
    }
    
    try:
        # Fonction pour obtenir des statistiques sur les notifications
        notification_stats_function = """
        CREATE OR REPLACE FUNCTION get_enrichment_notification_stats()
        RETURNS TABLE(
            channel_name TEXT,
            total_notifications BIGINT,
            last_notification_time TIMESTAMP
        ) AS $
        BEGIN
            -- Cette fonction fournirait des statistiques si PostgreSQL 
            -- maintenait un historique des notifications (ce qu'il ne fait pas par défaut)
            -- Pour l'instant, elle retourne des informations sur les triggers actifs
            
            RETURN QUERY
            SELECT 
                t.trigger_name::TEXT as channel_name,
                0::BIGINT as total_notifications,
                NOW()::TIMESTAMP as last_notification_time
            FROM information_schema.triggers t
            WHERE t.trigger_name LIKE '%_change_trigger';
        END;
        $ LANGUAGE plpgsql;
        """
        
        db.execute(text(notification_stats_function))
        result["functions_created"].append("get_enrichment_notification_stats")
        
        # Fonction pour tester manuellement les notifications
        test_notification_function = """
        CREATE OR REPLACE FUNCTION test_enrichment_notification(
            channel_name TEXT DEFAULT 'test_enrichment',
            test_payload TEXT DEFAULT '{"test": true}'
        )
        RETURNS BOOLEAN AS $
        BEGIN
            PERFORM pg_notify(channel_name, test_payload);
            RETURN TRUE;
        EXCEPTION WHEN OTHERS THEN
            RETURN FALSE;
        END;
        $ LANGUAGE plpgsql;
        """
        
        db.execute(text(test_notification_function))
        result["functions_created"].append("test_enrichment_notification")
        
        # Fonction pour vérifier l'état des triggers
        trigger_health_function = """
        CREATE OR REPLACE FUNCTION check_enrichment_triggers_health()
        RETURNS TABLE(
            trigger_name TEXT,
            table_name TEXT,
            is_enabled BOOLEAN,
            function_exists BOOLEAN
        ) AS $
        BEGIN
            RETURN QUERY
            SELECT 
                t.trigger_name::TEXT,
                t.event_object_table::TEXT,
                (t.status = 'ENABLED')::BOOLEAN,
                (r.routine_name IS NOT NULL)::BOOLEAN
            FROM information_schema.triggers t
            LEFT JOIN information_schema.routines r 
                ON r.routine_name = REPLACE(t.action_statement, 'EXECUTE FUNCTION ', '')
                AND r.routine_name = REPLACE(r.routine_name, '()', '')
            WHERE t.trigger_name LIKE '%_change_trigger'
            ORDER BY t.trigger_name;
        END;
        $ LANGUAGE plpgsql;
        """
        
        db.execute(text(trigger_health_function))
        result["functions_created"].append("check_enrichment_triggers_health")
        
        db.commit()
        result["status"] = "success"
        ctx_logger.info(f"Fonctions helper installées: {result['functions_created']}")
        
    except Exception as e:
        db.rollback()
        error_msg = f"Erreur lors de l'installation des fonctions helper: {str(e)}"
        ctx_logger.error(error_msg, exc_info=True)
        result["status"] = "error"
        result["errors"].append(error_msg)
    
    return result

def get_trigger_installation_summary(db: Session) -> Dict[str, Any]:
    """
    Retourne un résumé complet de l'état d'installation des triggers.
    
    Args:
        db: Session de base de données
        
    Returns:
        Dict: Résumé complet de l'installation
    """
    summary = {
        "installation_status": check_triggers_installed(db),
        "notification_channels": NOTIFICATION_CHANNELS,
        "database_info": {},
        "recommendations": []
    }
    
    try:
        # Informations sur la base de données
        db_info_query = """
        SELECT 
            version() as postgres_version,
            current_database() as database_name,
            current_user as current_user,
            inet_server_addr() as server_address
        """
        
        db_info = db.execute(text(db_info_query)).first()
        if db_info:
            summary["database_info"] = {
                "postgres_version": db_info.postgres_version,
                "database_name": db_info.database_name,
                "current_user": db_info.current_user,
                "server_address": str(db_info.server_address) if db_info.server_address else "localhost"
            }
        
        # Générer des recommandations
        if not summary["installation_status"]["all_installed"]:
            summary["recommendations"].append(
                "Exécuter install_triggers() pour installer les triggers manquants"
            )
        
        # Vérifier les permissions
        try:
            db.execute(text("SELECT pg_notify('test_channel', 'test')"))
            summary["can_notify"] = True
        except Exception:
            summary["can_notify"] = False
            summary["recommendations"].append(
                "L'utilisateur actuel n'a pas les permissions pour pg_notify"
            )
        
    except Exception as e:
        summary["error"] = str(e)
    
    return summary