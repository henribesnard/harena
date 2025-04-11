# sync_service/api/endpoints/webhooks.py
from fastapi import APIRouter, Depends, HTTPException, Header, Request, status
from sqlalchemy.orm import Session
from typing import Dict, Any, Optional

from user_service.db.session import get_db
from user_service.core.config import settings
from sync_service.services import webhook_handler

router = APIRouter()

WEBHOOK_SECRET = settings.BRIDGE_WEBHOOK_SECRET

@router.post("/bridge", status_code=status.HTTP_200_OK)
async def receive_bridge_webhook(
    request: Request,
    db: Session = Depends(get_db),
    bridge_signature: Optional[str] = Header(None, alias="BridgeApi-Signature")
):
    """
    Endpoint pour recevoir les webhooks de Bridge API.
    Vérifie la signature et traite les événements.
    """
    # Récupérer le corps de la requête
    raw_payload = await request.body()
    
    # Pour le débogage, imprimez les en-têtes et le payload
    print("Headers:", dict(request.headers))
    print("Payload:", raw_payload.decode())
    
    try:
        payload = await request.json()
    except Exception as e:
        print(f"Erreur de parsing JSON: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid JSON payload"
        )
    
    # Vérifier la signature si fournie
    if bridge_signature:
        try:
            if not webhook_handler.verify_webhook_signature(
                raw_payload.decode(), bridge_signature, WEBHOOK_SECRET
            ):
                print(f"Signature invalide. Fournie: {bridge_signature}, Secret: {WEBHOOK_SECRET[:5]}...")
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid webhook signature"
                )
        except Exception as e:
            print(f"Erreur lors de la vérification de signature: {str(e)}")
            # Pour le débogage, acceptez quand même le webhook
            # En production, il faudrait lever une exception ici
    else:
        print("Aucune signature trouvée dans les en-têtes")
    
    # Traiter l'événement (même sans signature pour le débogage)
    try:
        await webhook_handler.process_webhook(db, payload, bridge_signature)
    except Exception as e:
        print(f"Erreur de traitement du webhook: {str(e)}")
        # Renvoyer quand même 200 OK pour éviter les réessais
    
    # Renvoyer 200 OK rapidement pour éviter les tentatives
    return {"status": "received"}