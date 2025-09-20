from fastapi import APIRouter

from app.api import auth, boards, workspaces, assets
# Import media router for audio upload functionality
from app.api import media
# Import test router for agent and knowledge base testing
from app.api import agent
# Import chat router for conversation management
from app.api import chat
# Temporarily comment out problematic imports until database is set up
# from app.api import users

router = APIRouter()
router.include_router(auth.router, prefix="/auth", tags=["auth"])
router.include_router(boards.router)
router.include_router(workspaces.router)
router.include_router(assets.router)
router.include_router(agent.router)
router.include_router(chat.router)
# router.include_router(users.router, prefix="/users", tags=["users"])
