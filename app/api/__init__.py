from fastapi import APIRouter

from app.api import auth, boards, workspaces, assets
# Import media router for audio upload functionality
from app.api import media
# Import test router for agent and knowledge base testing
from app.api import test_router_agent
# Temporarily comment out problematic imports until database is set up
# from app.api import users, chat

router = APIRouter()
router.include_router(auth.router, prefix="/auth", tags=["auth"])
router.include_router(boards.router)
router.include_router(workspaces.router)
router.include_router(assets.router)
router.include_router(media.router)
router.include_router(test_router_agent.router)
# router.include_router(chat.router)
# router.include_router(users.router, prefix="/users", tags=["users"])
