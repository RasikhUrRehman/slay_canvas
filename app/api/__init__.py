from fastapi import APIRouter

from app.api import agent, assets_v2, auth, boards, chat, workspaces

# Temporarily comment out problematic imports until database is set up
# from app.api import users

router = APIRouter()
router.include_router(auth.router, prefix="/auth", tags=["auth"])
router.include_router(boards.router)
router.include_router(workspaces.router)
router.include_router(assets_v2.router)
router.include_router(agent.router)
router.include_router(chat.router)
# router.include_router(users.router, prefix="/users", tags=["users"])
