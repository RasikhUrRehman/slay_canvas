from fastapi import APIRouter

from app.api import auth, boards, workspaces, assets
# Temporarily comment out problematic imports until database is set up
# from app.api import users, chat, media

router = APIRouter()
router.include_router(auth.router, prefix="/auth", tags=["auth"])
router.include_router(boards.router)
router.include_router(workspaces.router)
router.include_router(assets.router)
# router.include_router(media.router)
# router.include_router(chat.router)
# router.include_router(users.router, prefix="/users", tags=["users"])
