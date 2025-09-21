from fastapi import APIRouter

# Import media router for audio upload functionality
# Import test router for agent and knowledge base testing
# Import chat router for conversation management
from app.api import agent, assets, auth, boards, chat, media, workspaces

# Temporarily comment out problematic imports until database is set up
# from app.api import users

router = APIRouter()

# health check endpoint
@router.get("/health", tags=["health"])
async def health_check():
    return {"status": "healthy", "message": "API is running"}

router.include_router(auth.router, prefix="/auth", tags=["auth"])
router.include_router(boards.router)
router.include_router(workspaces.router)
router.include_router(assets.router)
router.include_router(agent.router)
router.include_router(chat.router)
# router.include_router(users.router, prefix="/users", tags=["users"])
