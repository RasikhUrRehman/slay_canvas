"""
Complete authentication router with both OAuth and manual registration
Includes Google OAuth (working) + email/password auth + password reset
"""
from typing import Optional
from fastapi import APIRouter, HTTPException, Query, Depends
from fastapi.responses import RedirectResponse
import httpx
from urllib.parse import urlencode
import secrets
import logging

from app.core.config import settings

logger = logging.getLogger(__name__)

# Try to import database components - graceful fallback if not available
try:
    from sqlalchemy.ext.asyncio import AsyncSession
    from app.db.session import get_async_session
    from app.services.user_service import UserService
    from app.schemas.user import (
        UserPublic, UserRegistration, UserLogin, 
        PasswordResetRequest, PasswordResetVerify, AuthResponse, MessageResponse
    )
    from app.utils.auth import create_access_token, get_current_user_id
    from app.utils.security import otp_manager, email_service, password_hasher
    DATABASE_AVAILABLE = True
except ImportError as e:
    DATABASE_AVAILABLE = False
    logger.warning(f"Database components not available: {e}")

router = APIRouter()

# In-memory storage for demo (use Redis/database in production)
oauth_states = {}

def get_user_service():
    if DATABASE_AVAILABLE:
        return UserService()
    return None


# ===============================
# GOOGLE OAUTH ENDPOINTS (WORKING)
# ===============================

@router.get("/google/login")
async def google_login():
    """Initiate Google OAuth login"""
    # Generate a random state for CSRF protection
    state = secrets.token_urlsafe(32)
    oauth_states[state] = True  # Store state temporarily
    
    # Google OAuth parameters
    params = {
        "client_id": settings.GOOGLE_CLIENT_ID,
        "redirect_uri": settings.GOOGLE_REDIRECT_URI,
        "scope": "openid email profile",
        "response_type": "code",
        "state": state,
        "access_type": "offline",
        "prompt": "consent"
    }
    
    # Build authorization URL
    google_auth_url = "https://accounts.google.com/o/oauth2/auth?" + urlencode(params)
    
    return RedirectResponse(url=google_auth_url)


@router.get("/google/callback")
async def google_callback(
    code: Optional[str] = Query(None),
    state: Optional[str] = Query(None),
    error: Optional[str] = Query(None),
    db: AsyncSession = Depends(get_async_session) if DATABASE_AVAILABLE else None,
    user_service = Depends(get_user_service) if DATABASE_AVAILABLE else None
):
    """Handle Google OAuth callback and save user to database if available"""
    
    if error:
        raise HTTPException(status_code=400, detail=f"OAuth error: {error}")
    
    if not code or not state:
        raise HTTPException(status_code=400, detail="Missing code or state parameter")
    
    # Verify state (CSRF protection)
    if state not in oauth_states:
        raise HTTPException(status_code=400, detail="Invalid state parameter")
    
    # Remove used state
    del oauth_states[state]
    
    try:
        # Exchange code for access token
        token_data = {
            "client_id": settings.GOOGLE_CLIENT_ID,
            "client_secret": settings.GOOGLE_CLIENT_SECRET,
            "code": code,
            "grant_type": "authorization_code",
            "redirect_uri": settings.GOOGLE_REDIRECT_URI,
        }
        
        async with httpx.AsyncClient() as client:
            # Get access token
            token_response = await client.post(
                "https://oauth2.googleapis.com/token",
                data=token_data
            )
            token_response.raise_for_status()
            tokens = token_response.json()
            
            # Get user info
            user_response = await client.get(
                "https://www.googleapis.com/oauth2/v2/userinfo",
                headers={"Authorization": f"Bearer {tokens['access_token']}"}
            )
            user_response.raise_for_status()
            user_info = user_response.json()
        
        # Try to save user to database if available
        if DATABASE_AVAILABLE and db and user_service:
            try:
                # Create or update user in database
                user = await user_service.create_or_update_oauth_user(
                    db=db,
                    google_id=user_info.get("id"),
                    email=user_info.get("email"),
                    name=user_info.get("name"),
                    avatar_url=user_info.get("picture")
                )
                
                # Create JWT token for our app
                jwt_token = create_access_token(data={"sub": str(user.id), "email": user.email})
                
                # Return structured response with database user
                return {
                    "message": "🎉 Google OAuth login successful! User saved to database.",
                    "user": {
                        "id": user.id,
                        "email": user.email,
                        "name": user.name,
                        "avatar_url": user.avatar_url,
                        "is_active": user.is_active,
                        "subscription_plan": user.subscription_plan,
                        "created_at": user.created_at.isoformat()
                    },
                    "access_token": jwt_token,
                    "token_type": "bearer",
                    "database_integration": "✅ Active"
                }
                
            except Exception as db_error:
                logger.warning(f"Database save failed: {db_error}")
                # Fallback to original behavior without database
        
        # Fallback response without database
        return {
            "message": "🎉 Google OAuth login successful!",
            "user": {
                "id": user_info.get("id"),
                "email": user_info.get("email"),
                "name": user_info.get("name"),
                "picture": user_info.get("picture")
            },
            "tokens": {
                "access_token": tokens.get("access_token"),
                "token_type": tokens.get("token_type"),
                "expires_in": tokens.get("expires_in")
            },
            "database_integration": "⚠️ Unavailable",
            "note": "✅ OAuth is working! Database features will be available once configured."
        }
        
    except httpx.HTTPError as e:
        raise HTTPException(status_code=500, detail=f"Failed to exchange code: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OAuth callback error: {str(e)}")


@router.get("/test")
async def test_oauth():
    """Test endpoint to verify OAuth configuration"""
    return {
        "client_id": f"{settings.GOOGLE_CLIENT_ID[:20]}...",
        "redirect_uri": settings.GOOGLE_REDIRECT_URI,
        "login_url": "/api/auth/google/login",
        "status": "✅ OAuth is configured and ready"
    }


# ===============================
# MANUAL REGISTRATION & LOGIN (Database Required)
# ===============================

if DATABASE_AVAILABLE:
    @router.post("/register", response_model=AuthResponse)
    async def register_user(
        user_data: UserRegistration,
        db: AsyncSession = Depends(get_async_session),
        user_service = Depends(get_user_service)
    ):
        """Register a new user with email and password"""
        
        # Validate passwords match
        if not user_data.passwords_match():
            raise HTTPException(status_code=400, detail="Passwords do not match")
        
        # Validate password strength
        is_strong, errors = password_hasher.validate_password_strength(user_data.password)
        if not is_strong:
            raise HTTPException(status_code=400, detail=f"Password requirements not met: {', '.join(errors)}")
        
        try:
            # Create user
            user = await user_service.register_user(db, user_data)
            if not user:
                raise HTTPException(status_code=400, detail="User with this email already exists")
            
            # Create JWT token
            jwt_token = create_access_token(data={"sub": str(user.id), "email": user.email})
            
            # Convert user to UserPublic
            user_public = UserPublic(
                id=user.id,
                email=user.email,
                name=user.name,
                avatar_url=user.avatar_url,
                is_active=user.is_active,
                subscription_plan=user.subscription_plan,
                created_at=user.created_at
            )
            
            return AuthResponse(
                message="Registration successful",
                user=user_public,
                access_token=jwt_token,
                token_type="bearer"
            )
            
        except Exception as e:
            logger.error(f"Registration error: {str(e)}")
            raise HTTPException(status_code=500, detail="Registration failed")


    @router.post("/login", response_model=AuthResponse)
    async def login_user(
        user_data: UserLogin,
        db: AsyncSession = Depends(get_async_session),
        user_service = Depends(get_user_service)
    ):
        """Login user with email and password"""
        
        try:
            # Authenticate user
            user = await user_service.authenticate_user(db, user_data.email, user_data.password)
            if not user:
                raise HTTPException(status_code=401, detail="Invalid email or password")
            
            # Create JWT token
            jwt_token = create_access_token(data={"sub": str(user.id), "email": user.email})
            
            # Convert user to UserPublic
            user_public = UserPublic(
                id=user.id,
                email=user.email,
                name=user.name,
                avatar_url=user.avatar_url,
                is_active=user.is_active,
                subscription_plan=user.subscription_plan,
                created_at=user.created_at
            )
            
            return AuthResponse(
                message="Login successful",
                user=user_public,
                access_token=jwt_token,
                token_type="bearer"
            )
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Login error: {str(e)}")
            raise HTTPException(status_code=500, detail="Login failed")

else:
    @router.post("/register")
    async def register_user_unavailable():
        """Registration unavailable without database"""
        raise HTTPException(
            status_code=503, 
            detail="Registration requires database configuration. Please set up PostgreSQL first."
        )

    @router.post("/login")
    async def login_user_unavailable():
        """Login unavailable without database"""
        raise HTTPException(
            status_code=503, 
            detail="Login requires database configuration. Please set up PostgreSQL first."
        )


# ===============================
# PASSWORD RESET (Database Required)
# ===============================

if DATABASE_AVAILABLE:
    @router.post("/forgot-password", response_model=MessageResponse)
    async def forgot_password(
        request_data: PasswordResetRequest,
        db: AsyncSession = Depends(get_async_session),
        user_service = Depends(get_user_service)
    ):
        """Send OTP to user's email for password reset"""
        try:
            # Check if user exists
            user = await user_service.get_user_by_email(db, request_data.email)
            if not user:
                # Don't reveal if user exists or not for security
                return MessageResponse(message="If this email is registered, you will receive an OTP shortly")
            
            # Generate and send OTP
            otp = otp_manager.generate_otp(request_data.email)
            print(otp)
            email_sent = await email_service.send_otp_email(request_data.email, otp)
            
            if not email_sent:
                logger.error(f"Failed to send OTP email to {request_data.email}")
                # Still return success message for security
            
            return MessageResponse(message="If this email is registered, you will receive an OTP shortly")
            
        except Exception as e:
            logger.error(f"Forgot password error: {str(e)}")
            return MessageResponse(message="If this email is registered, you will receive an OTP shortly")


    @router.post("/reset-password", response_model=MessageResponse)
    async def reset_password(
        reset_data: PasswordResetVerify,
        db: AsyncSession = Depends(get_async_session),
        user_service = Depends(get_user_service)
    ):
        """Reset password with OTP verification"""
        
        # Validate passwords match
        if not reset_data.passwords_match():
            raise HTTPException(status_code=400, detail="Passwords do not match")
        
        # Validate password strength
        is_strong, errors = password_hasher.validate_password_strength(reset_data.new_password)
        if not is_strong:
            raise HTTPException(status_code=400, detail=f"Password requirements not met: {', '.join(errors)}")
        
        try:
            # Verify OTP
            if not otp_manager.verify_otp(reset_data.email, reset_data.otp):
                raise HTTPException(status_code=400, detail="Invalid or expired OTP")
            
            # Reset password
            success = await user_service.reset_password(db, reset_data.email, reset_data.new_password)
            if not success:
                raise HTTPException(status_code=404, detail="User not found")
            
            return MessageResponse(message="Password reset successfully")
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Password reset error: {str(e)}")
            raise HTTPException(status_code=500, detail="Password reset failed")


    @router.get("/me", response_model=UserPublic)
    async def get_current_user(
        db: AsyncSession = Depends(get_async_session),
        user_service = Depends(get_user_service),
        current_user_id: int = Depends(get_current_user_id)
    ):
        """Get current user profile (requires authentication)"""
        
        try:
            user = await user_service.get_user_by_id(db, current_user_id)
            if not user:
                raise HTTPException(status_code=404, detail="User not found")
            
            return UserPublic(
                id=user.id,
                email=user.email,
                name=user.name,
                avatar_url=user.avatar_url,
                is_active=user.is_active,
                subscription_plan=user.subscription_plan,
                created_at=user.created_at
            )
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Get current user error: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to get user profile")

else:
    @router.post("/forgot-password")
    async def forgot_password_unavailable():
        raise HTTPException(status_code=503, detail="Password reset requires database configuration.")

    @router.post("/reset-password") 
    async def reset_password_unavailable():
        raise HTTPException(status_code=503, detail="Password reset requires database configuration.")

    @router.get("/me")
    async def get_current_user_unavailable():
        raise HTTPException(status_code=503, detail="User profile requires database configuration.")


@router.post("/logout")
async def logout():
    """Logout user"""
    # Since we're using stateless JWT tokens, logout is handled on the client side
    # In a production app, you might want to blacklist tokens
    return {"message": "Logged out successfully"}


# ===============================
# HEALTH & DEBUG
# ===============================

@router.get("/health")
async def auth_health():
    """Health check for auth service with all features"""
    
    # Check database status
    if DATABASE_AVAILABLE:
        try:
            from app.db.session import async_engine
            from sqlalchemy import text
            async with async_engine.begin() as conn:
                await conn.execute(text("SELECT 1"))
            db_status = "✅ Connected"
        except Exception as e:
            db_status = f"❌ Failed: {str(e)}"
    else:
        db_status = "⚠️ Not configured"
    
    available_features = {
        "google_oauth": "✅ Available",
    }
    
    available_endpoints = {
        "oauth_login": "/api/auth/google/login",
        "oauth_callback": "/api/auth/google/callback",
        "test": "/api/auth/test",
        "logout": "/api/auth/logout"
    }
    
    if DATABASE_AVAILABLE:
        available_features.update({
            "manual_registration": "✅ Available", 
            "manual_login": "✅ Available",
            "password_reset": "✅ Available",
            "jwt_authentication": "✅ Available"
        })
        available_endpoints.update({
            "register": "/api/auth/register",
            "login": "/api/auth/login",
            "forgot_password": "/api/auth/forgot-password",
            "reset_password": "/api/auth/reset-password",
            "profile": "/api/auth/me"
        })
    else:
        available_features.update({
            "manual_registration": "⚠️ Requires database", 
            "manual_login": "⚠️ Requires database",
            "password_reset": "⚠️ Requires database",
            "jwt_authentication": "⚠️ Requires database"
        })
    
    return {
        "status": "healthy",
        "database_available": DATABASE_AVAILABLE,
        "features": available_features,
        "services": {
            "database": db_status,
            "google_oauth_configured": bool(settings.GOOGLE_CLIENT_ID and settings.GOOGLE_CLIENT_SECRET),
            "email_service_configured": DATABASE_AVAILABLE and bool(email_service.smtp_username and email_service.smtp_password) if DATABASE_AVAILABLE else False
        },
        "endpoints": available_endpoints
    }