"""
Authentication API with database integration
Handles Google OAuth login, manual registration, login, and password reset
"""
import logging
import secrets
from typing import Any, Dict

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import JSONResponse, RedirectResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.db.session import get_async_session
from app.schemas.user import (
    AuthResponse,
    MessageResponse,
    PasswordResetRequest,
    PasswordResetVerify,
    UserInDB,
    UserLogin,
    UserPublic,
    UserRegistration,
)
from app.services.otp_service import OTPService
from app.services.user_service import UserService
from app.utils.auth import create_access_token, get_current_user_id
from app.utils.security import email_service, password_hasher

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/auth", tags=["authentication"])

# OAuth state storage (in production, use Redis or database)
oauth_states: Dict[str, bool] = {}

def get_user_service() -> UserService:
    return UserService()

def get_otp_service() -> OTPService:
    return OTPService()

@router.get("/google/login")
async def google_login():
    """Initiate Google OAuth login"""
    state = secrets.token_urlsafe(32)
    oauth_states[state] = True
    
    google_auth_url = (
        f"https://accounts.google.com/o/oauth2/auth?"
        f"client_id={settings.GOOGLE_CLIENT_ID}&"
        f"redirect_uri={settings.GOOGLE_REDIRECT_URI}&"
        f"scope=openid email profile&"
        f"response_type=code&"
        f"state={state}"
    )
    
    return RedirectResponse(url=google_auth_url)

@router.get("/google/callback")
async def google_callback(
    code: str,
    state: str,
    db: AsyncSession = Depends(get_async_session),
    user_service: UserService = Depends(get_user_service)
):
    """Handle Google OAuth callback and save user to database"""
    
    # Validate state to prevent CSRF
    if state not in oauth_states:
        raise HTTPException(status_code=400, detail="Invalid state parameter")
    
    # Remove used state
    del oauth_states[state]
    
    try:
        # Exchange code for tokens
        async with httpx.AsyncClient() as client:
            token_response = await client.post(
                "https://oauth2.googleapis.com/token",
                data={
                    "client_id": settings.GOOGLE_CLIENT_ID,
                    "client_secret": settings.GOOGLE_CLIENT_SECRET,
                    "code": code,
                    "grant_type": "authorization_code",
                    "redirect_uri": settings.GOOGLE_REDIRECT_URI,
                }
            )
            
        if token_response.status_code != 200:
            logger.error(f"Token exchange failed: {token_response.text}")
            raise HTTPException(status_code=400, detail="Failed to exchange code for token")
        
        token_data = token_response.json()
        access_token = token_data.get("access_token")
        
        if not access_token:
            raise HTTPException(status_code=400, detail="No access token received")
        
        # Get user info from Google
        async with httpx.AsyncClient() as client:
            user_response = await client.get(
                "https://www.googleapis.com/oauth2/v2/userinfo",
                headers={"Authorization": f"Bearer {access_token}"}
            )
            
        if user_response.status_code != 200:
            logger.error(f"User info fetch failed: {user_response.text}")
            raise HTTPException(status_code=400, detail="Failed to get user information")
        
        user_data = user_response.json()
        
        # Extract user information
        google_id = user_data.get("id")
        email = user_data.get("email")
        name = user_data.get("name")
        avatar_url = user_data.get("picture")
        
        if not google_id or not email:
            raise HTTPException(status_code=400, detail="Incomplete user data from Google")
        
        # Create or update user in database
        user = await user_service.create_or_update_oauth_user(
            db=db,
            google_id=google_id,
            email=email,
            name=name,
            avatar_url=avatar_url
        )
        
        if not user:
            raise HTTPException(status_code=500, detail="Failed to create or update user")
        
        # Create JWT token for our app
        jwt_token = create_access_token(data={"sub": str(user.id), "email": user.email})
        
        # Return user data and token
        return JSONResponse(
            content={
                "message": "Login successful",
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
                "token_type": "bearer"
            }
        )
        
    except Exception as e:
        logger.error(f"OAuth callback error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Authentication failed: {str(e)}")

# Manual Registration and Login Endpoints

@router.post("/register", response_model=AuthResponse)
async def register_user(
    user_data: UserRegistration,
    db: AsyncSession = Depends(get_async_session),
    user_service: UserService = Depends(get_user_service)
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
    user_service: UserService = Depends(get_user_service)
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


# Password Reset Endpoints

@router.post("/forgot-password", response_model=MessageResponse)
async def forgot_password(
    request_data: PasswordResetRequest,
    db: AsyncSession = Depends(get_async_session),
    user_service: UserService = Depends(get_user_service),
    otp_service: OTPService = Depends(get_otp_service)
):
    """Send OTP to user's email for password reset"""
    
    try:
        # Check if user exists
        user = await user_service.get_user_by_email(db, request_data.email)
        if not user:
            # Don't reveal if user exists or not for security
            return MessageResponse(message="If this email is registered, you will receive an OTP shortly")
        
        # Generate and send OTP
        otp = await otp_service.generate_otp(db, request_data.email)
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
    user_service: UserService = Depends(get_user_service),
    otp_service: OTPService = Depends(get_otp_service)
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
        # Verify OTP and get associated email
        email = await otp_service.verify_otp_and_get_email(db, reset_data.otp)
        if not email:
            raise HTTPException(status_code=400, detail="Invalid or expired OTP")
        
        # Reset password using the email from OTP
        success = await user_service.reset_password(db, email, reset_data.new_password)
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
    user_service: UserService = Depends(get_user_service),
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


@router.post("/logout", response_model=MessageResponse)
async def logout():
    """Logout user"""
    # Since we're using stateless JWT tokens, logout is handled on the client side
    # In a production app, you might want to blacklist tokens
    return MessageResponse(message="Logged out successfully")


# Health check endpoint
@router.get("/health")
async def auth_health():
    """Health check for auth service"""
    return {
        "status": "healthy",
        "google_oauth_configured": bool(settings.GOOGLE_CLIENT_ID and settings.GOOGLE_CLIENT_SECRET),
        "email_service_configured": bool(email_service.smtp_username and email_service.smtp_password),
        "features": {
            "oauth_login": True,
            "manual_registration": True,
            "password_reset": True
        }
    }
