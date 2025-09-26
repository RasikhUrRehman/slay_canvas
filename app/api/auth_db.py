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
        logger.warning(f"Invalid OAuth state parameter received: {state}")
        raise HTTPException(
            status_code=400, 
            detail={
                "error": "invalid_state",
                "message": "Invalid state parameter. This may indicate a CSRF attack or expired session.",
                "code": "AUTH_INVALID_STATE"
            }
        )
    
    # Remove used state
    del oauth_states[state]
    
    try:
        # Exchange code for tokens
        async with httpx.AsyncClient() as client:
            try:
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
            except httpx.TimeoutException:
                logger.error("Timeout while exchanging OAuth code for token")
                raise HTTPException(
                    status_code=408,
                    detail={
                        "error": "timeout",
                        "message": "Request to Google OAuth service timed out. Please try again.",
                        "code": "AUTH_TIMEOUT"
                    }
                )
            except httpx.RequestError as e:
                logger.error(f"Network error during token exchange: {str(e)}")
                raise HTTPException(
                    status_code=503,
                    detail={
                        "error": "network_error",
                        "message": "Unable to connect to Google OAuth service. Please try again later.",
                        "code": "AUTH_NETWORK_ERROR"
                    }
                )
            
        if token_response.status_code != 200:
            error_detail = token_response.text
            logger.error(f"Token exchange failed with status {token_response.status_code}: {error_detail}")
            
            # Parse Google's error response if possible
            try:
                error_data = token_response.json()
                error_type = error_data.get("error", "unknown")
                error_description = error_data.get("error_description", "Token exchange failed")
            except:
                error_type = "token_exchange_failed"
                error_description = f"HTTP {token_response.status_code}: Failed to exchange authorization code"
            
            raise HTTPException(
                status_code=400,
                detail={
                    "error": error_type,
                    "message": f"Google OAuth error: {error_description}",
                    "code": "AUTH_TOKEN_EXCHANGE_FAILED"
                }
            )
        
        try:
            token_data = token_response.json()
        except Exception as e:
            logger.error(f"Failed to parse token response JSON: {str(e)}")
            raise HTTPException(
                status_code=502,
                detail={
                    "error": "invalid_response",
                    "message": "Invalid response format from Google OAuth service",
                    "code": "AUTH_INVALID_TOKEN_RESPONSE"
                }
            )
        
        access_token = token_data.get("access_token")
        if not access_token:
            logger.error(f"No access token in response: {token_data}")
            raise HTTPException(
                status_code=502,
                detail={
                    "error": "missing_access_token",
                    "message": "No access token received from Google OAuth service",
                    "code": "AUTH_MISSING_ACCESS_TOKEN"
                }
            )
        
        # Get user info from Google
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                user_response = await client.get(
                    "https://www.googleapis.com/oauth2/v2/userinfo",
                    headers={"Authorization": f"Bearer {access_token}"}
                )
            except httpx.TimeoutException:
                logger.error("Timeout while fetching user info from Google")
                raise HTTPException(
                    status_code=408,
                    detail={
                        "error": "timeout",
                        "message": "Request to Google user info service timed out. Please try again.",
                        "code": "AUTH_USER_INFO_TIMEOUT"
                    }
                )
            except httpx.RequestError as e:
                logger.error(f"Network error during user info fetch: {str(e)}")
                raise HTTPException(
                    status_code=503,
                    detail={
                        "error": "network_error",
                        "message": "Unable to fetch user information from Google. Please try again later.",
                        "code": "AUTH_USER_INFO_NETWORK_ERROR"
                    }
                )
            
        if user_response.status_code != 200:
            logger.error(f"User info fetch failed with status {user_response.status_code}: {user_response.text}")
            raise HTTPException(
                status_code=502,
                detail={
                    "error": "user_info_failed",
                    "message": f"Failed to get user information from Google (HTTP {user_response.status_code})",
                    "code": "AUTH_USER_INFO_FAILED"
                }
            )
        
        try:
            user_data = user_response.json()
        except Exception as e:
            logger.error(f"Failed to parse user info response JSON: {str(e)}")
            raise HTTPException(
                status_code=502,
                detail={
                    "error": "invalid_user_response",
                    "message": "Invalid user information format from Google",
                    "code": "AUTH_INVALID_USER_RESPONSE"
                }
            )
        
        # Extract user information
        google_id = user_data.get("id")
        email = user_data.get("email")
        name = user_data.get("name")
        avatar_url = user_data.get("picture")
        
        if not google_id or not email:
            logger.error(f"Incomplete user data from Google: {user_data}")
            missing_fields = []
            if not google_id:
                missing_fields.append("id")
            if not email:
                missing_fields.append("email")
            
            raise HTTPException(
                status_code=502,
                detail={
                    "error": "incomplete_user_data",
                    "message": f"Incomplete user data from Google. Missing: {', '.join(missing_fields)}",
                    "code": "AUTH_INCOMPLETE_USER_DATA"
                }
            )
        
        # Create or update user in database
        try:
            user = await user_service.create_or_update_oauth_user(
                db=db,
                google_id=google_id,
                email=email,
                name=name,
                avatar_url=avatar_url
            )
        except Exception as e:
            logger.error(f"Database error while creating/updating OAuth user: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "database_error",
                    "message": "Failed to save user information to database",
                    "code": "AUTH_DATABASE_ERROR"
                }
            )
        
        if not user:
            logger.error(f"User service returned None for email: {email}")
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "user_creation_failed",
                    "message": "Failed to create or update user account",
                    "code": "AUTH_USER_CREATION_FAILED"
                }
            )
        
        # Create JWT token for our app
        try:
            jwt_token = create_access_token(data={"sub": str(user.id), "email": user.email})
        except Exception as e:
            logger.error(f"Failed to create JWT token: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "token_creation_failed",
                    "message": "Failed to create authentication token",
                    "code": "AUTH_TOKEN_CREATION_FAILED"
                }
            )
        
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
        
    except HTTPException:
        # Re-raise HTTPExceptions as they already have proper error formatting
        raise
    except Exception as e:
        logger.error(f"Unexpected OAuth callback error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error": "internal_server_error",
                "message": "An unexpected error occurred during authentication",
                "code": "AUTH_INTERNAL_ERROR"
            }
        )

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
        raise HTTPException(
            status_code=400,
            detail={
                "error": "password_mismatch",
                "message": "Password and confirm password do not match",
                "code": "AUTH_PASSWORD_MISMATCH"
            }
        )
    
    # Validate password strength
    is_strong, errors = password_hasher.validate_password_strength(user_data.password)
    if not is_strong:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "weak_password",
                "message": f"Password requirements not met: {', '.join(errors)}",
                "code": "AUTH_WEAK_PASSWORD",
                "validation_errors": errors
            }
        )
    
    try:
        # Create user
        try:
            user = await user_service.register_user(db, user_data)
        except Exception as e:
            # Check if it's a duplicate email error
            error_str = str(e).lower()
            if "duplicate" in error_str or "unique constraint" in error_str or "already exists" in error_str:
                logger.warning(f"Registration attempt with existing email: {user_data.email}")
                raise HTTPException(
                    status_code=409,
                    detail={
                        "error": "email_already_exists",
                        "message": "An account with this email address already exists",
                        "code": "AUTH_EMAIL_EXISTS"
                    }
                )
            else:
                logger.error(f"Database error during user registration: {str(e)}")
                raise HTTPException(
                    status_code=500,
                    detail={
                        "error": "database_error",
                        "message": "Failed to create user account due to database error",
                        "code": "AUTH_DATABASE_ERROR"
                    }
                )
        
        if not user:
            logger.warning(f"User service returned None for registration: {user_data.email}")
            raise HTTPException(
                status_code=409,
                detail={
                    "error": "email_already_exists",
                    "message": "An account with this email address already exists",
                    "code": "AUTH_EMAIL_EXISTS"
                }
            )
        
        # Create JWT token
        try:
            jwt_token = create_access_token(data={"sub": str(user.id), "email": user.email})
        except Exception as e:
            logger.error(f"Failed to create JWT token during registration: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "token_creation_failed",
                    "message": "Account created successfully but failed to generate authentication token",
                    "code": "AUTH_TOKEN_CREATION_FAILED"
                }
            )
        
        # Convert user to UserPublic
        try:
            user_public = UserPublic(
                id=user.id,
                email=user.email,
                name=user.name,
                avatar_url=user.avatar_url,
                is_active=user.is_active,
                subscription_plan=user.subscription_plan,
                created_at=user.created_at
            )
        except Exception as e:
            logger.error(f"Failed to convert user to UserPublic: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "user_serialization_failed",
                    "message": "Account created successfully but failed to format user data",
                    "code": "AUTH_USER_SERIALIZATION_FAILED"
                }
            )
        
        return AuthResponse(
            message="Registration successful",
            user=user_public,
            access_token=jwt_token,
            token_type="bearer"
        )
        
    except HTTPException:
        # Re-raise HTTPExceptions as they already have proper error formatting
        raise
    except Exception as e:
        logger.error(f"Unexpected registration error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error": "internal_server_error",
                "message": "An unexpected error occurred during registration",
                "code": "AUTH_INTERNAL_ERROR"
            }
        )


@router.post("/login", response_model=AuthResponse)
async def login_user(
    user_data: UserLogin,
    db: AsyncSession = Depends(get_async_session),
    user_service: UserService = Depends(get_user_service)
):
    """Login user with email and password"""
    
    try:
        # Authenticate user
        try:
            user = await user_service.authenticate_user(db, user_data.email, user_data.password)
        except Exception as e:
            logger.error(f"Database error during authentication: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "database_error",
                    "message": "Authentication service temporarily unavailable",
                    "code": "AUTH_DATABASE_ERROR"
                }
            )
        
        if not user:
            logger.warning(f"Failed login attempt for email: {user_data.email}")
            raise HTTPException(
                status_code=401,
                detail={
                    "error": "invalid_credentials",
                    "message": "Invalid email or password",
                    "code": "AUTH_INVALID_CREDENTIALS"
                }
            )
        
        # Check if user account is active
        if not user.is_active:
            logger.warning(f"Login attempt for inactive account: {user_data.email}")
            raise HTTPException(
                status_code=403,
                detail={
                    "error": "account_inactive",
                    "message": "Your account has been deactivated. Please contact support.",
                    "code": "AUTH_ACCOUNT_INACTIVE"
                }
            )
        
        # Create JWT token
        try:
            jwt_token = create_access_token(data={"sub": str(user.id), "email": user.email})
        except Exception as e:
            logger.error(f"Failed to create JWT token during login: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "token_creation_failed",
                    "message": "Authentication successful but failed to generate access token",
                    "code": "AUTH_TOKEN_CREATION_FAILED"
                }
            )
        
        # Convert user to UserPublic
        try:
            user_public = UserPublic(
                id=user.id,
                email=user.email,
                name=user.name,
                avatar_url=user.avatar_url,
                is_active=user.is_active,
                subscription_plan=user.subscription_plan,
                created_at=user.created_at
            )
        except Exception as e:
            logger.error(f"Failed to convert user to UserPublic during login: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "user_serialization_failed",
                    "message": "Authentication successful but failed to format user data",
                    "code": "AUTH_USER_SERIALIZATION_FAILED"
                }
            )
        
        return AuthResponse(
            message="Login successful",
            user=user_public,
            access_token=jwt_token,
            token_type="bearer"
        )
        
    except HTTPException:
        # Re-raise HTTPExceptions as they already have proper error formatting
        raise
    except Exception as e:
        logger.error(f"Unexpected login error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error": "internal_server_error",
                "message": "An unexpected error occurred during login",
                "code": "AUTH_INTERNAL_ERROR"
            }
        )


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
        try:
            user = await user_service.get_user_by_email(db, request_data.email)
        except Exception as e:
            logger.error(f"Database error while checking user existence: {str(e)}")
            # Still return success message for security (don't reveal database errors)
            return MessageResponse(message="If this email is registered, you will receive an OTP shortly")
        
        if not user:
            logger.info(f"Password reset requested for non-existent email: {request_data.email}")
            # Don't reveal if user exists or not for security
            return MessageResponse(message="If this email is registered, you will receive an OTP shortly")
        
        # Check if user account is active
        if not user.is_active:
            logger.warning(f"Password reset requested for inactive account: {request_data.email}")
            # Still return success message for security
            return MessageResponse(message="If this email is registered, you will receive an OTP shortly")
        
        # Generate and send OTP
        try:
            otp = await otp_service.generate_otp(db, request_data.email)
        except Exception as e:
            logger.error(f"Failed to generate OTP for {request_data.email}: {str(e)}")
            # Still return success message for security
            return MessageResponse(message="If this email is registered, you will receive an OTP shortly")
        
        if not otp:
            logger.error(f"OTP service returned None for {request_data.email}")
            # Still return success message for security
            return MessageResponse(message="If this email is registered, you will receive an OTP shortly")
        
        try:
            email_sent = await email_service.send_otp_email(request_data.email, otp)
        except Exception as e:
            logger.error(f"Failed to send OTP email to {request_data.email}: {str(e)}")
            # Still return success message for security
            return MessageResponse(message="If this email is registered, you will receive an OTP shortly")
        
        if not email_sent:
            logger.error(f"Email service returned False for OTP email to {request_data.email}")
            # Still return success message for security
        
        logger.info(f"Password reset OTP sent successfully to {request_data.email}")
        return MessageResponse(message="If this email is registered, you will receive an OTP shortly")
        
    except Exception as e:
        logger.error(f"Unexpected error in forgot password: {str(e)}", exc_info=True)
        # Always return success message for security, even on unexpected errors
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
        raise HTTPException(
            status_code=400,
            detail={
                "error": "password_mismatch",
                "message": "New password and confirm password do not match",
                "code": "AUTH_PASSWORD_MISMATCH"
            }
        )
    
    # Validate password strength
    is_strong, errors = password_hasher.validate_password_strength(reset_data.new_password)
    if not is_strong:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "weak_password",
                "message": f"Password requirements not met: {', '.join(errors)}",
                "code": "AUTH_WEAK_PASSWORD",
                "validation_errors": errors
            }
        )
    
    try:
        # Verify OTP and get associated email
        try:
            email = await otp_service.verify_otp_and_get_email(db, reset_data.otp)
        except Exception as e:
            logger.error(f"Database error during OTP verification: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "database_error",
                    "message": "OTP verification service temporarily unavailable",
                    "code": "AUTH_DATABASE_ERROR"
                }
            )
        
        if not email:
            logger.warning(f"Invalid OTP provided: {reset_data.otp[:4]}****")
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "invalid_otp",
                    "message": "Invalid or expired OTP. Please request a new password reset.",
                    "code": "AUTH_INVALID_OTP"
                }
            )
        
        # Reset password using the email from OTP
        try:
            success = await user_service.reset_password(db, email, reset_data.new_password)
        except Exception as e:
            logger.error(f"Database error during password reset for {email}: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "database_error",
                    "message": "Password reset service temporarily unavailable",
                    "code": "AUTH_DATABASE_ERROR"
                }
            )
        
        if not success:
            logger.error(f"Password reset failed for email {email} - user not found")
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "user_not_found",
                    "message": "User account not found. The account may have been deleted.",
                    "code": "AUTH_USER_NOT_FOUND"
                }
            )
        
        logger.info(f"Password reset successful for email: {email}")
        return MessageResponse(message="Password reset successfully")
        
    except HTTPException:
        # Re-raise HTTPExceptions as they already have proper error formatting
        raise
    except Exception as e:
        logger.error(f"Unexpected password reset error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error": "internal_server_error",
                "message": "An unexpected error occurred during password reset",
                "code": "AUTH_INTERNAL_ERROR"
            }
        )


@router.get("/me", response_model=UserPublic)
async def get_current_user(
    db: AsyncSession = Depends(get_async_session),
    user_service: UserService = Depends(get_user_service),
    current_user_id: int = Depends(get_current_user_id)
):
    """Get current user profile (requires authentication)"""
    
    try:
        try:
            user = await user_service.get_user_by_id(db, current_user_id)
        except Exception as e:
            logger.error(f"Database error during user profile retrieval for user {current_user_id}: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "database_error",
                    "message": "User profile service temporarily unavailable",
                    "code": "AUTH_DATABASE_ERROR"
                }
            )
        
        if not user:
            logger.warning(f"User profile not found for user ID: {current_user_id}")
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "user_not_found",
                    "message": "User profile not found. The account may have been deleted.",
                    "code": "AUTH_USER_NOT_FOUND"
                }
            )
        
        try:
            return UserPublic(
                id=user.id,
                email=user.email,
                name=user.name,
                avatar_url=user.avatar_url,
                is_active=user.is_active,
                subscription_plan=user.subscription_plan,
                created_at=user.created_at
            )
        except Exception as e:
            logger.error(f"Error serializing user profile for user {current_user_id}: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "serialization_error",
                    "message": "Failed to format user profile data",
                    "code": "AUTH_SERIALIZATION_ERROR"
                }
            )
        
    except HTTPException:
        # Re-raise HTTPExceptions as they already have proper error formatting
        raise
    except Exception as e:
        logger.error(f"Unexpected error getting user profile for user {current_user_id}: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error": "internal_server_error",
                "message": "An unexpected error occurred while retrieving user profile",
                "code": "AUTH_INTERNAL_ERROR"
            }
        )


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
    try:
        # Check configuration status
        google_oauth_configured = bool(settings.GOOGLE_CLIENT_ID and settings.GOOGLE_CLIENT_SECRET)
        email_service_configured = bool(email_service.smtp_username and email_service.smtp_password)
        
        return {
            "status": "healthy",
            "google_oauth_configured": google_oauth_configured,
            "email_service_configured": email_service_configured,
            "features": {
                "oauth_login": True,
                "manual_registration": True,
                "password_reset": True
            }
        }
    except Exception as e:
        logger.error(f"Health check error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error": "health_check_failed",
                "message": "Authentication service health check failed",
                "code": "AUTH_HEALTH_ERROR"
            }
        )
