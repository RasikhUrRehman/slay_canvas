from datetime import datetime, timedelta
from typing import Any, Dict, Optional

import httpx
from authlib.integrations.httpx_client import AsyncOAuth2Client
from fastapi import HTTPException, status
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models.user import User
from app.schemas.user import GoogleUserInfo, OAuthUserCreate, UserCreate
from app.services.user_service import UserService

# Configure bcrypt with explicit settings to handle compatibility issues
pwd_context = CryptContext(
    schemes=["bcrypt"], 
    deprecated="auto",
    bcrypt__rounds=12,
    bcrypt__ident="2b"
)


class AuthService:
    def __init__(self):
        self.user_service = UserService()
    
    def _validate_password_length(self, password: str) -> str:
        """Validate and truncate password if needed for bcrypt compatibility."""
        if not password:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Password cannot be empty"
            )
        
        # bcrypt has a 72-byte limit, truncate if necessary
        password_bytes = password.encode('utf-8')
        if len(password_bytes) > 72:
            # Truncate to 72 bytes
            password = password_bytes[:72].decode('utf-8', errors='ignore')
        
        return password
        
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash."""
        try:
            plain_password = self._validate_password_length(plain_password)
            return pwd_context.verify(plain_password, hashed_password)
        except Exception as e:
            print(f"Password verification error: {e}")
            return False
    
    def get_password_hash(self, password: str) -> str:
        """Generate password hash."""
        try:
            password = self._validate_password_length(password)
            return pwd_context.hash(password)
        except Exception as e:
            print(f"Password hashing error: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Error processing password"
            )
    
    def create_access_token(self, data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
        """Create JWT access token."""
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
        
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.JWT_ALGORITHM)
        return encoded_jwt
    
    async def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify JWT token and return payload."""
        try:
            payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.JWT_ALGORITHM])
            return payload
        except JWTError:
            return None
    
    async def authenticate_user(self, db: AsyncSession, email: str, password: str) -> Optional[User]:
        """Authenticate user with email and password."""
        try:
            user = await self.user_service.get_user_by_email(db, email)
            if not user or not user.hashed_password:
                return None
            if not self.verify_password(password, user.hashed_password):
                return None
            return user
        except Exception as e:
            print(f"Authentication error: {e}")
            return None
    
    async def register_user(self, db: AsyncSession, user_data: UserCreate) -> User:
        """Register a new user."""
        try:
            # Check if user already exists
            existing_user = await self.user_service.get_user_by_email(db, user_data.email)
            if existing_user:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Email already registered"
                )
            
            # Hash password if provided
            hashed_password = None
            if user_data.password:
                hashed_password = self.get_password_hash(user_data.password)
            
            # Create user
            user = User(
                email=user_data.email,
                full_name=user_data.full_name,
                hashed_password=hashed_password,
                is_active=True,
                is_verified=False,
                provider="email" if user_data.password else None
            )
            
            db.add(user)
            await db.commit()
            await db.refresh(user)
            return user
        except HTTPException:
            # Re-raise HTTP exceptions as-is
            raise
        except Exception as e:
            print(f"Registration error: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Registration failed. Please try again."
            )


class GoogleOAuthService:
    def __init__(self):
        self.client_id = settings.GOOGLE_CLIENT_ID
        self.client_secret = settings.GOOGLE_CLIENT_SECRET
        self.redirect_uri = settings.GOOGLE_REDIRECT_URI
        self.auth_service = AuthService()
        
    def get_authorization_url(self) -> str:
        """Get Google OAuth authorization URL."""
        if not self.client_id or not self.client_secret:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Google OAuth not configured"
            )
        
        oauth_client = AsyncOAuth2Client(
            client_id=self.client_id,
            redirect_uri=self.redirect_uri,
        )
        
        authorization_url, state = oauth_client.create_authorization_url(
            'https://accounts.google.com/o/oauth2/auth',
            scope=['openid', 'email', 'profile']
        )
        
        return authorization_url
    
    async def exchange_code_for_token(self, code: str) -> Dict[str, Any]:
        """Exchange authorization code for access token."""
        oauth_client = AsyncOAuth2Client(
            client_id=self.client_id,
            client_secret=self.client_secret,
            redirect_uri=self.redirect_uri,
        )
        
        token = await oauth_client.fetch_token(
            'https://oauth2.googleapis.com/token',
            code=code
        )
        
        return token
    
    async def get_user_info(self, access_token: str) -> GoogleUserInfo:
        """Get user info from Google API."""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                'https://www.googleapis.com/oauth2/v2/userinfo',
                headers={'Authorization': f'Bearer {access_token}'}
            )
            
            if response.status_code != 200:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Failed to fetch user info from Google"
                )
            
            user_data = response.json()
            return GoogleUserInfo(**user_data)
    
    async def authenticate_or_create_user(self, db: AsyncSession, google_user: GoogleUserInfo) -> User:
        """Authenticate existing user or create new one from Google OAuth."""
        user_service = UserService()
        
        # First, try to find user by Google ID
        user = await user_service.get_user_by_google_id(db, google_user.id)
        
        if not user:
            # Try to find user by email
            user = await user_service.get_user_by_email(db, google_user.email)
            
            if user:
                # User exists with email but no Google ID, link accounts
                user.google_id = google_user.id
                user.provider = "google"
                user.avatar_url = google_user.picture
                user.is_verified = google_user.verified_email
                user.last_login = datetime.utcnow()
                await db.commit()
                await db.refresh(user)
            else:
                # Create new user
                oauth_user_data = OAuthUserCreate(
                    email=google_user.email,
                    full_name=google_user.name,
                    google_id=google_user.id,
                    avatar_url=google_user.picture,
                    provider="google"
                )
                
                user = User(
                    email=oauth_user_data.email,
                    full_name=oauth_user_data.full_name,
                    google_id=oauth_user_data.google_id,
                    avatar_url=oauth_user_data.avatar_url,
                    provider=oauth_user_data.provider,
                    is_active=True,
                    is_verified=google_user.verified_email,
                    last_login=datetime.utcnow()
                )
                
                db.add(user)
                await db.commit()
                await db.refresh(user)
        else:
            # Update last login and user info
            user.last_login = datetime.utcnow()
            user.avatar_url = google_user.picture
            user.full_name = google_user.name
            user.is_verified = google_user.verified_email
            await db.commit()
            await db.refresh(user)
        
        return user


# Legacy function for backward compatibility
def create_access_token(subject: str, expires_delta: int = 60 * 60 * 24 * 7) -> str:
    """Legacy function - use AuthService.create_access_token instead."""
    auth_service = AuthService()
    return auth_service.create_access_token(
        {"sub": subject}, 
        timedelta(seconds=expires_delta)
    )


# Dependency to get current user from JWT token
async def get_current_user(db: AsyncSession, token: str) -> User:
    """Get current user from JWT token."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    auth_service = AuthService()
    payload = await auth_service.verify_token(token)
    
    if payload is None:
        raise credentials_exception
    
    user_id: int = payload.get("sub")
    if user_id is None:
        raise credentials_exception
    
    user_service = UserService()
    user = await user_service.get_user_by_id(db, user_id)
    
    if user is None:
        raise credentials_exception
    
    return user


async def login_or_create_google_user(userinfo: dict, db: AsyncSession = None) -> str:
    """Given Google userinfo, ensure a local user exists and return a JWT."""
    email = userinfo.get('email')
    async with db.begin():
        result = await db.execute(select(User).where(User.email == email))
        user = result.scalar_one_or_none()
        if not user:
            user = User(
                email=email,
                full_name=userinfo.get('name'),
                avatar_url=userinfo.get('picture'),
                provider='google',
            )
            db.add(user)
            await db.flush()
    token = create_access_token(subject=str(user.id))
    return token
