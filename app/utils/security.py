import hashlib
import logging
import os
import secrets
import smtplib
from datetime import datetime, timedelta
from typing import Optional

from passlib.context import CryptContext

logger = logging.getLogger(__name__)

# Password hashing context using bcrypt
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class PasswordHasher:
    """Utility class for password hashing and verification using bcrypt"""
    
    @staticmethod
    def hash_password(password: str) -> str:
        """Hash a password using bcrypt"""
        return pwd_context.hash(password)
    
    @staticmethod
    def verify_password(password: str, hashed_password: str) -> bool:
        """Verify a password against its hash using bcrypt"""
        return pwd_context.verify(password, hashed_password)
    
    @staticmethod
    def validate_password_strength(password: str) -> tuple[bool, list[str]]:
        """Validate password strength and return errors if any"""
        errors = []
        
        if len(password) < 8:
            errors.append("Password must be at least 8 characters long")
        
        if not any(c.isupper() for c in password):
            errors.append("Password must contain at least one uppercase letter")
        
        if not any(c.islower() for c in password):
            errors.append("Password must contain at least one lowercase letter")
        
        if not any(c.isdigit() for c in password):
            errors.append("Password must contain at least one number")
        
        if not any(c in "!@#$%^&*(),.?\":{}|<>" for c in password):
            errors.append("Password must contain at least one special character")
        
        return len(errors) == 0, errors


class OTPManager:
    """Utility class for OTP generation and verification using database storage"""
    
    @staticmethod
    def generate_otp(email: str, length: int = 6, expiry_minutes: int = 10) -> str:
        """
        DEPRECATED: This method is kept for backward compatibility.
        Use OTPService.generate_otp() with database session instead.
        """
        import warnings
        warnings.warn(
            "OTPManager.generate_otp is deprecated. Use OTPService.generate_otp with database session.",
            DeprecationWarning,
            stacklevel=2
        )
        # Generate OTP for backward compatibility (will be logged but not stored)
        otp = ''.join([str(secrets.randbelow(10)) for _ in range(length)])
        logger.warning(f"Using deprecated OTPManager.generate_otp for {email}. OTP: {otp}")
        return otp
    
    @staticmethod
    def verify_otp(email: str, otp: str, max_attempts: int = 3) -> bool:
        """
        DEPRECATED: This method is kept for backward compatibility.
        Use OTPService.verify_otp() with database session instead.
        """
        import warnings
        warnings.warn(
            "OTPManager.verify_otp is deprecated. Use OTPService.verify_otp with database session.",
            DeprecationWarning,
            stacklevel=2
        )
        logger.warning(f"Using deprecated OTPManager.verify_otp for {email}")
        return False
    
    @staticmethod
    def clear_otp(email: str):
        """
        DEPRECATED: This method is kept for backward compatibility.
        Use OTPService.clear_otp() with database session instead.
        """
        import warnings
        warnings.warn(
            "OTPManager.clear_otp is deprecated. Use OTPService.clear_otp with database session.",
            DeprecationWarning,
            stacklevel=2
        )
        logger.warning(f"Using deprecated OTPManager.clear_otp for {email}")
    
    @staticmethod
    def is_otp_valid(email: str) -> bool:
        """
        DEPRECATED: This method is kept for backward compatibility.
        Use OTPService.is_otp_valid() with database session instead.
        """
        import warnings
        warnings.warn(
            "OTPManager.is_otp_valid is deprecated. Use OTPService.is_otp_valid with database session.",
            DeprecationWarning,
            stacklevel=2
        )
        logger.warning(f"Using deprecated OTPManager.is_otp_valid for {email}")
        return False


class EmailService:
    """Email service for sending OTPs and notifications"""
    
    def __init__(self):
        self.smtp_server = os.getenv("SMTP_SERVER", "")
        self.smtp_port = int(os.getenv("SMTP_PORT", "587"))
        self.smtp_username = os.getenv("SMTP_USERNAME", "")
        self.smtp_password = os.getenv("SMTP_PASSWORD", "")
        
    async def send_otp_email(self, email: str, otp: str) -> bool:
        """Send OTP email to user"""
        try:
            # Always display OTP in console for development backup
            print("\n" + "="*60)
            print("🔐 PASSWORD RESET OTP")
            print("="*60)
            print(f"📧 Email: {email}")
            print(f"🔢 OTP: {otp}")
            print(f"⏰ Valid for: 10 minutes")
            print("="*60)
            
            # Try to send actual email if SMTP is configured
            if self.smtp_username and self.smtp_password:
                try:
                    # Import here to avoid any conflicts
                    from email.mime.text import MIMEText
                    
                    # Create email message
                    message = f"""Hi there!

You requested a password reset for your Slay Canvas account.

Your One-Time Password (OTP) is: {otp}

This OTP will expire in 10 minutes.

If you didn't request this password reset, please ignore this email.

Best regards,
Slay Canvas Team"""
                    
                    msg = MIMEText(message, 'plain')
                    msg['Subject'] = "Password Reset OTP - Slay Canvas"
                    msg['From'] = self.smtp_username
                    msg['To'] = email
                    
                    # Send email using Gmail SMTP
                    with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                        server.starttls()  # Enable TLS encryption
                        server.login(self.smtp_username, self.smtp_password)
                        server.send_message(msg)
                    
                    print("✅ OTP sent to your email inbox!")
                    print("📧 Check your email for the OTP")
                    logger.info(f"📧 Email with OTP sent successfully to {email}")
                    
                except Exception as email_error:
                    logger.error(f"Failed to send email: {email_error}")
                    print(f"⚠️ Email sending failed: {email_error}")
                    print("📱 Please use the OTP from console above")
            else:
                print("📧 SMTP not configured - using console display only")
                print("💡 To enable email: add SMTP credentials to .env file")
            
            print("="*60 + "\n")
            logger.info(f"🔐 PASSWORD RESET OTP for {email}: {otp} (valid for 10 minutes)")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to send OTP email to {email}: {str(e)}")
            return False


# Global instances
password_hasher = PasswordHasher()
otp_manager = OTPManager()
email_service = EmailService()
