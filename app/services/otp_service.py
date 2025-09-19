import secrets
from datetime import datetime, timedelta

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.otp import OTPVerification


class OTPService:
    """Service for managing OTP operations with database storage"""
    
    @staticmethod
    async def generate_otp(
        db: AsyncSession, 
        email: str, 
        length: int = 6, 
        expiry_minutes: int = 10
    ) -> str:
        """Generate and store OTP in database"""
        
        # Generate random OTP
        otp = ''.join([str(secrets.randbelow(10)) for _ in range(length)])
        
        # Calculate expiry time
        expires_at = datetime.utcnow() + timedelta(minutes=expiry_minutes)
        
        # Delete any existing OTPs for this email
        await OTPService.clear_otp(db, email)
        
        # Create new OTP record
        otp_record = OTPVerification(
            email=email,
            otp=otp,
            expires_at=expires_at,
            used=False
        )
        
        db.add(otp_record)
        await db.commit()
        await db.refresh(otp_record)
        
        return otp
    
    @staticmethod
    async def verify_otp(db: AsyncSession, email: str, otp: str) -> bool:
        """Verify OTP from database"""
        
        # Find the OTP record
        result = await db.execute(
            select(OTPVerification)
            .where(
                OTPVerification.email == email,
                OTPVerification.otp == otp,
                ~OTPVerification.used,
                OTPVerification.expires_at > datetime.utcnow()
            )
        )
        otp_record = result.scalar_one_or_none()
        
        if not otp_record:
            return False
        
        # Mark OTP as used
        otp_record.used = True
        await db.commit()
        
        return True
    
    @staticmethod
    async def verify_otp_and_get_email(db: AsyncSession, otp: str) -> str | None:
        """Verify OTP and return the associated email if valid"""
        
        # Find the OTP record by OTP code only
        result = await db.execute(
            select(OTPVerification)
            .where(
                OTPVerification.otp == otp,
                ~OTPVerification.used,
                OTPVerification.expires_at > datetime.utcnow()
            )
        )
        otp_record = result.scalar_one_or_none()
        
        if not otp_record:
            return None
        
        # Mark OTP as used
        otp_record.used = True
        await db.commit()
        
        return otp_record.email
    
    @staticmethod
    async def clear_otp(db: AsyncSession, email: str) -> None:
        """Clear all OTPs for an email"""
        await db.execute(
            delete(OTPVerification).where(OTPVerification.email == email)
        )
        await db.commit()
    
    @staticmethod
    async def is_otp_valid(db: AsyncSession, email: str) -> bool:
        """Check if there's a valid unused OTP for an email"""
        result = await db.execute(
            select(OTPVerification)
            .where(
                OTPVerification.email == email,
                ~OTPVerification.used,
                OTPVerification.expires_at > datetime.utcnow()
            )
        )
        return result.scalar_one_or_none() is not None
    
    @staticmethod
    async def cleanup_expired_otps(db: AsyncSession) -> int:
        """Clean up expired OTPs and return count of deleted records"""
        result = await db.execute(
            delete(OTPVerification)
            .where(OTPVerification.expires_at < datetime.utcnow())
        )
        await db.commit()
        return result.rowcount
