"""
Test script to verify database and OAuth functionality
"""
import asyncio
import httpx
from sqlalchemy import text
from app.db.session import async_engine, async_session_factory

async def test_database_connection():
    """Test database connection"""
    print("🔍 Testing database connection...")
    try:
        async with async_engine.begin() as conn:
            result = await conn.execute(text("SELECT 1 as test"))
            row = result.fetchone()
            print(f"✅ Database connection successful: {row}")
        return True
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return False

async def check_users_table():
    """Check if users table exists and show structure"""
    print("\n🔍 Checking users table...")
    try:
        async with async_engine.begin() as conn:
            # Check if table exists
            result = await conn.execute(text("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    AND table_name = 'users'
                );
            """))
            exists = result.fetchone()[0]
            
            if exists:
                print("✅ Users table exists")
                
                # Get table structure
                result = await conn.execute(text("""
                    SELECT column_name, data_type, is_nullable
                    FROM information_schema.columns
                    WHERE table_name = 'users'
                    ORDER BY ordinal_position;
                """))
                columns = result.fetchall()
                
                print("📋 Table structure:")
                for col in columns:
                    print(f"  - {col[0]}: {col[1]} ({'NULL' if col[2] == 'YES' else 'NOT NULL'})")
                
                # Count existing users
                result = await conn.execute(text("SELECT COUNT(*) FROM users"))
                count = result.fetchone()[0]
                print(f"👥 Current users count: {count}")
                
                # Show sample users if any
                if count > 0:
                    result = await conn.execute(text("SELECT id, email, name, google_id FROM users LIMIT 5"))
                    users = result.fetchall()
                    print("📄 Sample users:")
                    for user in users:
                        print(f"  - ID: {user[0]}, Email: {user[1]}, Name: {user[2]}, Google ID: {user[3]}")
            else:
                print("❌ Users table does not exist")
        return exists
    except Exception as e:
        print(f"❌ Error checking users table: {e}")
        return False

async def test_oauth_endpoints():
    """Test OAuth endpoints availability"""
    print("\n🔍 Testing OAuth endpoints...")
    
    endpoints = [
        "http://localhost:8000/",
        "http://localhost:8000/health",
        "http://localhost:8000/auth/health"
    ]
    
    async with httpx.AsyncClient() as client:
        for endpoint in endpoints:
            try:
                response = await client.get(endpoint, timeout=5.0)
                print(f"✅ {endpoint}: {response.status_code}")
                if endpoint.endswith("/health"):
                    data = response.json()
                    print(f"   📊 Health data: {data}")
            except Exception as e:
                print(f"❌ {endpoint}: {e}")

async def main():
    """Main test function"""
    print("🧪 Slay Canvas Database & OAuth Test")
    print("=" * 50)
    
    # Test database connection
    db_ok = await test_database_connection()
    
    if db_ok:
        # Check users table
        await check_users_table()
    
    # Test OAuth endpoints
    await test_oauth_endpoints()
    
    print("\n" + "=" * 50)
    print("🎯 Next Steps:")
    print("1. Visit http://localhost:8000/auth/google/login to test OAuth")
    print("2. After login, check the database again to see if user was saved")
    print("3. Use http://localhost:8000/docs to explore all API endpoints")

if __name__ == "__main__":
    asyncio.run(main())
