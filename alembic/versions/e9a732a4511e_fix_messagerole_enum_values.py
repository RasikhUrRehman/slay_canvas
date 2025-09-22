"""fix_messagerole_enum_values

Revision ID: e9a732a4511e
Revises: ecd19be23516
Create Date: 2025-09-22 23:10:23.222471

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'e9a732a4511e'
down_revision = 'ecd19be23516'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # First, create the new enum type with lowercase values
    op.execute("CREATE TYPE messagerole_new AS ENUM ('user', 'agent')")
    
    # Add a temporary column with the new enum type
    op.execute("ALTER TABLE messages ADD COLUMN role_new messagerole_new")
    
    # Update the new column based on the old column values
    op.execute("UPDATE messages SET role_new = 'user' WHERE role = 'USER'")
    op.execute("UPDATE messages SET role_new = 'agent' WHERE role = 'AGENT'")
    
    # Drop the old column and rename the new one
    op.execute("ALTER TABLE messages DROP COLUMN role")
    op.execute("ALTER TABLE messages RENAME COLUMN role_new TO role")
    
    # Drop the old enum type and rename the new one
    op.execute("DROP TYPE messagerole")
    op.execute("ALTER TYPE messagerole_new RENAME TO messagerole")


def downgrade() -> None:
    # Create the old enum type with uppercase values
    op.execute("CREATE TYPE messagerole_old AS ENUM ('USER', 'AGENT')")
    
    # Add a temporary column with the old enum type
    op.execute("ALTER TABLE messages ADD COLUMN role_old messagerole_old")
    
    # Update the temporary column based on current values
    op.execute("UPDATE messages SET role_old = 'USER' WHERE role = 'user'")
    op.execute("UPDATE messages SET role_old = 'AGENT' WHERE role = 'agent'")
    
    # Drop the current column and rename the temporary one
    op.execute("ALTER TABLE messages DROP COLUMN role")
    op.execute("ALTER TABLE messages RENAME COLUMN role_old TO role")
    
    # Drop the new enum type and rename the old one back
    op.execute("DROP TYPE messagerole")
    op.execute("ALTER TYPE messagerole_old RENAME TO messagerole")
