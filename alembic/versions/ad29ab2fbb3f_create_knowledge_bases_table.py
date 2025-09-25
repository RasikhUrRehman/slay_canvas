"""create_knowledge_bases_table

Revision ID: ad29ab2fbb3f
Revises: e9a732a4511e
Create Date: 2025-09-23 00:24:39.642463

"""
import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision = 'ad29ab2fbb3f'
down_revision = 'e9a732a4511e'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create knowledge_bases table
    op.create_table('knowledge_bases',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('name', sa.String(), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('collection_name', sa.String(), nullable=False),
        sa.Column('chunk_size', sa.Integer(), nullable=True),
        sa.Column('chunk_overlap', sa.Integer(), nullable=True),
        sa.Column('embedding_model', sa.String(), nullable=True),
        sa.Column('settings', sa.JSON(), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=True),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('workspace_id', sa.Integer(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
        sa.ForeignKeyConstraint(['workspace_id'], ['workspaces.id'], ),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('collection_name')
    )
    op.create_index(op.f('ix_knowledge_bases_id'), 'knowledge_bases', ['id'], unique=False)
    op.create_index(op.f('ix_knowledge_bases_name'), 'knowledge_bases', ['name'], unique=False)
    op.create_index(op.f('ix_knowledge_bases_collection_name'), 'knowledge_bases', ['collection_name'], unique=False)
    op.create_index(op.f('ix_knowledge_bases_user_id'), 'knowledge_bases', ['user_id'], unique=False)
    
    # Add knowledge_base_id column to conversations table
    op.add_column('conversations', sa.Column('knowledge_base_id', sa.Integer(), nullable=True))
    op.create_index(op.f('ix_conversations_knowledge_base_id'), 'conversations', ['knowledge_base_id'], unique=False)
    op.create_foreign_key(None, 'conversations', 'knowledge_bases', ['knowledge_base_id'], ['id'])


def downgrade() -> None:
    # Remove foreign key and column from conversations
    op.drop_constraint(None, 'conversations', type_='foreignkey')
    op.drop_index(op.f('ix_conversations_knowledge_base_id'), table_name='conversations')
    op.drop_column('conversations', 'knowledge_base_id')
    
    # Drop knowledge_bases table
    op.drop_index(op.f('ix_knowledge_bases_user_id'), table_name='knowledge_bases')
    op.drop_index(op.f('ix_knowledge_bases_collection_name'), table_name='knowledge_bases')
    op.drop_index(op.f('ix_knowledge_bases_name'), table_name='knowledge_bases')
    op.drop_index(op.f('ix_knowledge_bases_id'), table_name='knowledge_bases')
    op.drop_table('knowledge_bases')
