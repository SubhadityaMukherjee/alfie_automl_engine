# Tests to add for app/tabular_automl/db.py
# engine config from env: respects DATABASE_URL and sets check_same_thread=False for SQLite.
# metadata creation: Base.metadata.create_all creates automl_sessions table.
# model schema: AutoMLSession columns and nullability match expectations.
# created_at default: auto-populates on insert (UTC, non-null).
# SessionLocal CRUD: insert, fetch, and rollback on exception work.
# primary key/uniqueness: duplicate session_id insert fails as expected.