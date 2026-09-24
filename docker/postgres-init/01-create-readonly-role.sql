-- Postgres init script — runs once on first container start.
-- Creates a dedicated SELECT-only role that the application sandbox connects as.
-- This is defense-in-depth: the application layer (guardrails) blocks
-- destructive SQL, and the DB itself rejects writes from this user.

-- Wait for the main DB to exist (created by POSTGRES_DB env var).
\connect text2sql

-- Create the readonly role. Idempotent.
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'text2sql_ro') THEN
        CREATE ROLE text2sql_ro LOGIN PASSWORD 'text2sql_ro';
    END IF;
END
$$;

-- Grant SELECT on every existing table; future tables are handled by an
-- event trigger below.
GRANT CONNECT ON DATABASE text2sql TO text2sql_ro;
GRANT USAGE ON SCHEMA public TO text2sql_ro;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO text2sql_ro;
GRANT SELECT ON ALL SEQUENCES IN SCHEMA public TO text2sql_ro;

-- Auto-grant SELECT on any future tables to the readonly role. This is what
-- keeps the role usable after the seed script creates tables.
CREATE OR REPLACE FUNCTION grant_select_on_new_tables() RETURNS event_trigger AS $$
DECLARE
    obj record;
BEGIN
    FOR obj IN SELECT * FROM pg_event_trigger_ddl_commands() WHERE object_type IN ('table', 'view')
    LOOP
        EXECUTE format('GRANT SELECT ON %s TO text2sql_ro', obj.object_identity);
    END LOOP;
END
$$ LANGUAGE plpgsql;

-- Drop any existing trigger with the same name (defensive) so we can re-run.
DROP EVENT TRIGGER IF EXISTS text2sql_ro_auto_grant;
CREATE EVENT TRIGGER text2sql_ro_auto_grant
    ON ddl_command_end
    WHEN TAG IN ('CREATE TABLE', 'CREATE VIEW', 'CREATE TABLE AS')
    EXECUTE FUNCTION grant_select_on_new_tables();

-- Belt-and-braces: explicitly revoke write privileges in case future GRANTs
-- widen the role's permissions.
REVOKE INSERT, UPDATE, DELETE, TRUNCATE, REFERENCES, TRIGGER ON ALL TABLES IN SCHEMA public FROM text2sql_ro;
REVOKE CREATE ON SCHEMA public FROM text2sql_ro;
