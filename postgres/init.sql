-- Create mlflow database
CREATE DATABASE mlflow;

-- Grant all privileges to the user for this database
GRANT ALL PRIVILEGES ON DATABASE mlflow TO ${POSTGRES_USER};