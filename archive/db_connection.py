import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("POSTGRES_CONNECTION_STRING")

def get_connection():
    conn = psycopg2.connect(DATABASE_URL)
    return conn