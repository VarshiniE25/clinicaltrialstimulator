"""
database/db_connection.py
==========================
Returns a MySQL connection.  Credentials are loaded from config.py
which lives in the project root.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import mysql.connector
import config


def get_connection():
    """Create and return a MySQL connection using settings in config.py."""
    return mysql.connector.connect(
        host=config.DB_HOST,
        port=config.DB_PORT,
        user=config.DB_USER,
        password=config.DB_PASSWORD,
        database=config.DB_NAME,
        autocommit=False,
        auth_plugin='mysql_native_password',
    )
