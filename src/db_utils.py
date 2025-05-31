import sqlite3
import os

DB_PATH = "/app/predictions/predictions.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            forecast_date TEXT NOT NULL,
            temperature REAL NOT NULL,
            humidity REAL NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    conn.commit()
    conn.close()

def save_prediction(forecast_date, temperature, humidity):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO predictions (forecast_date, temperature, humidity)
        VALUES (?, ?, ?);
    """, (forecast_date, temperature, humidity))
    conn.commit()
    conn.close()

def get_latest_prediction():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT forecast_date, temperature, humidity
        FROM predictions
        ORDER BY created_at DESC
        LIMIT 1;
    """)
    row = cursor.fetchone()
    conn.close()
    return row
