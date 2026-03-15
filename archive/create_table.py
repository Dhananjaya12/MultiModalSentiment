from db_connection import get_connection

conn = get_connection()
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS mosei_samples (

    id              TEXT PRIMARY KEY,
    text            TEXT,
    audio_features  FLOAT[][],   -- shape (500, 74)  : 500 timesteps, 74 audio features
    vision_features FLOAT[][],   -- shape (500, 713) : 500 timesteps, 713 vision features
    sentiment_label FLOAT        -- continuous score -3 to +3, NOT INT

);
""")

conn.commit()

cursor.close()
conn.close()

print("Table created successfully!")