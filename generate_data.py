import mysql.connector
import random
from datetime import datetime, timedelta

# Connect to DB
conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="Shravi@123",
    database="order_db"
)

cursor = conn.cursor()

# Sample data
payment_types = ["COD", "Prepaid"]
statuses = ["Delivered", "Returned"]
customer_ids = [1, 2]

start_date = datetime(2026, 1, 1)

# Generate 500 rows
for i in range(300, 800):   # 500 rows

    order_date = start_date + timedelta(days=random.randint(0, 90))
    
    delay_days = random.randint(1, 10)
    delivery_date = order_date + timedelta(days=delay_days)

    payment = random.choice(payment_types)

    attempts = random.randint(1, 5)

    # Logic: higher delay & COD → more chance of return
    if payment == "COD" and delay_days > 5:
        status = "Returned"
    else:
        status = random.choice(statuses)

    cursor.execute("""
        INSERT INTO orders 
        (order_id, customer_id, order_date, delivery_date, payment_type, attempts, status, delay, risk, action)
        VALUES (%s,%s,%s,%s,%s,%s,%s,NULL,NULL,NULL)
    """, (
        i,
        random.choice(customer_ids),
        order_date.strftime('%Y-%m-%d'),
        delivery_date.strftime('%Y-%m-%d'),
        payment,
        attempts,
        status
    ))

conn.commit()
print("✅ 500 rows inserted!")

cursor.close()
conn.close()