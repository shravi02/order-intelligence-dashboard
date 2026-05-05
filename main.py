import mysql.connector

# Connect to DB
conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="Shravi@123",
    database="order_db"
)

cursor = conn.cursor(dictionary=True)

# Fetch orders
cursor.execute("SELECT * FROM orders")
orders = cursor.fetchall()

# Functions
def calculate_delay(order_date, delivery_date):
    return (delivery_date - order_date).days

def rto_risk(delay, attempts, payment):
    if payment == "COD" and delay > 4 and attempts > 2:
        return "High"
    elif delay > 3 or attempts > 2:
        return "Medium"
    else:
        return "Low"

def decision(risk):
    if risk == "High":
        return "Verify Order"
    elif risk == "Medium":
        return "Prioritize Delivery"
    else:
        return "Normal"

# Process orders
for order in orders:
    delay = calculate_delay(order['order_date'], order['delivery_date'])
    risk = rto_risk(delay, order['attempts'], order['payment_type'])
    action = decision(risk)

    cursor.execute("""
        UPDATE orders
        SET delay=%s, risk=%s, action=%s
        WHERE order_id=%s
    """, (delay, risk, action, order['order_id']))

# Save updates
conn.commit()

print("✅ Done!")

# =========================
# 📊 BUSINESS METRICS
# =========================

# RTO %
cursor.execute("SELECT COUNT(*) FROM orders WHERE status='Returned'")
rto = cursor.fetchone()['COUNT(*)']

cursor.execute("SELECT COUNT(*) FROM orders")
total = cursor.fetchone()['COUNT(*)']

print("Total Orders:", total)
print("Returned Orders:", rto)
print("RTO %:", (rto/total)*100)

# Average delay
cursor.execute("SELECT AVG(delay) FROM orders")
avg_delay = cursor.fetchone()['AVG(delay)']
print("Average Delay:", round(avg_delay, 2))

# High risk orders
cursor.execute("SELECT COUNT(*) FROM orders WHERE risk='High'")
high_risk = cursor.fetchone()['COUNT(*)']
print("High Risk Orders:", high_risk)

# =========================
# 📍 REGION ANALYSIS (if added)
# =========================

cursor.execute("""
SELECT region, COUNT(*) as total,
SUM(CASE WHEN status='Returned' THEN 1 ELSE 0 END) as returns
FROM orders
GROUP BY region
""")

regions = cursor.fetchall()

for r in regions:
    print(f"Region: {r['region']}, Orders: {r['total']}, Returns: {r['returns']}")

# =========================
# 🔥 TOP HIGH RISK ORDERS
# =========================

cursor.execute("""
SELECT order_id, delay, attempts
FROM orders
WHERE risk='High'
ORDER BY delay DESC
LIMIT 5
""")

top_orders = cursor.fetchall()

print("\nTop High-Risk Orders:")
for o in top_orders:
    print(o)

# =========================
# 🔒 CLOSE CONNECTION (LAST LINE ONLY)
# =========================

cursor.close()
conn.close()