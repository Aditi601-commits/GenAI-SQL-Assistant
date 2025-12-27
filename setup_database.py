import sqlite3
import random
from faker import Faker
from datetime import datetime, timedelta

# Initialize Faker for generating Indian names/addresses
fake = Faker('en_IN')

def create_database():
    # Connect to SQLite database (creates the file if it doesn't exist)
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()

    print("1. Creating Tables...")
    
    # 1. Products Table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS products (
        product_id INTEGER PRIMARY KEY,
        name TEXT,
        category TEXT,
        price INTEGER,
        stock_quantity INTEGER
    )
    ''')

    # 2. Customers Table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS customers (
        customer_id INTEGER PRIMARY KEY,
        name TEXT,
        city TEXT,
        email TEXT,
        signup_date TEXT
    )
    ''')

    # 3. Sales Table 
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS sales (
        sale_id INTEGER PRIMARY KEY,
        customer_id INTEGER,
        product_id INTEGER,
        quantity INTEGER,
        total_amount INTEGER,
        sale_date TEXT,
        FOREIGN KEY (customer_id) REFERENCES customers (customer_id),
        FOREIGN KEY (product_id) REFERENCES products (product_id)
    )
    ''')
    
    conn.commit()
    return conn

def generate_data(conn):
    cursor = conn.cursor()
    print("2. Generating Fake Data...")

    # --- Generate Products ---
    categories = ['Electronics', 'Fashion', 'Home & Kitchen', 'Books', 'Beauty']
    product_names = {
        'Electronics': ['iPhone 15', 'Samsung Galaxy S24', 'Sony WH-1000XM5', 'Dell XPS 13', 'iPad Air', 'OnePlus 11', 'HP Spectre x360', 'Apple Watch Series 9'],
        'Fashion': ['Men Slim Fit Jeans', 'Nike Air Max', 'Ray-Ban Aviator', 'Puma T-Shirt', 'Women Kurta', 'Levi\'s Jacket', 'Adidas Sneakers'],
        'Home & Kitchen': ['Philips Air Fryer', 'Dyson Vacuum', 'Milton Water Bottle', 'Prestige Cooker', 'Havells Mixer Grinder', 'Crompton Ceiling Fan'],
        'Books': ['Atomic Habits', 'The Alchemist', 'Python Crash Course', 'Rich Dad Poor Dad', 'The Subtle Art of Not Giving a Damn', 'Deep Work'],
        'Beauty': ['Nivea Face Wash', 'Loreal Shampoo', 'Maybelline Lipstick', 'Lakme Foundation', 'The Body Shop Body Butter', 'Garnier Face Pack']
    }

    products = []
    for cat, items in product_names.items():
        for item in items:
            price = random.randint(500, 150000) if cat == 'Electronics' else random.randint(200, 5000)
            products.append((item, cat, price, random.randint(10, 100)))

    cursor.executemany('INSERT INTO products (name, category, price, stock_quantity) VALUES (?, ?, ?, ?)', products)
    print(f"   - Added {len(products)} products.")

    # --- Generate Customers ---
    customers = []
    for _ in range(100): # 100 Customers
        customers.append((
            fake.name(),
            fake.city(),
            fake.email(),
            fake.date_between(start_date='-2y', end_date='today').isoformat()
        ))
    
    cursor.executemany('INSERT INTO customers (name, city, email, signup_date) VALUES (?, ?, ?, ?)', customers)
    print(f"   - Added 100 customers.")

    # --- Generate Sales ---
    # Valid IDs to link them
    cursor.execute('SELECT product_id, price FROM products')
    product_list = cursor.fetchall() # List of (id, price)
    
    cursor.execute('SELECT customer_id FROM customers')
    customer_ids = [row[0] for row in cursor.fetchall()]

    sales = []
    for _ in range(1000): # 1000 Sales transactions
        prod_id, price = random.choice(product_list)
        cust_id = random.choice(customer_ids)
        qty = random.randint(1, 3)
        total = price * qty
        date = fake.date_between(start_date='-1y', end_date='today').isoformat()
        
        sales.append((cust_id, prod_id, qty, total, date))

    cursor.executemany('INSERT INTO sales (customer_id, product_id, quantity, total_amount, sale_date) VALUES (?, ?, ?, ?, ?)', sales)
    print(f"   - Added 1000 sales records.")

    conn.commit()
    print("3. Database Setup Complete!")

if __name__ == "__main__":
    connection = create_database()
    generate_data(connection)
    connection.close()