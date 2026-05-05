CREATE DATABASE order_db;
USE order_db;

CREATE TABLE orders (
  order_id INT PRIMARY KEY,
  customer_id INT,
  order_date DATE,
  delivery_date DATE,
  payment_type VARCHAR(10),
  attempts INT,
  status VARCHAR(20)
);

CREATE TABLE customers (
  customer_id INT PRIMARY KEY,
  city VARCHAR(50),
  state VARCHAR(50)
);

ALTER TABLE orders ADD delay INT;
ALTER TABLE orders ADD risk VARCHAR(10);
ALTER TABLE orders ADD action VARCHAR(20);