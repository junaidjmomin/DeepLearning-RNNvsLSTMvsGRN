CREATE DATABASE IF NOT EXISTS studentdb;
USE studentdb;
CREATE TABLE IF NOT EXISTS students (
  id INT AUTO_INCREMENT PRIMARY KEY,
  name VARCHAR(100) NOT NULL,
  email VARCHAR(150) NOT NULL UNIQUE,
  marks INT NOT NULL
);
-- Check saved records after using the form:
SELECT id, name, email, marks FROM students ORDER BY id DESC;
