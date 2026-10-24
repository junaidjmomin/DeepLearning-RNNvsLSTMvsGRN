package com.guestlecture;

import java.sql.*;
import java.util.ArrayList;
import java.util.List;

public class StudentRepository {
    private static final String URL = "jdbc:mysql://localhost:3306/studentdb";

    private Connection connect() throws SQLException {
        String user = System.getenv("DB_USER");
        String password = System.getenv("DB_PASSWORD");
        if (user == null || user.isBlank() || password == null) {
            throw new SQLException("Set DB_USER and DB_PASSWORD environment variables before running.");
        }
        return DriverManager.getConnection(URL, user, password);
    }

    public void save(String name, String email, int marks) throws SQLException {
        String sql = "INSERT INTO students (name, email, marks) VALUES (?, ?, ?)";
        try (Connection con = connect(); PreparedStatement ps = con.prepareStatement(sql)) {
            ps.setString(1, name);
            ps.setString(2, email);
            ps.setInt(3, marks);
            ps.executeUpdate();
        }
    }

    public List<String> findAll() throws SQLException {
        List<String> students = new ArrayList<>();
        String sql = "SELECT id, name, email, marks FROM students ORDER BY id DESC";
        try (Connection con = connect();
             PreparedStatement ps = con.prepareStatement(sql);
             ResultSet rs = ps.executeQuery()) {
            while (rs.next()) {
                students.add("#" + rs.getInt("id") + " | " + rs.getString("name")
                    + " | " + rs.getString("email") + " | " + rs.getInt("marks") + " marks");
            }
        }
        return students;
    }
}
