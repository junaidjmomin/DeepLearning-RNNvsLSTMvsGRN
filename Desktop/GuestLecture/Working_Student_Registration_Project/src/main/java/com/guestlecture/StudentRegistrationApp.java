package com.guestlecture;

import javafx.application.Application;
import javafx.geometry.Insets;
import javafx.scene.Scene;
import javafx.scene.control.*;
import javafx.scene.layout.VBox;
import javafx.stage.Stage;
import java.sql.SQLException;

public class StudentRegistrationApp extends Application {
    private final StudentRepository repository = new StudentRepository();
    private final ListView<String> savedStudents = new ListView<>();
    private final Label status = new Label("Enter student details, then click Save.");

    @Override
    public void start(Stage stage) {
        Label title = new Label("Student Registration Form");
        title.setStyle("-fx-font-size: 22px; -fx-font-weight: bold;");
        TextField name = new TextField();
        name.setPromptText("Full name");
        TextField email = new TextField();
        email.setPromptText("Email address");
        TextField marks = new TextField();
        marks.setPromptText("Marks (0-100)");

        Button save = new Button("Save Student");
        save.setMaxWidth(Double.MAX_VALUE);
        save.setOnAction(event -> {
            String n = name.getText().trim();
            String e = email.getText().trim();
            String m = marks.getText().trim();
            if (n.isEmpty() || e.isEmpty() || m.isEmpty()) {
                status.setText("All fields are required."); return;
            }
            if (!e.matches("^[^\\s@]+@[^\\s@]+\\.[^\\s@]+$")) {
                status.setText("Enter a valid email address."); return;
            }
            int score;
            try { score = Integer.parseInt(m); }
            catch (NumberFormatException ex) {
                status.setText("Marks must be a whole number."); return;
            }
            if (score < 0 || score > 100) {
                status.setText("Marks must be between 0 and 100."); return;
            }
            try {
                repository.save(n, e, score);
                status.setText("Saved successfully: " + n);
                name.clear(); email.clear(); marks.clear();
                refresh();
            } catch (SQLException ex) {
                status.setText("Database error: " + ex.getMessage());
            }
        });

        Button reload = new Button("Refresh Saved Students");
        reload.setOnAction(event -> refresh());
        savedStudents.setPrefHeight(180);
        VBox root = new VBox(10, title, new Label("Name"), name,
            new Label("Email"), email, new Label("Marks"), marks,
            save, status, new Label("Saved Students"), reload, savedStudents);
        root.setPadding(new Insets(20));
        root.setStyle("-fx-font-family: 'Arial'; -fx-font-size: 13px;");
        stage.setTitle("JavaFX + JDBC | Student Registration");
        stage.setScene(new Scene(root, 520, 620));
        stage.show();
        refresh();
    }

    private void refresh() {
        try { savedStudents.getItems().setAll(repository.findAll()); }
        catch (SQLException ex) { status.setText("Database error: " + ex.getMessage()); }
    }

    public static void main(String[] args) { launch(args); }
}
