# Student Registration — JavaFX + JDBC + MySQL

## Requirements
- JDK 21 (not just a JRE)
- Maven 3.9+ (`mvn -version`)
- MySQL Server 8+ running locally, and MySQL Workbench or the mysql CLI
- Internet on first Maven build to fetch JavaFX and MySQL Connector/J

Maven downloads JavaFX and Connector/J automatically; no manual JAR paths or `--module-path` arguments are required for this Maven project. The uploaded teaching handouts instead describe manual SDK/JAR configuration.

## Database preparation
Open MySQL Workbench, connect to your local server, open `sql/setup.sql`, and run the CREATE DATABASE and CREATE TABLE statements. This creates `studentdb.students` with `id`, `name`, `email`, `marks`.

For a classroom demo, you may use your local MySQL account. A separate low-privilege MySQL account is recommended for real deployments. Do not commit credentials to source control.

## PowerShell (Windows)
In the project root:
```powershell
$env:DB_USER="root"
$env:DB_PASSWORD="YOUR_MYSQL_PASSWORD"
mvn clean javafx:run
```
Environment variables apply to this PowerShell session. Set the actual password locally; do not project it on the classroom screen.

## macOS / Linux
```bash
export DB_USER=root
export DB_PASSWORD='YOUR_MYSQL_PASSWORD'
mvn clean javafx:run
```

## Test
1. Type name `Amit`, email `amit@example.com`, marks `85`.
2. Click Save Student. Expect a success message and a row in Saved Students.
3. In Workbench run `USE studentdb; SELECT * FROM students;`.
4. Try an empty name, nonnumeric marks, marks 101, and a duplicate email. These should not insert records.

## Architecture
`StudentRegistrationApp` (JavaFX UI and validation) -> `StudentRepository` (JDBC and PreparedStatement) -> MySQL `studentdb.students`.

## Notes
- JavaFX runs database calls on its UI thread in this minimal teaching example; real applications should use a JavaFX `Task` or background executor to prevent UI freezes.
- The email check is basic classroom validation, not comprehensive address verification.
- `PreparedStatement` binds values and avoids SQL injection via input concatenation.
- `Class.forName("com.mysql.cj.jdbc.Driver")` is optional for modern JDBC with Connector/J on the classpath; older course notes include it explicitly.
- JavaFX and JDBC are distinct: JavaFX builds the desktop UI; JDBC communicates with the database.
