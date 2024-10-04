# `database.py`

## Design Considerations and Enhancements

- **Singleton Pattern**: The `DatabaseOperations` class uses a thread-safe Singleton pattern (`SingletonMeta`) to ensure only one instance of the database connection pool is created. This reduces overhead and improves performance, especially in multi-threaded environments.
- **Connection Pooling**: By utilizing connection pooling (`psycopg2.pool`), the module efficiently manages multiple database connections, reusing connections for performance optimization and minimizing the overhead of creating new connections.
- **SSH Tunneling for Secure Access**: The `sshtunnel` library is used to securely connect to the database via an SSH tunnel, ensuring that sensitive data remains protected during transmission.
- **Environment-Based Configuration**: The module loads database credentials and configuration details from environment variables using `dotenv`, keeping sensitive information out of the codebase and supporting environment-specific configurations.
- **Error Handling and Logging**: The module integrates with the `loguru` logging library to provide detailed logging of database operations, facilitating easier debugging and monitoring.

---

## Module Overview

The `database.py` module is responsible for managing the connection to the PostgreSQL database, providing secure access via SSH tunneling and efficient connection handling through connection pooling. It includes functionalities for executing database queries, managing transactions, and handling the results of queries.

### Key Libraries and Dependencies

- **`psycopg2`**: A PostgreSQL adapter for Python that provides the interface for executing SQL queries and managing connections.
  - [Psycopg2 Documentation](https://www.psycopg.org/docs/)
- **`sshtunnel`**: A library used for creating secure SSH tunnels to remote databases.
  - [SSHTunnel Documentation](https://sshtunnel.readthedocs.io/en/latest/)
- **`pandas`**: Provides powerful data manipulation capabilities, which are often used for handling database query results.
  - [Pandas Documentation](https://pandas.pydata.org/)

---

## Classes and Methods

### `SingletonMeta`

This class implements the Singleton pattern to ensure that only one instance of a class can be created. It is thread-safe and employs a locking mechanism to prevent race conditions in multi-threaded environments.

### `DatabaseOperations`

This class is responsible for managing the PostgreSQL database connection and executing queries. It uses the Singleton pattern to ensure that the connection pool is only initialized once. 

#### Constructor: `__init__(self) -> None`

Initializes the database connection pool and SSH tunnel. This method is called only once due to the Singleton pattern, ensuring that the connection pool is reused.

#### Method: `get_connection(self) -> psycopg2.extensions.connection`

Returns a connection from the connection pool. This method ensures that the connection pool is used efficiently, providing a new or existing connection for database operations.

- **Returns**: A `psycopg2` connection object.

```python
def get_connection(self) -> psycopg2.extensions.connection:
    """
    Retrieves a connection from the connection pool for executing database queries.
    """
```

#### Method: `execute_query(self, worker_name: str, query: str, params: tuple, operation: str = 'fetch') -> Optional[Union[List[Dict], int]]`

Executes a SQL query using a connection from the pool. This method can be used to retrieve results from the database in the form of a list of dictionaries, where each dictionary represents a row in the result set.

- **Parameters**:
  - `query` (str): The SQL query to execute.
  - `params` (Optional[tuple]): Parameters to pass to the SQL query (default: None).
  - `operation` (str): The type of operation ('fetch', 'commit', 'execute'). Defaults to 'fetch'.
  
- **Returns**: A list of dictionaries representing the query result.

```python
def execute_query(self, worker_name: str, query: str, params: tuple, operation: str = 'fetch') -> Optional[Union[List[Dict], int]]:
    """
    Executes a SQL query and returns the result as a list of dictionaries.
    """
```

#### Method: `close(self) -> None`

Closes the database connection pool and SSH tunnel, ensuring that all resources are properly released.

- **Returns**: None

```python
def close(self) -> None:
    """
    Closes the database connection pool and SSH tunnel.
    """
```

---

## Usage Example

Here is a basic example of how to use the `DatabaseOperations` class to execute a SQL query.

```python
from database import DatabaseOperations

# Instantiate the DatabaseOperations class (Singleton)
db_ops = DatabaseOperations()

# Connect to the database
db_ops.connect()

# Execute a query
query = "SELECT * FROM users WHERE id = %s"
params = (1,)
result = db_ops.execute_query(query, params)

# Print the result
for row in result:
    print(row)

# Close the connection
db_ops.close()
```
