# Tempo: Productivity and Time Management Agent Scaffold

This repository provides a ready-to-use implementation scaffold for Tempo, a productivity and time management agent. Includes Docker configuration, database setup, API scaffolding, and frontend components to accelerate development process.

## Prerequisites

Before you begin, ensure you have the following software installed:

-   **Git**: For cloning the repository. ([Installation Guide](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git))
-   **Docker**: For running containerized applications. ([Installation Guide](https://docs.docker.com/engine/install/))
-   **Docker Compose**: For defining and running multi-container Docker applications. ([Installation Guide](https://docs.docker.com/compose/install/))

## Getting Started

1. Clone this repository:
   ```bash
   git clone <repository_url>
   ```
   Replace `<repository_url>` with the actual URL of this repository.
2. **Install Prerequisites**: Ensure you have installed all the software listed in the "Prerequisites" section above.
3. **Configure Environment**:
   - This project relies on environment variables for its configuration. A template, `.env.example`, is available in the project's root directory.
   - Create your own environment file by copying the template:
     ```bash
     cp .env.example .env
     ```
   - Modify the variables in your new `.env` file to suit your setup. Detailed explanations of these variables are in the "Configuration" section.
4. **Start Services**: Launch the entire application stack using Docker Compose:
   ```bash
   sudo docker-compose up
   ```
5. **Access Services**:
   - **API Documentation**: Available at [http://localhost:8000/docs](http://localhost:8000/docs)
   - **Frontend Dashboard**: Available at [http://localhost:3000](http://localhost:3000)

## Configuration

Effective configuration is key to tailoring this Tempo scaffold to your needs. Environment variables allow customization of database credentials, API ports, service integrations, and more, without altering the core codebase.

### Setting up .env file

1.  **Template File**: A file named `.env.example` is located in the root directory of the project. This file serves as a template listing the available environment variables and their default or example values.
2.  **Create `.env` File**: Make a copy of `.env.example` and name it `.env`. You can do this with the following command in your terminal:
    ```bash
    cp .env.example .env
    ```
3.  **Customize Variables**: Open the `.env` file in a text editor and modify the variable values according to your local setup or deployment environment.

### Key Environment Variables

Below is a list of key environment variables. While `docker-compose.yml` provides defaults for some, it's best to define them explicitly in your `.env` file.

**Database (PostgreSQL):**

*   `POSTGRES_USER`: Username for the PostgreSQL database.
    *   The `docker-compose.yml` file uses `tempo` by default for the service.
    *   The `.env.example` suggests `admin`.
    *   **Recommendation**: Use `tempo` for consistency with the Docker setup, or ensure `db/init-postgres.sql` and `docker-compose.yml` are updated if you choose a different username.
*   `POSTGRES_PASSWORD`: Password for the PostgreSQL database. Set a strong password, especially for production. (A default is provided in `docker-compose.yml` if not set in `.env`).
*   `POSTGRES_DB`: Name of the PostgreSQL database (e.g., `tempo_db`).
*   `DB_HOST`: Hostname for the database server. Defaults to `db` (the service name in Docker Compose). Change this if your PostgreSQL instance is external.
*   `DB_PORT`: Port for the database server. Defaults to `5432`.

**Backend API:**

*   `API_HOST`: Host address for the backend API to bind to. `0.0.0.0` allows access from any network interface.
*   `API_PORT`: Port for the backend API (e.g., `8000`).
*   `SECRET_KEY`: A secret key for security functions like signing JWTs. **It is crucial to set a strong, unique secret for production environments.** (A default is provided in `docker-compose.yml`).
*   `DATABASE_URL`: The full connection string for the database, typically constructed from other `POSTGRES_*` and `DB_*` variables. Example: `postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@${DB_HOST}:${DB_PORT}/${POSTGRES_DB}`.

**Frontend (React):**

*   `FRONTEND_PORT`: Port for the React development server (e.g., `3000`).
*   `REACT_APP_API_BASE_URL`: The base URL for the backend API that the frontend will connect to (e.g., `http://localhost:8000`). This is used by the frontend application.
    *   Note: `docker-compose.yml` sets `REACT_APP_API_URL` for the frontend service, which might be used internally within the Docker network. For local development and access via your browser, `REACT_APP_API_BASE_URL` in your `.env` file is typically what you'll configure. Ensure these are aligned if both are used.

**Message Broker (RabbitMQ):**

*   `RABBITMQ_DEFAULT_USER`: Username for RabbitMQ. (Default in `docker-compose.yml` is `tempo`).
*   `RABBITMQ_PASSWORD`: Password for RabbitMQ. Set a strong password. (A default is provided in `docker-compose.yml` if not set in `.env`).
*   `RABBITMQ_URL`: The full connection string for RabbitMQ, typically constructed. Example: `amqp://${RABBITMQ_DEFAULT_USER}:${RABBITMQ_PASSWORD}@rabbitmq:5672/`. (The hostname `rabbitmq` refers to the service name in Docker Compose).

**Important Security Note**: Always ensure your `.env` file is added to your project's `.gitignore` file to prevent committing sensitive credentials to your version control system. The provided `.gitignore` file should already include `.env`.

## Directory Structure

The project's root directory is structured as follows:
```
.
├── README.md                   # This documentation
├── docker-compose.yml          # Container orchestration
├── .env.example                # Environment variables template
├── backend/                    # Contains the FastAPI backend application code.
│   ├── Dockerfile              # Python service container
│   ├── requirements.txt        # Python dependencies
│   ├── app/                    # Application code
│   │   ├── main.py             # FastAPI entry point
│   │   ├── models/             # Pydantic data models
│   │   ├── routers/            # API endpoints
│   │   ├── services/           # Business logic
│   │   ├── ml/                 # Machine learning components
│   │   └── utils/              # Helper functions
├── frontend/                   # Contains the React frontend application code.
│   ├── Dockerfile              # Node.js container
│   ├── package.json            # JS dependencies
│   ├── src/                    # React application
│   │   ├── components/         # UI components
│   │   ├── pages/              # Page layouts
│   │   ├── services/           # API clients
│   │   └── utils/              # Helper functions
├── db/                         # Contains database initialization scripts.
│   ├── init-postgres.sql       # PostgreSQL schema
│   └── init-timescaledb.sql    # TimescaleDB extensions
└── scripts/                    # Contains utility scripts for development and setup.
    ├── setup.sh                # Initial setup script
    └── seed-data.py            # Sample data generator
```

## Included Components

This implementation tool provides:

1.  **Docker Environment**: Pre-configured containers for all services (backend, frontend, database, message broker). This ensures a consistent development and deployment environment using `docker-compose`.
2.  **Database Setup**: PostgreSQL with the TimescaleDB extension for efficient time-series data handling. Includes an initial schema definition in `db/init-postgres.sql` and TimescaleDB setup in `db/init-timescaledb.sql`.
3.  **Redis Configuration**: Configured for caching to improve API response times and for real-time operations like session management or task queues.
4.  **FastAPI Backend**: A Python-based backend using FastAPI, providing a structured and high-performance API. Includes Pydantic models for data validation (in `backend/app/models/`), API endpoint definitions (in `backend/app/routers/`), and service logic (in `backend/app/services/`).
5.  **React Frontend**: A JavaScript-based frontend built with React. Includes reusable UI components (in `frontend/src/components/`), page layouts (in `frontend/src/pages/`), and services for interacting with the backend API (in `frontend/src/services/`).
6.  **ML Scaffolding**: A basic structure within the backend (`backend/app/ml/`) for integrating machine learning models, particularly for time-series analysis and prediction tasks relevant to Tempo.
7.  **RabbitMQ Integration**: RabbitMQ is set up as a message broker for asynchronous task processing and event-driven communication between services. This helps in decoupling components and improving scalability.

## Phase 1 Implementation Focus

This scaffold is pre-configured for a Phase 1 Minimum Viable Product (MVP) implementation of Tempo, focusing on:

- Core calendar and task management functionality
- Basic time tracking and analysis
- Simple productivity metrics
- User authentication framework
- Essential integrations with external services

## Next Steps

After deploying this scaffold:

1. Customize the data models for your specific requirements
2. Implement your ML models in the provided structure
3. Extend the API endpoints for additional functionality
4. Customize the frontend components for your UI needs
5. Add your specific external integrations

## Scaling to Later Phases

This scaffold is designed with future scalability in mind. Features and architectural changes anticipated for later phases (Phase 2 and 3) include:

-   **TimescaleDB Advanced Features**: Leveraging more of TimescaleDB's capabilities for complex time-series queries and data management (currently enabled with minimal configuration).
-   **Kafka Integration**: Introducing Kafka for more robust, high-throughput event streaming (current `docker-compose.yml` may contain commented-out Kafka configurations as a starting point).
-   **Kubernetes Deployment**: Future plans include providing Kubernetes deployment templates (e.g., in a `kubernetes/` directory, to be added later) for orchestrating the application in a cloud-native environment.
-   **Advanced ML Pipeline**: Developing a more sophisticated machine learning pipeline (e.g., within `backend/app/ml/advanced/`, to be added later) for advanced analytics and predictive features.

## Support and Further Documentation

For more specific details on each component, refer to any README files or documentation within their respective directories (e.g., `backend/README.md`, `frontend/README.md` if they exist).
