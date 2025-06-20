# Morph AI

A Django-based web application with user authentication, PostgreSQL database, and modern Bootstrap frontend.

## Features

- Custom User model with email authentication
- PostgreSQL database integration
- Bootstrap 5 responsive frontend
- User registration and login system
- Media and static file handling
- GitHub Actions CI/CD pipeline
- Security best practices

## Setup Instructions

### Prerequisites

- Python 3.11+
- PostgreSQL
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd morph_ai
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Setup environment variables**
   ```bash
   cp .env.example .env
   # Edit .env file with your database credentials and secret key
   ```

5. **Setup PostgreSQL database**
   ```sql
   CREATE DATABASE morph_ai;
   CREATE USER your_username WITH PASSWORD 'your_password';
   GRANT ALL PRIVILEGES ON DATABASE morph_ai TO your_username;
   ```

6. **Run migrations**
   ```bash
   python manage.py makemigrations
   python manage.py migrate
   ```

7. **Create superuser**
   ```bash
   python manage.py createsuperuser
   ```

8. **Collect static files**
   ```bash
   python manage.py collectstatic
   ```

9. **Run development server**
   ```bash
   python manage.py runserver
   ```

## Project Structure

```
morph_ai/
├── accounts/              # User authentication app
├── morph_ai/             # Main project settings
├── templates/            # HTML templates
├── static/              # Static files (CSS, JS, images)
├── media/               # User uploaded files
├── .github/workflows/   # GitHub Actions CI/CD
├── requirements.txt     # Python dependencies
├── .env.example        # Environment variables template
└── manage.py           # Django management script
```

## Environment Variables

Copy `.env.example` to `.env` and configure:

- `SECRET_KEY`: Django secret key
- `DEBUG`: Debug mode (True/False)
- `ALLOWED_HOSTS`: Comma-separated allowed hosts
- `DB_NAME`: Database name
- `DB_USER`: Database user
- `DB_PASSWORD`: Database password
- `DB_HOST`: Database host
- `DB_PORT`: Database port

## Deployment

The project includes GitHub Actions workflow for CI/CD. Configure your deployment platform with the necessary environment variables.

## Development

### Running Tests
```bash
python manage.py test
```

### Creating Migrations
```bash
python manage.py makemigrations
python manage.py migrate
```

### Accessing Admin Panel
Visit `/admin/` and use your superuser credentials.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests
5. Submit a pull request

## License

This project is licensed under the MIT License.