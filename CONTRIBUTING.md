# Contributing to Speculative Cascade

We welcome contributions to the Speculative Cascade project! This document provides guidelines for contributing.

## Development Setup

1. **Clone the repository**:
```bash
git clone https://github.com/anacronic-io/speculative-cascade.git
cd speculative-cascade
```

2. **Create a virtual environment**:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
pip install -e .  # Install in development mode
```

4. **Install pre-commit hooks**:
```bash
pip install pre-commit
pre-commit install
```

## Code Style

We use the following tools for code quality:

- **Black**: Code formatting
- **isort**: Import sorting
- **flake8**: Linting
- **mypy**: Type checking

Run before committing:
```bash
black src/ tests/
isort src/ tests/
flake8 src/ tests/
mypy src/
```

## Testing

Run tests with pytest:
```bash
pytest tests/ -v
pytest tests/ --cov=speculative_cascade  # With coverage
```

## Contribution Process

1. **Fork the repository** on GitHub
2. **Create a feature branch**: `git checkout -b feature/my-feature`
3. **Make your changes** with clear commit messages
4. **Add tests** for new functionality
5. **Run tests** and ensure they pass
6. **Update documentation** if needed
7. **Submit a pull request**

## Pull Request Guidelines

- Provide a clear description of the changes
- Reference any related issues
- Ensure all tests pass
- Maintain or improve code coverage
- Follow the existing code style

## Reporting Issues

When reporting issues, please include:

- Python version
- JAX/TPU version
- Operating system
- Clear steps to reproduce
- Expected vs actual behavior
- Error messages/stack traces

## Areas for Contribution

We especially welcome contributions in:

- Performance optimizations
- Additional model architectures
- Benchmark improvements
- Documentation enhancements
- Bug fixes

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Contact

- GitHub Issues: https://github.com/anacronic-io/speculative-cascade/issues
- Email: marco@anachroni.co
